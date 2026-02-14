use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use jupiter_core::align::phase_correlation::{
    align_frames_gpu_with_progress, align_frames_with_progress, compute_offset,
    compute_offset_gpu, shift_frame,
};
use jupiter_core::color::debayer::{debayer, is_bayer, luminance, DebayerMethod};
use jupiter_core::color::process::process_color_parallel;
use jupiter_core::compute::create_backend;
use jupiter_core::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame, QualityScore};
use jupiter_core::io::crop::crop_ser;
use jupiter_core::io::image_io::{save_color_image, save_image};
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::{DebayerConfig, QualityMetric, StackMethod};
use jupiter_core::pipeline::{apply_filter_step, run_pipeline_reported, PipelineOutput, PipelineStage};
use jupiter_core::consts::{COLOR_CHANNEL_COUNT, LOW_MEMORY_THRESHOLD_BYTES};
use jupiter_core::quality::gradient::{rank_frames_gradient, rank_frames_gradient_color_streaming, rank_frames_gradient_streaming};
use jupiter_core::quality::laplacian::{rank_frames, rank_frames_color_streaming, rank_frames_streaming};
use jupiter_core::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use jupiter_core::sharpen::wavelet;
use jupiter_core::stack::drizzle::drizzle_stack;
use jupiter_core::stack::mean::mean_stack;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::multi_point::{multi_point_stack, multi_point_stack_color};
use jupiter_core::stack::sigma_clip::sigma_clip_stack;

use crate::messages::{WorkerCommand, WorkerResult};
use crate::progress::ChannelProgressReporter;

/// Cached intermediate results living on the worker thread.
struct PipelineCache {
    file_path: Option<PathBuf>,
    is_color: bool,
    is_streaming: bool,
    /// Stored color mode from the SER header, needed for re-reading color frames in streaming mode.
    color_mode: Option<ColorMode>,
    /// Stored debayer method, needed for re-reading Bayer frames in streaming mode.
    debayer_method: Option<DebayerMethod>,
    all_frames: Option<Vec<Frame>>,
    all_color_frames: Option<Vec<ColorFrame>>,
    ranked: Option<Vec<(usize, QualityScore)>>,
    stacked: Option<Frame>,
    stacked_color: Option<ColorFrame>,
    sharpened: Option<Frame>,
    sharpened_color: Option<ColorFrame>,
    filtered: Option<Frame>,
    filtered_color: Option<ColorFrame>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            file_path: None,
            is_color: false,
            is_streaming: false,
            color_mode: None,
            debayer_method: None,
            all_frames: None,
            all_color_frames: None,
            ranked: None,
            stacked: None,
            stacked_color: None,
            sharpened: None,
            sharpened_color: None,
            filtered: None,
            filtered_color: None,
        }
    }

    /// Latest available output for display/saving.
    fn latest_output(&self) -> Option<PipelineOutput> {
        if self.is_color {
            self.filtered_color.as_ref()
                .map(|cf| PipelineOutput::Color(cf.clone()))
                .or_else(|| self.sharpened_color.as_ref().map(|cf| PipelineOutput::Color(cf.clone())))
                .or_else(|| self.stacked_color.as_ref().map(|cf| PipelineOutput::Color(cf.clone())))
        } else {
            self.filtered.as_ref()
                .map(|f| PipelineOutput::Mono(f.clone()))
                .or_else(|| self.sharpened.as_ref().map(|f| PipelineOutput::Mono(f.clone())))
                .or_else(|| self.stacked.as_ref().map(|f| PipelineOutput::Mono(f.clone())))
        }
    }

    fn invalidate_downstream(&mut self) {
        self.stacked = None;
        self.stacked_color = None;
        self.sharpened = None;
        self.sharpened_color = None;
        self.filtered = None;
        self.filtered_color = None;
    }
}

/// Spawn the worker thread. Returns the command sender.
pub fn spawn_worker(
    result_tx: mpsc::Sender<WorkerResult>,
    ctx: egui::Context,
) -> mpsc::Sender<WorkerCommand> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>();

    std::thread::Builder::new()
        .name("jupiter-worker".into())
        .spawn(move || {
            worker_loop(cmd_rx, result_tx, ctx);
        })
        .expect("Failed to spawn worker thread");

    cmd_tx
}

fn send(tx: &mpsc::Sender<WorkerResult>, ctx: &egui::Context, result: WorkerResult) {
    let _ = tx.send(result);
    ctx.request_repaint();
}

fn send_log(tx: &mpsc::Sender<WorkerResult>, ctx: &egui::Context, msg: impl Into<String>) {
    send(tx, ctx, WorkerResult::Log { message: msg.into() });
}

fn send_error(tx: &mpsc::Sender<WorkerResult>, ctx: &egui::Context, msg: impl Into<String>) {
    send(tx, ctx, WorkerResult::Error { message: msg.into() });
}

fn worker_loop(
    cmd_rx: mpsc::Receiver<WorkerCommand>,
    tx: mpsc::Sender<WorkerResult>,
    ctx: egui::Context,
) {
    let mut cache = PipelineCache::new();

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            WorkerCommand::LoadFileInfo { path } => {
                handle_load_file_info(&path, &tx, &ctx);
            }
            WorkerCommand::PreviewFrame { path, frame_index } => {
                handle_preview_frame(&path, frame_index, &tx, &ctx);
            }
            WorkerCommand::LoadAndScore { path, metric, debayer } => {
                handle_load_and_score(&path, &metric, &debayer, &mut cache, &tx, &ctx);
            }
            WorkerCommand::Stack {
                select_percentage,
                method,
                device,
            } => {
                handle_stack(select_percentage, &method, &device, &mut cache, &tx, &ctx);
            }
            WorkerCommand::Sharpen { config, device } => {
                handle_sharpen(&config, &device, &mut cache, &tx, &ctx);
            }
            WorkerCommand::ApplyFilters { filters } => {
                handle_apply_filters(&filters, &mut cache, &tx, &ctx);
            }
            WorkerCommand::RunAll { config } => {
                handle_run_all(&config, &mut cache, &tx, &ctx);
            }
            WorkerCommand::SaveImage { path } => {
                handle_save_image(&path, &cache, &tx, &ctx);
            }
            WorkerCommand::CropAndSave {
                source_path,
                output_path,
                crop,
            } => {
                handle_crop_and_save(&source_path, &output_path, &crop, &tx, &ctx);
            }
        }
    }
}

fn handle_load_file_info(
    path: &Path,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match SerReader::open(path) {
        Ok(reader) => {
            let info = reader.source_info(path);
            send(tx, ctx, WorkerResult::FileInfo {
                path: path.to_path_buf(),
                info,
            });
        }
        Err(e) => send_error(tx, ctx, format!("Failed to open file: {e}")),
    }
}

fn handle_preview_frame(
    path: &Path,
    frame_index: usize,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let reader = match SerReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open file: {e}"));
            return;
        }
    };

    let color_mode = reader.header.color_mode();

    let output = if is_bayer(&color_mode) {
        match reader.read_frame_color(frame_index, &DebayerMethod::Bilinear) {
            Ok(cf) => PipelineOutput::Color(cf),
            Err(e) => {
                send_error(tx, ctx, format!("Failed to read frame {frame_index}: {e}"));
                return;
            }
        }
    } else if matches!(color_mode, ColorMode::RGB | ColorMode::BGR) {
        match reader.read_frame_rgb(frame_index) {
            Ok(cf) => PipelineOutput::Color(cf),
            Err(e) => {
                send_error(tx, ctx, format!("Failed to read frame {frame_index}: {e}"));
                return;
            }
        }
    } else {
        match reader.read_frame(frame_index) {
            Ok(frame) => PipelineOutput::Mono(frame),
            Err(e) => {
                send_error(tx, ctx, format!("Failed to read frame {frame_index}: {e}"));
                return;
            }
        }
    };

    send(tx, ctx, WorkerResult::FramePreview {
        output,
        index: frame_index,
    });
}

fn handle_load_and_score(
    path: &Path,
    metric: &QualityMetric,
    debayer_config: &Option<DebayerConfig>,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Reading,
        items_done: None,
        items_total: None,
    });
    send_log(tx, ctx, "Reading frames...");

    let reader = match SerReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open file: {e}"));
            return;
        }
    };

    let color_mode = reader.header.color_mode();
    let use_color = debayer_config.is_some()
        && (is_bayer(&color_mode) || matches!(color_mode, ColorMode::RGB | ColorMode::BGR));
    let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);
    let total = reader.frame_count();

    // Check if we should use streaming mode (large files)
    let channels: usize = if use_color { COLOR_CHANNEL_COUNT } else { 1 };
    let decoded_bytes = reader.header.width as usize
        * reader.header.height as usize
        * std::mem::size_of::<f32>()
        * channels
        * total;
    let use_streaming = decoded_bytes > LOW_MEMORY_THRESHOLD_BYTES;

    if use_streaming {
        let debayer_method = debayer_config
            .as_ref()
            .map(|c| c.method.clone())
            .unwrap_or_default();

        let mode_label = if use_color { "color " } else { "" };
        send_log(tx, ctx, format!("Scoring {total} {mode_label}frames (streaming)..."));
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::QualityAssessment,
            items_done: Some(0),
            items_total: Some(total),
        });

        let ranked = if use_color {
            match metric {
                QualityMetric::Laplacian => match rank_frames_color_streaming(&reader, &color_mode, &debayer_method) {
                    Ok(r) => r,
                    Err(e) => { send_error(tx, ctx, format!("Streaming color scoring failed: {e}")); return; }
                },
                QualityMetric::Gradient => match rank_frames_gradient_color_streaming(&reader, &color_mode, &debayer_method) {
                    Ok(r) => r,
                    Err(e) => { send_error(tx, ctx, format!("Streaming color scoring failed: {e}")); return; }
                },
            }
        } else {
            match metric {
                QualityMetric::Laplacian => match rank_frames_streaming(&reader) {
                    Ok(r) => r,
                    Err(e) => { send_error(tx, ctx, format!("Streaming scoring failed: {e}")); return; }
                },
                QualityMetric::Gradient => match rank_frames_gradient_streaming(&reader) {
                    Ok(r) => r,
                    Err(e) => { send_error(tx, ctx, format!("Streaming scoring failed: {e}")); return; }
                },
            }
        };

        let ranked_preview: Vec<(usize, f64)> = ranked
            .iter()
            .take(20)
            .map(|(i, s)| (*i, s.composite))
            .collect();

        cache.file_path = Some(path.to_path_buf());
        cache.is_color = use_color;
        cache.is_streaming = true;
        cache.color_mode = if use_color { Some(color_mode.clone()) } else { None };
        cache.debayer_method = if use_color { Some(debayer_method) } else { None };
        cache.all_frames = None; // streaming: no cached frames
        cache.all_color_frames = None;
        cache.ranked = Some(ranked);
        cache.invalidate_downstream();

        send_log(tx, ctx, format!("Scored {total} {mode_label}frames (streaming mode)"));
        send(tx, ctx, WorkerResult::LoadAndScoreComplete {
            frame_count: total,
            ranked_preview,
        });
        return;
    }

    // Eager mode: load all frames
    let frames: Vec<Frame> = match reader.frames().collect::<jupiter_core::error::Result<_>>() {
        Ok(f) => f,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to read frames: {e}"));
            return;
        }
    };

    // Debayer if needed
    let (scoring_frames, color_frames) = if use_color {
        send_log(tx, ctx, format!("Debayering {total} frames..."));
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Debayering,
            items_done: None,
            items_total: Some(total),
        });

        let debayer_method = debayer_config
            .as_ref()
            .map(|c| c.method.clone())
            .unwrap_or_default();

        let color_frames: Vec<ColorFrame> = if is_rgb_bgr {
            match (0..total).map(|i| reader.read_frame_rgb(i)).collect::<jupiter_core::error::Result<_>>() {
                Ok(cf) => cf,
                Err(e) => {
                    send_error(tx, ctx, format!("Failed to read RGB frames: {e}"));
                    return;
                }
            }
        } else {
            frames.iter().map(|frame| {
                debayer(&frame.data, &color_mode, &debayer_method, frame.original_bit_depth)
                    .expect("is_bayer check should guarantee success")
            }).collect()
        };

        let lum_frames: Vec<Frame> = color_frames.iter().map(luminance).collect();
        (lum_frames, Some(color_frames))
    } else {
        (frames.clone(), None)
    };

    send_log(tx, ctx, format!("Read {total} frames, scoring quality..."));
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::QualityAssessment,
        items_done: Some(0),
        items_total: Some(total),
    });

    let ranked = match metric {
        QualityMetric::Laplacian => rank_frames(&scoring_frames),
        QualityMetric::Gradient => rank_frames_gradient(&scoring_frames),
    };

    let ranked_preview: Vec<(usize, f64)> = ranked
        .iter()
        .take(20)
        .map(|(i, s)| (*i, s.composite))
        .collect();

    // Update cache — invalidate downstream
    cache.file_path = Some(path.to_path_buf());
    cache.is_color = use_color;
    cache.is_streaming = false;
    cache.color_mode = None;
    cache.debayer_method = None;
    cache.all_frames = Some(scoring_frames);
    cache.all_color_frames = color_frames;
    cache.ranked = Some(ranked);
    cache.invalidate_downstream();

    send_log(tx, ctx, format!("Scored {total} frames"));
    send(tx, ctx, WorkerResult::LoadAndScoreComplete {
        frame_count: total,
        ranked_preview,
    });
}

fn handle_stack(
    select_percentage: f32,
    method: &StackMethod,
    device: &jupiter_core::compute::DevicePreference,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    // Multi-point has its own flow (color or mono)
    if let StackMethod::MultiPoint(ref mp_config) = method {
        let file_path = match &cache.file_path {
            Some(p) => p.clone(),
            None => {
                send_error(tx, ctx, "No file loaded. Run Score Frames first.");
                return;
            }
        };
        send_log(tx, ctx, "Multi-point stacking...");
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: None,
            items_total: None,
        });
        let start = Instant::now();
        let reader = match SerReader::open(&file_path) {
            Ok(r) => r,
            Err(e) => {
                send_error(tx, ctx, format!("Failed to open file: {e}"));
                return;
            }
        };

        if cache.is_color {
            let color_mode = match reader.header.color_mode() {
                ColorMode::Mono => {
                    send_error(tx, ctx, "Expected color source but got mono");
                    return;
                }
                mode => mode,
            };
            let debayer_method = cache.debayer_method.clone().unwrap_or_default();
            match multi_point_stack_color(&reader, mp_config, &color_mode, &debayer_method, |_| {}) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    cache.stacked_color = Some(result.clone());
                    cache.stacked = None;
                    cache.sharpened = None;
                    cache.sharpened_color = None;
                    cache.filtered = None;
                    cache.filtered_color = None;
                    let output = PipelineOutput::Color(result);
                    send_log(tx, ctx, format!("Multi-point color stacking complete in {:.1}s", elapsed.as_secs_f32()));
                    send(tx, ctx, WorkerResult::StackComplete { result: output, elapsed });
                }
                Err(e) => send_error(tx, ctx, format!("Multi-point color stacking failed: {e}")),
            }
        } else {
            match multi_point_stack(&reader, mp_config, |_| {}) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    cache.stacked = Some(result.clone());
                    cache.stacked_color = None;
                    cache.sharpened = None;
                    cache.sharpened_color = None;
                    cache.filtered = None;
                    cache.filtered_color = None;
                    let output = PipelineOutput::Mono(result);
                    send_log(tx, ctx, format!("Multi-point stacking complete in {:.1}s", elapsed.as_secs_f32()));
                    send(tx, ctx, WorkerResult::StackComplete { result: output, elapsed });
                }
                Err(e) => send_error(tx, ctx, format!("Multi-point stacking failed: {e}")),
            }
        }
        return;
    }

    let ranked = match &cache.ranked {
        Some(r) => r,
        None => {
            send_error(tx, ctx, "Frames not scored. Run Score Frames first.");
            return;
        }
    };

    let start = Instant::now();

    // Streaming mode: all_frames is None, must load selected frames from disk
    let is_streaming = cache.is_streaming;

    let (total, selected_indices) = if is_streaming {
        // Determine total from ranked scores (all frames were scored)
        let total = ranked.len();
        let keep = (total as f32 * select_percentage).ceil() as usize;
        let keep = keep.max(1).min(total);
        let indices: Vec<usize> = ranked.iter().take(keep).map(|(i, _)| *i).collect();
        (total, indices)
    } else {
        let total = cache.all_frames.as_ref().unwrap().len();
        let keep = (total as f32 * select_percentage).ceil() as usize;
        let keep = keep.max(1).min(total);
        let indices: Vec<usize> = ranked.iter().take(keep).map(|(i, _)| *i).collect();
        (total, indices)
    };

    let frame_count = selected_indices.len();

    // Load selected frames and color frames
    let (selected_frames, selected_color): (Vec<Frame>, Option<Vec<ColorFrame>>) = if is_streaming {
        let file_path = match &cache.file_path {
            Some(p) => p.clone(),
            None => {
                send_error(tx, ctx, "No file loaded. Run Score Frames first.");
                return;
            }
        };
        let reader = match SerReader::open(&file_path) {
            Ok(r) => r,
            Err(e) => {
                send_error(tx, ctx, format!("Failed to open file: {e}"));
                return;
            }
        };
        send_log(tx, ctx, format!("Loading {frame_count} selected frames from disk..."));

        if cache.is_color {
            // Color streaming: read color frames from disk, derive luminance for alignment
            let color_mode = cache.color_mode.as_ref().unwrap();
            let debayer_method = cache.debayer_method.as_ref().unwrap();
            let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);

            let color_frames: Vec<ColorFrame> = match selected_indices.iter().map(|&i| {
                if is_rgb_bgr {
                    reader.read_frame_rgb(i)
                } else {
                    reader.read_frame_color(i, debayer_method)
                }
            }).collect::<jupiter_core::error::Result<_>>() {
                Ok(cf) => cf,
                Err(e) => {
                    send_error(tx, ctx, format!("Failed to read selected color frames: {e}"));
                    return;
                }
            };

            let lum_frames: Vec<Frame> = color_frames.iter().map(luminance).collect();
            (lum_frames, Some(color_frames))
        } else {
            // Mono streaming: read mono frames from disk
            let mono_frames: Vec<Frame> = match selected_indices.iter()
                .map(|&i| reader.read_frame(i))
                .collect::<jupiter_core::error::Result<_>>()
            {
                Ok(frames) => frames,
                Err(e) => {
                    send_error(tx, ctx, format!("Failed to read selected frames: {e}"));
                    return;
                }
            };
            (mono_frames, None)
        }
    } else {
        // Eager mode: frames already in cache
        let frames = cache.all_frames.as_ref().unwrap();
        let mono: Vec<Frame> = selected_indices.iter().map(|&i| frames[i].clone()).collect();

        let color = if cache.is_color {
            cache.all_color_frames.as_ref().map(|cfs| {
                selected_indices.iter().map(|&i| cfs[i].clone()).collect()
            })
        } else {
            None
        };

        (mono, color)
    };

    send_log(tx, ctx, format!("Selected {frame_count}/{total} frames, aligning..."));
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Alignment,
        items_done: Some(0),
        items_total: Some(frame_count),
    });

    let backend = create_backend(device);

    // Compute offsets on luminance
    let reference = &selected_frames[0];
    let offsets: Vec<AlignmentOffset> = selected_frames
        .iter()
        .enumerate()
        .map(|(i, frame)| {
            if i == 0 {
                AlignmentOffset::default()
            } else if backend.is_gpu() {
                compute_offset_gpu(&reference.data, &frame.data, backend.as_ref())
                    .unwrap_or_default()
            } else {
                compute_offset(reference, frame).unwrap_or_default()
            }
        })
        .collect();

    // Drizzle path
    if let StackMethod::Drizzle(ref drizzle_config) = method {
        send_log(tx, ctx, "Drizzle stacking...");
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: None,
            items_total: None,
        });

        let quality_scores: Vec<f64> = if drizzle_config.quality_weighted {
            selected_indices.iter()
                .map(|&i| ranked.iter().find(|(idx, _)| *idx == i).unwrap().1.composite)
                .collect()
        } else {
            Vec::new()
        };
        let scores = if drizzle_config.quality_weighted {
            Some(quality_scores.as_slice())
        } else {
            None
        };

        if let Some(ref color_frames) = selected_color {
            // Color drizzle: per-channel
            let red_frames: Vec<Frame> = color_frames.iter().map(|cf| cf.red.clone()).collect();
            let green_frames: Vec<Frame> = color_frames.iter().map(|cf| cf.green.clone()).collect();
            let blue_frames: Vec<Frame> = color_frames.iter().map(|cf| cf.blue.clone()).collect();

            let (dr, (dg, db)) = rayon::join(
                || drizzle_stack(&red_frames, &offsets, drizzle_config, scores),
                || rayon::join(
                    || drizzle_stack(&green_frames, &offsets, drizzle_config, scores),
                    || drizzle_stack(&blue_frames, &offsets, drizzle_config, scores),
                ),
            );
            match (dr, dg, db) {
                (Ok(r), Ok(g), Ok(b)) => {
                    let elapsed = start.elapsed();
                    let cf = ColorFrame { red: r, green: g, blue: b };
                    cache.stacked_color = Some(cf.clone());
                    cache.stacked = None;
                    cache.sharpened = None;
                    cache.sharpened_color = None;
                    cache.filtered = None;
                    cache.filtered_color = None;
                    let output = PipelineOutput::Color(cf);
                    send_log(tx, ctx, format!("Color drizzle complete in {:.1}s", elapsed.as_secs_f32()));
                    send(tx, ctx, WorkerResult::StackComplete { result: output, elapsed });
                }
                _ => send_error(tx, ctx, "Color drizzle stacking failed"),
            }
        } else {
            match drizzle_stack(&selected_frames, &offsets, drizzle_config, scores) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    cache.stacked = Some(result.clone());
                    cache.stacked_color = None;
                    cache.sharpened = None;
                    cache.sharpened_color = None;
                    cache.filtered = None;
                    cache.filtered_color = None;
                    let output = PipelineOutput::Mono(result);
                    send_log(tx, ctx, format!("Drizzle complete in {:.1}s", elapsed.as_secs_f32()));
                    send(tx, ctx, WorkerResult::StackComplete { result: output, elapsed });
                }
                Err(e) => send_error(tx, ctx, format!("Drizzle stacking failed: {e}")),
            }
        }
        return;
    }

    // Standard path: shift + stack
    if let Some(ref color_frames) = selected_color {
        // Color: shift per-channel, stack per-channel
        let aligned_color: Vec<ColorFrame> = color_frames.iter().zip(offsets.iter())
            .map(|(cf, offset)| ColorFrame {
                red: shift_frame(&cf.red, offset),
                green: shift_frame(&cf.green, offset),
                blue: shift_frame(&cf.blue, offset),
            })
            .collect();

        send_log(tx, ctx, "Stacking color channels...");
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: None,
            items_total: None,
        });

        let red_frames: Vec<Frame> = aligned_color.iter().map(|cf| cf.red.clone()).collect();
        let green_frames: Vec<Frame> = aligned_color.iter().map(|cf| cf.green.clone()).collect();
        let blue_frames: Vec<Frame> = aligned_color.iter().map(|cf| cf.blue.clone()).collect();

        let stack_fn = |frames: &[Frame]| -> jupiter_core::error::Result<Frame> {
            match method {
                StackMethod::Mean => mean_stack(frames),
                StackMethod::Median => median_stack(frames),
                StackMethod::SigmaClip(params) => sigma_clip_stack(frames, params),
                _ => unreachable!(),
            }
        };

        let (sr, (sg, sb)) = rayon::join(
            || stack_fn(&red_frames),
            || rayon::join(|| stack_fn(&green_frames), || stack_fn(&blue_frames)),
        );
        match (sr, sg, sb) {
            (Ok(r), Ok(g), Ok(b)) => {
                let elapsed = start.elapsed();
                let cf = ColorFrame { red: r, green: g, blue: b };
                cache.stacked_color = Some(cf.clone());
                cache.stacked = None;
                cache.sharpened = None;
                cache.sharpened_color = None;
                cache.filtered = None;
                cache.filtered_color = None;
                let output = PipelineOutput::Color(cf);
                send_log(tx, ctx, format!("Color stacking complete in {:.1}s", elapsed.as_secs_f32()));
                send(tx, ctx, WorkerResult::StackComplete { result: output, elapsed });
            }
            _ => send_error(tx, ctx, "Color stacking failed"),
        }
    } else {
        // Mono: align then stack
        let aligned = if frame_count > 1 {
            let tx_clone = tx.clone();
            let ctx_clone = ctx.clone();
            if backend.is_gpu() {
                match align_frames_gpu_with_progress(&selected_frames, 0, backend.clone(), move |done| {
                    let _ = tx_clone.send(WorkerResult::Progress {
                        stage: PipelineStage::Alignment,
                        items_done: Some(done),
                        items_total: Some(frame_count),
                    });
                    ctx_clone.request_repaint();
                }) {
                    Ok(a) => a,
                    Err(e) => { send_error(tx, ctx, format!("Alignment failed: {e}")); return; }
                }
            } else {
                match align_frames_with_progress(&selected_frames, 0, move |done| {
                    let _ = tx_clone.send(WorkerResult::Progress {
                        stage: PipelineStage::Alignment,
                        items_done: Some(done),
                        items_total: Some(frame_count),
                    });
                    ctx_clone.request_repaint();
                }) {
                    Ok(a) => a,
                    Err(e) => { send_error(tx, ctx, format!("Alignment failed: {e}")); return; }
                }
            }
        } else {
            selected_frames
        };

        send_log(tx, ctx, "Stacking...");
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: None,
            items_total: None,
        });

        let result = match method {
            StackMethod::Mean => mean_stack(&aligned),
            StackMethod::Median => median_stack(&aligned),
            StackMethod::SigmaClip(params) => sigma_clip_stack(&aligned, params),
            _ => unreachable!(),
        };

        match result {
            Ok(result) => {
                let elapsed = start.elapsed();
                cache.stacked = Some(result.clone());
                cache.stacked_color = None;
                cache.sharpened = None;
                cache.sharpened_color = None;
                cache.filtered = None;
                cache.filtered_color = None;
                let output = PipelineOutput::Mono(result);
                send_log(tx, ctx, format!("Stacking complete in {:.1}s", elapsed.as_secs_f32()));
                send(tx, ctx, WorkerResult::StackComplete { result: output, elapsed });
            }
            Err(e) => send_error(tx, ctx, format!("Stacking failed: {e}")),
        }
    }
}

fn handle_sharpen(
    config: &jupiter_core::pipeline::config::SharpeningConfig,
    device: &jupiter_core::compute::DevicePreference,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, "Sharpening...");
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Sharpening,
        items_done: None,
        items_total: None,
    });

    let backend = create_backend(device);

    let sharpen_mono = |frame: &Frame| -> Frame {
        let mut result = frame.clone();
        if let Some(ref deconv_config) = config.deconvolution {
            if backend.is_gpu() {
                result = deconvolve_gpu(&result, deconv_config, &*backend);
            } else {
                result = deconvolve(&result, deconv_config);
            }
        }
        wavelet::sharpen(&result, &config.wavelet)
    };

    if cache.is_color {
        let stacked_color = match &cache.stacked_color {
            Some(cf) => cf.clone(),
            None => {
                send_error(tx, ctx, "No stacked color frame. Run Stack first.");
                return;
            }
        };
        let sharpened = process_color_parallel(&stacked_color, |frame| sharpen_mono(frame));
        let elapsed = start.elapsed();
        cache.sharpened_color = Some(sharpened.clone());
        cache.sharpened = None;
        cache.filtered = None;
        cache.filtered_color = None;
        let output = PipelineOutput::Color(sharpened);
        send_log(tx, ctx, format!("Color sharpening complete in {:.1}s", elapsed.as_secs_f32()));
        send(tx, ctx, WorkerResult::SharpenComplete { result: output, elapsed });
    } else {
        let stacked = match &cache.stacked {
            Some(f) => f.clone(),
            None => {
                send_error(tx, ctx, "No stacked frame. Run Stack first.");
                return;
            }
        };
        let result = sharpen_mono(&stacked);
        let elapsed = start.elapsed();
        cache.sharpened = Some(result.clone());
        cache.sharpened_color = None;
        cache.filtered = None;
        cache.filtered_color = None;
        let output = PipelineOutput::Mono(result);
        send_log(tx, ctx, format!("Sharpening complete in {:.1}s", elapsed.as_secs_f32()));
        send(tx, ctx, WorkerResult::SharpenComplete { result: output, elapsed });
    }
}

fn handle_apply_filters(
    filters: &[jupiter_core::pipeline::config::FilterStep],
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, format!("Applying {} filters...", filters.len()));
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Filtering,
        items_done: Some(0),
        items_total: Some(filters.len()),
    });

    if cache.is_color {
        let base = cache.sharpened_color.as_ref()
            .or(cache.stacked_color.as_ref());
        let base = match base {
            Some(cf) => cf.clone(),
            None => {
                send_error(tx, ctx, "No color frame to filter. Run Stack or Sharpen first.");
                return;
            }
        };

        let mut result = base;
        for (i, step) in filters.iter().enumerate() {
            result = process_color_parallel(&result, |frame| apply_filter_step(frame, step));
            send(tx, ctx, WorkerResult::Progress {
                stage: PipelineStage::Filtering,
                items_done: Some(i + 1),
                items_total: Some(filters.len()),
            });
        }

        let elapsed = start.elapsed();
        cache.filtered_color = Some(result.clone());
        cache.filtered = None;
        let output = PipelineOutput::Color(result);
        send_log(tx, ctx, format!("{} color filters applied in {:.1}s", filters.len(), elapsed.as_secs_f32()));
        send(tx, ctx, WorkerResult::FilterComplete { result: output, elapsed });
    } else {
        let base = cache.sharpened.as_ref()
            .or(cache.stacked.as_ref());
        let base = match base {
            Some(f) => f.clone(),
            None => {
                send_error(tx, ctx, "No frame to filter. Run Stack or Sharpen first.");
                return;
            }
        };

        let mut result = base;
        for (i, step) in filters.iter().enumerate() {
            result = apply_filter_step(&result, step);
            send(tx, ctx, WorkerResult::Progress {
                stage: PipelineStage::Filtering,
                items_done: Some(i + 1),
                items_total: Some(filters.len()),
            });
        }

        let elapsed = start.elapsed();
        cache.filtered = Some(result.clone());
        cache.filtered_color = None;
        let output = PipelineOutput::Mono(result);
        send_log(tx, ctx, format!("{} filters applied in {:.1}s", filters.len(), elapsed.as_secs_f32()));
        send(tx, ctx, WorkerResult::FilterComplete { result: output, elapsed });
    }
}

fn handle_run_all(
    config: &jupiter_core::pipeline::config::PipelineConfig,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, "Running full pipeline...");

    let backend = create_backend(&config.device);
    let reporter = Arc::new(ChannelProgressReporter::new(tx.clone(), ctx.clone()));

    match run_pipeline_reported(config, backend, reporter) {
        Ok(output) => {
            let elapsed = start.elapsed();
            cache.file_path = Some(config.input.clone());
            match &output {
                PipelineOutput::Mono(frame) => {
                    cache.is_color = false;
                    cache.stacked = Some(frame.clone());
                    cache.stacked_color = None;
                    cache.sharpened = if config.sharpening.is_some() { Some(frame.clone()) } else { None };
                    cache.sharpened_color = None;
                    cache.filtered = if !config.filters.is_empty() { Some(frame.clone()) } else { None };
                    cache.filtered_color = None;
                }
                PipelineOutput::Color(cf) => {
                    cache.is_color = true;
                    cache.stacked_color = Some(cf.clone());
                    cache.stacked = None;
                    cache.sharpened_color = if config.sharpening.is_some() { Some(cf.clone()) } else { None };
                    cache.sharpened = None;
                    cache.filtered_color = if !config.filters.is_empty() { Some(cf.clone()) } else { None };
                    cache.filtered = None;
                }
            }
            send_log(tx, ctx, format!("Pipeline complete in {:.1}s", elapsed.as_secs_f32()));
            send(tx, ctx, WorkerResult::PipelineComplete { result: output, elapsed });
        }
        Err(e) => send_error(tx, ctx, format!("Pipeline failed: {e}")),
    }
}

fn handle_save_image(
    path: &Path,
    cache: &PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let output = match cache.latest_output() {
        Some(o) => o,
        None => {
            send_error(tx, ctx, "No frame to save.");
            return;
        }
    };

    let result = match &output {
        PipelineOutput::Mono(frame) => save_image(frame, path),
        PipelineOutput::Color(cf) => save_color_image(cf, path),
    };

    match result {
        Ok(()) => {
            send_log(tx, ctx, format!("Saved to {}", path.display()));
            send(tx, ctx, WorkerResult::ImageSaved { path: path.to_path_buf() });
        }
        Err(e) => send_error(tx, ctx, format!("Failed to save: {e}")),
    }
}

fn handle_crop_and_save(
    source_path: &Path,
    output_path: &Path,
    crop: &jupiter_core::io::crop::CropRect,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, format!(
        "Cropping to {}x{} at ({},{})...",
        crop.width, crop.height, crop.x, crop.y
    ));

    let reader = match SerReader::open(source_path) {
        Ok(r) => r,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open source: {e}"));
            return;
        }
    };

    let total = reader.frame_count();
    let tx_progress = tx.clone();
    let ctx_progress = ctx.clone();

    match crop_ser(&reader, output_path, crop, |done, total| {
        let _ = tx_progress.send(WorkerResult::Progress {
            stage: PipelineStage::Cropping,
            items_done: Some(done),
            items_total: Some(total),
        });
        ctx_progress.request_repaint();
    }) {
        Ok(()) => {
            let elapsed = start.elapsed();
            send_log(tx, ctx, format!(
                "Cropped {total} frames in {:.1}s -> {}",
                elapsed.as_secs_f32(),
                output_path.display()
            ));
            send(tx, ctx, WorkerResult::CropComplete {
                output_path: output_path.to_path_buf(),
                elapsed,
            });
        }
        Err(e) => send_error(tx, ctx, format!("Crop failed: {e}")),
    }
}
