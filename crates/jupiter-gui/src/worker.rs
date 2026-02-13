use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use jupiter_core::align::phase_correlation::{
    align_frames_gpu_with_progress, align_frames_with_progress, compute_offset,
    compute_offset_gpu,
};
use jupiter_core::compute::create_backend;
use jupiter_core::frame::{AlignmentOffset, Frame, QualityScore};
use jupiter_core::io::image_io::save_image;
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::{QualityMetric, StackMethod};
use jupiter_core::pipeline::{apply_filter_step, run_pipeline_reported, PipelineStage};
use jupiter_core::quality::gradient::rank_frames_gradient;
use jupiter_core::quality::laplacian::rank_frames;
use jupiter_core::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use jupiter_core::sharpen::wavelet;
use jupiter_core::stack::drizzle::drizzle_stack;
use jupiter_core::stack::mean::mean_stack;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::multi_point::multi_point_stack;
use jupiter_core::stack::sigma_clip::sigma_clip_stack;

use crate::messages::{WorkerCommand, WorkerResult};
use crate::progress::ChannelProgressReporter;

/// Cached intermediate results living on the worker thread.
struct PipelineCache {
    file_path: Option<PathBuf>,
    all_frames: Option<Vec<Frame>>,
    ranked: Option<Vec<(usize, QualityScore)>>,
    stacked: Option<Frame>,
    sharpened: Option<Frame>,
    filtered: Option<Frame>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            file_path: None,
            all_frames: None,
            ranked: None,
            stacked: None,
            sharpened: None,
            filtered: None,
        }
    }

    /// Latest available frame for display/saving.
    fn latest_frame(&self) -> Option<&Frame> {
        self.filtered
            .as_ref()
            .or(self.sharpened.as_ref())
            .or(self.stacked.as_ref())
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
            WorkerCommand::LoadAndScore { path, metric } => {
                handle_load_and_score(&path, &metric, &mut cache, &tx, &ctx);
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
        }
    }
}

fn handle_load_file_info(
    path: &PathBuf,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match SerReader::open(path) {
        Ok(reader) => {
            let info = reader.source_info(path);
            send(tx, ctx, WorkerResult::FileInfo {
                path: path.clone(),
                info,
            });
        }
        Err(e) => send_error(tx, ctx, format!("Failed to open file: {e}")),
    }
}

fn handle_preview_frame(
    path: &PathBuf,
    frame_index: usize,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match SerReader::open(path) {
        Ok(reader) => match reader.read_frame(frame_index) {
            Ok(frame) => {
                send(tx, ctx, WorkerResult::FramePreview {
                    frame,
                    index: frame_index,
                });
            }
            Err(e) => send_error(tx, ctx, format!("Failed to read frame {frame_index}: {e}")),
        },
        Err(e) => send_error(tx, ctx, format!("Failed to open file: {e}")),
    }
}

fn handle_load_and_score(
    path: &PathBuf,
    metric: &QualityMetric,
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

    let frames: Vec<Frame> = match reader.frames().collect::<jupiter_core::error::Result<_>>() {
        Ok(f) => f,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to read frames: {e}"));
            return;
        }
    };

    let total = frames.len();
    send_log(tx, ctx, format!("Read {total} frames, scoring quality..."));
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::QualityAssessment,
        items_done: Some(0),
        items_total: Some(total),
    });

    let ranked = match metric {
        QualityMetric::Laplacian => rank_frames(&frames),
        QualityMetric::Gradient => rank_frames_gradient(&frames),
    };

    let ranked_preview: Vec<(usize, f64)> = ranked
        .iter()
        .take(20)
        .map(|(i, s)| (*i, s.composite))
        .collect();

    // Update cache — invalidate downstream
    cache.file_path = Some(path.clone());
    cache.all_frames = Some(frames);
    cache.ranked = Some(ranked);
    cache.stacked = None;
    cache.sharpened = None;
    cache.filtered = None;

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
    // Multi-point has its own flow
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
        match multi_point_stack(&reader, mp_config, |_| {}) {
            Ok(result) => {
                let elapsed = start.elapsed();
                cache.stacked = Some(result.clone());
                cache.sharpened = None;
                cache.filtered = None;
                send_log(tx, ctx, format!("Multi-point stacking complete in {:.1}s", elapsed.as_secs_f32()));
                send(tx, ctx, WorkerResult::StackComplete { result, elapsed });
            }
            Err(e) => send_error(tx, ctx, format!("Multi-point stacking failed: {e}")),
        }
        return;
    }

    let frames = match &cache.all_frames {
        Some(f) => f,
        None => {
            send_error(tx, ctx, "No frames loaded. Run Score Frames first.");
            return;
        }
    };
    let ranked = match &cache.ranked {
        Some(r) => r,
        None => {
            send_error(tx, ctx, "Frames not scored. Run Score Frames first.");
            return;
        }
    };

    let start = Instant::now();
    let total = frames.len();
    let keep = (total as f32 * select_percentage).ceil() as usize;
    let keep = keep.max(1).min(total);
    let selected_indices: Vec<usize> = ranked.iter().take(keep).map(|(i, _)| *i).collect();
    let selected_frames: Vec<Frame> = selected_indices.iter().map(|&i| frames[i].clone()).collect();
    let frame_count = selected_frames.len();

    send_log(tx, ctx, format!("Selected {frame_count}/{total} frames, aligning..."));
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Alignment,
        items_done: Some(0),
        items_total: Some(frame_count),
    });

    let backend = create_backend(device);

    // Drizzle path: compute offsets without shifting
    if let StackMethod::Drizzle(ref drizzle_config) = method {
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

        send_log(tx, ctx, "Drizzle stacking...");
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: None,
            items_total: None,
        });

        let quality_scores: Vec<f64> = if drizzle_config.quality_weighted {
            selected_indices
                .iter()
                .map(|&i| {
                    ranked.iter().find(|(idx, _)| *idx == i).unwrap().1.composite
                })
                .collect()
        } else {
            Vec::new()
        };
        let scores = if drizzle_config.quality_weighted {
            Some(quality_scores.as_slice())
        } else {
            None
        };

        match drizzle_stack(&selected_frames, &offsets, drizzle_config, scores) {
            Ok(result) => {
                let elapsed = start.elapsed();
                cache.stacked = Some(result.clone());
                cache.sharpened = None;
                cache.filtered = None;
                send_log(tx, ctx, format!("Drizzle stacking complete in {:.1}s", elapsed.as_secs_f32()));
                send(tx, ctx, WorkerResult::StackComplete { result, elapsed });
            }
            Err(e) => send_error(tx, ctx, format!("Drizzle stacking failed: {e}")),
        }
        return;
    }

    // Standard path: align then stack
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
                Err(e) => {
                    send_error(tx, ctx, format!("Alignment failed: {e}"));
                    return;
                }
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
                Err(e) => {
                    send_error(tx, ctx, format!("Alignment failed: {e}"));
                    return;
                }
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
            cache.sharpened = None;
            cache.filtered = None;
            send_log(tx, ctx, format!("Stacking complete in {:.1}s", elapsed.as_secs_f32()));
            send(tx, ctx, WorkerResult::StackComplete { result, elapsed });
        }
        Err(e) => send_error(tx, ctx, format!("Stacking failed: {e}")),
    }
}

fn handle_sharpen(
    config: &jupiter_core::pipeline::config::SharpeningConfig,
    device: &jupiter_core::compute::DevicePreference,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let stacked = match &cache.stacked {
        Some(f) => f.clone(),
        None => {
            send_error(tx, ctx, "No stacked frame. Run Stack first.");
            return;
        }
    };

    let start = Instant::now();
    send_log(tx, ctx, "Sharpening...");
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Sharpening,
        items_done: None,
        items_total: None,
    });

    let backend = create_backend(device);
    let mut result = stacked;

    if let Some(ref deconv_config) = config.deconvolution {
        if backend.is_gpu() {
            result = deconvolve_gpu(&result, deconv_config, &*backend);
        } else {
            result = deconvolve(&result, deconv_config);
        }
        send_log(tx, ctx, "Deconvolution complete");
    }

    result = wavelet::sharpen(&result, &config.wavelet);

    let elapsed = start.elapsed();
    cache.sharpened = Some(result.clone());
    cache.filtered = None;
    send_log(tx, ctx, format!("Sharpening complete in {:.1}s", elapsed.as_secs_f32()));
    send(tx, ctx, WorkerResult::SharpenComplete { result, elapsed });
}

fn handle_apply_filters(
    filters: &[jupiter_core::pipeline::config::FilterStep],
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    // Start from sharpened if available, else stacked
    let base = cache
        .sharpened
        .as_ref()
        .or(cache.stacked.as_ref());

    let base = match base {
        Some(f) => f.clone(),
        None => {
            send_error(tx, ctx, "No frame to filter. Run Stack or Sharpen first.");
            return;
        }
    };

    let start = Instant::now();
    send_log(tx, ctx, format!("Applying {} filters...", filters.len()));
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Filtering,
        items_done: Some(0),
        items_total: Some(filters.len()),
    });

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
    send_log(tx, ctx, format!("{} filters applied in {:.1}s", filters.len(), elapsed.as_secs_f32()));
    send(tx, ctx, WorkerResult::FilterComplete { result, elapsed });
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
        Ok(result) => {
            let elapsed = start.elapsed();
            cache.file_path = Some(config.input.clone());
            cache.stacked = Some(result.clone());
            cache.sharpened = if config.sharpening.is_some() {
                Some(result.clone())
            } else {
                None
            };
            cache.filtered = if !config.filters.is_empty() {
                Some(result.clone())
            } else {
                None
            };
            send_log(tx, ctx, format!("Pipeline complete in {:.1}s", elapsed.as_secs_f32()));
            send(tx, ctx, WorkerResult::PipelineComplete { result, elapsed });
        }
        Err(e) => send_error(tx, ctx, format!("Pipeline failed: {e}")),
    }
}

fn handle_save_image(
    path: &PathBuf,
    cache: &PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let frame = match cache.latest_frame() {
        Some(f) => f,
        None => {
            send_error(tx, ctx, "No frame to save.");
            return;
        }
    };

    match save_image(frame, path) {
        Ok(()) => {
            send_log(tx, ctx, format!("Saved to {}", path.display()));
            send(tx, ctx, WorkerResult::ImageSaved { path: path.clone() });
        }
        Err(e) => send_error(tx, ctx, format!("Failed to save: {e}")),
    }
}
