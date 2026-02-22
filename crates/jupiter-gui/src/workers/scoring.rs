use std::path::Path;
use std::sync::mpsc;

use jupiter_core::color::debayer::{debayer, is_bayer, luminance};
use jupiter_core::consts::{COLOR_CHANNEL_COUNT, LOW_MEMORY_THRESHOLD_BYTES};
use jupiter_core::detection::{detect_planet_in_frame, DetectionConfig};
use jupiter_core::frame::{ColorFrame, ColorMode, Frame};
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::{DebayerConfig, QualityMetric};
use jupiter_core::pipeline::PipelineStage;
use jupiter_core::quality::gradient::{
    rank_frames_gradient_color_streaming_with_progress,
    rank_frames_gradient_streaming_with_progress, rank_frames_gradient_with_progress,
};
use jupiter_core::quality::laplacian::{
    rank_frames_color_streaming_with_progress, rank_frames_streaming_with_progress,
    rank_frames_with_progress,
};

use crate::messages::WorkerResult;

use super::{make_progress_callback, send, send_error, send_log, PipelineCache};

pub(super) fn handle_load_and_score(
    path: &Path,
    metric: &QualityMetric,
    debayer_config: &Option<DebayerConfig>,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    send(
        tx,
        ctx,
        WorkerResult::Progress {
            stage: PipelineStage::Reading,
            items_done: None,
            items_total: None,
        },
    );
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
            .map(|c| c.method)
            .unwrap_or_default();

        let mode_label = if use_color { "color " } else { "" };
        send_log(
            tx,
            ctx,
            format!("Scoring {total} {mode_label}frames (streaming)..."),
        );
        send(
            tx,
            ctx,
            WorkerResult::Progress {
                stage: PipelineStage::QualityAssessment,
                items_done: Some(0),
                items_total: Some(total),
            },
        );

        let streaming_progress =
            make_progress_callback(tx, ctx, PipelineStage::QualityAssessment, total);

        let ranked = if use_color {
            match metric {
                QualityMetric::Laplacian => {
                    match rank_frames_color_streaming_with_progress(
                        &reader,
                        &color_mode,
                        &debayer_method,
                        &streaming_progress,
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            send_error(tx, ctx, format!("Streaming color scoring failed: {e}"));
                            return;
                        }
                    }
                }
                QualityMetric::Gradient => {
                    match rank_frames_gradient_color_streaming_with_progress(
                        &reader,
                        &color_mode,
                        &debayer_method,
                        &streaming_progress,
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            send_error(tx, ctx, format!("Streaming color scoring failed: {e}"));
                            return;
                        }
                    }
                }
            }
        } else {
            match metric {
                QualityMetric::Laplacian => {
                    match rank_frames_streaming_with_progress(&reader, &streaming_progress) {
                        Ok(r) => r,
                        Err(e) => {
                            send_error(tx, ctx, format!("Streaming scoring failed: {e}"));
                            return;
                        }
                    }
                }
                QualityMetric::Gradient => {
                    match rank_frames_gradient_streaming_with_progress(&reader, &streaming_progress)
                    {
                        Ok(r) => r,
                        Err(e) => {
                            send_error(tx, ctx, format!("Streaming scoring failed: {e}"));
                            return;
                        }
                    }
                }
            }
        };

        let ranked_preview: Vec<(usize, f64)> =
            ranked.iter().map(|(i, s)| (*i, s.composite)).collect();

        // Detect planet diameter on first frame for auto AP size
        let detected_planet_diameter = reader
            .read_frame(0)
            .ok()
            .and_then(|f| detect_planet_diameter(&f));

        cache.file_path = Some(path.to_path_buf());
        cache.is_color = use_color;
        cache.is_streaming = true;
        cache.color_mode = if use_color {
            Some(color_mode.clone())
        } else {
            None
        };
        cache.debayer_method = if use_color {
            Some(debayer_method)
        } else {
            None
        };
        cache.all_frames = None; // streaming: no cached frames
        cache.all_color_frames = None;
        cache.ranked = Some(ranked);
        cache.invalidate_downstream();

        send_log(
            tx,
            ctx,
            format!("Scored {total} {mode_label}frames (streaming mode)"),
        );
        send(
            tx,
            ctx,
            WorkerResult::LoadAndScoreComplete {
                frame_count: total,
                ranked_preview,
                detected_planet_diameter,
            },
        );
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
        send(
            tx,
            ctx,
            WorkerResult::Progress {
                stage: PipelineStage::Debayering,
                items_done: None,
                items_total: Some(total),
            },
        );

        let debayer_method = debayer_config
            .as_ref()
            .map(|c| c.method)
            .unwrap_or_default();

        let color_frames: Vec<ColorFrame> = if is_rgb_bgr {
            match (0..total)
                .map(|i| reader.read_frame_as_color(i, &debayer_method))
                .collect::<jupiter_core::error::Result<_>>()
            {
                Ok(cf) => cf,
                Err(e) => {
                    send_error(tx, ctx, format!("Failed to read RGB frames: {e}"));
                    return;
                }
            }
        } else {
            frames
                .iter()
                .map(|frame| {
                    debayer(
                        &frame.data,
                        &color_mode,
                        &debayer_method,
                        frame.original_bit_depth,
                    )
                    .expect("is_bayer check should guarantee success")
                })
                .collect()
        };

        let lum_frames: Vec<Frame> = color_frames.iter().map(luminance).collect();
        (lum_frames, Some(color_frames))
    } else {
        (frames.clone(), None)
    };

    send_log(tx, ctx, format!("Read {total} frames, scoring quality..."));
    send(
        tx,
        ctx,
        WorkerResult::Progress {
            stage: PipelineStage::QualityAssessment,
            items_done: Some(0),
            items_total: Some(total),
        },
    );

    let eager_progress = make_progress_callback(tx, ctx, PipelineStage::QualityAssessment, total);

    let ranked = match metric {
        QualityMetric::Laplacian => rank_frames_with_progress(&scoring_frames, &eager_progress),
        QualityMetric::Gradient => {
            rank_frames_gradient_with_progress(&scoring_frames, &eager_progress)
        }
    };

    let ranked_preview: Vec<(usize, f64)> = ranked.iter().map(|(i, s)| (*i, s.composite)).collect();

    // Detect planet diameter on first frame for auto AP size
    let detected_planet_diameter = detect_planet_diameter(&scoring_frames[0]);

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
    send(
        tx,
        ctx,
        WorkerResult::LoadAndScoreComplete {
            frame_count: total,
            ranked_preview,
            detected_planet_diameter,
        },
    );
}

/// Run planet detection on a single frame and return the diameter (max of bbox width/height).
fn detect_planet_diameter(frame: &Frame) -> Option<usize> {
    let config = DetectionConfig::default();
    let detection = detect_planet_in_frame(&frame.data, 0, &config)?;
    Some(detection.bbox_width.max(detection.bbox_height))
}
