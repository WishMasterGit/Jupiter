use std::sync::mpsc;
use std::time::Instant;

use jupiter_core::align::compute_offset_configured;
use jupiter_core::color::debayer::luminance;
use jupiter_core::compute::create_backend;
use jupiter_core::frame::{AlignmentOffset, ColorFrame, Frame};
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::AlignmentConfig;
use jupiter_core::pipeline::PipelineStage;

use crate::messages::WorkerResult;

use super::{send, send_error, send_log, PipelineCache};

pub(super) fn handle_align(
    select_percentage: f32,
    alignment_config: &AlignmentConfig,
    device: &jupiter_core::compute::DevicePreference,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let ranked = match &cache.ranked {
        Some(r) => r,
        None => {
            send_error(tx, ctx, "Frames not scored. Run Score Frames first.");
            return;
        }
    };

    let start = Instant::now();
    let is_streaming = cache.is_streaming;

    let (total, selected_indices) = if is_streaming {
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

    // Extract quality scores for the selected frames (for drizzle weighting)
    let quality_scores: Vec<f64> = ranked
        .iter()
        .take(frame_count)
        .map(|(_, score)| score.composite)
        .collect();

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
        send_log(
            tx,
            ctx,
            format!("Loading {frame_count} selected frames from disk..."),
        );

        if cache.is_color {
            let debayer_method = cache.debayer_method.as_ref().unwrap();

            let color_frames: Vec<ColorFrame> = match selected_indices
                .iter()
                .map(|&i| reader.read_frame_as_color(i, debayer_method))
                .collect::<jupiter_core::error::Result<_>>()
            {
                Ok(cf) => cf,
                Err(e) => {
                    send_error(
                        tx,
                        ctx,
                        format!("Failed to read selected color frames: {e}"),
                    );
                    return;
                }
            };

            let lum_frames: Vec<Frame> = color_frames.iter().map(luminance).collect();
            (lum_frames, Some(color_frames))
        } else {
            let mono_frames: Vec<Frame> = match selected_indices
                .iter()
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
        let frames = cache.all_frames.as_ref().unwrap();
        let mono: Vec<Frame> = selected_indices
            .iter()
            .map(|&i| frames[i].clone())
            .collect();

        let color = if cache.is_color {
            cache
                .all_color_frames
                .as_ref()
                .map(|cfs| selected_indices.iter().map(|&i| cfs[i].clone()).collect())
        } else {
            None
        };

        (mono, color)
    };

    send_log(
        tx,
        ctx,
        format!("Selected {frame_count}/{total} frames, aligning..."),
    );
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Alignment,
        items_done: Some(0),
        items_total: Some(frame_count),
    });

    let backend = create_backend(device);
    let reference = &selected_frames[0];

    // Compute offsets with per-frame progress
    let mut offsets = Vec::with_capacity(selected_frames.len());
    offsets.push(AlignmentOffset::default()); // reference frame
    for (i, frame) in selected_frames[1..].iter().enumerate() {
        let offset = compute_offset_configured(
            &reference.data,
            &frame.data,
            alignment_config,
            backend.as_ref(),
        )
        .unwrap_or_default();
        offsets.push(offset);
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Alignment,
            items_done: Some(i + 1),
            items_total: Some(frame_count),
        });
    }

    // Cache alignment results
    cache.selected_frames = Some(selected_frames);
    cache.selected_color_frames = selected_color;
    cache.alignment_offsets = Some(offsets);
    cache.selected_quality_scores = Some(quality_scores);
    cache.invalidate_from_stack();

    let elapsed = start.elapsed();
    send_log(
        tx,
        ctx,
        format!(
            "Aligned {frame_count} frames in {:.1}s",
            elapsed.as_secs_f32()
        ),
    );
    send(tx, ctx, WorkerResult::AlignComplete {
        frame_count,
        elapsed,
    });
}
