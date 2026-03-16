use std::sync::mpsc;
use std::time::Instant;

use jupiter_core::align::shift_frame;
use jupiter_core::frame::{ColorFrame, Frame};
use jupiter_core::pipeline::config::StackMethod;
use jupiter_core::pipeline::{PipelineOutput, PipelineStage};
use jupiter_core::stack::drizzle::{drizzle_single_frame, DrizzleConfig};
use jupiter_core::stack::mean::mean_stack_with_progress;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::sigma_clip::sigma_clip_stack;

use crate::messages::WorkerResult;

use super::super::{make_progress_callback_with_detail, send, send_error, send_log, PipelineCache};

pub(crate) fn handle_standard(
    method: &StackMethod,
    drizzle: Option<&DrizzleConfig>,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let selected_frames = match &cache.selected_frames {
        Some(f) => f,
        None => {
            send_error(tx, ctx, "Frames not aligned. Run Align Frames first.");
            return;
        }
    };
    let offsets = match &cache.alignment_offsets {
        Some(o) => o,
        None => {
            send_error(tx, ctx, "No alignment offsets. Run Align Frames first.");
            return;
        }
    };

    let start = Instant::now();

    if let Some(ref color_frames) = cache.selected_color_frames {
        // Color: transform per-channel (drizzle or shift), stack per-channel
        let color_frame_count = color_frames.len();
        let transform_detail = if drizzle.is_some() {
            "Drizzle-projecting frames"
        } else {
            "Shifting frames"
        };

        // Report transform progress
        let transform_progress = make_progress_callback_with_detail(
            tx,
            ctx,
            PipelineStage::Stacking,
            color_frame_count,
            Some(transform_detail.into()),
        );

        let transformed_color: Vec<ColorFrame> = match drizzle {
            Some(dc) => {
                send_log(tx, ctx, "Drizzle-projecting color frames...");
                match color_frames
                    .iter()
                    .zip(offsets.iter())
                    .enumerate()
                    .map(|(i, (cf, offset))| {
                        let result = (|| {
                            Ok(ColorFrame {
                                red: drizzle_single_frame(&cf.red, offset, dc)?,
                                green: drizzle_single_frame(&cf.green, offset, dc)?,
                                blue: drizzle_single_frame(&cf.blue, offset, dc)?,
                            })
                        })();
                        transform_progress(i + 1);
                        result
                    })
                    .collect::<jupiter_core::error::Result<Vec<_>>>()
                {
                    Ok(frames) => frames,
                    Err(e) => {
                        send_error(tx, ctx, format!("Drizzle projection failed: {e}"));
                        return;
                    }
                }
            }
            None => {
                send_log(tx, ctx, "Shifting color frames...");
                color_frames
                    .iter()
                    .zip(offsets.iter())
                    .enumerate()
                    .map(|(i, (cf, offset))| {
                        let result = ColorFrame {
                            red: shift_frame(&cf.red, offset),
                            green: shift_frame(&cf.green, offset),
                            blue: shift_frame(&cf.blue, offset),
                        };
                        transform_progress(i + 1);
                        result
                    })
                    .collect()
            }
        };

        send_log(tx, ctx, "Stacking color channels...");

        let red_frames: Vec<Frame> = transformed_color.iter().map(|cf| cf.red.clone()).collect();
        let green_frames: Vec<Frame> = transformed_color
            .iter()
            .map(|cf| cf.green.clone())
            .collect();
        let blue_frames: Vec<Frame> = transformed_color.iter().map(|cf| cf.blue.clone()).collect();

        let stacking_detail = match method {
            StackMethod::Mean => "Mean stacking",
            StackMethod::Median => "Combining pixels (median)",
            StackMethod::SigmaClip(_) => "Combining pixels (sigma clip)",
            _ => "Stacking",
        };

        let color_stack_progress = make_progress_callback_with_detail(
            tx,
            ctx,
            PipelineStage::Stacking,
            color_frame_count,
            Some(stacking_detail.into()),
        );

        let stack_fn =
            |frames: &[Frame], on_progress: &dyn Fn(usize)| -> jupiter_core::error::Result<Frame> {
                match method {
                    StackMethod::Mean => mean_stack_with_progress(frames, on_progress),
                    StackMethod::Median => {
                        let r = median_stack(frames);
                        on_progress(frames.len());
                        r
                    }
                    StackMethod::SigmaClip(params) => {
                        let r = sigma_clip_stack(frames, params);
                        on_progress(frames.len());
                        r
                    }
                    _ => unreachable!(),
                }
            };

        let (sr, (sg, sb)) = rayon::join(
            || stack_fn(&red_frames, &color_stack_progress),
            || {
                rayon::join(
                    || stack_fn(&green_frames, &|_| {}),
                    || stack_fn(&blue_frames, &|_| {}),
                )
            },
        );
        match (sr, sg, sb) {
            (Ok(r), Ok(g), Ok(b)) => {
                let elapsed = start.elapsed();
                let output = PipelineOutput::Color(ColorFrame {
                    red: r,
                    green: g,
                    blue: b,
                });
                // In streaming mode, persist to disk and free RAM
                if cache.is_streaming {
                    if let Some(disk_result) = cache.store_output_to_disk(&output) {
                        cache.set_stacked_disk(disk_result);
                    } else {
                        cache.set_stacked(output.clone());
                    }
                } else {
                    cache.set_stacked(output.clone());
                }
                send_log(
                    tx,
                    ctx,
                    format!("Color stacking complete in {:.1}s", elapsed.as_secs_f32()),
                );
                send(
                    tx,
                    ctx,
                    WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    },
                );
            }
            _ => send_error(tx, ctx, "Color stacking failed"),
        }
    } else {
        // Mono: transform (drizzle or shift) then stack
        let frame_count = selected_frames.len();
        let transform_detail = if drizzle.is_some() {
            "Drizzle-projecting frames"
        } else {
            "Shifting frames"
        };

        let transform_progress = make_progress_callback_with_detail(
            tx,
            ctx,
            PipelineStage::Stacking,
            frame_count,
            Some(transform_detail.into()),
        );

        let transformed: Vec<Frame> = match drizzle {
            Some(dc) => {
                send_log(tx, ctx, "Drizzle-projecting frames...");
                match selected_frames
                    .iter()
                    .zip(offsets.iter())
                    .enumerate()
                    .map(|(i, (frame, offset))| {
                        let result = drizzle_single_frame(frame, offset, dc);
                        transform_progress(i + 1);
                        result
                    })
                    .collect::<jupiter_core::error::Result<Vec<_>>>()
                {
                    Ok(frames) => frames,
                    Err(e) => {
                        send_error(tx, ctx, format!("Drizzle projection failed: {e}"));
                        return;
                    }
                }
            }
            None => {
                send_log(tx, ctx, "Shifting frames...");
                selected_frames
                    .iter()
                    .zip(offsets.iter())
                    .enumerate()
                    .map(|(i, (frame, offset))| {
                        let result = shift_frame(frame, offset);
                        transform_progress(i + 1);
                        result
                    })
                    .collect()
            }
        };

        let stacking_detail = match method {
            StackMethod::Mean => "Mean stacking",
            StackMethod::Median => "Combining pixels (median)",
            StackMethod::SigmaClip(_) => "Combining pixels (sigma clip)",
            _ => "Stacking",
        };

        send_log(tx, ctx, "Stacking...");

        let stacking_progress = make_progress_callback_with_detail(
            tx,
            ctx,
            PipelineStage::Stacking,
            frame_count,
            Some(stacking_detail.into()),
        );

        let result = match method {
            StackMethod::Mean => mean_stack_with_progress(&transformed, stacking_progress),
            StackMethod::Median => {
                let r = median_stack(&transformed);
                stacking_progress(frame_count);
                r
            }
            StackMethod::SigmaClip(params) => {
                let r = sigma_clip_stack(&transformed, params);
                stacking_progress(frame_count);
                r
            }
            _ => unreachable!(),
        };

        match result {
            Ok(result) => {
                let elapsed = start.elapsed();
                let output = PipelineOutput::Mono(result);
                // In streaming mode, persist to disk and free RAM
                if cache.is_streaming {
                    if let Some(disk_result) = cache.store_output_to_disk(&output) {
                        cache.set_stacked_disk(disk_result);
                    } else {
                        cache.set_stacked(output.clone());
                    }
                } else {
                    cache.set_stacked(output.clone());
                }
                send_log(
                    tx,
                    ctx,
                    format!("Stacking complete in {:.1}s", elapsed.as_secs_f32()),
                );
                send(
                    tx,
                    ctx,
                    WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    },
                );
            }
            Err(e) => send_error(tx, ctx, format!("Stacking failed: {e}")),
        }
    }
}
