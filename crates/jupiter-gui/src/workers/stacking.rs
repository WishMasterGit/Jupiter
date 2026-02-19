use std::sync::mpsc;
use std::time::Instant;

use jupiter_core::align::shift_frame;
use jupiter_core::frame::{ColorFrame, ColorMode, Frame};
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::StackMethod;
use jupiter_core::pipeline::{PipelineOutput, PipelineStage};
use jupiter_core::stack::drizzle::drizzle_stack_with_progress;
use jupiter_core::stack::mean::mean_stack_with_progress;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::multi_point::{multi_point_stack, multi_point_stack_color};
use jupiter_core::stack::sigma_clip::sigma_clip_stack;

use crate::messages::WorkerResult;

use super::{make_progress_callback, send, send_error, send_log, PipelineCache};

pub(super) fn handle_stack(
    method: &StackMethod,
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
            match multi_point_stack_color(
                &reader,
                mp_config,
                &color_mode,
                &debayer_method,
                |_| {},
            ) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let output = PipelineOutput::Color(result);
                    cache.set_stacked(output.clone());
                    send_log(
                        tx,
                        ctx,
                        format!(
                            "Multi-point color stacking complete in {:.1}s",
                            elapsed.as_secs_f32()
                        ),
                    );
                    send(tx, ctx, WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    });
                }
                Err(e) => {
                    send_error(tx, ctx, format!("Multi-point color stacking failed: {e}"))
                }
            }
        } else {
            match multi_point_stack(&reader, mp_config, |_| {}) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let output = PipelineOutput::Mono(result);
                    cache.set_stacked(output.clone());
                    send_log(
                        tx,
                        ctx,
                        format!(
                            "Multi-point stacking complete in {:.1}s",
                            elapsed.as_secs_f32()
                        ),
                    );
                    send(tx, ctx, WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    });
                }
                Err(e) => send_error(tx, ctx, format!("Multi-point stacking failed: {e}")),
            }
        }
        return;
    }

    // Non-multi-point: use cached alignment data
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
    let frame_count = selected_frames.len();

    // Drizzle path — uses unshifted frames + offsets
    if let StackMethod::Drizzle(ref drizzle_config) = method {
        send_log(tx, ctx, "Drizzle stacking...");
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: Some(0),
            items_total: Some(frame_count),
        });

        let scores = if drizzle_config.quality_weighted {
            cache.selected_quality_scores.as_deref()
        } else {
            None
        };

        let drizzle_progress =
            make_progress_callback(tx, ctx, PipelineStage::Stacking, frame_count);

        if let Some(ref color_frames) = cache.selected_color_frames {
            let red_frames: Vec<Frame> = color_frames.iter().map(|cf| cf.red.clone()).collect();
            let green_frames: Vec<Frame> =
                color_frames.iter().map(|cf| cf.green.clone()).collect();
            let blue_frames: Vec<Frame> = color_frames.iter().map(|cf| cf.blue.clone()).collect();

            let (dr, (dg, db)) = rayon::join(
                || {
                    drizzle_stack_with_progress(
                        &red_frames,
                        offsets,
                        drizzle_config,
                        scores,
                        drizzle_progress,
                    )
                },
                || {
                    rayon::join(
                        || {
                            drizzle_stack_with_progress(
                                &green_frames,
                                offsets,
                                drizzle_config,
                                scores,
                                |_| {},
                            )
                        },
                        || {
                            drizzle_stack_with_progress(
                                &blue_frames,
                                offsets,
                                drizzle_config,
                                scores,
                                |_| {},
                            )
                        },
                    )
                },
            );
            match (dr, dg, db) {
                (Ok(r), Ok(g), Ok(b)) => {
                    let elapsed = start.elapsed();
                    let output = PipelineOutput::Color(ColorFrame {
                        red: r,
                        green: g,
                        blue: b,
                    });
                    cache.set_stacked(output.clone());
                    send_log(
                        tx,
                        ctx,
                        format!("Color drizzle complete in {:.1}s", elapsed.as_secs_f32()),
                    );
                    send(tx, ctx, WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    });
                }
                _ => send_error(tx, ctx, "Color drizzle stacking failed"),
            }
        } else {
            match drizzle_stack_with_progress(
                selected_frames,
                offsets,
                drizzle_config,
                scores,
                drizzle_progress,
            ) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let output = PipelineOutput::Mono(result);
                    cache.set_stacked(output.clone());
                    send_log(
                        tx,
                        ctx,
                        format!("Drizzle complete in {:.1}s", elapsed.as_secs_f32()),
                    );
                    send(tx, ctx, WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    });
                }
                Err(e) => send_error(tx, ctx, format!("Drizzle stacking failed: {e}")),
            }
        }
        return;
    }

    // Standard path: shift frames using cached offsets, then stack
    if let Some(ref color_frames) = cache.selected_color_frames {
        // Color: shift per-channel, stack per-channel
        let aligned_color: Vec<ColorFrame> = color_frames
            .iter()
            .zip(offsets.iter())
            .map(|(cf, offset)| ColorFrame {
                red: shift_frame(&cf.red, offset),
                green: shift_frame(&cf.green, offset),
                blue: shift_frame(&cf.blue, offset),
            })
            .collect();

        send_log(tx, ctx, "Stacking color channels...");
        let color_frame_count = aligned_color.len();
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: Some(0),
            items_total: Some(color_frame_count),
        });

        let red_frames: Vec<Frame> = aligned_color.iter().map(|cf| cf.red.clone()).collect();
        let green_frames: Vec<Frame> = aligned_color.iter().map(|cf| cf.green.clone()).collect();
        let blue_frames: Vec<Frame> = aligned_color.iter().map(|cf| cf.blue.clone()).collect();

        let color_stack_progress =
            make_progress_callback(tx, ctx, PipelineStage::Stacking, color_frame_count);

        let stack_fn =
            |frames: &[Frame],
             on_progress: &dyn Fn(usize)|
             -> jupiter_core::error::Result<Frame> {
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
                cache.set_stacked(output.clone());
                send_log(
                    tx,
                    ctx,
                    format!(
                        "Color stacking complete in {:.1}s",
                        elapsed.as_secs_f32()
                    ),
                );
                send(tx, ctx, WorkerResult::StackComplete {
                    result: output,
                    elapsed,
                });
            }
            _ => send_error(tx, ctx, "Color stacking failed"),
        }
    } else {
        // Mono: shift then stack
        let aligned: Vec<Frame> = selected_frames
            .iter()
            .zip(offsets.iter())
            .map(|(frame, offset)| shift_frame(frame, offset))
            .collect();

        send_log(tx, ctx, "Stacking...");
        let frame_count = aligned.len();
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: Some(0),
            items_total: Some(frame_count),
        });

        let stacking_progress =
            make_progress_callback(tx, ctx, PipelineStage::Stacking, frame_count);

        let result = match method {
            StackMethod::Mean => mean_stack_with_progress(&aligned, stacking_progress),
            StackMethod::Median => {
                let r = median_stack(&aligned);
                stacking_progress(frame_count);
                r
            }
            StackMethod::SigmaClip(params) => {
                let r = sigma_clip_stack(&aligned, params);
                stacking_progress(frame_count);
                r
            }
            _ => unreachable!(),
        };

        match result {
            Ok(result) => {
                let elapsed = start.elapsed();
                let output = PipelineOutput::Mono(result);
                cache.set_stacked(output.clone());
                send_log(
                    tx,
                    ctx,
                    format!("Stacking complete in {:.1}s", elapsed.as_secs_f32()),
                );
                send(tx, ctx, WorkerResult::StackComplete {
                    result: output,
                    elapsed,
                });
            }
            Err(e) => send_error(tx, ctx, format!("Stacking failed: {e}")),
        }
    }
}
