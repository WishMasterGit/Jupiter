use std::sync::mpsc;
use std::time::Instant;

use jupiter_core::frame::{ColorFrame, Frame};
use jupiter_core::stack::drizzle::DrizzleConfig;
use jupiter_core::pipeline::{PipelineOutput, PipelineStage};
use jupiter_core::stack::drizzle::drizzle_stack_with_progress;

use crate::messages::WorkerResult;

use super::super::{make_progress_callback, send, send_error, send_log, PipelineCache};

pub(crate) fn handle_drizzle(
    drizzle_config: &DrizzleConfig,
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
    let frame_count = selected_frames.len();

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
}
