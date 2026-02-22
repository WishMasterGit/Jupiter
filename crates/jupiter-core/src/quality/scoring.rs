use rayon::prelude::*;

use crate::color::debayer::{luminance, DebayerMethod};
use crate::consts::STREAMING_BATCH_SIZE;
use crate::error::Result;
use crate::frame::{ColorMode, Frame, QualityScore};
use crate::io::ser::SerReader;

/// Score all mono frames streaming in batches from a SER reader.
///
/// Each batch of [`STREAMING_BATCH_SIZE`] frames is decoded, scored in parallel
/// via `score_fn`, then dropped before the next batch. This avoids holding all N
/// frames in memory simultaneously.
///
/// `make_quality_score` converts the raw f64 score into a `QualityScore`.
///
/// An optional `on_progress` callback is called with the total items scored so far
/// after each batch.
pub fn rank_frames_streaming_generic(
    reader: &SerReader,
    score_fn: fn(&Frame) -> f64,
    make_quality_score: fn(f64) -> QualityScore,
    on_progress: Option<&dyn Fn(usize)>,
) -> Result<Vec<(usize, QualityScore)>> {
    let total = reader.frame_count();
    let mut scores: Vec<(usize, QualityScore)> = Vec::with_capacity(total);

    for batch_start in (0..total).step_by(STREAMING_BATCH_SIZE) {
        let batch_end = (batch_start + STREAMING_BATCH_SIZE).min(total);
        let batch: Vec<(usize, Frame)> = (batch_start..batch_end)
            .map(|i| Ok((i, reader.read_frame(i)?)))
            .collect::<Result<_>>()?;

        let batch_scores: Vec<(usize, QualityScore)> = batch
            .par_iter()
            .map(|(i, frame)| {
                let s = score_fn(frame);
                (*i, make_quality_score(s))
            })
            .collect();

        scores.extend(batch_scores);
        if let Some(progress) = on_progress {
            progress(scores.len());
        }
    }

    scores.sort_by(|a, b| b.1.composite.total_cmp(&a.1.composite));
    Ok(scores)
}

/// Score all color frames streaming in batches from a SER reader.
///
/// For each batch: read raw frames, debayer (or split RGB), convert to
/// luminance, score in parallel via `score_fn`, then drop the batch.
pub fn rank_frames_color_streaming_generic(
    reader: &SerReader,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
    score_fn: fn(&Frame) -> f64,
    make_quality_score: fn(f64) -> QualityScore,
    on_progress: Option<&dyn Fn(usize)>,
) -> Result<Vec<(usize, QualityScore)>> {
    let total = reader.frame_count();
    let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);
    let mut scores: Vec<(usize, QualityScore)> = Vec::with_capacity(total);

    for batch_start in (0..total).step_by(STREAMING_BATCH_SIZE) {
        let batch_end = (batch_start + STREAMING_BATCH_SIZE).min(total);
        let batch: Vec<(usize, Frame)> = (batch_start..batch_end)
            .map(|i| {
                let color_frame = if is_rgb_bgr {
                    reader.read_frame_rgb(i)?
                } else {
                    reader.read_frame_color(i, debayer_method)?
                };
                Ok((i, luminance(&color_frame)))
            })
            .collect::<Result<_>>()?;

        let batch_scores: Vec<(usize, QualityScore)> = batch
            .par_iter()
            .map(|(i, frame)| {
                let s = score_fn(frame);
                (*i, make_quality_score(s))
            })
            .collect();

        scores.extend(batch_scores);
        if let Some(progress) = on_progress {
            progress(scores.len());
        }
    }

    scores.sort_by(|a, b| b.1.composite.total_cmp(&a.1.composite));
    Ok(scores)
}
