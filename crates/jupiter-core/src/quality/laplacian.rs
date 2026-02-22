use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array2;
use rayon::prelude::*;

use crate::color::debayer::DebayerMethod;
use crate::error::Result;
use crate::frame::{ColorMode, Frame, QualityScore};
use crate::io::ser::SerReader;
use crate::quality::scoring::{rank_frames_color_streaming_generic, rank_frames_streaming_generic};

/// Compute Laplacian variance of a frame — higher means sharper.
///
/// Convolves with the 3x3 Laplacian kernel:
///   0  1  0
///   1 -4  1
///   0  1  0
/// Then returns the variance of the result.
pub fn laplacian_variance(frame: &Frame) -> f64 {
    laplacian_variance_array(&frame.data)
}

pub fn laplacian_variance_array(data: &Array2<f32>) -> f64 {
    let (h, w) = data.dim();
    if h < 3 || w < 3 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let count = ((h - 2) * (w - 2)) as f64;

    for row in 1..h - 1 {
        for col in 1..w - 1 {
            let lap = -4.0 * data[[row, col]] as f64
                + data[[row - 1, col]] as f64
                + data[[row + 1, col]] as f64
                + data[[row, col - 1]] as f64
                + data[[row, col + 1]] as f64;
            sum += lap;
            sum_sq += lap * lap;
        }
    }

    let mean = sum / count;
    sum_sq / count - mean * mean
}

fn make_laplacian_quality_score(lv: f64) -> QualityScore {
    QualityScore {
        laplacian_variance: lv,
        composite: lv,
    }
}

/// Score all frames and return (index, QualityScore) sorted by quality descending.
pub fn rank_frames(frames: &[Frame]) -> Vec<(usize, QualityScore)> {
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| (i, make_laplacian_quality_score(laplacian_variance(f))))
        .collect();

    scores.sort_by(|a, b| b.1.composite.total_cmp(&a.1.composite));
    scores
}

/// Score all frames with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` as each frame is scored.
pub fn rank_frames_with_progress(
    frames: &[Frame],
    on_progress: impl Fn(usize) + Send + Sync,
) -> Vec<(usize, QualityScore)> {
    let done = AtomicUsize::new(0);
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let lv = laplacian_variance(f);
            let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(completed);
            (i, make_laplacian_quality_score(lv))
        })
        .collect();

    scores.sort_by(|a, b| b.1.composite.total_cmp(&a.1.composite));
    scores
}

/// Select indices of the best `percentage` (0.0..1.0) of frames.
pub fn select_best(frames: &[Frame], percentage: f32) -> Vec<usize> {
    let ranked = rank_frames(frames);
    let keep = ((frames.len() as f32 * percentage.clamp(0.0, 1.0)).ceil()) as usize;
    ranked.into_iter().take(keep).map(|(i, _)| i).collect()
}

/// Score all frames by reading them in batches from the SER reader.
pub fn rank_frames_streaming(reader: &SerReader) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_streaming_generic(
        reader,
        laplacian_variance,
        make_laplacian_quality_score,
        None,
    )
}

/// Score all frames streaming with per-frame progress reporting.
pub fn rank_frames_streaming_with_progress(
    reader: &SerReader,
    on_progress: impl Fn(usize),
) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_streaming_generic(
        reader,
        laplacian_variance,
        make_laplacian_quality_score,
        Some(&on_progress),
    )
}

/// Score color frames by reading them in batches from the SER reader.
pub fn rank_frames_color_streaming(
    reader: &SerReader,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_color_streaming_generic(
        reader,
        color_mode,
        debayer_method,
        laplacian_variance,
        make_laplacian_quality_score,
        None,
    )
}

/// Score color frames streaming with per-frame progress reporting.
pub fn rank_frames_color_streaming_with_progress(
    reader: &SerReader,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
    on_progress: impl Fn(usize),
) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_color_streaming_generic(
        reader,
        color_mode,
        debayer_method,
        laplacian_variance,
        make_laplacian_quality_score,
        Some(&on_progress),
    )
}
