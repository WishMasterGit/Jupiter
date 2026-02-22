use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array2;
use rayon::prelude::*;

use crate::color::debayer::DebayerMethod;
use crate::error::Result;
use crate::frame::{ColorMode, Frame, QualityScore};
use crate::io::ser::SerReader;
use crate::quality::scoring::{rank_frames_color_streaming_generic, rank_frames_streaming_generic};

/// Compute Sobel gradient magnitude image.
///
/// Returns an `Array2<f32>` of the same dimensions as input. The 1-pixel
/// border is zero (Sobel kernel needs a 3x3 neighborhood).
pub fn gradient_magnitude_array(data: &Array2<f32>) -> Array2<f32> {
    let (h, w) = data.dim();
    let mut result = Array2::<f32>::zeros((h, w));

    if h < 3 || w < 3 {
        return result;
    }

    for row in 1..h - 1 {
        for col in 1..w - 1 {
            let gx = -data[[row - 1, col - 1]] as f64 + data[[row - 1, col + 1]] as f64
                - 2.0 * data[[row, col - 1]] as f64
                + 2.0 * data[[row, col + 1]] as f64
                - data[[row + 1, col - 1]] as f64
                + data[[row + 1, col + 1]] as f64;

            let gy = -data[[row - 1, col - 1]] as f64
                - 2.0 * data[[row - 1, col]] as f64
                - data[[row - 1, col + 1]] as f64
                + data[[row + 1, col - 1]] as f64
                + 2.0 * data[[row + 1, col]] as f64
                + data[[row + 1, col + 1]] as f64;

            result[[row, col]] = (gx * gx + gy * gy).sqrt() as f32;
        }
    }

    result
}

/// Compute gradient magnitude quality score on raw array data.
pub fn gradient_score_array(data: &Array2<f32>) -> f64 {
    let (h, w) = data.dim();
    if h < 3 || w < 3 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let count = ((h - 2) * (w - 2)) as f64;

    for row in 1..h - 1 {
        for col in 1..w - 1 {
            let gx = -data[[row - 1, col - 1]] as f64 + data[[row - 1, col + 1]] as f64
                - 2.0 * data[[row, col - 1]] as f64
                + 2.0 * data[[row, col + 1]] as f64
                - data[[row + 1, col - 1]] as f64
                + data[[row + 1, col + 1]] as f64;

            let gy = -data[[row - 1, col - 1]] as f64
                - 2.0 * data[[row - 1, col]] as f64
                - data[[row - 1, col + 1]] as f64
                + data[[row + 1, col - 1]] as f64
                + 2.0 * data[[row + 1, col]] as f64
                + data[[row + 1, col + 1]] as f64;

            sum += (gx * gx + gy * gy).sqrt();
        }
    }

    sum / count
}

/// Compute gradient magnitude quality score using Sobel operator.
///
/// Sobel kernels:
///   Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
///   Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
///
/// Score = mean of sqrt(Gx^2 + Gy^2). Higher = sharper.
pub fn gradient_score(frame: &Frame) -> f64 {
    gradient_score_array(&frame.data)
}

fn make_gradient_quality_score(gs: f64) -> QualityScore {
    QualityScore {
        laplacian_variance: 0.0,
        composite: gs,
    }
}

/// Score all frames using gradient metric and return sorted by quality descending.
pub fn rank_frames_gradient(frames: &[Frame]) -> Vec<(usize, QualityScore)> {
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| (i, make_gradient_quality_score(gradient_score(f))))
        .collect();

    scores.sort_by(|a, b| b.1.composite.total_cmp(&a.1.composite));
    scores
}

/// Score all frames using gradient metric with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` as each frame is scored.
pub fn rank_frames_gradient_with_progress(
    frames: &[Frame],
    on_progress: impl Fn(usize) + Send + Sync,
) -> Vec<(usize, QualityScore)> {
    let done = AtomicUsize::new(0);
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let gs = gradient_score(f);
            let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(completed);
            (i, make_gradient_quality_score(gs))
        })
        .collect();

    scores.sort_by(|a, b| b.1.composite.total_cmp(&a.1.composite));
    scores
}

/// Score all frames using gradient metric by reading in batches from the SER reader.
pub fn rank_frames_gradient_streaming(reader: &SerReader) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_streaming_generic(reader, gradient_score, make_gradient_quality_score, None)
}

/// Score all frames using gradient metric streaming with per-frame progress reporting.
pub fn rank_frames_gradient_streaming_with_progress(
    reader: &SerReader,
    on_progress: impl Fn(usize),
) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_streaming_generic(
        reader,
        gradient_score,
        make_gradient_quality_score,
        Some(&on_progress),
    )
}

/// Score color frames using gradient metric by reading in batches from the SER reader.
pub fn rank_frames_gradient_color_streaming(
    reader: &SerReader,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_color_streaming_generic(
        reader,
        color_mode,
        debayer_method,
        gradient_score,
        make_gradient_quality_score,
        None,
    )
}

/// Score color frames using gradient metric streaming with per-frame progress reporting.
pub fn rank_frames_gradient_color_streaming_with_progress(
    reader: &SerReader,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
    on_progress: impl Fn(usize),
) -> Result<Vec<(usize, QualityScore)>> {
    rank_frames_color_streaming_generic(
        reader,
        color_mode,
        debayer_method,
        gradient_score,
        make_gradient_quality_score,
        Some(&on_progress),
    )
}
