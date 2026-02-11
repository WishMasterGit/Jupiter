use ndarray::Array2;
use rayon::prelude::*;

use crate::frame::{Frame, QualityScore};

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

fn laplacian_variance_array(data: &Array2<f32>) -> f64 {
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

/// Score all frames and return (index, QualityScore) sorted by quality descending.
pub fn rank_frames(frames: &[Frame]) -> Vec<(usize, QualityScore)> {
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let lv = laplacian_variance(f);
            (
                i,
                QualityScore {
                    laplacian_variance: lv,
                    composite: lv,
                },
            )
        })
        .collect();

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    scores
}

/// Select indices of the best `percentage` (0.0..1.0) of frames.
pub fn select_best(frames: &[Frame], percentage: f32) -> Vec<usize> {
    let ranked = rank_frames(frames);
    let keep = ((frames.len() as f32 * percentage.clamp(0.0, 1.0)).ceil()) as usize;
    ranked.into_iter().take(keep).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_flat_image_has_zero_variance() {
        let data = Array2::<f32>::from_elem((10, 10), 0.5);
        let frame = Frame::new(data, 8);
        let lv = laplacian_variance(&frame);
        assert!(lv.abs() < 1e-10, "Flat image should have ~0 Laplacian variance");
    }

    #[test]
    fn test_sharp_beats_blurry() {
        // "Sharp" image: alternating pixels
        let mut sharp_data = Array2::<f32>::zeros((16, 16));
        for row in 0..16 {
            for col in 0..16 {
                sharp_data[[row, col]] = if (row + col) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let sharp = Frame::new(sharp_data, 8);

        // "Blurry" image: smooth gradient
        let mut blurry_data = Array2::<f32>::zeros((16, 16));
        for row in 0..16 {
            for col in 0..16 {
                blurry_data[[row, col]] = (row as f32 + col as f32) / 30.0;
            }
        }
        let blurry = Frame::new(blurry_data, 8);

        let sharp_score = laplacian_variance(&sharp);
        let blurry_score = laplacian_variance(&blurry);

        assert!(
            sharp_score > blurry_score,
            "Sharp image ({sharp_score}) should score higher than blurry ({blurry_score})"
        );
    }

    #[test]
    fn test_rank_and_select() {
        let flat = Frame::new(Array2::from_elem((10, 10), 0.5), 8);

        let mut sharp_data = Array2::<f32>::zeros((10, 10));
        for r in 0..10 {
            for c in 0..10 {
                sharp_data[[r, c]] = if (r + c) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let sharp = Frame::new(sharp_data, 8);

        let frames = vec![flat.clone(), sharp, flat];
        let ranked = rank_frames(&frames);

        // Sharpest frame (index 1) should be first
        assert_eq!(ranked[0].0, 1);

        // 34% of 3 frames = ceil(1.02) = 2 frames
        let selected = select_best(&frames, 0.34);
        assert_eq!(selected.len(), 2);
        // Best frame (index 1) should be first in selection
        assert_eq!(selected[0], 1);
    }
}
