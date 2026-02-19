use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::consts::{
    DEFAULT_AUTOCROP_PADDING_FRACTION, DEFAULT_AUTOCROP_SAMPLE_COUNT,
    DEFAULT_AUTOCROP_SIGMA_MULTIPLIER, OTSU_HISTOGRAM_BINS,
};
use crate::error::{JupiterError, Result};
use crate::io::crop::CropRect;
use crate::io::ser::SerReader;

/// Method used to separate the planet from the sky background.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ThresholdMethod {
    /// Threshold = mean + sigma_multiplier * stddev.
    MeanPlusSigma,
    /// Otsu's method: minimizes intra-class variance on a bimodal histogram.
    Otsu,
    /// User-specified fixed threshold in [0.0, 1.0].
    Fixed(f32),
}

impl Default for ThresholdMethod {
    fn default() -> Self {
        Self::MeanPlusSigma
    }
}

/// Configuration for automatic planet detection and cropping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoCropConfig {
    /// Number of frames to sample for detection (evenly spaced).
    pub sample_count: usize,
    /// Padding around detected bounding box as fraction of planet diameter.
    pub padding_fraction: f32,
    /// Thresholding method.
    pub threshold_method: ThresholdMethod,
    /// Sigma multiplier for MeanPlusSigma method.
    pub sigma_multiplier: f32,
}

impl Default for AutoCropConfig {
    fn default() -> Self {
        Self {
            sample_count: DEFAULT_AUTOCROP_SAMPLE_COUNT,
            padding_fraction: DEFAULT_AUTOCROP_PADDING_FRACTION,
            threshold_method: ThresholdMethod::default(),
            sigma_multiplier: DEFAULT_AUTOCROP_SIGMA_MULTIPLIER,
        }
    }
}

/// Detect the planet in a SER file and return a crop rectangle that contains it.
///
/// The algorithm samples one or more frames, thresholds bright pixels,
/// finds a bounding box, adds padding, and clamps to image bounds.
pub fn auto_detect_crop(reader: &SerReader, config: &AutoCropConfig) -> Result<CropRect> {
    let frame_count = reader.frame_count();
    if frame_count == 0 {
        return Err(JupiterError::Pipeline(
            "Auto-crop: SER file has no frames".into(),
        ));
    }

    let sample = sample_frames(reader, config.sample_count)?;
    let (h, w) = (sample.nrows(), sample.ncols());

    let threshold = compute_threshold(&sample, &config.threshold_method, config.sigma_multiplier);

    let bbox = find_bounding_box(&sample, threshold).ok_or_else(|| {
        JupiterError::Pipeline("Auto-crop: no planet detected above threshold".into())
    })?;

    let (min_row, max_row, min_col, max_col) = bbox;
    let bbox_h = max_row - min_row + 1;
    let bbox_w = max_col - min_col + 1;
    let diameter = bbox_h.max(bbox_w);
    let pad = (diameter as f32 * config.padding_fraction).ceil() as usize;

    // Expand by padding, clamped to image bounds.
    let x = min_col.saturating_sub(pad);
    let y = min_row.saturating_sub(pad);
    let x2 = (max_col + pad + 1).min(w);
    let y2 = (max_row + pad + 1).min(h);

    let crop = CropRect {
        x: x as u32,
        y: y as u32,
        width: (x2 - x) as u32,
        height: (y2 - y) as u32,
    };

    crop.validated(w as u32, h as u32, &reader.header.color_mode())
}

/// Read and average `count` evenly-spaced frames from the SER file.
fn sample_frames(reader: &SerReader, count: usize) -> Result<Array2<f32>> {
    let total = reader.frame_count();
    let count = count.clamp(1, total);

    if count == 1 {
        let idx = total / 2;
        let frame = reader.read_frame(idx)?;
        return Ok(frame.data);
    }

    // Evenly spaced indices across the video.
    let indices: Vec<usize> = (0..count)
        .map(|i| i * (total - 1) / (count - 1))
        .collect();

    let first = reader.read_frame(indices[0])?;
    let mut acc = first.data.mapv(|v| v as f64);

    for &idx in &indices[1..] {
        let frame = reader.read_frame(idx)?;
        acc += &frame.data.mapv(|v| v as f64);
    }

    let n = count as f64;
    Ok(acc.mapv(|v| (v / n) as f32))
}

/// Compute the threshold value using the configured method.
fn compute_threshold(data: &Array2<f32>, method: &ThresholdMethod, sigma_mul: f32) -> f32 {
    match method {
        ThresholdMethod::MeanPlusSigma => {
            let (mean, std) = compute_mean_stddev(data);
            (mean + sigma_mul as f64 * std) as f32
        }
        ThresholdMethod::Otsu => otsu_threshold(data),
        ThresholdMethod::Fixed(v) => *v,
    }
}

/// Compute mean and standard deviation of pixel values.
fn compute_mean_stddev(data: &Array2<f32>) -> (f64, f64) {
    let n = data.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let sum: f64 = data.iter().map(|&v| v as f64).sum();
    let mean = sum / n;
    let var: f64 = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Otsu's thresholding: find the value that minimizes intra-class variance.
fn otsu_threshold(data: &Array2<f32>) -> f32 {
    let bins = OTSU_HISTOGRAM_BINS;
    let mut histogram = vec![0u64; bins];

    for &v in data.iter() {
        let bin = ((v.clamp(0.0, 1.0) * (bins - 1) as f32) as usize).min(bins - 1);
        histogram[bin] += 1;
    }

    let total = data.len() as f64;
    let mut sum_all: f64 = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_all += i as f64 * count as f64;
    }

    let mut weight_bg: f64 = 0.0;
    let mut sum_bg: f64 = 0.0;
    let mut best_variance = 0.0_f64;
    let mut best_bin = 0usize;

    for (i, &count) in histogram.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }
        sum_bg += i as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_all - sum_bg) / weight_fg;
        let between_variance = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);

        if between_variance > best_variance {
            best_variance = between_variance;
            best_bin = i;
        }
    }

    (best_bin as f32 + 0.5) / bins as f32
}

/// Find the bounding box of all pixels above `threshold`.
/// Returns `Some((min_row, max_row, min_col, max_col))` or `None` if no pixels qualify.
fn find_bounding_box(
    data: &Array2<f32>,
    threshold: f32,
) -> Option<(usize, usize, usize, usize)> {
    let (h, w) = (data.nrows(), data.ncols());
    let mut min_row = h;
    let mut max_row = 0;
    let mut min_col = w;
    let mut max_col = 0;

    for row in 0..h {
        for col in 0..w {
            if data[[row, col]] > threshold {
                min_row = min_row.min(row);
                max_row = max_row.max(row);
                min_col = min_col.min(col);
                max_col = max_col.max(col);
            }
        }
    }

    if max_row >= min_row && max_col >= min_col {
        Some((min_row, max_row, min_col, max_col))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mean_stddev() {
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let (mean, std) = compute_mean_stddev(&data);
        assert!((mean - 0.5).abs() < 1e-6);
        assert!((std - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_otsu_bimodal() {
        // Bimodal: half at 0.2, half at 0.8
        let mut data = Array2::zeros((10, 10));
        for row in 0..5 {
            for col in 0..10 {
                data[[row, col]] = 0.2;
            }
        }
        for row in 5..10 {
            for col in 0..10 {
                data[[row, col]] = 0.8;
            }
        }
        let t = otsu_threshold(&data);
        // Otsu places threshold at boundary of lower mode.
        // For planet detection, we need t < bright mode (0.8).
        assert!(t < 0.8, "Otsu threshold {t} should be below bright mode");
        // The bright pixels (0.8) should be above the threshold.
        assert!(0.8 > t, "Bright pixels should exceed threshold");
    }

    #[test]
    fn test_find_bounding_box_some() {
        let mut data = Array2::zeros((10, 10));
        data[[3, 4]] = 0.8;
        data[[6, 7]] = 0.9;
        let bbox = find_bounding_box(&data, 0.5);
        assert_eq!(bbox, Some((3, 6, 4, 7)));
    }

    #[test]
    fn test_find_bounding_box_none() {
        let data = Array2::zeros((10, 10));
        assert_eq!(find_bounding_box(&data, 0.5), None);
    }
}
