pub mod components;
pub mod detection;
pub mod morphology;
pub mod temporal;

use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::consts::{
    AUTOCROP_FALLBACK_FRAME_COUNT, AUTOCROP_MIN_VALID_DETECTIONS, DEFAULT_AUTOCROP_BLUR_SIGMA,
    DEFAULT_AUTOCROP_MIN_AREA, DEFAULT_AUTOCROP_PADDING_FRACTION, DEFAULT_AUTOCROP_SAMPLE_COUNT,
    DEFAULT_AUTOCROP_SIGMA_MULTIPLIER, OTSU_HISTOGRAM_BINS, PARALLEL_FRAME_THRESHOLD,
};
use crate::error::{JupiterError, Result};
use crate::io::crop::CropRect;
use crate::io::ser::SerReader;

use self::detection::{detect_planet_in_frame, FrameDetection};
use self::temporal::{analyze_detections, compute_crop_rect};

/// Method used to separate the planet from the sky background.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum ThresholdMethod {
    /// Threshold = mean + sigma_multiplier * stddev.
    MeanPlusSigma,
    /// Otsu's method: minimizes intra-class variance on a bimodal histogram.
    #[default]
    Otsu,
    /// User-specified fixed threshold in [0.0, 1.0].
    Fixed(f32),
}

/// Configuration for automatic planet detection and cropping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoCropConfig {
    /// Number of frames to sample for detection (evenly spaced).
    #[serde(default = "default_sample_count")]
    pub sample_count: usize,
    /// Padding around detected bounding box as fraction of planet diameter.
    #[serde(default = "default_padding_fraction")]
    pub padding_fraction: f32,
    /// Thresholding method.
    #[serde(default)]
    pub threshold_method: ThresholdMethod,
    /// Sigma multiplier for MeanPlusSigma method.
    #[serde(default = "default_sigma_multiplier")]
    pub sigma_multiplier: f32,
    /// Gaussian blur sigma for noise suppression before thresholding.
    #[serde(default = "default_blur_sigma")]
    pub blur_sigma: f32,
    /// Minimum connected component area (pixels) to be a planet candidate.
    #[serde(default = "default_min_area")]
    pub min_area: usize,
    /// Round crop size to next multiple of 32 for FFT efficiency.
    #[serde(default = "default_true")]
    pub align_to_fft: bool,
}

fn default_sample_count() -> usize {
    DEFAULT_AUTOCROP_SAMPLE_COUNT
}
fn default_padding_fraction() -> f32 {
    DEFAULT_AUTOCROP_PADDING_FRACTION
}
fn default_sigma_multiplier() -> f32 {
    DEFAULT_AUTOCROP_SIGMA_MULTIPLIER
}
fn default_blur_sigma() -> f32 {
    DEFAULT_AUTOCROP_BLUR_SIGMA
}
fn default_min_area() -> usize {
    DEFAULT_AUTOCROP_MIN_AREA
}
fn default_true() -> bool {
    true
}

impl Default for AutoCropConfig {
    fn default() -> Self {
        Self {
            sample_count: DEFAULT_AUTOCROP_SAMPLE_COUNT,
            padding_fraction: DEFAULT_AUTOCROP_PADDING_FRACTION,
            threshold_method: ThresholdMethod::default(),
            sigma_multiplier: DEFAULT_AUTOCROP_SIGMA_MULTIPLIER,
            blur_sigma: DEFAULT_AUTOCROP_BLUR_SIGMA,
            min_area: DEFAULT_AUTOCROP_MIN_AREA,
            align_to_fft: true,
        }
    }
}

/// Detect the planet in a SER file and return a crop rectangle that contains it.
///
/// Uses multi-frame sampling, connected component analysis, temporal
/// filtering, and drift-aware crop sizing for robust detection.
pub fn auto_detect_crop(reader: &SerReader, config: &AutoCropConfig) -> Result<CropRect> {
    let frame_count = reader.frame_count();
    if frame_count == 0 {
        return Err(JupiterError::Pipeline(
            "Auto-crop: SER file has no frames".into(),
        ));
    }

    let (h, w) = (reader.header.height, reader.header.width);

    // Compute evenly-spaced frame indices.
    let sample_count = config.sample_count.clamp(1, frame_count);
    let indices: Vec<usize> = if sample_count == 1 {
        vec![frame_count / 2]
    } else {
        (0..sample_count)
            .map(|i| i * (frame_count - 1) / (sample_count - 1))
            .collect()
    };

    // Phase 1: per-frame detection with optional parallelism.
    let detections: Vec<Option<FrameDetection>> =
        if indices.len() >= PARALLEL_FRAME_THRESHOLD {
            indices
                .par_iter()
                .map(|&idx| {
                    reader
                        .read_frame(idx)
                        .ok()
                        .and_then(|f| detect_planet_in_frame(&f.data, idx, config))
                })
                .collect()
        } else {
            indices
                .iter()
                .map(|&idx| {
                    reader
                        .read_frame(idx)
                        .ok()
                        .and_then(|f| detect_planet_in_frame(&f.data, idx, config))
                })
                .collect()
        };

    let valid: Vec<FrameDetection> = detections.into_iter().flatten().collect();

    // Phase 2-3: temporal analysis or fallback.
    if valid.len() >= AUTOCROP_MIN_VALID_DETECTIONS {
        let analysis = analyze_detections(&valid);
        compute_crop_rect(&analysis, w, h, config, &reader.header.color_mode())
    } else {
        // Fallback: median-combine center frames, detect on the composite.
        detect_fallback(reader, config)
    }
}

/// Fallback detection when multi-frame analysis has too few valid detections.
///
/// Median-combines several center frames, runs single-frame detection, and
/// retries with progressively lower thresholds if needed.
fn detect_fallback(reader: &SerReader, config: &AutoCropConfig) -> Result<CropRect> {
    let total = reader.frame_count();
    let (h, w) = (reader.header.height, reader.header.width);
    let n = AUTOCROP_FALLBACK_FRAME_COUNT.min(total);

    // Read center frames.
    let center = total / 2;
    let half = n / 2;
    let start = center.saturating_sub(half);
    let mut frames: Vec<Array2<f32>> = Vec::with_capacity(n);
    for i in start..start + n {
        let idx = i.min(total - 1);
        if let Ok(frame) = reader.read_frame(idx) {
            frames.push(frame.data);
        }
    }

    if frames.is_empty() {
        return Err(JupiterError::Pipeline(
            "Auto-crop fallback: could not read any frames".into(),
        ));
    }

    // Median-combine pixel-wise.
    let combined = median_combine(&frames);

    // Try detection on the composite.
    if let Some(det) = detect_planet_in_frame(&combined, center, config) {
        let analysis = analyze_detections(&[det]);
        return compute_crop_rect(&analysis, w, h, config, &reader.header.color_mode());
    }

    // Retry with progressively lower thresholds.
    for &multiplier in &[0.8_f32, 0.6] {
        let mut lowered = config.clone();
        lowered.threshold_method = ThresholdMethod::Fixed(
            otsu_threshold(&crate::filters::gaussian_blur::gaussian_blur_array(
                &combined,
                config.blur_sigma,
            )) * multiplier,
        );
        if let Some(det) = detect_planet_in_frame(&combined, center, &lowered) {
            let analysis = analyze_detections(&[det]);
            return compute_crop_rect(&analysis, w, h, config, &reader.header.color_mode());
        }
    }

    Err(JupiterError::Pipeline(
        "Auto-crop: no planet detected after fallback attempts".into(),
    ))
}

/// Pixel-wise median of multiple frames.
fn median_combine(frames: &[Array2<f32>]) -> Array2<f32> {
    let (h, w) = frames[0].dim();
    let n = frames.len();
    let mut result = Array2::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            let mut vals: Vec<f32> = frames.iter().map(|f| f[[row, col]]).collect();
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            result[[row, col]] = if n % 2 == 1 {
                vals[n / 2]
            } else {
                (vals[n / 2 - 1] + vals[n / 2]) * 0.5
            };
        }
    }

    result
}

// --- Threshold helpers (reused by detection.rs) ---

/// Compute the threshold value using the configured method.
pub(crate) fn compute_threshold(
    data: &Array2<f32>,
    method: &ThresholdMethod,
    sigma_mul: f32,
) -> f32 {
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
pub(crate) fn compute_mean_stddev(data: &Array2<f32>) -> (f64, f64) {
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
pub(crate) fn otsu_threshold(data: &Array2<f32>) -> f32 {
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
