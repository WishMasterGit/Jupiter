use ndarray::Array2;
use rayon::prelude::*;

use crate::consts::{
    AUTOCROP_FALLBACK_FRAME_COUNT, AUTOCROP_MIN_VALID_DETECTIONS, PARALLEL_FRAME_THRESHOLD,
};
use crate::error::{JupiterError, Result};
use crate::io::crop::CropRect;
use crate::io::ser::SerReader;

use super::config::{AutoCropConfig, ThresholdMethod};
use super::detection::{detect_planet_in_frame, FrameDetection};
use super::temporal::{analyze_detections, compute_crop_rect};
use super::threshold::otsu_threshold;

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
    let detections: Vec<Option<FrameDetection>> = if indices.len() >= PARALLEL_FRAME_THRESHOLD {
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
