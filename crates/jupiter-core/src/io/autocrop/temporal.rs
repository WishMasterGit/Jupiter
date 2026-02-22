use crate::consts::{
    AUTOCROP_SIGMA_CLIP_ITERATIONS, AUTOCROP_SIGMA_CLIP_THRESHOLD, AUTOCROP_SIZE_ALIGNMENT,
};
use crate::detection::planet::FrameDetection;
use crate::error::Result;
use crate::frame::ColorMode;
use crate::io::crop::CropRect;

use super::config::AutoCropConfig;

/// Result of temporal filtering on multi-frame detections.
#[derive(Clone, Debug)]
pub struct TemporalAnalysis {
    /// Median centroid X (column) from valid frames.
    pub median_cx: f64,
    /// Median centroid Y (row) from valid frames.
    pub median_cy: f64,
    /// Drift range in X (max - min) among valid frames.
    pub drift_range_x: f64,
    /// Drift range in Y (max - min) among valid frames.
    pub drift_range_y: f64,
    /// Median bounding box diameter across valid frames.
    pub median_diameter: f64,
    /// Number of valid detections after sigma-clipping.
    pub valid_count: usize,
}

/// Analyze multi-frame detections: sigma-clip outlier centroids, compute
/// drift range and median diameter.
pub fn analyze_detections(detections: &[FrameDetection]) -> TemporalAnalysis {
    let n = detections.len();
    assert!(n > 0, "analyze_detections requires at least one detection");

    let cx_vals: Vec<f64> = detections.iter().map(|d| d.cx).collect();
    let cy_vals: Vec<f64> = detections.iter().map(|d| d.cy).collect();
    let diameters: Vec<f64> = detections
        .iter()
        .map(|d| d.bbox_width.max(d.bbox_height) as f64)
        .collect();

    // Sigma-clip centroids to reject outliers.
    let mut valid = vec![true; n];
    sigma_clip_1d(&cx_vals, &mut valid);
    sigma_clip_1d(&cy_vals, &mut valid);

    // Collect surviving values.
    let mut clean_cx: Vec<f64> = Vec::new();
    let mut clean_cy: Vec<f64> = Vec::new();
    let mut clean_diameters: Vec<f64> = Vec::new();

    for i in 0..n {
        if valid[i] {
            clean_cx.push(cx_vals[i]);
            clean_cy.push(cy_vals[i]);
            clean_diameters.push(diameters[i]);
        }
    }

    // If too few survived, use all detections.
    if clean_cx.len() < 2 {
        clean_cx = cx_vals;
        clean_cy = cy_vals;
        clean_diameters = diameters;
    }

    let median_cx = median_f64(&mut clean_cx);
    let median_cy = median_f64(&mut clean_cy);
    let median_diameter = median_f64(&mut clean_diameters);

    let drift_range_x = max_f64(&clean_cx) - min_f64(&clean_cx);
    let drift_range_y = max_f64(&clean_cy) - min_f64(&clean_cy);

    TemporalAnalysis {
        median_cx,
        median_cy,
        drift_range_x,
        drift_range_y,
        median_diameter,
        valid_count: clean_cx.len(),
    }
}

/// Compute the crop rectangle from temporal analysis results.
pub fn compute_crop_rect(
    analysis: &TemporalAnalysis,
    frame_width: u32,
    frame_height: u32,
    config: &AutoCropConfig,
    color_mode: &ColorMode,
) -> Result<CropRect> {
    let diameter = analysis.median_diameter;
    let padding = diameter * config.padding_fraction as f64;

    let mut crop_w = diameter + analysis.drift_range_x + 2.0 * padding;
    let mut crop_h = diameter + analysis.drift_range_y + 2.0 * padding;

    // Make square.
    let size = crop_w.max(crop_h);
    crop_w = size;
    crop_h = size;

    // Round up to alignment.
    let align = if config.align_to_fft {
        AUTOCROP_SIZE_ALIGNMENT
    } else {
        1
    };
    let w = round_up(crop_w.ceil() as u32, align).min(frame_width);
    let h = round_up(crop_h.ceil() as u32, align).min(frame_height);

    // Center on the median centroid, clamped to image bounds.
    let cx = analysis.median_cx;
    let cy = analysis.median_cy;

    let x = ((cx - w as f64 / 2.0).round().max(0.0) as u32).min(frame_width.saturating_sub(w));
    let y = ((cy - h as f64 / 2.0).round().max(0.0) as u32).min(frame_height.saturating_sub(h));

    let crop = CropRect {
        x,
        y,
        width: w,
        height: h,
    };

    crop.validated(frame_width, frame_height, color_mode)
}

/// Iterative sigma-clipping on paired centroid values.
/// Marks entries as invalid if they deviate more than threshold * stddev from the median.
fn sigma_clip_1d(values: &[f64], valid: &mut [bool]) {
    let threshold = AUTOCROP_SIGMA_CLIP_THRESHOLD;

    for _ in 0..AUTOCROP_SIGMA_CLIP_ITERATIONS {
        let active: Vec<f64> = values
            .iter()
            .zip(valid.iter())
            .filter_map(|(&v, &ok)| if ok { Some(v) } else { None })
            .collect();

        if active.len() < 3 {
            break;
        }

        let mean = active.iter().sum::<f64>() / active.len() as f64;
        let variance =
            active.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / active.len() as f64;
        let stddev = variance.sqrt();

        if stddev < 1e-10 {
            break;
        }

        for (i, &v) in values.iter().enumerate() {
            if valid[i] && (v - mean).abs() > threshold * stddev {
                valid[i] = false;
            }
        }
    }
}

fn median_f64(vals: &mut [f64]) -> f64 {
    vals.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = vals.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) * 0.5
    }
}

fn min_f64(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::INFINITY, f64::min)
}

fn max_f64(vals: &[f64]) -> f64 {
    vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

fn round_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    value.div_ceil(align) * align
}
