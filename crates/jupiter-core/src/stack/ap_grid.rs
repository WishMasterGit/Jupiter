use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::align::phase_correlation::bilinear_sample;
use crate::consts::{AUTO_AP_DIVISOR, AUTO_AP_SIZE_ALIGN, AUTO_AP_SIZE_MAX, AUTO_AP_SIZE_MIN};
use crate::frame::AlignmentOffset;
use crate::pipeline::config::QualityMetric;

/// Local stacking method for per-AP patches.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum LocalStackMethod {
    #[default]
    Mean,
    Median,
    SigmaClip {
        sigma: f32,
        iterations: usize,
    },
}

/// Configuration for multi-alignment-point stacking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiPointConfig {
    /// Region size in pixels (default: 64).
    pub ap_size: usize,
    /// Padding for local alignment FFT search (default: 16).
    pub search_radius: usize,
    /// Fraction of frames to select per AP (default: 0.25).
    pub select_percentage: f32,
    /// Skip APs where reference mean brightness is below this (default: 0.05).
    pub min_brightness: f32,
    /// Quality metric for per-AP scoring.
    pub quality_metric: QualityMetric,
    /// Local stacking method for each AP.
    #[serde(default)]
    pub local_stack_method: LocalStackMethod,
}

impl Default for MultiPointConfig {
    fn default() -> Self {
        Self {
            ap_size: 64,
            search_radius: 16,
            select_percentage: 0.25,
            min_brightness: 0.05,
            quality_metric: QualityMetric::Laplacian,
            local_stack_method: LocalStackMethod::Mean,
        }
    }
}

/// Compute an appropriate AP size from a detected planet diameter.
///
/// Divides the diameter by [`AUTO_AP_DIVISOR`], clamps to
/// [`AUTO_AP_SIZE_MIN`]..=[`AUTO_AP_SIZE_MAX`], and rounds down to a
/// multiple of [`AUTO_AP_SIZE_ALIGN`].
pub fn auto_ap_size(planet_diameter: usize) -> usize {
    let raw = planet_diameter / AUTO_AP_DIVISOR;
    let clamped = raw.clamp(AUTO_AP_SIZE_MIN, AUTO_AP_SIZE_MAX);
    (clamped / AUTO_AP_SIZE_ALIGN) * AUTO_AP_SIZE_ALIGN
}

/// Fallback: compute AP size from frame dimensions when no planet is detected.
pub fn auto_ap_size_from_frame(width: usize, height: usize) -> usize {
    let dim = width.min(height);
    auto_ap_size(dim)
}

/// A single alignment point on the grid.
#[derive(Clone, Debug)]
pub struct AlignmentPoint {
    /// Center row in the image.
    pub cy: usize,
    /// Center column in the image.
    pub cx: usize,
    /// Index in the AP list.
    pub index: usize,
}

/// The grid of alignment points.
#[derive(Clone, Debug)]
pub struct ApGrid {
    pub points: Vec<AlignmentPoint>,
    pub ap_size: usize,
}

/// Extract a square sub-region from `data`, centered at (cy, cx) with edge clamping.
pub fn extract_region(data: &Array2<f32>, cy: usize, cx: usize, half_size: usize) -> Array2<f32> {
    let (h, w) = data.dim();
    let size = half_size * 2;
    let mut region = Array2::<f32>::zeros((size, size));

    for dr in 0..size {
        for dc in 0..size {
            let r = (cy + dr).saturating_sub(half_size).min(h - 1);
            let c = (cx + dc).saturating_sub(half_size).min(w - 1);
            region[[dr, dc]] = data[[r, c]];
        }
    }

    region
}

/// Extract a square sub-region with a global offset applied via bilinear interpolation.
pub fn extract_region_shifted(
    data: &Array2<f32>,
    cy: usize,
    cx: usize,
    half_size: usize,
    offset: &AlignmentOffset,
) -> Array2<f32> {
    let size = half_size * 2;
    let mut region = Array2::<f32>::zeros((size, size));

    for dr in 0..size {
        for dc in 0..size {
            let src_y = (cy as f64 + dr as f64 - half_size as f64) - offset.dy;
            let src_x = (cx as f64 + dc as f64 - half_size as f64) - offset.dx;
            region[[dr, dc]] = bilinear_sample(data, src_y, src_x);
        }
    }

    region
}

/// Build the AP grid over the reference frame.
/// APs are placed with stride = ap_size/2 (50% overlap).
/// APs with mean brightness below `min_brightness` are skipped.
pub fn build_ap_grid(reference: &Array2<f32>, config: &MultiPointConfig) -> ApGrid {
    let (h, w) = reference.dim();
    let half = config.ap_size / 2;
    let stride = half; // 50% overlap

    let mut points = Vec::new();
    let mut index = 0;

    // Place APs starting from half (center of first AP)
    let mut cy = half;
    while cy + half <= h {
        let mut cx = half;
        while cx + half <= w {
            // Check mean brightness in reference
            let region = extract_region(reference, cy, cx, half);
            let mean_brightness = region.mean().unwrap_or(0.0);

            if mean_brightness >= config.min_brightness {
                points.push(AlignmentPoint { cy, cx, index });
                index += 1;
            }

            cx += stride;
        }
        cy += stride;
    }

    ApGrid {
        points,
        ap_size: config.ap_size,
    }
}
