use ndarray::Array2;

use crate::consts::AUTOCROP_BORDER_STRIP_WIDTH;
use crate::filters::gaussian_blur::gaussian_blur_array;

use super::components::{connected_components, touches_border};
use super::config::DetectionConfig;
use super::morphology::morphological_opening;
use super::threshold::compute_threshold;

/// Per-frame detection result.
#[derive(Clone, Debug)]
pub struct FrameDetection {
    /// Index of the frame in the SER file.
    pub frame_index: usize,
    /// Intensity-weighted centroid X (column) position.
    pub cx: f64,
    /// Intensity-weighted centroid Y (row) position.
    pub cy: f64,
    /// Bounding box width of the detected component.
    pub bbox_width: usize,
    /// Bounding box height of the detected component.
    pub bbox_height: usize,
    /// Area in pixels of the detected component.
    pub area: usize,
}

/// Detect the planet in a single frame.
///
/// Pipeline: Gaussian blur -> threshold -> morphological opening ->
/// connected component analysis -> select largest -> validate -> IWC centroid.
///
/// Returns `None` if no valid planet candidate is found.
pub fn detect_planet_in_frame(
    data: &Array2<f32>,
    frame_index: usize,
    config: &DetectionConfig,
) -> Option<FrameDetection> {
    let (h, w) = data.dim();
    if h == 0 || w == 0 {
        return None;
    }

    // Step 1: Gaussian blur for noise suppression.
    let blurred = gaussian_blur_array(data, config.blur_sigma);

    // Step 2: Compute threshold on blurred frame.
    let threshold = compute_threshold(&blurred, &config.threshold_method, config.sigma_multiplier);

    // Step 3: Create binary mask.
    let mask = blurred.mapv(|v| v > threshold);

    // Step 4: Morphological opening to remove hot pixels.
    let opened = morphological_opening(&mask);

    // Step 5: Connected component analysis.
    let components = connected_components(&opened);
    if components.is_empty() {
        return None;
    }

    // Step 6: Select the largest component.
    let largest = &components[0];

    // Step 7: Reject if area too small or touches border.
    if largest.area < config.min_area {
        return None;
    }
    if touches_border(largest.bbox, h, w) {
        return None;
    }

    // Step 8: Compute intensity-weighted centroid on the ORIGINAL frame
    // with background subtraction, using the component mask.
    let (min_row, max_row, min_col, max_col) = largest.bbox;
    let bg = estimate_background(data, h, w);
    let (cy, cx) = masked_centroid(data, &opened, min_row, max_row, min_col, max_col, bg);

    let bbox_width = max_col - min_col + 1;
    let bbox_height = max_row - min_row + 1;

    Some(FrameDetection {
        frame_index,
        cx,
        cy,
        bbox_width,
        bbox_height,
        area: largest.area,
    })
}

/// Estimate background level as the median of pixels in the border strip.
fn estimate_background(data: &Array2<f32>, h: usize, w: usize) -> f32 {
    let strip = AUTOCROP_BORDER_STRIP_WIDTH.min(h / 2).min(w / 2);
    if strip == 0 {
        return 0.0;
    }

    let mut border_pixels = Vec::new();

    for row in 0..h {
        for col in 0..w {
            if row < strip || row >= h - strip || col < strip || col >= w - strip {
                border_pixels.push(data[[row, col]]);
            }
        }
    }

    if border_pixels.is_empty() {
        return 0.0;
    }

    border_pixels.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    border_pixels[border_pixels.len() / 2]
}

/// Compute intensity-weighted centroid within the opened mask's bounding box,
/// with background subtraction.
///
/// Returns `(center_row, center_col)` in f64 pixel coordinates.
fn masked_centroid(
    data: &Array2<f32>,
    mask: &Array2<bool>,
    min_row: usize,
    max_row: usize,
    min_col: usize,
    max_col: usize,
    background: f32,
) -> (f64, f64) {
    let mut sum_row = 0.0_f64;
    let mut sum_col = 0.0_f64;
    let mut sum_weight = 0.0_f64;

    for row in min_row..=max_row {
        for col in min_col..=max_col {
            if mask[[row, col]] {
                let weight = (data[[row, col]] - background).max(0.0) as f64;
                sum_row += row as f64 * weight;
                sum_col += col as f64 * weight;
                sum_weight += weight;
            }
        }
    }

    if sum_weight > 0.0 {
        (sum_row / sum_weight, sum_col / sum_weight)
    } else {
        // Fallback: geometric center of bounding box.
        (
            (min_row + max_row) as f64 / 2.0,
            (min_col + max_col) as f64 / 2.0,
        )
    }
}
