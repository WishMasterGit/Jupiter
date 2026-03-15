use ndarray::Array2;
use rayon::prelude::*;

use crate::consts::{
    PARALLEL_FRAME_THRESHOLD, PRE_CENTER_MIN_DETECTION_FRACTION, PRE_CENTER_MIN_OVERLAP_FRACTION,
};
use crate::detection::config::DetectionConfig;
use crate::detection::planet::detect_planet_in_frame;
use crate::error::Result;
use crate::frame::{AlignmentOffset, ColorFrame, Frame};
use crate::io::crop::CropRect;
use crate::io::ser::SerReader;

use super::phase_correlation::shift_frame;

/// Detect planet centroids in a slice of in-memory frames.
///
/// Returns `(cy, cx)` for each frame, or `None` if detection failed.
pub fn detect_centroids(frames: &[Frame], config: &DetectionConfig) -> Vec<Option<(f64, f64)>> {
    if frames.len() >= PARALLEL_FRAME_THRESHOLD {
        frames
            .par_iter()
            .enumerate()
            .map(|(i, frame)| {
                detect_planet_in_frame(&frame.data, i, config).map(|det| (det.cy, det.cx))
            })
            .collect()
    } else {
        frames
            .iter()
            .enumerate()
            .map(|(i, frame)| {
                detect_planet_in_frame(&frame.data, i, config).map(|det| (det.cy, det.cx))
            })
            .collect()
    }
}

/// Detect planet centroids by streaming frames from a SER reader.
pub fn detect_centroids_streaming(
    reader: &SerReader,
    indices: &[usize],
    config: &DetectionConfig,
) -> Result<Vec<Option<(f64, f64)>>> {
    let results: Vec<Option<(f64, f64)>> = indices
        .iter()
        .map(|&idx| {
            let frame = reader.read_frame(idx).ok()?;
            detect_planet_in_frame(&frame.data, idx, config).map(|det| (det.cy, det.cx))
        })
        .collect();
    Ok(results)
}

/// Compute centering offsets to shift each frame's planet to the image center.
///
/// Returns `None` if too few frames have successful detections
/// (below `PRE_CENTER_MIN_DETECTION_FRACTION`).
/// For individual failed detections, the median centroid is used as fallback.
pub fn compute_centering_offsets(
    centroids: &[Option<(f64, f64)>],
    h: usize,
    w: usize,
) -> Option<Vec<AlignmentOffset>> {
    let valid: Vec<(f64, f64)> = centroids.iter().filter_map(|c| *c).collect();
    let detection_fraction = valid.len() as f32 / centroids.len() as f32;

    if detection_fraction < PRE_CENTER_MIN_DETECTION_FRACTION {
        return None;
    }

    // Compute median centroid as fallback for failed detections
    let fallback = median_centroid(&valid);
    let center_y = h as f64 / 2.0;
    let center_x = w as f64 / 2.0;

    let offsets = centroids
        .iter()
        .map(|centroid| {
            let (cy, cx) = centroid.unwrap_or(fallback);
            AlignmentOffset {
                dx: center_x - cx,
                dy: center_y - cy,
            }
        })
        .collect();

    Some(offsets)
}

/// Compute the common valid overlap region after all frames are shifted by their offsets.
///
/// Returns `None` if the overlap area is less than `PRE_CENTER_MIN_OVERLAP_FRACTION`
/// of the original frame area.
pub fn compute_common_overlap(offsets: &[AlignmentOffset], h: usize, w: usize) -> Option<CropRect> {
    // For each shifted frame, the valid region in the output coordinate space is:
    //   rows: [max(0, dy) .. h + min(0, dy)]
    //   cols: [max(0, dx) .. w + min(0, dx)]
    // We need the intersection of all these regions.
    let mut min_x = 0.0_f64;
    let mut min_y = 0.0_f64;
    let mut max_x = w as f64;
    let mut max_y = h as f64;

    for offset in offsets {
        // After shifting by (dx, dy), source pixel (sy, sx) maps to output (sy+dy, sx+dx).
        // Valid output rows: [max(0, dy) .. h-1+min(0, dy)]
        let valid_min_y = offset.dy.max(0.0);
        let valid_max_y = (h as f64) + offset.dy.min(0.0);
        let valid_min_x = offset.dx.max(0.0);
        let valid_max_x = (w as f64) + offset.dx.min(0.0);

        min_y = min_y.max(valid_min_y);
        max_y = max_y.min(valid_max_y);
        min_x = min_x.max(valid_min_x);
        max_x = max_x.min(valid_max_x);
    }

    // Snap to integer pixel boundaries (round inward)
    let x = min_x.ceil() as u32;
    let y = min_y.ceil() as u32;
    let x_end = max_x.floor() as u32;
    let y_end = max_y.floor() as u32;

    if x_end <= x || y_end <= y {
        return None;
    }

    let crop_w = x_end - x;
    let crop_h = y_end - y;
    let overlap_area = (crop_w as f64) * (crop_h as f64);
    let original_area = (h as f64) * (w as f64);

    if overlap_area / original_area < PRE_CENTER_MIN_OVERLAP_FRACTION as f64 {
        return None;
    }

    Some(CropRect {
        x,
        y,
        width: crop_w,
        height: crop_h,
    })
}

/// Crop a frame to a `CropRect` subregion.
pub fn crop_frame(frame: &Frame, crop: &CropRect) -> Frame {
    let data = crop_array(&frame.data, crop);
    Frame::new(data, frame.original_bit_depth)
}

/// Crop a raw array to a `CropRect` subregion.
fn crop_array(data: &Array2<f32>, crop: &CropRect) -> Array2<f32> {
    let y = crop.y as usize;
    let x = crop.x as usize;
    let h = crop.height as usize;
    let w = crop.width as usize;
    data.slice(ndarray::s![y..y + h, x..x + w]).to_owned()
}

/// Shift each frame by its centering offset, then optionally crop to the common overlap.
pub fn pre_center_frames(
    frames: &[Frame],
    offsets: &[AlignmentOffset],
    crop: Option<&CropRect>,
) -> Vec<Frame> {
    let apply = |frame: &Frame, offset: &AlignmentOffset| {
        let shifted = shift_frame(frame, offset);
        match crop {
            Some(rect) => crop_frame(&shifted, rect),
            None => shifted,
        }
    };

    if frames.len() >= PARALLEL_FRAME_THRESHOLD {
        frames
            .par_iter()
            .zip(offsets.par_iter())
            .map(|(frame, offset)| apply(frame, offset))
            .collect()
    } else {
        frames
            .iter()
            .zip(offsets.iter())
            .map(|(frame, offset)| apply(frame, offset))
            .collect()
    }
}

/// Shift each color frame by its centering offset, then optionally crop.
pub fn pre_center_color_frames(
    frames: &[ColorFrame],
    offsets: &[AlignmentOffset],
    crop: Option<&CropRect>,
) -> Vec<ColorFrame> {
    let apply = |cf: &ColorFrame, offset: &AlignmentOffset| {
        let sr = shift_frame(&cf.red, offset);
        let sg = shift_frame(&cf.green, offset);
        let sb = shift_frame(&cf.blue, offset);
        match crop {
            Some(rect) => ColorFrame {
                red: crop_frame(&sr, rect),
                green: crop_frame(&sg, rect),
                blue: crop_frame(&sb, rect),
            },
            None => ColorFrame {
                red: sr,
                green: sg,
                blue: sb,
            },
        }
    };

    if frames.len() >= PARALLEL_FRAME_THRESHOLD {
        frames
            .par_iter()
            .zip(offsets.par_iter())
            .map(|(cf, offset)| apply(cf, offset))
            .collect()
    } else {
        frames
            .iter()
            .zip(offsets.iter())
            .map(|(cf, offset)| apply(cf, offset))
            .collect()
    }
}

/// Compute the median of a set of 2D centroids (component-wise median).
fn median_centroid(centroids: &[(f64, f64)]) -> (f64, f64) {
    if centroids.is_empty() {
        return (0.0, 0.0);
    }
    let mut ys: Vec<f64> = centroids.iter().map(|(cy, _)| *cy).collect();
    let mut xs: Vec<f64> = centroids.iter().map(|(_, cx)| *cx).collect();
    ys.sort_unstable_by(|a, b| a.total_cmp(b));
    xs.sort_unstable_by(|a, b| a.total_cmp(b));
    (ys[ys.len() / 2], xs[xs.len() / 2])
}
