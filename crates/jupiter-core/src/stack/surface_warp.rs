use std::collections::HashMap;

use ndarray::Array2;
use rayon::prelude::*;
use tracing::info;

use crate::align::phase_correlation::{bilinear_sample, compute_offset_with_confidence};
use crate::color::debayer::DebayerMethod;
use crate::color::process::{read_color_frame, read_luminance_frame};
use crate::consts::{MEAN_REFERENCE_KEEP_FRACTION, MIN_CORRELATION_CONFIDENCE};
use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame};
use crate::io::ser::SerReader;
use crate::quality::score_with_metric;
use crate::stack::ap_grid::{
    build_ap_grid, extract_region, extract_region_shifted, ApGrid, MultiPointConfig,
};
use crate::stack::reference::{build_mean_reference, build_mean_reference_color};

/// Configuration for surface-model warping stacking.
///
/// Reuses many of the same parameters as multi-point stacking, but the
/// stacking step is fundamentally different: instead of blending independent
/// patches, each frame is warped with a smooth per-pixel deformation field
/// derived from the AP shift grid, then stacked as a whole image.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SurfaceWarpConfig {
    /// Region size in pixels for alignment points (default: 64).
    pub ap_size: usize,
    /// Padding for local alignment FFT search (default: 16).
    pub search_radius: usize,
    /// Fraction of frames to select (0.0–1.0, default: 0.25).
    pub select_percentage: f32,
    /// Skip APs where reference mean brightness is below this (default: 0.05).
    pub min_brightness: f32,
    /// Quality metric for frame scoring.
    pub quality_metric: crate::pipeline::config::QualityMetric,
}

impl Default for SurfaceWarpConfig {
    fn default() -> Self {
        Self {
            ap_size: 64,
            search_radius: 16,
            select_percentage: 0.25,
            min_brightness: 0.05,
            quality_metric: crate::pipeline::config::QualityMetric::Laplacian,
        }
    }
}

/// Interpolate per-AP shifts into a smooth per-pixel deformation field using
/// bilinear interpolation on the AP grid.
///
/// Returns `(shift_y, shift_x)` arrays of shape `(h, w)` giving the total
/// shift (global + local) at every pixel.
pub fn interpolate_shift_field(
    grid: &ApGrid,
    local_offsets: &HashMap<usize, AlignmentOffset>,
    global_offset: &AlignmentOffset,
    h: usize,
    w: usize,
) -> (Array2<f64>, Array2<f64>) {
    // Build a regular grid of AP positions and their total shifts.
    // Collect unique sorted row/col centres.
    let half = grid.ap_size / 2;
    let stride = half; // same as in build_ap_grid

    let mut row_positions: Vec<usize> = Vec::new();
    let mut col_positions: Vec<usize> = Vec::new();
    {
        let mut cy = half;
        while cy + half <= h {
            row_positions.push(cy);
            cy += stride;
        }
        let mut cx = half;
        while cx + half <= w {
            col_positions.push(cx);
            cx += stride;
        }
    }
    let grid_rows = row_positions.len();
    let grid_cols = col_positions.len();

    // Build 2D grid of shifts (row_idx, col_idx) -> (dy, dx).
    // APs that failed confidence check won't be in local_offsets; we use
    // global_offset only for those.
    let mut shift_dy_grid = Array2::<f64>::zeros((grid_rows, grid_cols));
    let mut shift_dx_grid = Array2::<f64>::zeros((grid_rows, grid_cols));

    for ap in &grid.points {
        // Find grid indices for this AP
        let ri = row_positions.iter().position(|&r| r == ap.cy);
        let ci = col_positions.iter().position(|&c| c == ap.cx);
        if let (Some(ri), Some(ci)) = (ri, ci) {
            if let Some(local) = local_offsets.get(&ap.index) {
                shift_dy_grid[[ri, ci]] = global_offset.dy + local.dy;
                shift_dx_grid[[ri, ci]] = global_offset.dx + local.dx;
            } else {
                // AP was rejected or not computed — fall back to global
                shift_dy_grid[[ri, ci]] = global_offset.dy;
                shift_dx_grid[[ri, ci]] = global_offset.dx;
            }
        }
    }

    // Fill grid cells with no AP (below min_brightness) using global offset
    // (they're already zeros, replace with global)
    for ri in 0..grid_rows {
        for ci in 0..grid_cols {
            let cy = row_positions[ri];
            let cx = col_positions[ci];
            let has_ap = grid.points.iter().any(|ap| ap.cy == cy && ap.cx == cx);
            if !has_ap {
                shift_dy_grid[[ri, ci]] = global_offset.dy;
                shift_dx_grid[[ri, ci]] = global_offset.dx;
            }
        }
    }

    // Bilinear interpolation to per-pixel level
    let mut field_dy = Array2::<f64>::zeros((h, w));
    let mut field_dx = Array2::<f64>::zeros((h, w));

    if grid_rows < 2 || grid_cols < 2 {
        // Degenerate: uniform shift
        field_dy.fill(global_offset.dy);
        field_dx.fill(global_offset.dx);
        return (field_dy, field_dx);
    }

    for row in 0..h {
        // Find bounding grid rows
        let (ri0, ri1, fy) = find_interval(&row_positions, row);
        for col in 0..w {
            let (ci0, ci1, fx) = find_interval(&col_positions, col);

            // Bilinear interpolation
            let dy00 = shift_dy_grid[[ri0, ci0]];
            let dy10 = shift_dy_grid[[ri0, ci1]];
            let dy01 = shift_dy_grid[[ri1, ci0]];
            let dy11 = shift_dy_grid[[ri1, ci1]];
            field_dy[[row, col]] = dy00 * (1.0 - fx) * (1.0 - fy)
                + dy10 * fx * (1.0 - fy)
                + dy01 * (1.0 - fx) * fy
                + dy11 * fx * fy;

            let dx00 = shift_dx_grid[[ri0, ci0]];
            let dx10 = shift_dx_grid[[ri0, ci1]];
            let dx01 = shift_dx_grid[[ri1, ci0]];
            let dx11 = shift_dx_grid[[ri1, ci1]];
            field_dx[[row, col]] = dx00 * (1.0 - fx) * (1.0 - fy)
                + dx10 * fx * (1.0 - fy)
                + dx01 * (1.0 - fx) * fy
                + dx11 * fx * fy;
        }
    }

    (field_dy, field_dx)
}

/// Find the bracketing interval and interpolation fraction for `val` in a
/// sorted list of positions.
fn find_interval(positions: &[usize], val: usize) -> (usize, usize, f64) {
    let n = positions.len();
    if n == 0 {
        return (0, 0, 0.0);
    }
    if val <= positions[0] {
        return (0, 0, 0.0);
    }
    if val >= positions[n - 1] {
        return (n - 1, n - 1, 0.0);
    }
    // Binary search for interval
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if positions[mid] <= val {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = (positions[hi] - positions[lo]) as f64;
    let frac = if span > 0.0 {
        (val - positions[lo]) as f64 / span
    } else {
        0.0
    };
    (lo, hi, frac)
}

/// Warp a frame using a per-pixel deformation field via bilinear sampling.
pub fn warp_frame(
    data: &Array2<f32>,
    shift_field_y: &Array2<f64>,
    shift_field_x: &Array2<f64>,
) -> Array2<f32> {
    let (h, w) = data.dim();
    let mut result = Array2::<f32>::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            let src_y = row as f64 - shift_field_y[[row, col]];
            let src_x = col as f64 - shift_field_x[[row, col]];
            result[[row, col]] = bilinear_sample(data, src_y, src_x);
        }
    }

    result
}

/// Compute local AP shifts for one frame, with confidence filtering.
///
/// Returns a map from AP index to local offset for APs that pass the
/// confidence threshold.
fn compute_local_shifts(
    frame_data: &Array2<f32>,
    reference: &Array2<f32>,
    grid: &ApGrid,
    global_offset: &AlignmentOffset,
    search_radius: usize,
) -> HashMap<usize, AlignmentOffset> {
    let half = grid.ap_size / 2;
    let search_half = half + search_radius;
    let mut shifts = HashMap::new();

    for ap in &grid.points {
        let ref_search = extract_region(reference, ap.cy, ap.cx, search_half);
        let tgt_search =
            extract_region_shifted(frame_data, ap.cy, ap.cx, search_half, global_offset);

        let (local_offset, confidence) = compute_offset_with_confidence(&ref_search, &tgt_search)
            .unwrap_or((AlignmentOffset::default(), 0.0));

        if confidence >= MIN_CORRELATION_CONFIDENCE {
            shifts.insert(ap.index, local_offset);
        }
    }

    shifts
}

/// Top-level orchestrator for surface-model warping stacking (mono).
///
/// Pipeline:
/// 1. Global align all frames vs frame 0
/// 2. Build mean reference from top-quality frames
/// 3. Build AP grid on mean reference
/// 4. Score all frames globally, select top N%
/// 5. For each selected frame: compute local shifts → interpolate → warp → accumulate
/// 6. Quality-weighted mean of all warped frames
pub fn surface_warp_stack<F>(
    reader: &SerReader,
    config: &SurfaceWarpConfig,
    mut on_progress: F,
) -> Result<Frame>
where
    F: FnMut(f32),
{
    let total_frames = reader.frame_count();
    if total_frames == 0 {
        return Err(JupiterError::EmptySequence);
    }

    let reference = reader.read_frame(0)?;
    let (h, w) = reference.data.dim();

    // Step 1: Global alignment
    info!("Surface warp: global alignment of {} frames", total_frames);
    let rest_offsets: Vec<Result<AlignmentOffset>> = (1..total_frames)
        .into_par_iter()
        .map(|i| {
            let frame = reader.read_frame(i)?;
            crate::align::phase_correlation::compute_offset(&reference, &frame)
        })
        .collect();

    let mut global_offsets = Vec::with_capacity(total_frames);
    global_offsets.push(AlignmentOffset::default());
    for r in rest_offsets {
        global_offsets.push(r?);
    }
    on_progress(0.1);

    // Step 2: Mean reference
    let mp_config = to_mp_config(config);
    info!("Surface warp: building mean reference");
    let mean_ref = build_mean_reference(
        reader,
        &global_offsets,
        &config.quality_metric,
        MEAN_REFERENCE_KEEP_FRACTION,
    )?;
    on_progress(0.2);

    // Step 3: AP grid
    let grid = build_ap_grid(&mean_ref, &mp_config);
    info!("Surface warp: {} alignment points", grid.points.len());
    if grid.points.is_empty() {
        return Err(JupiterError::Pipeline("No alignment points created".into()));
    }

    // Step 4: Score + select
    let selected = score_and_select_frames(reader, &global_offsets, config)?;
    let frame_count = selected.len();
    info!("Surface warp: selected {} frames", frame_count);
    on_progress(0.3);

    // Step 5–6: Warp + accumulate (quality-weighted)
    info!("Surface warp: warping and stacking {} frames", frame_count);
    let mut accumulator = Array2::<f64>::zeros((h, w));
    let mut total_weight: f64 = 0.0;

    for (i, &(frame_idx, quality_score)) in selected.iter().enumerate() {
        let frame = reader.read_frame(frame_idx)?;

        let local_shifts = compute_local_shifts(
            &frame.data,
            &mean_ref,
            &grid,
            &global_offsets[frame_idx],
            config.search_radius,
        );

        let (field_dy, field_dx) =
            interpolate_shift_field(&grid, &local_shifts, &global_offsets[frame_idx], h, w);

        let warped = warp_frame(&frame.data, &field_dy, &field_dx);

        let weight = quality_score.max(0.0);
        total_weight += weight;
        for row in 0..h {
            for col in 0..w {
                accumulator[[row, col]] += warped[[row, col]] as f64 * weight;
            }
        }

        on_progress(0.3 + 0.65 * (i + 1) as f32 / frame_count as f32);
    }

    let result = if total_weight > 1e-15 {
        accumulator.mapv(|v| (v / total_weight) as f32)
    } else {
        accumulator.mapv(|v| v as f32)
    };

    on_progress(1.0);
    Ok(Frame::new(result, reference.original_bit_depth))
}

/// Top-level orchestrator for surface-model warping stacking (color).
pub fn surface_warp_stack_color<F>(
    reader: &SerReader,
    config: &SurfaceWarpConfig,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
    mut on_progress: F,
) -> Result<ColorFrame>
where
    F: FnMut(f32),
{
    let total_frames = reader.frame_count();
    if total_frames == 0 {
        return Err(JupiterError::EmptySequence);
    }

    // Step 1: Get luminance reference for alignment
    let ref_color = read_color_frame(reader, 0, color_mode, debayer_method)?;
    let ref_lum = crate::color::debayer::luminance(&ref_color);
    let (h, w) = ref_lum.data.dim();
    let bit_depth = ref_color.red.original_bit_depth;

    // Step 2: Global alignment on luminance
    info!(
        "Surface warp color: global alignment of {} frames",
        total_frames
    );
    let rest_offsets: Vec<Result<AlignmentOffset>> = (1..total_frames)
        .into_par_iter()
        .map(|i| {
            let lum = read_luminance_frame(reader, i, color_mode, debayer_method)?;
            crate::align::phase_correlation::compute_offset(&ref_lum, &lum)
        })
        .collect();

    let mut global_offsets = Vec::with_capacity(total_frames);
    global_offsets.push(AlignmentOffset::default());
    for r in rest_offsets {
        global_offsets.push(r?);
    }
    on_progress(0.1);

    // Step 3: Mean reference (luminance)
    let mp_config = to_mp_config(config);
    info!("Surface warp color: building mean reference");
    let mean_ref = build_mean_reference_color(
        reader,
        &global_offsets,
        &config.quality_metric,
        MEAN_REFERENCE_KEEP_FRACTION,
        color_mode,
        debayer_method,
    )?;
    on_progress(0.2);

    // Step 4: AP grid
    let grid = build_ap_grid(&mean_ref, &mp_config);
    info!("Surface warp color: {} alignment points", grid.points.len());
    if grid.points.is_empty() {
        return Err(JupiterError::Pipeline("No alignment points created".into()));
    }

    // Step 5: Score + select (on luminance)
    let selected =
        score_and_select_frames_color(reader, &global_offsets, config, color_mode, debayer_method)?;
    let frame_count = selected.len();
    info!("Surface warp color: selected {} frames", frame_count);
    on_progress(0.3);

    // Step 6: Warp + accumulate per channel
    let mut acc_r = Array2::<f64>::zeros((h, w));
    let mut acc_g = Array2::<f64>::zeros((h, w));
    let mut acc_b = Array2::<f64>::zeros((h, w));
    let mut total_weight: f64 = 0.0;

    for (i, &(frame_idx, quality_score)) in selected.iter().enumerate() {
        // Read color frame
        let cf = read_color_frame(reader, frame_idx, color_mode, debayer_method)?;
        let lum = crate::color::debayer::luminance(&cf);

        // Compute local shifts on luminance
        let local_shifts = compute_local_shifts(
            &lum.data,
            &mean_ref,
            &grid,
            &global_offsets[frame_idx],
            config.search_radius,
        );

        let (field_dy, field_dx) =
            interpolate_shift_field(&grid, &local_shifts, &global_offsets[frame_idx], h, w);

        // Warp each channel
        let warped_r = warp_frame(&cf.red.data, &field_dy, &field_dx);
        let warped_g = warp_frame(&cf.green.data, &field_dy, &field_dx);
        let warped_b = warp_frame(&cf.blue.data, &field_dy, &field_dx);

        let weight = quality_score.max(0.0);
        total_weight += weight;
        for row in 0..h {
            for col in 0..w {
                acc_r[[row, col]] += warped_r[[row, col]] as f64 * weight;
                acc_g[[row, col]] += warped_g[[row, col]] as f64 * weight;
                acc_b[[row, col]] += warped_b[[row, col]] as f64 * weight;
            }
        }

        on_progress(0.3 + 0.65 * (i + 1) as f32 / frame_count as f32);
    }

    let finalize = |acc: Array2<f64>| -> Array2<f32> {
        if total_weight > 1e-15 {
            acc.mapv(|v| (v / total_weight) as f32)
        } else {
            acc.mapv(|v| v as f32)
        }
    };

    let result_r = finalize(acc_r);
    let result_g = finalize(acc_g);
    let result_b = finalize(acc_b);

    on_progress(1.0);
    Ok(ColorFrame {
        red: Frame::new(result_r, bit_depth),
        green: Frame::new(result_g, bit_depth),
        blue: Frame::new(result_b, bit_depth),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert SurfaceWarpConfig to a MultiPointConfig for shared helpers.
fn to_mp_config(config: &SurfaceWarpConfig) -> MultiPointConfig {
    MultiPointConfig {
        ap_size: config.ap_size,
        search_radius: config.search_radius,
        select_percentage: config.select_percentage,
        min_brightness: config.min_brightness,
        quality_metric: config.quality_metric,
        ..Default::default()
    }
}

/// Score all frames globally and return the top N% as `(frame_index, score)`.
fn score_and_select_frames(
    reader: &SerReader,
    _offsets: &[AlignmentOffset],
    config: &SurfaceWarpConfig,
) -> Result<Vec<(usize, f64)>> {
    let total = reader.frame_count();

    let mut scores: Vec<(usize, f64)> = (0..total)
        .map(|i| {
            let frame = reader.read_frame(i).unwrap();
            let score = score_with_metric(&frame.data, &config.quality_metric);
            (i, score)
        })
        .collect();

    scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    let keep = ((total as f32 * config.select_percentage).ceil() as usize)
        .max(1)
        .min(total);
    scores.truncate(keep);
    Ok(scores)
}

/// Score all color frames on luminance and return top N%.
fn score_and_select_frames_color(
    reader: &SerReader,
    _offsets: &[AlignmentOffset],
    config: &SurfaceWarpConfig,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
) -> Result<Vec<(usize, f64)>> {
    let total = reader.frame_count();

    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(total);
    for i in 0..total {
        let lum = read_luminance_frame(reader, i, color_mode, debayer_method)?;
        let score = score_with_metric(&lum.data, &config.quality_metric);
        scores.push((i, score));
    }

    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    let keep = ((total as f32 * config.select_percentage).ceil() as usize)
        .max(1)
        .min(total);
    scores.truncate(keep);
    Ok(scores)
}
