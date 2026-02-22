use std::collections::{BTreeSet, HashMap};

use ndarray::Array2;
use rayon::prelude::*;
use tracing::info;

use crate::color::debayer::{luminance, DebayerMethod};
use crate::color::process::{read_color_frame, read_luminance_frame};
use crate::consts::MEAN_REFERENCE_KEEP_FRACTION;
use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame};
use crate::io::ser::SerReader;
use crate::quality::score_with_metric;
use crate::stack::ap_local::{stack_ap_cached, stack_ap_cached_color};

// Re-export public types so external code can continue to use `stack::multi_point::*`.
pub use crate::stack::ap_grid::{
    auto_ap_size, auto_ap_size_from_frame, build_ap_grid, extract_region, extract_region_shifted,
    AlignmentPoint, ApGrid, LocalStackMethod, MultiPointConfig,
};
pub use crate::stack::ap_local::blend_ap_stacks;
pub use crate::stack::reference::{build_mean_reference, build_mean_reference_color};

/// Score all APs across all frames using frame-major loop (read each frame once).
/// Returns `quality_matrix[ap_index]` = Vec of (frame_index, score), sorted descending.
pub fn score_all_aps(
    reader: &SerReader,
    grid: &ApGrid,
    offsets: &[AlignmentOffset],
    config: &MultiPointConfig,
) -> Result<Vec<Vec<(usize, f64)>>> {
    let total_frames = reader.frame_count();
    let num_aps = grid.points.len();
    let half = config.ap_size / 2;

    // quality_matrix[ap][frame] = score
    let mut quality_matrix: Vec<Vec<f64>> = vec![vec![0.0; total_frames]; num_aps];

    // Frame-major: read each frame once, score all APs
    for frame_idx in 0..total_frames {
        let frame = reader.read_frame(frame_idx)?;

        for ap in &grid.points {
            let region =
                extract_region_shifted(&frame.data, ap.cy, ap.cx, half, &offsets[frame_idx]);

            let score = score_with_metric(&region, &config.quality_metric);

            quality_matrix[ap.index][frame_idx] = score;
        }
    }

    // For each AP, sort frames by score descending, return top N
    let keep_count = ((total_frames as f32 * config.select_percentage).ceil() as usize)
        .max(1)
        .min(total_frames);

    let mut result = Vec::with_capacity(num_aps);
    for ap_scores in &quality_matrix {
        let mut indexed: Vec<(usize, f64)> =
            ap_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(keep_count);
        result.push(indexed);
    }

    Ok(result)
}

/// Top-level orchestrator for multi-alignment-point stacking.
///
/// Pipeline:
/// 1. Global align all frames vs frame 0 (offsets only, no shifting)
/// 2. Build mean reference from top-quality frames
/// 3. Build AP grid on mean reference
/// 4. Score quality per-AP per-frame (frame-major loop)
/// 5. Select best frames per-AP
/// 6. Local align + stack each AP (with confidence check + quality weighting)
/// 7. Blend AP stacks with cosine weighting
pub fn multi_point_stack<F>(
    reader: &SerReader,
    config: &MultiPointConfig,
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

    // Step 1: Global alignment — compute offsets in parallel
    info!(
        "Computing global alignment offsets for {} frames",
        total_frames
    );
    let rest_offsets: Vec<Result<AlignmentOffset>> = (1..total_frames)
        .into_par_iter()
        .map(|i| {
            let frame = reader.read_frame(i)?;
            crate::align::phase_correlation::compute_offset(&reference, &frame)
        })
        .collect();

    let mut global_offsets = Vec::with_capacity(total_frames);
    global_offsets.push(AlignmentOffset::default());
    for offset_result in rest_offsets {
        global_offsets.push(offset_result?);
    }
    on_progress(0.1);

    // Step 2: Build mean reference from top-quality frames
    info!(
        "Building mean reference from top {}% frames",
        (MEAN_REFERENCE_KEEP_FRACTION * 100.0) as u32
    );
    let mean_ref = build_mean_reference(
        reader,
        &global_offsets,
        &config.quality_metric,
        MEAN_REFERENCE_KEEP_FRACTION,
    )?;
    on_progress(0.2);

    // Step 3: Build AP grid on mean reference
    info!("Building alignment point grid (ap_size={})", config.ap_size);
    let grid = build_ap_grid(&mean_ref, config);
    info!("Created {} alignment points", grid.points.len());

    if grid.points.is_empty() {
        return Err(JupiterError::Pipeline(
            "No alignment points created (image may be too dark or too small)".into(),
        ));
    }

    // Step 4: Per-AP quality scoring (frame-major)
    info!(
        "Scoring {} APs across {} frames",
        grid.points.len(),
        total_frames
    );
    let ap_selections = score_all_aps(reader, &grid, &global_offsets, config)?;
    on_progress(0.4);

    // Step 5 & 6: Per-AP local alignment + stacking (parallel)
    info!("Stacking {} alignment points", grid.points.len());

    // Pre-read all frames needed by any AP into a cache for parallel access
    let needed_frames: BTreeSet<usize> = ap_selections
        .iter()
        .flat_map(|sel| sel.iter().map(|(idx, _)| *idx))
        .collect();

    let mut frame_cache: HashMap<usize, Frame> = HashMap::with_capacity(needed_frames.len());
    for &idx in &needed_frames {
        frame_cache.insert(idx, reader.read_frame(idx)?);
    }

    let ap_stacks: Vec<(AlignmentPoint, Array2<f32>)> = grid
        .points
        .par_iter()
        .map(|ap| {
            let stacked_patch = stack_ap_cached(
                &frame_cache,
                ap,
                &ap_selections[ap.index],
                &global_offsets,
                &mean_ref,
                config,
            );
            (ap.clone(), stacked_patch)
        })
        .collect();
    on_progress(0.9);

    // Step 7: Blend
    info!("Blending alignment point stacks");
    let blended = blend_ap_stacks(&ap_stacks, h, w, config.ap_size);
    on_progress(1.0);

    Ok(Frame::new(blended, reference.original_bit_depth))
}

/// Score all APs across all frames for color input.
///
/// For each frame: read -> debayer (or split RGB) -> luminance -> score all APs -> drop color.
/// Returns the same structure as `score_all_aps`: `quality_matrix[ap_index]` = sorted `(frame_index, score)`.
fn score_all_aps_color(
    reader: &SerReader,
    grid: &ApGrid,
    offsets: &[AlignmentOffset],
    config: &MultiPointConfig,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
) -> Result<Vec<Vec<(usize, f64)>>> {
    let total_frames = reader.frame_count();
    let num_aps = grid.points.len();
    let half = config.ap_size / 2;

    let mut quality_matrix: Vec<Vec<f64>> = vec![vec![0.0; total_frames]; num_aps];

    for frame_idx in 0..total_frames {
        // Read and convert to luminance — only one color frame in memory at a time
        let lum = read_luminance_frame(reader, frame_idx, color_mode, debayer_method)?;

        for ap in &grid.points {
            let region = extract_region_shifted(&lum.data, ap.cy, ap.cx, half, &offsets[frame_idx]);

            let score = score_with_metric(&region, &config.quality_metric);

            quality_matrix[ap.index][frame_idx] = score;
        }
    }

    let keep_count = ((total_frames as f32 * config.select_percentage).ceil() as usize)
        .max(1)
        .min(total_frames);

    let mut result = Vec::with_capacity(num_aps);
    for ap_scores in &quality_matrix {
        let mut indexed: Vec<(usize, f64)> =
            ap_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(keep_count);
        result.push(indexed);
    }

    Ok(result)
}

/// Top-level orchestrator for multi-alignment-point stacking with color support.
///
/// Pipeline:
/// 1. Read frame 0 -> debayer -> luminance as reference
/// 2. Global alignment on luminance
/// 3. Build mean reference from top-quality frames (luminance)
/// 4. Build AP grid on mean reference
/// 5. Score APs on luminance (frame-major, memory-efficient)
/// 6. Build frame cache with (luminance, ColorFrame) for needed frames
/// 7. Per-AP local alignment (on luminance) + stack (R/G/B independently)
/// 8. Blend AP stacks per channel
/// 9. Return ColorFrame
pub fn multi_point_stack_color<F>(
    reader: &SerReader,
    config: &MultiPointConfig,
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

    // Step 1: Read reference frame and get luminance
    let ref_color = read_color_frame(reader, 0, color_mode, debayer_method)?;
    let ref_lum = luminance(&ref_color);
    let (h, w) = ref_lum.data.dim();

    // Step 2: Global alignment — compute offsets on luminance
    info!(
        "Computing global alignment offsets for {} color frames",
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
    for offset_result in rest_offsets {
        global_offsets.push(offset_result?);
    }
    on_progress(0.1);

    // Step 3: Build mean reference from top-quality frames (luminance)
    info!(
        "Building mean reference from top {}% color frames",
        (MEAN_REFERENCE_KEEP_FRACTION * 100.0) as u32
    );
    let mean_ref = build_mean_reference_color(
        reader,
        &global_offsets,
        &config.quality_metric,
        MEAN_REFERENCE_KEEP_FRACTION,
        color_mode,
        debayer_method,
    )?;
    on_progress(0.2);

    // Step 4: Build AP grid on mean reference
    info!("Building alignment point grid (ap_size={})", config.ap_size);
    let grid = build_ap_grid(&mean_ref, config);
    info!("Created {} alignment points", grid.points.len());

    if grid.points.is_empty() {
        return Err(JupiterError::Pipeline(
            "No alignment points created (image may be too dark or too small)".into(),
        ));
    }

    // Step 5: Per-AP quality scoring on luminance (frame-major, memory-efficient)
    info!(
        "Scoring {} APs across {} color frames",
        grid.points.len(),
        total_frames
    );
    let ap_selections = score_all_aps_color(
        reader,
        &grid,
        &global_offsets,
        config,
        color_mode,
        debayer_method,
    )?;
    on_progress(0.4);

    // Step 6: Build frame cache — (luminance, ColorFrame) for needed frames
    info!("Loading selected color frames into cache");
    let needed_frames: BTreeSet<usize> = ap_selections
        .iter()
        .flat_map(|sel| sel.iter().map(|(idx, _)| *idx))
        .collect();

    let mut frame_cache: HashMap<usize, (Frame, ColorFrame)> =
        HashMap::with_capacity(needed_frames.len());
    for &idx in &needed_frames {
        let cf = read_color_frame(reader, idx, color_mode, debayer_method)?;
        let lum = luminance(&cf);
        frame_cache.insert(idx, (lum, cf));
    }

    // Step 7: Per-AP local alignment + stacking (parallel)
    info!("Stacking {} alignment points (color)", grid.points.len());
    type ColorApStack = (AlignmentPoint, (Array2<f32>, Array2<f32>, Array2<f32>));
    let ap_stacks: Vec<ColorApStack> = grid
        .points
        .par_iter()
        .map(|ap| {
            let stacked = stack_ap_cached_color(
                &frame_cache,
                ap,
                &ap_selections[ap.index],
                &global_offsets,
                &mean_ref,
                config,
            );
            (ap.clone(), stacked)
        })
        .collect();
    on_progress(0.9);

    // Step 8: Blend per channel
    info!("Blending color alignment point stacks");
    let bit_depth = ref_color.red.original_bit_depth;

    let red_stacks: Vec<(AlignmentPoint, Array2<f32>)> = ap_stacks
        .iter()
        .map(|(ap, (r, _, _))| (ap.clone(), r.clone()))
        .collect();
    let green_stacks: Vec<(AlignmentPoint, Array2<f32>)> = ap_stacks
        .iter()
        .map(|(ap, (_, g, _))| (ap.clone(), g.clone()))
        .collect();
    let blue_stacks: Vec<(AlignmentPoint, Array2<f32>)> = ap_stacks
        .iter()
        .map(|(ap, (_, _, b))| (ap.clone(), b.clone()))
        .collect();

    let (blended_r, (blended_g, blended_b)) = rayon::join(
        || blend_ap_stacks(&red_stacks, h, w, config.ap_size),
        || {
            rayon::join(
                || blend_ap_stacks(&green_stacks, h, w, config.ap_size),
                || blend_ap_stacks(&blue_stacks, h, w, config.ap_size),
            )
        },
    );
    on_progress(1.0);

    Ok(ColorFrame {
        red: Frame::new(blended_r, bit_depth),
        green: Frame::new(blended_g, bit_depth),
        blue: Frame::new(blended_b, bit_depth),
    })
}
