use std::collections::{BTreeSet, HashMap};

use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::align::phase_correlation::{bilinear_sample, compute_offset_array};
use crate::color::debayer::{debayer, luminance, DebayerMethod};
use crate::consts::EPSILON;
use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame};
use crate::io::ser::SerReader;
use crate::pipeline::config::QualityMetric;
use crate::quality::gradient::gradient_score_array;
use crate::quality::laplacian::laplacian_variance_array;

/// Local stacking method for per-AP patches.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum LocalStackMethod {
    #[default]
    Mean,
    Median,
    SigmaClip { sigma: f32, iterations: usize },
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

            let score = match config.quality_metric {
                QualityMetric::Laplacian => laplacian_variance_array(&region),
                QualityMetric::Gradient => gradient_score_array(&region),
            };

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
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(keep_count);
        result.push(indexed);
    }

    Ok(result)
}

/// Stack one AP using pre-cached frames (for parallel execution).
fn stack_ap_cached(
    frame_cache: &HashMap<usize, Frame>,
    ap: &AlignmentPoint,
    selected_frames: &[(usize, f64)],
    global_offsets: &[AlignmentOffset],
    reference_data: &Array2<f32>,
    config: &MultiPointConfig,
) -> Array2<f32> {
    let half = config.ap_size / 2;
    let search_half = half + config.search_radius;

    let ref_search = extract_region(reference_data, ap.cy, ap.cx, search_half);

    let mut patches: Vec<Array2<f32>> = Vec::with_capacity(selected_frames.len());

    for &(frame_idx, _) in selected_frames {
        let frame = match frame_cache.get(&frame_idx) {
            Some(f) => f,
            None => continue,
        };

        let tgt_search = extract_region_shifted(
            &frame.data,
            ap.cy,
            ap.cx,
            search_half,
            &global_offsets[frame_idx],
        );

        let local_offset = compute_offset_array(&ref_search, &tgt_search).unwrap_or_default();

        let search_size = search_half * 2;
        let center = search_size as f64 / 2.0;
        let patch_half = half;
        let patch_size = patch_half * 2;
        let mut patch = Array2::<f32>::zeros((patch_size, patch_size));

        for dr in 0..patch_size {
            for dc in 0..patch_size {
                let src_y = center + dr as f64 - patch_half as f64 - local_offset.dy;
                let src_x = center + dc as f64 - patch_half as f64 - local_offset.dx;
                patch[[dr, dc]] = bilinear_sample_2d(&tgt_search, src_y, src_x);
            }
        }

        patches.push(patch);
    }

    match config.local_stack_method {
        LocalStackMethod::Mean => mean_stack_arrays(&patches),
        LocalStackMethod::Median => median_stack_arrays(&patches),
        LocalStackMethod::SigmaClip { sigma, iterations } => {
            sigma_clip_stack_arrays(&patches, sigma, iterations)
        }
    }
}

/// Bilinear sample from an Array2<f32> (uses f64 coordinates, returns 0 for out-of-bounds).
fn bilinear_sample_2d(data: &Array2<f32>, y: f64, x: f64) -> f32 {
    let (h, w) = data.dim();

    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;

    let sample = |r: i64, c: i64| -> f32 {
        if r >= 0 && r < h as i64 && c >= 0 && c < w as i64 {
            data[[r as usize, c as usize]]
        } else {
            0.0
        }
    };

    let v00 = sample(y0, x0);
    let v10 = sample(y0, x1);
    let v01 = sample(y1, x0);
    let v11 = sample(y1, x1);

    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

/// Mean-stack a set of Array2 patches.
fn mean_stack_arrays(patches: &[Array2<f32>]) -> Array2<f32> {
    let (h, w) = patches[0].dim();
    let n = patches.len() as f32;
    let mut sum = Array2::<f32>::zeros((h, w));
    for p in patches {
        sum += p;
    }
    sum /= n;
    sum
}

/// Median-stack a set of Array2 patches.
fn median_stack_arrays(patches: &[Array2<f32>]) -> Array2<f32> {
    let (h, w) = patches[0].dim();
    let n = patches.len();
    let mut result = Array2::<f32>::zeros((h, w));
    let mut vals = vec![0.0f32; n];

    for row in 0..h {
        for col in 0..w {
            for (i, p) in patches.iter().enumerate() {
                vals[i] = p[[row, col]];
            }
            result[[row, col]] = if n == 1 {
                vals[0]
            } else if n % 2 == 1 {
                let mid = n / 2;
                *vals
                    .select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap())
                    .1
            } else {
                let mid = n / 2;
                let (_, upper, _) =
                    vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
                let upper_val = *upper;
                let lower_val = vals[..mid]
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                (lower_val + upper_val) / 2.0
            };
        }
    }

    result
}

/// Sigma-clip mean stack of Array2 patches.
fn sigma_clip_stack_arrays(patches: &[Array2<f32>], sigma: f32, iterations: usize) -> Array2<f32> {
    let (h, w) = patches[0].dim();
    let n = patches.len();
    let mut result = Array2::<f32>::zeros((h, w));
    let mut vals = vec![0.0f32; n];
    let mut mask = vec![true; n];

    for row in 0..h {
        for col in 0..w {
            for (i, p) in patches.iter().enumerate() {
                vals[i] = p[[row, col]];
                mask[i] = true;
            }

            for _ in 0..iterations {
                let (mean, stddev) = masked_mean_stddev(&vals, &mask);
                if stddev < EPSILON {
                    break;
                }
                let lo = mean - sigma * stddev;
                let hi = mean + sigma * stddev;
                for i in 0..n {
                    if mask[i] && (vals[i] < lo || vals[i] > hi) {
                        mask[i] = false;
                    }
                }
            }

            let mut sum = 0.0f32;
            let mut count = 0u32;
            for i in 0..n {
                if mask[i] {
                    sum += vals[i];
                    count += 1;
                }
            }
            result[[row, col]] = if count > 0 {
                sum / count as f32
            } else {
                vals.iter().sum::<f32>() / n as f32
            };
        }
    }

    result
}

fn masked_mean_stddev(values: &[f32], mask: &[bool]) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for (i, &v) in values.iter().enumerate() {
        if mask[i] {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / count as f32;
    let mut var_sum = 0.0f32;
    for (i, &v) in values.iter().enumerate() {
        if mask[i] {
            let d = v - mean;
            var_sum += d * d;
        }
    }
    let stddev = (var_sum / count as f32).sqrt();
    (mean, stddev)
}

/// Blend per-AP stacked patches using raised-cosine (Hann) weighting.
/// With 50% overlap, cosine weights form a partition of unity.
pub fn blend_ap_stacks(
    stacks: &[(AlignmentPoint, Array2<f32>)],
    h: usize,
    w: usize,
    ap_size: usize,
) -> Array2<f32> {
    let mut weighted_sum = Array2::<f64>::zeros((h, w));
    let mut weight_sum = Array2::<f64>::zeros((h, w));
    let half = ap_size / 2;

    for (ap, patch) in stacks {
        let patch_size = patch.dim().0;
        let patch_half = patch_size / 2;

        for dr in 0..patch_size {
            for dc in 0..patch_size {
                let img_r = ap.cy + dr;
                let img_c = ap.cx + dc;

                // Convert to image coordinates
                let img_r = if img_r >= patch_half {
                    img_r - patch_half
                } else {
                    continue;
                };
                let img_c = if img_c >= patch_half {
                    img_c - patch_half
                } else {
                    continue;
                };

                if img_r >= h || img_c >= w {
                    continue;
                }

                let wy = hann_weight(dr, half);
                let wx = hann_weight(dc, half);
                let weight = (wy * wx) as f64;

                weighted_sum[[img_r, img_c]] += patch[[dr, dc]] as f64 * weight;
                weight_sum[[img_r, img_c]] += weight;
            }
        }
    }

    let mut result = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            result[[row, col]] = if weight_sum[[row, col]] > 1e-12 {
                (weighted_sum[[row, col]] / weight_sum[[row, col]]) as f32
            } else {
                0.0
            };
        }
    }

    result
}

/// Raised cosine (Hann) weight for position `pos` within a window of `half_size`.
/// Returns 1.0 at center, tapering to 0.0 at edges.
fn hann_weight(pos: usize, half_size: usize) -> f32 {
    let size = half_size * 2;
    if size == 0 {
        return 1.0;
    }
    let t = pos as f32 / size as f32;
    0.5 * (1.0 - (std::f32::consts::TAU * t).cos())
}

/// Top-level orchestrator for multi-alignment-point stacking.
///
/// Pipeline:
/// 1. Global align all frames vs frame 0 (offsets only, no shifting)
/// 2. Build AP grid on reference frame
/// 3. Score quality per-AP per-frame (frame-major loop)
/// 4. Select best frames per-AP
/// 5. Local align + stack each AP
/// 6. Blend AP stacks with cosine weighting
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

    // Step 2: Build AP grid
    info!("Building alignment point grid (ap_size={})", config.ap_size);
    let grid = build_ap_grid(&reference.data, config);
    info!("Created {} alignment points", grid.points.len());

    if grid.points.is_empty() {
        return Err(JupiterError::Pipeline(
            "No alignment points created (image may be too dark or too small)".into(),
        ));
    }

    // Step 3: Per-AP quality scoring (frame-major)
    info!(
        "Scoring {} APs across {} frames",
        grid.points.len(),
        total_frames
    );
    let ap_selections = score_all_aps(reader, &grid, &global_offsets, config)?;
    on_progress(0.4);

    // Step 4 & 5: Per-AP local alignment + stacking (parallel)
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
                &reference.data,
                config,
            );
            (ap.clone(), stacked_patch)
        })
        .collect();
    on_progress(0.9);

    // Step 6: Blend
    info!("Blending alignment point stacks");
    let blended = blend_ap_stacks(&ap_stacks, h, w, config.ap_size);
    on_progress(1.0);

    Ok(Frame::new(blended, reference.original_bit_depth))
}

/// Score all APs across all frames for color input.
///
/// For each frame: read → debayer (or split RGB) → luminance → score all APs → drop color.
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
    let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);

    let mut quality_matrix: Vec<Vec<f64>> = vec![vec![0.0; total_frames]; num_aps];

    for frame_idx in 0..total_frames {
        // Read and convert to luminance — only one color frame in memory at a time
        let lum = if is_rgb_bgr {
            let cf = reader.read_frame_rgb(frame_idx)?;
            luminance(&cf)
        } else {
            let raw_frame = reader.read_frame(frame_idx)?;
            let cf = debayer(&raw_frame.data, color_mode, debayer_method, raw_frame.original_bit_depth)
                .ok_or_else(|| JupiterError::Pipeline("Debayer failed for color mode".into()))?;
            luminance(&cf)
        };

        for ap in &grid.points {
            let region = extract_region_shifted(&lum.data, ap.cy, ap.cx, half, &offsets[frame_idx]);

            let score = match config.quality_metric {
                QualityMetric::Laplacian => laplacian_variance_array(&region),
                QualityMetric::Gradient => gradient_score_array(&region),
            };

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
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(keep_count);
        result.push(indexed);
    }

    Ok(result)
}

/// Stack one AP using pre-cached color frames.
///
/// Local alignment is computed on luminance. R/G/B patches are extracted and
/// stacked independently using the configured local method.
fn stack_ap_cached_color(
    frame_cache: &HashMap<usize, (Frame, ColorFrame)>,
    ap: &AlignmentPoint,
    selected_frames: &[(usize, f64)],
    global_offsets: &[AlignmentOffset],
    reference_lum: &Array2<f32>,
    config: &MultiPointConfig,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let half = config.ap_size / 2;
    let search_half = half + config.search_radius;

    let ref_search = extract_region(reference_lum, ap.cy, ap.cx, search_half);

    let mut red_patches: Vec<Array2<f32>> = Vec::with_capacity(selected_frames.len());
    let mut green_patches: Vec<Array2<f32>> = Vec::with_capacity(selected_frames.len());
    let mut blue_patches: Vec<Array2<f32>> = Vec::with_capacity(selected_frames.len());

    for &(frame_idx, _) in selected_frames {
        let (lum, color) = match frame_cache.get(&frame_idx) {
            Some(entry) => entry,
            None => continue,
        };

        // Local alignment on luminance
        let tgt_search = extract_region_shifted(
            &lum.data,
            ap.cy,
            ap.cx,
            search_half,
            &global_offsets[frame_idx],
        );

        let local_offset = compute_offset_array(&ref_search, &tgt_search).unwrap_or_default();

        let patch_half = half;
        let patch_size = patch_half * 2;

        // Combined offset for bilinear sampling from full-size color channels
        let combined_dy = global_offsets[frame_idx].dy + local_offset.dy;
        let combined_dx = global_offsets[frame_idx].dx + local_offset.dx;

        let mut r_patch = Array2::<f32>::zeros((patch_size, patch_size));
        let mut g_patch = Array2::<f32>::zeros((patch_size, patch_size));
        let mut b_patch = Array2::<f32>::zeros((patch_size, patch_size));

        for dr in 0..patch_size {
            for dc in 0..patch_size {
                let src_y = (ap.cy as f64 + dr as f64 - patch_half as f64) - combined_dy;
                let src_x = (ap.cx as f64 + dc as f64 - patch_half as f64) - combined_dx;
                r_patch[[dr, dc]] = bilinear_sample(&color.red.data, src_y, src_x);
                g_patch[[dr, dc]] = bilinear_sample(&color.green.data, src_y, src_x);
                b_patch[[dr, dc]] = bilinear_sample(&color.blue.data, src_y, src_x);
            }
        }

        red_patches.push(r_patch);
        green_patches.push(g_patch);
        blue_patches.push(b_patch);
    }

    let stack_fn = |patches: &[Array2<f32>]| -> Array2<f32> {
        match config.local_stack_method {
            LocalStackMethod::Mean => mean_stack_arrays(patches),
            LocalStackMethod::Median => median_stack_arrays(patches),
            LocalStackMethod::SigmaClip { sigma, iterations } => {
                sigma_clip_stack_arrays(patches, sigma, iterations)
            }
        }
    };

    // Stack R/G/B in parallel
    let (stacked_r, (stacked_g, stacked_b)) = rayon::join(
        || stack_fn(&red_patches),
        || rayon::join(|| stack_fn(&green_patches), || stack_fn(&blue_patches)),
    );

    (stacked_r, stacked_g, stacked_b)
}

/// Top-level orchestrator for multi-alignment-point stacking with color support.
///
/// Pipeline:
/// 1. Read frame 0 → debayer → luminance as reference
/// 2. Global alignment on luminance
/// 3. Build AP grid on reference luminance
/// 4. Score APs on luminance (frame-major, memory-efficient)
/// 5. Build frame cache with (luminance, ColorFrame) for needed frames
/// 6. Per-AP local alignment (on luminance) + stack (R/G/B independently)
/// 7. Blend AP stacks per channel
/// 8. Return ColorFrame
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

    let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);

    // Step 1: Read reference frame and get luminance
    let ref_color = if is_rgb_bgr {
        reader.read_frame_rgb(0)?
    } else {
        let raw = reader.read_frame(0)?;
        debayer(&raw.data, color_mode, debayer_method, raw.original_bit_depth)
            .ok_or_else(|| JupiterError::Pipeline("Debayer failed for reference frame".into()))?
    };
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
            let lum = if is_rgb_bgr {
                let cf = reader.read_frame_rgb(i)?;
                luminance(&cf)
            } else {
                let raw = reader.read_frame(i)?;
                let cf = debayer(&raw.data, color_mode, debayer_method, raw.original_bit_depth)
                    .ok_or_else(|| JupiterError::Pipeline("Debayer failed".into()))?;
                luminance(&cf)
            };
            crate::align::phase_correlation::compute_offset(&ref_lum, &lum)
        })
        .collect();

    let mut global_offsets = Vec::with_capacity(total_frames);
    global_offsets.push(AlignmentOffset::default());
    for offset_result in rest_offsets {
        global_offsets.push(offset_result?);
    }
    on_progress(0.1);

    // Step 3: Build AP grid on reference luminance
    info!("Building alignment point grid (ap_size={})", config.ap_size);
    let grid = build_ap_grid(&ref_lum.data, config);
    info!("Created {} alignment points", grid.points.len());

    if grid.points.is_empty() {
        return Err(JupiterError::Pipeline(
            "No alignment points created (image may be too dark or too small)".into(),
        ));
    }

    // Step 4: Per-AP quality scoring on luminance (frame-major, memory-efficient)
    info!(
        "Scoring {} APs across {} color frames",
        grid.points.len(),
        total_frames
    );
    let ap_selections = score_all_aps_color(reader, &grid, &global_offsets, config, color_mode, debayer_method)?;
    on_progress(0.4);

    // Step 5: Build frame cache — (luminance, ColorFrame) for needed frames
    info!("Loading selected color frames into cache");
    let needed_frames: BTreeSet<usize> = ap_selections
        .iter()
        .flat_map(|sel| sel.iter().map(|(idx, _)| *idx))
        .collect();

    let mut frame_cache: HashMap<usize, (Frame, ColorFrame)> =
        HashMap::with_capacity(needed_frames.len());
    for &idx in &needed_frames {
        let cf = if is_rgb_bgr {
            reader.read_frame_rgb(idx)?
        } else {
            let raw = reader.read_frame(idx)?;
            debayer(&raw.data, color_mode, debayer_method, raw.original_bit_depth)
                .ok_or_else(|| JupiterError::Pipeline("Debayer failed for cached frame".into()))?
        };
        let lum = luminance(&cf);
        frame_cache.insert(idx, (lum, cf));
    }

    // Step 6: Per-AP local alignment + stacking (parallel)
    info!("Stacking {} alignment points (color)", grid.points.len());
    let ap_stacks: Vec<(AlignmentPoint, (Array2<f32>, Array2<f32>, Array2<f32>))> = grid
        .points
        .par_iter()
        .map(|ap| {
            let stacked = stack_ap_cached_color(
                &frame_cache,
                ap,
                &ap_selections[ap.index],
                &global_offsets,
                &ref_lum.data,
                config,
            );
            (ap.clone(), stacked)
        })
        .collect();
    on_progress(0.9);

    // Step 7: Blend per channel
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
