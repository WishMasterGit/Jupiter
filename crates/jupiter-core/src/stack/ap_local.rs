use std::collections::HashMap;

use ndarray::Array2;

use crate::align::phase_correlation::{bilinear_sample, compute_offset_with_confidence};
use crate::consts::{EPSILON, MIN_CORRELATION_CONFIDENCE};
use crate::frame::{AlignmentOffset, ColorFrame, Frame};
use crate::stack::ap_grid::{
    extract_region, extract_region_shifted, AlignmentPoint, LocalStackMethod, MultiPointConfig,
};

/// Stack one AP using pre-cached frames (for parallel execution).
///
/// Includes correlation confidence check: frames whose local alignment
/// peak-to-mean ratio is below [`MIN_CORRELATION_CONFIDENCE`] are skipped.
/// When using Mean stacking, quality weights from `selected_frames` are applied.
pub(crate) fn stack_ap_cached(
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
    let mut weights: Vec<f64> = Vec::with_capacity(selected_frames.len());

    for &(frame_idx, quality_score) in selected_frames {
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

        // Local alignment with confidence check
        let (local_offset, confidence) =
            compute_offset_with_confidence(&ref_search, &tgt_search)
                .unwrap_or((AlignmentOffset::default(), 0.0));

        if confidence < MIN_CORRELATION_CONFIDENCE {
            continue; // skip unreliable alignment
        }

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
        weights.push(quality_score);
    }

    if patches.is_empty() {
        // All frames rejected by confidence check — fall back to reference region
        return extract_region(reference_data, ap.cy, ap.cx, half);
    }

    match config.local_stack_method {
        LocalStackMethod::Mean => mean_stack_arrays_weighted(&patches, &weights),
        LocalStackMethod::Median => median_stack_arrays(&patches),
        LocalStackMethod::SigmaClip { sigma, iterations } => {
            sigma_clip_stack_arrays(&patches, sigma, iterations)
        }
    }
}

/// Stack one AP using pre-cached color frames.
///
/// Local alignment is computed on luminance with confidence check.
/// R/G/B patches are extracted and stacked independently using the
/// configured local method (quality-weighted for Mean).
pub(crate) fn stack_ap_cached_color(
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
    let mut weights: Vec<f64> = Vec::with_capacity(selected_frames.len());

    for &(frame_idx, quality_score) in selected_frames {
        let (lum, color) = match frame_cache.get(&frame_idx) {
            Some(entry) => entry,
            None => continue,
        };

        // Local alignment on luminance with confidence check
        let tgt_search = extract_region_shifted(
            &lum.data,
            ap.cy,
            ap.cx,
            search_half,
            &global_offsets[frame_idx],
        );

        let (local_offset, confidence) =
            compute_offset_with_confidence(&ref_search, &tgt_search)
                .unwrap_or((AlignmentOffset::default(), 0.0));

        if confidence < MIN_CORRELATION_CONFIDENCE {
            continue; // skip unreliable alignment
        }

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
        weights.push(quality_score);
    }

    if red_patches.is_empty() {
        // All frames rejected — fall back to reference region (luminance as proxy)
        let fallback = extract_region(reference_lum, ap.cy, ap.cx, half);
        return (fallback.clone(), fallback.clone(), fallback);
    }

    let stack_fn = |patches: &[Array2<f32>], wts: &[f64]| -> Array2<f32> {
        match config.local_stack_method {
            LocalStackMethod::Mean => mean_stack_arrays_weighted(patches, wts),
            LocalStackMethod::Median => median_stack_arrays(patches),
            LocalStackMethod::SigmaClip { sigma, iterations } => {
                sigma_clip_stack_arrays(patches, sigma, iterations)
            }
        }
    };

    // Stack R/G/B in parallel
    let (stacked_r, (stacked_g, stacked_b)) = rayon::join(
        || stack_fn(&red_patches, &weights),
        || {
            rayon::join(
                || stack_fn(&green_patches, &weights),
                || stack_fn(&blue_patches, &weights),
            )
        },
    );

    (stacked_r, stacked_g, stacked_b)
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
pub(crate) fn mean_stack_arrays(patches: &[Array2<f32>]) -> Array2<f32> {
    let (h, w) = patches[0].dim();
    let n = patches.len() as f32;
    let mut sum = Array2::<f32>::zeros((h, w));
    for p in patches {
        sum += p;
    }
    sum /= n;
    sum
}

/// Quality-weighted mean stack. Each patch is weighted by its quality score.
pub(crate) fn mean_stack_arrays_weighted(patches: &[Array2<f32>], weights: &[f64]) -> Array2<f32> {
    assert_eq!(patches.len(), weights.len());
    let (rows, cols) = patches[0].dim();
    let mut weighted_sum = Array2::<f64>::zeros((rows, cols));
    let mut total_weight: f64 = 0.0;

    for (p, &wt) in patches.iter().zip(weights.iter()) {
        let weight = wt.max(0.0);
        total_weight += weight;
        for row in 0..rows {
            for col in 0..cols {
                weighted_sum[[row, col]] += p[[row, col]] as f64 * weight;
            }
        }
    }

    if total_weight < 1e-15 {
        return mean_stack_arrays(patches);
    }

    weighted_sum.mapv(|v| (v / total_weight) as f32)
}

/// Median-stack a set of Array2 patches.
pub(crate) fn median_stack_arrays(patches: &[Array2<f32>]) -> Array2<f32> {
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
                    .select_nth_unstable_by(mid, |a, b| a.total_cmp(b))
                    .1
            } else {
                let mid = n / 2;
                let (_, upper, _) =
                    vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                let upper_val = *upper;
                let lower_val = vals[..mid]
                    .iter()
                    .copied()
                    .max_by(|a, b| a.total_cmp(b))
                    .unwrap();
                (lower_val + upper_val) / 2.0
            };
        }
    }

    result
}

/// Sigma-clip mean stack of Array2 patches.
pub(crate) fn sigma_clip_stack_arrays(
    patches: &[Array2<f32>],
    sigma: f32,
    iterations: usize,
) -> Array2<f32> {
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
