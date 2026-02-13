use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::consts::PARALLEL_FRAME_THRESHOLD;
use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, Frame};
use crate::io::ser::SerReader;

/// Drop kernel shape for drizzle projection.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum DrizzleKernel {
    /// Square drop: uniform weight over pixel footprint.
    #[default]
    Square,
}

/// Configuration for drizzle super-resolution stacking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrizzleConfig {
    /// Output upscale factor (e.g., 2.0 = 2x resolution).
    pub scale: f32,
    /// Drop size as fraction of input pixel (0.0-1.0).
    /// Smaller = sharper but noisier; 0.6-0.8 is typical for planetary.
    pub pixfrac: f32,
    /// Weight each frame by its quality score during accumulation.
    #[serde(default = "default_true")]
    pub quality_weighted: bool,
    /// Drop kernel shape.
    #[serde(default)]
    pub kernel: DrizzleKernel,
}

fn default_true() -> bool {
    true
}

impl Default for DrizzleConfig {
    fn default() -> Self {
        Self {
            scale: 2.0,
            pixfrac: 0.7,
            quality_weighted: true,
            kernel: DrizzleKernel::default(),
        }
    }
}

impl std::fmt::Display for DrizzleKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrizzleKernel::Square => write!(f, "Square"),
        }
    }
}

/// Intermediate accumulation buffer for drizzle stacking.
struct DrizzleAccumulator {
    /// Accumulated pixel values at output resolution.
    data: Array2<f32>,
    /// Weight map tracking contribution per output pixel.
    weights: Array2<f32>,
    /// Output dimensions.
    out_height: usize,
    out_width: usize,
}

impl DrizzleAccumulator {
    fn new(in_height: usize, in_width: usize, scale: f32) -> Self {
        let out_height = (in_height as f32 * scale).ceil() as usize;
        let out_width = (in_width as f32 * scale).ceil() as usize;
        Self {
            data: Array2::zeros((out_height, out_width)),
            weights: Array2::zeros((out_height, out_width)),
            out_height,
            out_width,
        }
    }

    /// Merge another accumulator into this one (for parallel frame processing).
    fn merge(&mut self, other: &DrizzleAccumulator) {
        self.data += &other.data;
        self.weights += &other.weights;
    }

    /// Normalize by weight map, clamp to [0,1], and produce the final frame.
    fn finalize(self, bit_depth: u8) -> Frame {
        let mut result = self.data;
        let mut zero_weight_count: usize = 0;

        for (val, &weight) in result.iter_mut().zip(self.weights.iter()) {
            if weight > f32::EPSILON {
                *val /= weight;
            } else {
                *val = 0.0;
                zero_weight_count += 1;
            }
        }

        if zero_weight_count > 0 {
            warn!(
                "Drizzle: {} output pixels received no contributions",
                zero_weight_count
            );
        }

        result.mapv_inplace(|v| v.clamp(0.0, 1.0));
        Frame::new(result, bit_depth)
    }
}

/// Stack frames using the Drizzle algorithm for super-resolution reconstruction.
///
/// Each input pixel is projected onto a higher-resolution output grid using its
/// alignment offset. The pixel's contribution is "dropped" as a shrunk footprint
/// (controlled by `pixfrac`) and distributed to overlapping output pixels based
/// on geometric overlap area.
///
/// # Arguments
///
/// - `frames`: Input frames (must all have same dimensions)
/// - `offsets`: Alignment offset for each frame (length must match `frames`)
/// - `config`: Drizzle configuration (scale, pixfrac, kernel)
/// - `quality_scores`: Optional per-frame quality weights (length must match `frames`)
///
/// # Returns
///
/// A `Frame` with dimensions `(ceil(h*scale), ceil(w*scale))`.
pub fn drizzle_stack(
    frames: &[Frame],
    offsets: &[AlignmentOffset],
    config: &DrizzleConfig,
    quality_scores: Option<&[f64]>,
) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }
    if frames.len() != offsets.len() {
        return Err(JupiterError::Pipeline(
            "Frame count must match offset count".into(),
        ));
    }
    if config.scale <= 0.0 {
        return Err(JupiterError::Pipeline(format!(
            "Invalid drizzle scale: {}",
            config.scale
        )));
    }
    if config.pixfrac <= 0.0 || config.pixfrac > 1.0 {
        return Err(JupiterError::Pipeline(format!(
            "Invalid pixfrac: {} (must be in (0.0, 1.0])",
            config.pixfrac
        )));
    }

    let (h, w) = frames[0].data.dim();
    for frame in &frames[1..] {
        if frame.data.dim() != (h, w) {
            return Err(JupiterError::Pipeline("Frame size mismatch".into()));
        }
    }

    let bit_depth = frames[0].original_bit_depth;

    // Build per-frame weights from quality scores.
    let frame_weights: Vec<f32> = if config.quality_weighted {
        if let Some(scores) = quality_scores {
            scores.iter().map(|&s| s as f32).collect()
        } else {
            vec![1.0; frames.len()]
        }
    } else {
        vec![1.0; frames.len()]
    };

    let accumulator = if frames.len() >= PARALLEL_FRAME_THRESHOLD {
        // Parallel: each frame gets its own accumulator, then merge.
        let accumulators: Vec<DrizzleAccumulator> = frames
            .par_iter()
            .zip(offsets.par_iter())
            .zip(frame_weights.par_iter())
            .map(|((frame, offset), &weight)| {
                let mut acc = DrizzleAccumulator::new(h, w, config.scale);
                drizzle_frame_into(&frame.data, offset, config.scale, config.pixfrac, weight, &mut acc);
                acc
            })
            .collect();

        let mut final_acc = DrizzleAccumulator::new(h, w, config.scale);
        for acc in &accumulators {
            final_acc.merge(acc);
        }
        final_acc
    } else {
        // Sequential: single accumulator.
        let mut acc = DrizzleAccumulator::new(h, w, config.scale);
        for ((frame, offset), &weight) in frames.iter().zip(offsets.iter()).zip(frame_weights.iter())
        {
            drizzle_frame_into(&frame.data, offset, config.scale, config.pixfrac, weight, &mut acc);
        }
        acc
    };

    Ok(accumulator.finalize(bit_depth))
}

/// Project one input frame onto the output grid.
fn drizzle_frame_into(
    input: &Array2<f32>,
    offset: &AlignmentOffset,
    scale: f32,
    pixfrac: f32,
    frame_weight: f32,
    acc: &mut DrizzleAccumulator,
) {
    let (in_h, in_w) = input.dim();
    let scale_f64 = scale as f64;
    let drop_half = (pixfrac as f64 * scale_f64) / 2.0;

    for in_row in 0..in_h {
        for in_col in 0..in_w {
            let pixel_value = input[[in_row, in_col]];
            if pixel_value.abs() < f32::EPSILON {
                continue;
            }

            // Transform input pixel center to output grid coordinates.
            // Subtract offset because offset represents how much the target moved
            // relative to the reference.
            let aligned_y = in_row as f64 - offset.dy;
            let aligned_x = in_col as f64 - offset.dx;
            let out_y = aligned_y * scale_f64;
            let out_x = aligned_x * scale_f64;

            // Drop footprint bounds in output coordinates.
            let drop_y_min = out_y - drop_half;
            let drop_y_max = out_y + drop_half;
            let drop_x_min = out_x - drop_half;
            let drop_x_max = out_x + drop_half;

            // Output pixel range overlapped by this drop.
            let out_row_start = (drop_y_min.floor() as i64).max(0) as usize;
            let out_row_end = ((drop_y_max.ceil() as i64) as usize).min(acc.out_height);
            let out_col_start = (drop_x_min.floor() as i64).max(0) as usize;
            let out_col_end = ((drop_x_max.ceil() as i64) as usize).min(acc.out_width);

            for out_row in out_row_start..out_row_end {
                for out_col in out_col_start..out_col_end {
                    let overlap = compute_overlap(
                        out_row as f64,
                        out_col as f64,
                        drop_y_min,
                        drop_y_max,
                        drop_x_min,
                        drop_x_max,
                    );

                    if overlap > f32::EPSILON {
                        let contribution = pixel_value * overlap * frame_weight;
                        acc.data[[out_row, out_col]] += contribution;
                        acc.weights[[out_row, out_col]] += overlap * frame_weight;
                    }
                }
            }
        }
    }
}

/// Stack frames using the Drizzle algorithm by streaming one frame at a time.
///
/// Instead of taking a `&[Frame]`, reads each frame on-demand from the SER
/// reader. Memory usage: one decoded frame + one `DrizzleAccumulator` at output
/// resolution, regardless of frame count.
pub fn drizzle_stack_streaming(
    reader: &SerReader,
    frame_indices: &[usize],
    offsets: &[AlignmentOffset],
    config: &DrizzleConfig,
    quality_scores: Option<&[f64]>,
) -> Result<Frame> {
    if frame_indices.is_empty() {
        return Err(JupiterError::EmptySequence);
    }
    if frame_indices.len() != offsets.len() {
        return Err(JupiterError::Pipeline(
            "Frame count must match offset count".into(),
        ));
    }
    if config.scale <= 0.0 {
        return Err(JupiterError::Pipeline(format!(
            "Invalid drizzle scale: {}",
            config.scale
        )));
    }
    if config.pixfrac <= 0.0 || config.pixfrac > 1.0 {
        return Err(JupiterError::Pipeline(format!(
            "Invalid pixfrac: {} (must be in (0.0, 1.0])",
            config.pixfrac
        )));
    }

    let h = reader.header.height as usize;
    let w = reader.header.width as usize;
    let bit_depth = reader.header.pixel_depth as u8;

    let frame_weights: Vec<f32> = if config.quality_weighted {
        if let Some(scores) = quality_scores {
            scores.iter().map(|&s| s as f32).collect()
        } else {
            vec![1.0; frame_indices.len()]
        }
    } else {
        vec![1.0; frame_indices.len()]
    };

    // Sequential single-accumulator: read frame → drizzle → drop
    let mut acc = DrizzleAccumulator::new(h, w, config.scale);
    for (i, (&frame_idx, offset)) in frame_indices.iter().zip(offsets.iter()).enumerate() {
        let frame = reader.read_frame(frame_idx)?;
        drizzle_frame_into(
            &frame.data,
            offset,
            config.scale,
            config.pixfrac,
            frame_weights[i],
            &mut acc,
        );
        // frame dropped here — memory freed
    }

    Ok(acc.finalize(bit_depth))
}

/// Compute overlap area between a square drop and a unit output pixel.
///
/// The output pixel occupies `[out_row, out_row+1) x [out_col, out_col+1)`.
/// The drop occupies `[drop_y_min, drop_y_max) x [drop_x_min, drop_x_max)`.
fn compute_overlap(
    out_row: f64,
    out_col: f64,
    drop_y_min: f64,
    drop_y_max: f64,
    drop_x_min: f64,
    drop_x_max: f64,
) -> f32 {
    let pixel_y_min = out_row;
    let pixel_y_max = out_row + 1.0;
    let pixel_x_min = out_col;
    let pixel_x_max = out_col + 1.0;

    let y_overlap = (drop_y_max.min(pixel_y_max) - drop_y_min.max(pixel_y_min)).max(0.0);
    let x_overlap = (drop_x_max.min(pixel_x_max) - drop_x_min.max(pixel_x_min)).max(0.0);

    (y_overlap * x_overlap) as f32
}
