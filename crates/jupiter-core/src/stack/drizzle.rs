use ndarray::Array2;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, Frame};

/// Drop kernel shape for drizzle projection.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum DrizzleKernel {
    /// Square drop: uniform weight over pixel footprint.
    #[default]
    Square,
}

/// Configuration for drizzle super-resolution projection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DrizzleConfig {
    /// Output upscale factor (e.g., 2.0 = 2x resolution).
    pub scale: f32,
    /// Drop size as fraction of input pixel (0.0-1.0).
    /// Smaller = sharper but noisier; 0.6-0.8 is typical for planetary.
    pub pixfrac: f32,
    /// Drop kernel shape.
    #[serde(default)]
    pub kernel: DrizzleKernel,
}

impl Default for DrizzleConfig {
    fn default() -> Self {
        Self {
            scale: 2.0,
            pixfrac: 0.7,
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

impl std::fmt::Display for DrizzleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Drizzle ({}x, pixfrac={})", self.scale, self.pixfrac)
    }
}

/// Compute the output dimensions for a drizzled frame.
pub fn drizzle_output_dims(h: usize, w: usize, scale: f32) -> (usize, usize) {
    let out_h = (h as f32 * scale).ceil() as usize;
    let out_w = (w as f32 * scale).ceil() as usize;
    (out_h, out_w)
}

/// Drizzle a single frame onto a high-resolution grid using its alignment offset.
///
/// Returns a `Frame` with dimensions `(ceil(h*scale), ceil(w*scale))`.
///
/// # Arguments
///
/// - `frame`: Input frame
/// - `offset`: Alignment offset for this frame
/// - `config`: Drizzle configuration (scale, pixfrac, kernel)
pub fn drizzle_single_frame(
    frame: &Frame,
    offset: &AlignmentOffset,
    config: &DrizzleConfig,
) -> Result<Frame> {
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

    let (h, w) = frame.data.dim();
    let mut acc = DrizzleAccumulator::new(h, w, config.scale);
    drizzle_frame_into(
        &frame.data,
        offset,
        config.scale,
        config.pixfrac,
        1.0,
        &mut acc,
    );
    Ok(acc.finalize(frame.original_bit_depth))
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
        let (out_height, out_width) = drizzle_output_dims(in_height, in_width, scale);
        Self {
            data: Array2::zeros((out_height, out_width)),
            weights: Array2::zeros((out_height, out_width)),
            out_height,
            out_width,
        }
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
