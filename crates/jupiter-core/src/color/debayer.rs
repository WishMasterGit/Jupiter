use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::consts::{LUMINANCE_B, LUMINANCE_G, LUMINANCE_R};
use crate::frame::{ColorFrame, ColorMode, Frame};

/// Debayering (demosaicing) algorithm.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum DebayerMethod {
    /// Simple bilinear interpolation — fast, good for planetary stacking.
    #[default]
    Bilinear,
    /// Malvar-He-Cutler gradient-corrected — higher quality, moderate speed.
    MalvarHeCutler,
}

impl std::fmt::Display for DebayerMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bilinear => write!(f, "Bilinear"),
            Self::MalvarHeCutler => write!(f, "Malvar-He-Cutler"),
        }
    }
}

/// Check whether a `ColorMode` is a Bayer pattern.
pub fn is_bayer(mode: &ColorMode) -> bool {
    matches!(
        mode,
        ColorMode::BayerRGGB | ColorMode::BayerGRBG | ColorMode::BayerGBRG | ColorMode::BayerBGGR
    )
}

/// Debayer a raw Bayer mosaic into a `ColorFrame`.
///
/// Returns `None` if `mode` is not a Bayer pattern.
pub fn debayer(
    raw: &Array2<f32>,
    mode: &ColorMode,
    method: &DebayerMethod,
    bit_depth: u8,
) -> Option<ColorFrame> {
    if !is_bayer(mode) {
        return None;
    }
    Some(match method {
        DebayerMethod::Bilinear => debayer_bilinear(raw, mode, bit_depth),
        DebayerMethod::MalvarHeCutler => debayer_mhc(raw, mode, bit_depth),
    })
}

/// Compute luminance from a `ColorFrame` using ITU-R BT.601 weights.
pub fn luminance(color: &ColorFrame) -> Frame {
    let (h, w) = color.red.data.dim();
    let mut data = Array2::<f32>::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            data[[row, col]] = LUMINANCE_R * color.red.data[[row, col]]
                + LUMINANCE_G * color.green.data[[row, col]]
                + LUMINANCE_B * color.blue.data[[row, col]];
        }
    }

    Frame::new(data, color.red.original_bit_depth)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Which color sits at position (0,0) in the 2x2 Bayer cell.
#[derive(Clone, Copy)]
#[allow(clippy::upper_case_acronyms)]
enum BayerPhase {
    RGGB,
    GRBG,
    GBRG,
    BGGR,
}

impl BayerPhase {
    fn from_color_mode(mode: &ColorMode) -> Option<Self> {
        match mode {
            ColorMode::BayerRGGB => Some(Self::RGGB),
            ColorMode::BayerGRBG => Some(Self::GRBG),
            ColorMode::BayerGBRG => Some(Self::GBRG),
            ColorMode::BayerBGGR => Some(Self::BGGR),
            _ => None,
        }
    }

    /// Returns `(row_parity, col_parity)` of the red pixel within the 2x2 cell.
    fn red_position(self) -> (usize, usize) {
        match self {
            Self::RGGB => (0, 0),
            Self::GRBG => (0, 1),
            Self::GBRG => (1, 0),
            Self::BGGR => (1, 1),
        }
    }
}

/// Clamped indexing into the raw Bayer mosaic.
#[inline]
fn px(raw: &Array2<f32>, row: isize, col: isize) -> f32 {
    let (h, w) = raw.dim();
    let r = row.clamp(0, h as isize - 1) as usize;
    let c = col.clamp(0, w as isize - 1) as usize;
    raw[[r, c]]
}

// ---------------------------------------------------------------------------
// Bilinear demosaicing
// ---------------------------------------------------------------------------

fn debayer_bilinear(raw: &Array2<f32>, mode: &ColorMode, bit_depth: u8) -> ColorFrame {
    let phase = BayerPhase::from_color_mode(mode).expect("non-Bayer mode in debayer_bilinear");
    let (h, w) = raw.dim();
    let (r_row, r_col) = phase.red_position();

    let mut red = Array2::<f32>::zeros((h, w));
    let mut green = Array2::<f32>::zeros((h, w));
    let mut blue = Array2::<f32>::zeros((h, w));

    for row in 0..h {
        let ri = row as isize;
        let is_red_row = (row % 2) == r_row;
        for col in 0..w {
            let ci = col as isize;
            let is_red_col = (col % 2) == r_col;

            match (is_red_row, is_red_col) {
                // Red pixel position
                (true, true) => {
                    red[[row, col]] = raw[[row, col]];
                    green[[row, col]] = avg_cross(raw, ri, ci);
                    blue[[row, col]] = avg_diagonal(raw, ri, ci);
                }
                // Green on red row
                (true, false) => {
                    red[[row, col]] = avg_horizontal(raw, ri, ci);
                    green[[row, col]] = raw[[row, col]];
                    blue[[row, col]] = avg_vertical(raw, ri, ci);
                }
                // Green on blue row
                (false, true) => {
                    red[[row, col]] = avg_vertical(raw, ri, ci);
                    green[[row, col]] = raw[[row, col]];
                    blue[[row, col]] = avg_horizontal(raw, ri, ci);
                }
                // Blue pixel position
                (false, false) => {
                    red[[row, col]] = avg_diagonal(raw, ri, ci);
                    green[[row, col]] = avg_cross(raw, ri, ci);
                    blue[[row, col]] = raw[[row, col]];
                }
            }
        }
    }

    ColorFrame {
        red: Frame::new(red, bit_depth),
        green: Frame::new(green, bit_depth),
        blue: Frame::new(blue, bit_depth),
    }
}

/// Average of 4 cross (cardinal) neighbours.
#[inline]
fn avg_cross(raw: &Array2<f32>, r: isize, c: isize) -> f32 {
    (px(raw, r - 1, c) + px(raw, r + 1, c) + px(raw, r, c - 1) + px(raw, r, c + 1)) * 0.25
}

/// Average of 4 diagonal neighbours.
#[inline]
fn avg_diagonal(raw: &Array2<f32>, r: isize, c: isize) -> f32 {
    (px(raw, r - 1, c - 1)
        + px(raw, r - 1, c + 1)
        + px(raw, r + 1, c - 1)
        + px(raw, r + 1, c + 1))
        * 0.25
}

/// Average of left and right neighbours.
#[inline]
fn avg_horizontal(raw: &Array2<f32>, r: isize, c: isize) -> f32 {
    (px(raw, r, c - 1) + px(raw, r, c + 1)) * 0.5
}

/// Average of top and bottom neighbours.
#[inline]
fn avg_vertical(raw: &Array2<f32>, r: isize, c: isize) -> f32 {
    (px(raw, r - 1, c) + px(raw, r + 1, c)) * 0.5
}

// ---------------------------------------------------------------------------
// Malvar-He-Cutler (MHC) demosaicing
// ---------------------------------------------------------------------------
//
// Reference: "High-quality linear interpolation for demosaicing of
// Bayer-patterned color images" — Malvar, He, Cutler (2004).
//
// Five 5x5 kernels, each applied then divided by 8. The kernel indices
// correspond to the combination of (pixel_native_color, target_color).

// All MHC kernels below are scaled by 2 from the paper values and divided by 16.
// This avoids fractional kernel coefficients while maintaining exact arithmetic.

/// Green at a red or blue location (paper kernel * 2, /16).
const MHC_G_AT_RB: [[i32; 5]; 5] = [
    [0, 0, -2, 0, 0],
    [0, 0, 4, 0, 0],
    [-2, 4, 8, 4, -2],
    [0, 0, 4, 0, 0],
    [0, 0, -2, 0, 0],
];

/// Red at green in a red row / Blue at green in a blue row (paper kernel * 2, /16).
const MHC_RB_AT_G_SAME_ROW: [[i32; 5]; 5] = [
    [0, 0, 1, 0, 0],
    [0, -2, 0, -2, 0],
    [-2, 8, 10, 8, -2],
    [0, -2, 0, -2, 0],
    [0, 0, 1, 0, 0],
];

/// Red at green in a blue row / Blue at green in a red row (paper kernel * 2, /16).
const MHC_RB_AT_G_DIFF_ROW: [[i32; 5]; 5] = [
    [0, 0, -2, 0, 0],
    [0, -2, 8, -2, 0],
    [1, 0, 10, 0, 1],
    [0, -2, 8, -2, 0],
    [0, 0, -2, 0, 0],
];

/// Red at blue / Blue at red — diagonal (paper kernel, /16).
const MHC_RB_AT_BR: [[i32; 5]; 5] = [
    [0, 0, -3, 0, 0],
    [0, 4, 0, 4, 0],
    [-3, 0, 12, 0, -3],
    [0, 4, 0, 4, 0],
    [0, 0, -3, 0, 0],
];

/// Apply a 5x5 i32 kernel centred at (r,c) then divide by `divisor`.
#[inline]
fn apply_kernel(raw: &Array2<f32>, r: isize, c: isize, kernel: &[[i32; 5]; 5], divisor: f32) -> f32 {
    let mut sum = 0.0_f32;
    for (kr, krow) in kernel.iter().enumerate() {
        for (kc, &kval) in krow.iter().enumerate() {
            if kval != 0 {
                sum += kval as f32 * px(raw, r + kr as isize - 2, c + kc as isize - 2);
            }
        }
    }
    (sum / divisor).clamp(0.0, 1.0)
}

fn debayer_mhc(raw: &Array2<f32>, mode: &ColorMode, bit_depth: u8) -> ColorFrame {
    let phase = BayerPhase::from_color_mode(mode).expect("non-Bayer mode in debayer_mhc");
    let (h, w) = raw.dim();
    let (r_row, r_col) = phase.red_position();

    let mut red = Array2::<f32>::zeros((h, w));
    let mut green = Array2::<f32>::zeros((h, w));
    let mut blue = Array2::<f32>::zeros((h, w));

    for row in 0..h {
        let ri = row as isize;
        let is_red_row = (row % 2) == r_row;
        for col in 0..w {
            let ci = col as isize;
            let is_red_col = (col % 2) == r_col;

            const DIVISOR: f32 = 16.0;
            match (is_red_row, is_red_col) {
                // Red pixel position
                (true, true) => {
                    red[[row, col]] = raw[[row, col]];
                    green[[row, col]] = apply_kernel(raw, ri, ci, &MHC_G_AT_RB, DIVISOR);
                    blue[[row, col]] = apply_kernel(raw, ri, ci, &MHC_RB_AT_BR, DIVISOR);
                }
                // Green on red row (red neighbours are left/right)
                (true, false) => {
                    red[[row, col]] = apply_kernel(raw, ri, ci, &MHC_RB_AT_G_SAME_ROW, DIVISOR);
                    green[[row, col]] = raw[[row, col]];
                    blue[[row, col]] = apply_kernel(raw, ri, ci, &MHC_RB_AT_G_DIFF_ROW, DIVISOR);
                }
                // Green on blue row (blue neighbours are left/right)
                (false, true) => {
                    red[[row, col]] = apply_kernel(raw, ri, ci, &MHC_RB_AT_G_DIFF_ROW, DIVISOR);
                    green[[row, col]] = raw[[row, col]];
                    blue[[row, col]] = apply_kernel(raw, ri, ci, &MHC_RB_AT_G_SAME_ROW, DIVISOR);
                }
                // Blue pixel position
                (false, false) => {
                    red[[row, col]] = apply_kernel(raw, ri, ci, &MHC_RB_AT_BR, DIVISOR);
                    green[[row, col]] = apply_kernel(raw, ri, ci, &MHC_G_AT_RB, DIVISOR);
                    blue[[row, col]] = raw[[row, col]];
                }
            }
        }
    }

    ColorFrame {
        red: Frame::new(red, bit_depth),
        green: Frame::new(green, bit_depth),
        blue: Frame::new(blue, bit_depth),
    }
}
