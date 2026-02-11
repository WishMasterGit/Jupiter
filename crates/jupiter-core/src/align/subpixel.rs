use ndarray::Array2;

/// Refine peak location using paraboloid fitting on the 3x3 neighborhood.
///
/// Returns (delta_row, delta_col) as fractional pixel offsets from the integer peak.
pub fn refine_peak_paraboloid(
    correlation: &Array2<f64>,
    peak_row: usize,
    peak_col: usize,
) -> (f64, f64) {
    let (h, w) = correlation.dim();

    // Need 3x3 neighborhood — if peak is at edge, skip refinement
    if peak_row == 0 || peak_row >= h - 1 || peak_col == 0 || peak_col >= w - 1 {
        return (0.0, 0.0);
    }

    // 1D parabola fit in each direction
    // For row: fit parabola through (r-1), r, (r+1)
    let y_prev = correlation[[peak_row - 1, peak_col]];
    let y_curr = correlation[[peak_row, peak_col]];
    let y_next = correlation[[peak_row + 1, peak_col]];

    let delta_row = if (y_prev - 2.0 * y_curr + y_next).abs() > 1e-12 {
        (y_prev - y_next) / (2.0 * (y_prev - 2.0 * y_curr + y_next))
    } else {
        0.0
    };

    // For col: fit parabola through (c-1), c, (c+1)
    let x_prev = correlation[[peak_row, peak_col - 1]];
    let x_curr = correlation[[peak_row, peak_col]];
    let x_next = correlation[[peak_row, peak_col + 1]];

    let delta_col = if (x_prev - 2.0 * x_curr + x_next).abs() > 1e-12 {
        (x_prev - x_next) / (2.0 * (x_prev - 2.0 * x_curr + x_next))
    } else {
        0.0
    };

    // Clamp to within +/- 0.5 pixel
    (delta_row.clamp(-0.5, 0.5), delta_col.clamp(-0.5, 0.5))
}
