use crate::consts::EPSILON;
use crate::frame::Frame;

/// Linear histogram stretch: maps [black_point, white_point] → [0.0, 1.0].
pub fn histogram_stretch(frame: &Frame, black_point: f32, white_point: f32) -> Frame {
    let range = white_point - black_point;
    let range = if range.abs() < EPSILON { 1.0 } else { range };

    let data = frame.data.mapv(|v| ((v - black_point) / range).clamp(0.0, 1.0));
    Frame::new(data, frame.original_bit_depth)
}

/// Automatic histogram stretch using percentile-based black/white points.
///
/// `low_percentile` and `high_percentile` are in [0.0, 1.0].
/// Default: 0.001 (0.1%) and 0.999 (99.9%).
pub fn auto_stretch(frame: &Frame, low_percentile: f32, high_percentile: f32) -> Frame {
    let mut sorted: Vec<f32> = frame.data.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len();
    let lo_idx = ((n as f32 * low_percentile) as usize).min(n - 1);
    let hi_idx = ((n as f32 * high_percentile) as usize).min(n - 1);

    let black_point = sorted[lo_idx];
    let white_point = sorted[hi_idx];

    histogram_stretch(frame, black_point, white_point)
}
