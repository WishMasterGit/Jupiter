use ndarray::Array2;

use crate::align::phase_correlation::shift_array;
use crate::color::debayer::DebayerMethod;
use crate::color::process::read_luminance_frame;
use crate::error::Result;
use crate::frame::{AlignmentOffset, ColorMode};
use crate::io::ser::SerReader;
use crate::pipeline::config::QualityMetric;
use crate::quality::score_with_metric;

/// Build a mean reference frame from the top-quality frames.
///
/// 1. Globally aligns all frames vs frame 0
/// 2. Scores each frame with the configured quality metric
/// 3. Selects the top `keep_fraction` by quality
/// 4. Shifts and averages them to create a synthetic reference
///
/// This produces a much cleaner reference than using a single frame,
/// reducing bias toward one atmospheric state.
pub fn build_mean_reference(
    reader: &SerReader,
    offsets: &[AlignmentOffset],
    quality_metric: &QualityMetric,
    keep_fraction: f32,
) -> Result<Array2<f32>> {
    let total = reader.frame_count();

    // Score every frame
    let mut scores: Vec<(usize, f64)> = (0..total)
        .map(|i| {
            let frame = reader.read_frame(i).unwrap();
            let score = score_with_metric(&frame.data, quality_metric);
            (i, score)
        })
        .collect();

    scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    let keep = ((total as f32 * keep_fraction).ceil() as usize)
        .max(1)
        .min(total);
    scores.truncate(keep);

    let frame0 = reader.read_frame(0)?;
    let (h, w) = frame0.data.dim();
    let mut accumulator = Array2::<f64>::zeros((h, w));

    for &(idx, _) in &scores {
        let frame = reader.read_frame(idx)?;
        let shifted = if idx == 0 {
            frame.data
        } else {
            shift_array(&frame.data, &offsets[idx])
        };
        accumulator += &shifted.mapv(|v| v as f64);
    }

    let n = scores.len() as f64;
    Ok(accumulator.mapv(|v| (v / n) as f32))
}

/// Build a mean reference from color frames (returns luminance).
pub fn build_mean_reference_color(
    reader: &SerReader,
    offsets: &[AlignmentOffset],
    quality_metric: &QualityMetric,
    keep_fraction: f32,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
) -> Result<Array2<f32>> {
    let total = reader.frame_count();

    // Score every frame on luminance
    let mut scores: Vec<(usize, f64)> = Vec::with_capacity(total);
    for i in 0..total {
        let lum = read_luminance_frame(reader, i, color_mode, debayer_method)?;
        let score = score_with_metric(&lum.data, quality_metric);
        scores.push((i, score));
    }

    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    let keep = ((total as f32 * keep_fraction).ceil() as usize)
        .max(1)
        .min(total);
    scores.truncate(keep);

    // Average luminance of the best frames (shifted)
    let first_lum = read_luminance_frame(reader, 0, color_mode, debayer_method)?;
    let (h, w) = first_lum.data.dim();
    let mut accumulator = Array2::<f64>::zeros((h, w));

    for &(idx, _) in &scores {
        let lum = read_luminance_frame(reader, idx, color_mode, debayer_method)?;
        let shifted = if idx == 0 {
            lum.data
        } else {
            shift_array(&lum.data, &offsets[idx])
        };
        accumulator += &shifted.mapv(|v| v as f64);
    }

    let n = scores.len() as f64;
    Ok(accumulator.mapv(|v| (v / n) as f32))
}
