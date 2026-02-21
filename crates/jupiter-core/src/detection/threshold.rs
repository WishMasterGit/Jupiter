use ndarray::Array2;

use crate::consts::OTSU_HISTOGRAM_BINS;

use super::config::ThresholdMethod;

/// Compute the threshold value using the configured method.
pub fn compute_threshold(data: &Array2<f32>, method: &ThresholdMethod, sigma_mul: f32) -> f32 {
    match method {
        ThresholdMethod::MeanPlusSigma => {
            let (mean, std) = compute_mean_stddev(data);
            (mean + sigma_mul as f64 * std) as f32
        }
        ThresholdMethod::Otsu => otsu_threshold(data),
        ThresholdMethod::Fixed(v) => *v,
    }
}

/// Compute mean and standard deviation of pixel values.
pub fn compute_mean_stddev(data: &Array2<f32>) -> (f64, f64) {
    let n = data.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let sum: f64 = data.iter().map(|&v| v as f64).sum();
    let mean = sum / n;
    let var: f64 = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Otsu's thresholding: find the value that minimizes intra-class variance.
pub fn otsu_threshold(data: &Array2<f32>) -> f32 {
    let bins = OTSU_HISTOGRAM_BINS;
    let mut histogram = vec![0u64; bins];

    for &v in data.iter() {
        let bin = ((v.clamp(0.0, 1.0) * (bins - 1) as f32) as usize).min(bins - 1);
        histogram[bin] += 1;
    }

    let total = data.len() as f64;
    let mut sum_all: f64 = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_all += i as f64 * count as f64;
    }

    let mut weight_bg: f64 = 0.0;
    let mut sum_bg: f64 = 0.0;
    let mut best_variance = 0.0_f64;
    let mut best_bin = 0usize;

    for (i, &count) in histogram.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }
        sum_bg += i as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_all - sum_bg) / weight_fg;
        let between_variance = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);

        if between_variance > best_variance {
            best_variance = between_variance;
            best_bin = i;
        }
    }

    (best_bin as f32 + 0.5) / bins as f32
}
