pub mod gradient;
pub mod laplacian;
pub mod scoring;

use ndarray::Array2;

use crate::pipeline::config::QualityMetric;

/// Score an array using the specified quality metric.
pub fn score_with_metric(data: &Array2<f32>, metric: &QualityMetric) -> f64 {
    match metric {
        QualityMetric::Laplacian => laplacian::laplacian_variance_array(data),
        QualityMetric::Gradient => gradient::gradient_score_array(data),
    }
}
