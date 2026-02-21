use serde::{Deserialize, Serialize};

use crate::consts::{
    DEFAULT_AUTOCROP_BLUR_SIGMA, DEFAULT_AUTOCROP_MIN_AREA, DEFAULT_AUTOCROP_SIGMA_MULTIPLIER,
};

/// Method used to separate the planet from the sky background.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum ThresholdMethod {
    /// Threshold = mean + sigma_multiplier * stddev.
    MeanPlusSigma,
    /// Otsu's method: minimizes intra-class variance on a bimodal histogram.
    #[default]
    Otsu,
    /// User-specified fixed threshold in [0.0, 1.0].
    Fixed(f32),
}

/// Configuration for planet detection in a single frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Thresholding method.
    #[serde(default)]
    pub threshold_method: ThresholdMethod,
    /// Sigma multiplier for MeanPlusSigma method.
    #[serde(default = "default_sigma_multiplier")]
    pub sigma_multiplier: f32,
    /// Gaussian blur sigma for noise suppression before thresholding.
    #[serde(default = "default_blur_sigma")]
    pub blur_sigma: f32,
    /// Minimum connected component area (pixels) to be a planet candidate.
    #[serde(default = "default_min_area")]
    pub min_area: usize,
}

fn default_sigma_multiplier() -> f32 {
    DEFAULT_AUTOCROP_SIGMA_MULTIPLIER
}
fn default_blur_sigma() -> f32 {
    DEFAULT_AUTOCROP_BLUR_SIGMA
}
fn default_min_area() -> usize {
    DEFAULT_AUTOCROP_MIN_AREA
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            threshold_method: ThresholdMethod::default(),
            sigma_multiplier: DEFAULT_AUTOCROP_SIGMA_MULTIPLIER,
            blur_sigma: DEFAULT_AUTOCROP_BLUR_SIGMA,
            min_area: DEFAULT_AUTOCROP_MIN_AREA,
        }
    }
}
