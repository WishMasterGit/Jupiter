use serde::{Deserialize, Serialize};

use crate::consts::{
    DEFAULT_AUTOCROP_BLUR_SIGMA, DEFAULT_AUTOCROP_MIN_AREA, DEFAULT_AUTOCROP_PADDING_FRACTION,
    DEFAULT_AUTOCROP_SAMPLE_COUNT, DEFAULT_AUTOCROP_SIGMA_MULTIPLIER,
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

/// Configuration for automatic planet detection and cropping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoCropConfig {
    /// Number of frames to sample for detection (evenly spaced).
    #[serde(default = "default_sample_count")]
    pub sample_count: usize,
    /// Padding around detected bounding box as fraction of planet diameter.
    #[serde(default = "default_padding_fraction")]
    pub padding_fraction: f32,
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
    /// Round crop size to next multiple of 32 for FFT efficiency.
    #[serde(default = "default_true")]
    pub align_to_fft: bool,
}

fn default_sample_count() -> usize {
    DEFAULT_AUTOCROP_SAMPLE_COUNT
}
fn default_padding_fraction() -> f32 {
    DEFAULT_AUTOCROP_PADDING_FRACTION
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
fn default_true() -> bool {
    true
}

impl Default for AutoCropConfig {
    fn default() -> Self {
        Self {
            sample_count: DEFAULT_AUTOCROP_SAMPLE_COUNT,
            padding_fraction: DEFAULT_AUTOCROP_PADDING_FRACTION,
            threshold_method: ThresholdMethod::default(),
            sigma_multiplier: DEFAULT_AUTOCROP_SIGMA_MULTIPLIER,
            blur_sigma: DEFAULT_AUTOCROP_BLUR_SIGMA,
            min_area: DEFAULT_AUTOCROP_MIN_AREA,
            align_to_fft: true,
        }
    }
}
