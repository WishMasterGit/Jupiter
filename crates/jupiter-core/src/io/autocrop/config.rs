use serde::{Deserialize, Serialize};

use crate::consts::{DEFAULT_AUTOCROP_PADDING_FRACTION, DEFAULT_AUTOCROP_SAMPLE_COUNT};
use crate::detection::config::DetectionConfig;

/// Configuration for automatic planet detection and cropping.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoCropConfig {
    /// Number of frames to sample for detection (evenly spaced).
    #[serde(default = "default_sample_count")]
    pub sample_count: usize,
    /// Padding around detected bounding box as fraction of planet diameter.
    #[serde(default = "default_padding_fraction")]
    pub padding_fraction: f32,
    /// Planet detection parameters.
    #[serde(flatten)]
    pub detection: DetectionConfig,
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
fn default_true() -> bool {
    true
}

impl Default for AutoCropConfig {
    fn default() -> Self {
        Self {
            sample_count: DEFAULT_AUTOCROP_SAMPLE_COUNT,
            padding_fraction: DEFAULT_AUTOCROP_PADDING_FRACTION,
            detection: DetectionConfig::default(),
            align_to_fft: true,
        }
    }
}
