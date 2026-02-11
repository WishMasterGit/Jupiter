use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::sharpen::wavelet::WaveletParams;
use crate::stack::multi_point::MultiPointConfig;
use crate::stack::sigma_clip::SigmaClipParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    #[serde(default)]
    pub frame_selection: FrameSelectionConfig,
    #[serde(default)]
    pub stacking: StackingConfig,
    pub sharpening: Option<SharpeningConfig>,
    #[serde(default)]
    pub filters: Vec<FilterStep>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameSelectionConfig {
    /// Fraction of frames to keep (0.0..1.0).
    pub select_percentage: f32,
    /// Quality metric to use.
    #[serde(default)]
    pub metric: QualityMetric,
}

impl Default for FrameSelectionConfig {
    fn default() -> Self {
        Self {
            select_percentage: 0.25,
            metric: QualityMetric::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum QualityMetric {
    #[default]
    Laplacian,
    Gradient,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StackingConfig {
    pub method: StackMethod,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            method: StackMethod::Mean,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StackMethod {
    Mean,
    Median,
    SigmaClip(SigmaClipParams),
    MultiPoint(MultiPointConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SharpeningConfig {
    pub wavelet: WaveletParams,
}

/// A single post-processing filter step.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FilterStep {
    /// Linear histogram stretch with explicit black/white points.
    HistogramStretch { black_point: f32, white_point: f32 },
    /// Automatic histogram stretch using percentiles.
    AutoStretch {
        low_percentile: f32,
        high_percentile: f32,
    },
    /// Gamma correction.
    Gamma(f32),
    /// Brightness and contrast adjustment.
    BrightnessContrast { brightness: f32, contrast: f32 },
    /// Unsharp mask sharpening.
    UnsharpMask {
        radius: f32,
        amount: f32,
        threshold: f32,
    },
    /// Gaussian blur.
    GaussianBlur { sigma: f32 },
}
