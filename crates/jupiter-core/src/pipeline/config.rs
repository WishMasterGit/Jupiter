use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::sharpen::wavelet::WaveletParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    #[serde(default)]
    pub frame_selection: FrameSelectionConfig,
    #[serde(default)]
    pub stacking: StackingConfig,
    pub sharpening: Option<SharpeningConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameSelectionConfig {
    /// Fraction of frames to keep (0.0..1.0).
    pub select_percentage: f32,
}

impl Default for FrameSelectionConfig {
    fn default() -> Self {
        Self {
            select_percentage: 0.25,
        }
    }
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharpeningConfig {
    pub wavelet: WaveletParams,
}

impl Default for SharpeningConfig {
    fn default() -> Self {
        Self {
            wavelet: WaveletParams::default(),
        }
    }
}
