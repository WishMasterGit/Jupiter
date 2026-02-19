use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::color::debayer::DebayerMethod;
use crate::compute::DevicePreference;
use crate::consts::{DEFAULT_CENTROID_THRESHOLD, DEFAULT_ENHANCED_PHASE_UPSAMPLE, DEFAULT_PYRAMID_LEVELS};
use crate::sharpen::wavelet::WaveletParams;
use crate::stack::drizzle::DrizzleConfig;
use crate::stack::multi_point::{LocalStackMethod, MultiPointConfig};
use crate::stack::sigma_clip::SigmaClipParams;

/// Memory usage strategy for the pipeline.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum MemoryStrategy {
    /// Automatically choose based on estimated decoded data size.
    #[default]
    Auto,
    /// Load all frames at once (fastest, highest memory).
    Eager,
    /// Stream frames on demand (lower memory, may re-read from disk).
    LowMemory,
}

impl fmt::Display for MemoryStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryStrategy::Auto => write!(f, "Auto"),
            MemoryStrategy::Eager => write!(f, "Eager"),
            MemoryStrategy::LowMemory => write!(f, "Low Memory"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineConfig {
    #[serde(default)]
    pub input: PathBuf,
    #[serde(default)]
    pub output: PathBuf,
    #[serde(default)]
    pub device: DevicePreference,
    /// Memory usage strategy.
    #[serde(default)]
    pub memory: MemoryStrategy,
    /// Debayering configuration. `None` = auto-detect from SER header.
    /// Set to `Some(config)` to force a specific method.
    #[serde(default)]
    pub debayer: Option<DebayerConfig>,
    /// When true, force mono processing even for Bayer/RGB sources.
    #[serde(default)]
    pub force_mono: bool,
    #[serde(default)]
    pub frame_selection: FrameSelectionConfig,
    /// Alignment algorithm configuration.
    #[serde(default)]
    pub alignment: AlignmentConfig,
    #[serde(default)]
    pub stacking: StackingConfig,
    pub sharpening: Option<SharpeningConfig>,
    #[serde(default)]
    pub filters: Vec<FilterStep>,
}

/// Configuration for debayering (demosaicing) raw Bayer data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebayerConfig {
    /// Debayering algorithm.
    #[serde(default)]
    pub method: DebayerMethod,
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum QualityMetric {
    #[default]
    Laplacian,
    Gradient,
}

/// Alignment algorithm to use for frame registration.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum AlignmentMethod {
    /// Standard FFT phase correlation with paraboloid sub-pixel fitting.
    #[default]
    PhaseCorrelation,
    /// Enhanced phase correlation with upsampled matrix-multiply DFT
    /// (Guizar-Sicairos et al., 2008). Sub-pixel accuracy ~0.01 px.
    EnhancedPhaseCorrelation(EnhancedPhaseConfig),
    /// Intensity-weighted centroid alignment. Very fast, good for bright
    /// planetary disks.
    Centroid(CentroidConfig),
    /// Sobel gradient preprocessing + phase correlation. More robust
    /// in noisy conditions.
    GradientCorrelation,
    /// Coarse-to-fine Gaussian pyramid alignment. Handles large
    /// displacements that exceed FFT wrap-around.
    Pyramid(PyramidConfig),
}

/// Parameters for enhanced phase correlation (Guizar-Sicairos method).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnhancedPhaseConfig {
    /// Upsampling factor for the matrix-multiply DFT refinement stage.
    /// Higher = more precise but slower. 20 gives ~0.05 px, 100 gives ~0.01 px.
    #[serde(default = "default_upsample_factor")]
    pub upsample_factor: usize,
}

fn default_upsample_factor() -> usize {
    DEFAULT_ENHANCED_PHASE_UPSAMPLE
}

impl Default for EnhancedPhaseConfig {
    fn default() -> Self {
        Self {
            upsample_factor: DEFAULT_ENHANCED_PHASE_UPSAMPLE,
        }
    }
}

/// Parameters for centroid (center-of-gravity) alignment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CentroidConfig {
    /// Intensity threshold below which pixels are excluded from centroid
    /// computation (fraction of max brightness, 0.0-1.0).
    #[serde(default = "default_centroid_threshold")]
    pub threshold: f32,
}

fn default_centroid_threshold() -> f32 {
    DEFAULT_CENTROID_THRESHOLD
}

impl Default for CentroidConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_CENTROID_THRESHOLD,
        }
    }
}

/// Parameters for coarse-to-fine pyramid alignment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PyramidConfig {
    /// Number of pyramid levels (each level halves dimensions).
    #[serde(default = "default_pyramid_levels")]
    pub levels: usize,
}

fn default_pyramid_levels() -> usize {
    DEFAULT_PYRAMID_LEVELS
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            levels: DEFAULT_PYRAMID_LEVELS,
        }
    }
}

/// Alignment configuration for the pipeline.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AlignmentConfig {
    /// The alignment algorithm to use.
    #[serde(default)]
    pub method: AlignmentMethod,
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
    Drizzle(DrizzleConfig),
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SharpeningConfig {
    pub wavelet: WaveletParams,
    #[serde(default)]
    pub deconvolution: Option<DeconvolutionConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PsfModel {
    Gaussian { sigma: f32 },
    Kolmogorov { seeing: f32 },
    Airy { radius: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DeconvolutionMethod {
    RichardsonLucy { iterations: usize },
    Wiener { noise_ratio: f32 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeconvolutionConfig {
    pub method: DeconvolutionMethod,
    pub psf: PsfModel,
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

// --- Display implementations ---

impl fmt::Display for QualityMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QualityMetric::Laplacian => write!(f, "Laplacian"),
            QualityMetric::Gradient => write!(f, "Gradient"),
        }
    }
}

impl fmt::Display for AlignmentMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlignmentMethod::PhaseCorrelation => write!(f, "Phase Correlation"),
            AlignmentMethod::EnhancedPhaseCorrelation(cfg) => {
                write!(f, "Enhanced Phase (upsample={})", cfg.upsample_factor)
            }
            AlignmentMethod::Centroid(cfg) => {
                write!(f, "Centroid (threshold={})", cfg.threshold)
            }
            AlignmentMethod::GradientCorrelation => write!(f, "Gradient Correlation"),
            AlignmentMethod::Pyramid(cfg) => {
                write!(f, "Pyramid ({} levels)", cfg.levels)
            }
        }
    }
}

impl fmt::Display for StackMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StackMethod::Mean => write!(f, "Mean"),
            StackMethod::Median => write!(f, "Median"),
            StackMethod::SigmaClip(_) => write!(f, "Sigma Clip"),
            StackMethod::MultiPoint(_) => write!(f, "Multi-Point"),
            StackMethod::Drizzle(cfg) => {
                write!(f, "Drizzle ({}x, pixfrac={})", cfg.scale, cfg.pixfrac)
            }
        }
    }
}

impl fmt::Display for DeconvolutionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeconvolutionMethod::RichardsonLucy { iterations } => {
                write!(f, "Richardson-Lucy ({iterations} iterations)")
            }
            DeconvolutionMethod::Wiener { noise_ratio } => {
                write!(f, "Wiener (noise ratio={noise_ratio})")
            }
        }
    }
}

impl fmt::Display for PsfModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PsfModel::Gaussian { sigma } => write!(f, "Gaussian (\u{03c3}={sigma} px)"),
            PsfModel::Kolmogorov { seeing } => write!(f, "Kolmogorov (seeing={seeing} px)"),
            PsfModel::Airy { radius } => write!(f, "Airy (radius={radius} px)"),
        }
    }
}

impl fmt::Display for FilterStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterStep::HistogramStretch {
                black_point,
                white_point,
            } => write!(f, "Histogram Stretch (black={black_point}, white={white_point})"),
            FilterStep::AutoStretch {
                low_percentile,
                high_percentile,
            } => write!(f, "Auto Stretch (low={low_percentile}, high={high_percentile})"),
            FilterStep::Gamma(g) => write!(f, "Gamma ({g})"),
            FilterStep::BrightnessContrast {
                brightness,
                contrast,
            } => write!(f, "Brightness/Contrast (b={brightness}, c={contrast})"),
            FilterStep::UnsharpMask {
                radius,
                amount,
                threshold,
            } => write!(f, "Unsharp Mask (r={radius}, a={amount}, t={threshold})"),
            FilterStep::GaussianBlur { sigma } => write!(f, "Gaussian Blur (\u{03c3}={sigma})"),
        }
    }
}

impl fmt::Display for LocalStackMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocalStackMethod::Mean => write!(f, "Mean"),
            LocalStackMethod::Median => write!(f, "Median"),
            LocalStackMethod::SigmaClip { sigma, iterations } => {
                write!(f, "Sigma Clip (\u{03c3}={sigma}, {iterations} iter)")
            }
        }
    }
}
