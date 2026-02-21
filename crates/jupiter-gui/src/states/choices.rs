use std::fmt;

use jupiter_core::pipeline::config::FilterStep;

/// Alignment method selector (no associated data — just the discriminant).
#[derive(Clone, Copy, PartialEq, Default)]
pub enum AlignMethodChoice {
    #[default]
    PhaseCorrelation,
    EnhancedPhase,
    Centroid,
    GradientCorrelation,
    Pyramid,
}

impl AlignMethodChoice {
    pub const ALL: &[Self] = &[
        Self::PhaseCorrelation,
        Self::EnhancedPhase,
        Self::Centroid,
        Self::GradientCorrelation,
        Self::Pyramid,
    ];
}

impl fmt::Display for AlignMethodChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhaseCorrelation => write!(f, "Phase Correlation"),
            Self::EnhancedPhase => write!(f, "Enhanced Phase"),
            Self::Centroid => write!(f, "Centroid"),
            Self::GradientCorrelation => write!(f, "Gradient Correlation"),
            Self::Pyramid => write!(f, "Pyramid"),
        }
    }
}

/// Stacking method selector.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum StackMethodChoice {
    #[default]
    Mean,
    Median,
    SigmaClip,
    MultiPoint,
    Drizzle,
    SurfaceWarp,
}

impl StackMethodChoice {
    pub const ALL: &[Self] = &[
        Self::Mean,
        Self::Median,
        Self::SigmaClip,
        Self::MultiPoint,
        Self::Drizzle,
        Self::SurfaceWarp,
    ];
}

impl fmt::Display for StackMethodChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mean => write!(f, "Mean"),
            Self::Median => write!(f, "Median"),
            Self::SigmaClip => write!(f, "Sigma Clip"),
            Self::MultiPoint => write!(f, "Multi-Point"),
            Self::Drizzle => write!(f, "Drizzle"),
            Self::SurfaceWarp => write!(f, "Surface Warp"),
        }
    }
}

/// Deconvolution method selector.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum DeconvMethodChoice {
    #[default]
    RichardsonLucy,
    Wiener,
}

impl DeconvMethodChoice {
    pub const ALL: &[Self] = &[Self::RichardsonLucy, Self::Wiener];
}

impl fmt::Display for DeconvMethodChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RichardsonLucy => write!(f, "Richardson-Lucy"),
            Self::Wiener => write!(f, "Wiener"),
        }
    }
}

/// PSF model selector.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum PsfModelChoice {
    #[default]
    Gaussian,
    Kolmogorov,
    Airy,
}

impl PsfModelChoice {
    pub const ALL: &[Self] = &[Self::Gaussian, Self::Kolmogorov, Self::Airy];
}

impl fmt::Display for PsfModelChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gaussian => write!(f, "Gaussian"),
            Self::Kolmogorov => write!(f, "Kolmogorov"),
            Self::Airy => write!(f, "Airy"),
        }
    }
}

/// Filter type selector (for the "Add Filter" menu).
#[derive(Clone, Copy, PartialEq)]
pub enum FilterType {
    AutoStretch,
    HistogramStretch,
    Gamma,
    BrightnessContrast,
    UnsharpMask,
    GaussianBlur,
}

impl FilterType {
    pub const ALL: &[Self] = &[
        Self::AutoStretch,
        Self::HistogramStretch,
        Self::Gamma,
        Self::BrightnessContrast,
        Self::UnsharpMask,
        Self::GaussianBlur,
    ];

    /// Create a `FilterStep` with default parameters for this filter type.
    pub fn default_step(&self) -> FilterStep {
        match self {
            Self::AutoStretch => FilterStep::AutoStretch {
                low_percentile: 0.001,
                high_percentile: 0.999,
            },
            Self::HistogramStretch => FilterStep::HistogramStretch {
                black_point: 0.0,
                white_point: 1.0,
            },
            Self::Gamma => FilterStep::Gamma(1.0),
            Self::BrightnessContrast => FilterStep::BrightnessContrast {
                brightness: 0.0,
                contrast: 1.0,
            },
            Self::UnsharpMask => FilterStep::UnsharpMask {
                radius: 1.5,
                amount: 0.5,
                threshold: 0.0,
            },
            Self::GaussianBlur => FilterStep::GaussianBlur { sigma: 1.0 },
        }
    }
}

impl fmt::Display for FilterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AutoStretch => write!(f, "Auto Stretch"),
            Self::HistogramStretch => write!(f, "Histogram Stretch"),
            Self::Gamma => write!(f, "Gamma"),
            Self::BrightnessContrast => write!(f, "Brightness/Contrast"),
            Self::UnsharpMask => write!(f, "Unsharp Mask"),
            Self::GaussianBlur => write!(f, "Gaussian Blur"),
        }
    }
}
