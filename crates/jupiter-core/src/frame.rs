use ndarray::Array2;
use std::path::PathBuf;

/// A single grayscale image frame.
/// Pixel values are f32 in [0.0, 1.0].
#[derive(Clone, Debug)]
pub struct Frame {
    /// Pixel data, row-major, shape = (height, width)
    pub data: Array2<f32>,
    /// Original bit depth before conversion (8 or 16)
    pub original_bit_depth: u8,
    /// Optional per-frame metadata
    pub metadata: FrameMetadata,
}

impl Frame {
    pub fn new(data: Array2<f32>, bit_depth: u8) -> Self {
        Self {
            data,
            original_bit_depth: bit_depth,
            metadata: FrameMetadata::default(),
        }
    }

    pub fn width(&self) -> usize {
        self.data.ncols()
    }

    pub fn height(&self) -> usize {
        self.data.nrows()
    }
}

#[derive(Clone, Debug, Default)]
pub struct FrameMetadata {
    pub frame_index: usize,
    pub quality_score: Option<QualityScore>,
    pub timestamp_us: Option<u64>,
}

/// Quality assessment result for a single frame.
#[derive(Clone, Debug)]
pub struct QualityScore {
    pub laplacian_variance: f64,
    pub composite: f64,
}

/// Color image composed of separate channel frames.
#[derive(Clone, Debug)]
pub struct ColorFrame {
    pub red: Frame,
    pub green: Frame,
    pub blue: Frame,
}

/// Alignment offset for a frame relative to a reference.
#[derive(Clone, Debug, Default)]
pub struct AlignmentOffset {
    pub dx: f64,
    pub dy: f64,
}

/// Color/Bayer mode of the source data.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ColorMode {
    Mono,
    BayerRGGB,
    BayerGRBG,
    BayerGBRG,
    BayerBGGR,
    RGB,
    BGR,
}

/// Metadata about the source file.
#[derive(Clone, Debug)]
pub struct SourceInfo {
    pub filename: PathBuf,
    pub total_frames: usize,
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_mode: ColorMode,
    pub observer: Option<String>,
    pub telescope: Option<String>,
    pub instrument: Option<String>,
}
