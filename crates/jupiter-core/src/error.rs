use thiserror::Error;

#[derive(Error, Debug)]
pub enum JupiterError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid SER file: {0}")]
    InvalidSer(String),

    #[error("Invalid image dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("Frame index {index} out of range (total: {total})")]
    FrameIndexOutOfRange { index: usize, total: usize },

    #[error("Unsupported color mode: {0}")]
    UnsupportedColorMode(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Image format error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Empty frame sequence")]
    EmptySequence,

    #[error("GPU error: {0}")]
    GpuError(String),
}

pub type Result<T> = std::result::Result<T, JupiterError>;
