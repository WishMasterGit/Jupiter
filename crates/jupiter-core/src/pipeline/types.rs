use crate::color::debayer::luminance;
use crate::frame::{ColorFrame, Frame};

/// Pipeline processing stage, used for progress reporting.
#[derive(Clone, Copy, Debug)]
pub enum PipelineStage {
    Reading,
    Debayering,
    QualityAssessment,
    FrameSelection,
    Alignment,
    Stacking,
    Sharpening,
    Filtering,
    Writing,
    Cropping,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reading => write!(f, "Reading frames"),
            Self::Debayering => write!(f, "Debayering"),
            Self::QualityAssessment => write!(f, "Assessing quality"),
            Self::FrameSelection => write!(f, "Selecting best frames"),
            Self::Alignment => write!(f, "Aligning frames"),
            Self::Stacking => write!(f, "Stacking"),
            Self::Sharpening => write!(f, "Sharpening"),
            Self::Filtering => write!(f, "Applying filters"),
            Self::Writing => write!(f, "Writing output"),
            Self::Cropping => write!(f, "Cropping"),
        }
    }
}

/// Result of the pipeline — either mono or color.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum PipelineOutput {
    Mono(Frame),
    Color(ColorFrame),
}

impl PipelineOutput {
    /// Get a mono frame. Color output is converted to luminance.
    pub fn to_mono(&self) -> Frame {
        match self {
            Self::Mono(f) => f.clone(),
            Self::Color(cf) => luminance(cf),
        }
    }
}

/// Thread-safe progress reporting for the pipeline.
///
/// Implementors can use this to drive progress bars, logging, or any other
/// UI feedback. All methods have default no-op implementations.
pub trait ProgressReporter: Send + Sync {
    /// A new pipeline stage has started. `total_items` is the number of
    /// work items in this stage (e.g., frame count), if known.
    fn begin_stage(&self, _stage: PipelineStage, _total_items: Option<usize>) {}

    /// One work item within the current stage has completed.
    fn advance(&self, _items_done: usize) {}

    /// The current stage is finished.
    fn finish_stage(&self) {}
}

/// No-op progress reporter, used when `run_pipeline` delegates.
pub(super) struct NoOpReporter;
impl ProgressReporter for NoOpReporter {}
