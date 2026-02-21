use std::path::PathBuf;

use jupiter_core::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame, QualityScore};
use jupiter_core::pipeline::PipelineOutput;

use jupiter_core::color::debayer::DebayerMethod;

/// Cached intermediate results living on the worker thread.
pub(crate) struct PipelineCache {
    pub(crate) file_path: Option<PathBuf>,
    pub(crate) is_color: bool,
    pub(crate) is_streaming: bool,
    /// Stored color mode from the SER header, needed for re-reading color frames in streaming mode.
    pub(crate) color_mode: Option<ColorMode>,
    /// Stored debayer method, needed for re-reading Bayer frames in streaming mode.
    pub(crate) debayer_method: Option<DebayerMethod>,
    pub(crate) all_frames: Option<Vec<Frame>>,
    pub(crate) all_color_frames: Option<Vec<ColorFrame>>,
    pub(crate) ranked: Option<Vec<(usize, QualityScore)>>,
    /// Selected + aligned data (from Align stage).
    pub(crate) selected_frames: Option<Vec<Frame>>,
    pub(crate) selected_color_frames: Option<Vec<ColorFrame>>,
    pub(crate) alignment_offsets: Option<Vec<AlignmentOffset>>,
    /// Quality scores for the selected frames (same order as selected_frames).
    pub(crate) selected_quality_scores: Option<Vec<f64>>,
    /// Result after stacking (mono or color).
    pub(crate) stacked: Option<PipelineOutput>,
    /// Result after sharpening.
    pub(crate) sharpened: Option<PipelineOutput>,
    /// Result after filtering (final output).
    pub(crate) filtered: Option<PipelineOutput>,
}

impl PipelineCache {
    pub(super) fn new() -> Self {
        Self {
            file_path: None,
            is_color: false,
            is_streaming: false,
            color_mode: None,
            debayer_method: None,
            all_frames: None,
            all_color_frames: None,
            ranked: None,
            selected_frames: None,
            selected_color_frames: None,
            alignment_offsets: None,
            selected_quality_scores: None,
            stacked: None,
            sharpened: None,
            filtered: None,
        }
    }

    /// Latest available output for display/saving.
    pub(crate) fn latest_output(&self) -> Option<PipelineOutput> {
        self.filtered
            .clone()
            .or_else(|| self.sharpened.clone())
            .or_else(|| self.stacked.clone())
    }

    /// Set the stacked result and clear downstream stages.
    pub(crate) fn set_stacked(&mut self, output: PipelineOutput) {
        self.stacked = Some(output);
        self.sharpened = None;
        self.filtered = None;
    }

    pub(crate) fn invalidate_downstream(&mut self) {
        self.selected_frames = None;
        self.selected_color_frames = None;
        self.alignment_offsets = None;
        self.selected_quality_scores = None;
        self.invalidate_from_stack();
    }

    pub(crate) fn invalidate_from_stack(&mut self) {
        self.stacked = None;
        self.sharpened = None;
        self.filtered = None;
    }
}
