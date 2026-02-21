use std::path::PathBuf;
use std::time::Instant;

use jupiter_core::frame::SourceInfo;
use jupiter_core::pipeline::PipelineStage;

use super::crop::CropState;
use super::stage_status::Stages;

/// Overall UI state.
pub struct UIState {
    pub file_path: Option<PathBuf>,
    pub source_info: Option<SourceInfo>,
    pub preview_frame_index: usize,
    pub output_path: String,

    /// True when a multi-frame video (SER) is loaded; false for single images.
    pub is_video: bool,

    /// Which stage is currently running (None = idle).
    pub running_stage: Option<PipelineStage>,

    /// Unified pipeline stage status (completion + dirty tracking).
    pub stages: Stages,

    /// Ranked frame preview data from scoring.
    pub ranked_preview: Vec<(usize, f64)>,

    /// Log messages.
    pub log_messages: Vec<String>,

    /// Progress.
    pub progress_items_done: Option<usize>,
    pub progress_items_total: Option<usize>,

    /// Crop state.
    pub crop_state: CropState,

    /// Debounce timer for auto-sharpening on slider changes.
    pub sharpen_auto_pending: Option<Instant>,
}

impl Default for UIState {
    fn default() -> Self {
        Self {
            file_path: None,
            source_info: None,
            preview_frame_index: 0,
            output_path: String::new(),
            is_video: false,
            running_stage: None,
            stages: Stages::default(),
            ranked_preview: Vec::new(),
            log_messages: Vec::new(),
            progress_items_done: None,
            progress_items_total: None,
            crop_state: CropState::default(),
            sharpen_auto_pending: None,
        }
    }
}

impl UIState {
    pub fn is_busy(&self) -> bool {
        self.running_stage.is_some()
    }

    pub fn add_log(&mut self, msg: String) {
        self.log_messages.push(msg);
    }

    /// Mark sharpening and filter stages as stale, and start debounce timer.
    pub fn mark_dirty_from_sharpen(&mut self) {
        self.stages.mark_dirty_from(PipelineStage::Sharpening);
        self.sharpen_auto_pending = Some(Instant::now());
    }

    /// Mark filter stage as stale.
    pub fn mark_dirty_from_filter(&mut self) {
        self.stages.mark_dirty_from(PipelineStage::Filtering);
    }

    /// Reset all pipeline state (for file-load).
    pub fn reset_pipeline(&mut self) {
        self.stages.reset_all();
        self.ranked_preview.clear();
        self.crop_state = Default::default();
        self.clear_progress();
        self.sharpen_auto_pending = None;
    }

    /// Clear progress counters.
    pub fn clear_progress(&mut self) {
        self.progress_items_done = None;
        self.progress_items_total = None;
    }
}
