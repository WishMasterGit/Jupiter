use std::path::PathBuf;

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

    /// Detected planet diameter in pixels (from scoring step).
    pub detected_planet_diameter: Option<usize>,

    /// Set to true on mouse-up to trigger auto-sharpening.
    pub sharpen_requested: bool,

    /// Whether the viewport is showing a raw frame (true) or processed result (false).
    pub viewing_raw: bool,
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
            detected_planet_diameter: None,
            sharpen_requested: false,
            viewing_raw: true,
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

    /// Mark sharpening and downstream stages as stale (visual feedback while dragging).
    pub fn mark_dirty_from_sharpen(&mut self) {
        self.stages.mark_dirty_from(PipelineStage::Sharpening);
    }

    /// Request auto-sharpening (called on mouse-up / discrete control change).
    pub fn request_sharpen(&mut self) {
        self.sharpen_requested = true;
    }

    /// Mark filter stage as stale.
    pub fn mark_dirty_from_filter(&mut self) {
        self.stages.mark_dirty_from(PipelineStage::Filtering);
    }

    /// Reset all pipeline state (for file-load).
    pub fn reset_pipeline(&mut self) {
        self.stages.reset_all();
        self.ranked_preview.clear();
        self.detected_planet_diameter = None;
        self.crop_state = Default::default();
        self.clear_progress();
        self.sharpen_requested = false;
        self.viewing_raw = true;
    }

    /// Clear progress counters.
    pub fn clear_progress(&mut self) {
        self.progress_items_done = None;
        self.progress_items_total = None;
    }
}
