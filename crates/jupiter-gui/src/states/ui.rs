use std::path::PathBuf;

use jupiter_core::frame::SourceInfo;
use jupiter_core::pipeline::PipelineStage;

use super::crop::CropState;

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

    /// Cache status indicators.
    pub frames_scored: Option<usize>,
    pub ranked_preview: Vec<(usize, f64)>,
    pub align_status: Option<String>,
    pub stack_status: Option<String>,
    pub sharpen_status: bool,
    pub filter_status: Option<usize>,

    /// Log messages.
    pub log_messages: Vec<String>,

    /// Progress.
    pub progress_items_done: Option<usize>,
    pub progress_items_total: Option<usize>,

    /// Crop state.
    pub crop_state: CropState,

    /// Params changed since last run (stale indicators).
    pub score_params_dirty: bool,
    pub align_params_dirty: bool,
    pub stack_params_dirty: bool,
    pub sharpen_params_dirty: bool,
    pub filter_params_dirty: bool,
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
            frames_scored: None,
            ranked_preview: Vec::new(),
            align_status: None,
            stack_status: None,
            sharpen_status: false,
            filter_status: None,
            log_messages: Vec::new(),
            progress_items_done: None,
            progress_items_total: None,
            crop_state: CropState::default(),
            score_params_dirty: false,
            align_params_dirty: false,
            stack_params_dirty: false,
            sharpen_params_dirty: false,
            filter_params_dirty: false,
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

    /// Mark all pipeline stages as stale (e.g., debayer or metric changed).
    pub fn mark_dirty_from_score(&mut self) {
        self.score_params_dirty = true;
        self.align_params_dirty = true;
        self.stack_params_dirty = true;
        self.sharpen_params_dirty = true;
        self.filter_params_dirty = true;
    }

    /// Mark alignment and all downstream stages as stale.
    pub fn mark_dirty_from_align(&mut self) {
        self.align_params_dirty = true;
        self.stack_params_dirty = true;
        self.sharpen_params_dirty = true;
        self.filter_params_dirty = true;
    }

    /// Mark stacking and all downstream stages as stale.
    pub fn mark_dirty_from_stack(&mut self) {
        self.stack_params_dirty = true;
        self.sharpen_params_dirty = true;
        self.filter_params_dirty = true;
    }

    /// Mark sharpening and filter stages as stale.
    pub fn mark_dirty_from_sharpen(&mut self) {
        self.sharpen_params_dirty = true;
        self.filter_params_dirty = true;
    }

    /// Clear all dirty flags (e.g., after opening a new file).
    pub fn clear_all_dirty(&mut self) {
        self.score_params_dirty = false;
        self.align_params_dirty = false;
        self.stack_params_dirty = false;
        self.sharpen_params_dirty = false;
        self.filter_params_dirty = false;
    }
}
