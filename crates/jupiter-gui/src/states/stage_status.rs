use egui::Color32;
use jupiter_core::pipeline::PipelineStage;

const COLOR_COMPLETE: Color32 = Color32::from_rgb(80, 180, 80);

/// Per-stage completion + dirty tracking.
#[derive(Clone, Default)]
pub struct StageStatus {
    /// Some(label) when stage is complete; None when not run or invalidated.
    label: Option<String>,
    /// True when parameters changed since last run.
    dirty: bool,
}

impl StageStatus {
    /// Mark this stage as complete with a human-readable label.
    pub fn set_complete(&mut self, label: String) {
        self.label = Some(label);
        self.dirty = false;
    }

    /// Clear completion (stage not run or invalidated).
    pub fn clear(&mut self) {
        self.label = None;
    }

    /// Clear both completion and dirty flag.
    pub fn reset(&mut self) {
        self.label = None;
        self.dirty = false;
    }

    pub fn is_complete(&self) -> bool {
        self.label.is_some()
    }

    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Returns the header fill color:
    /// - green if completed and up-to-date
    /// - None (transparent) otherwise
    pub fn button_color(&self) -> Option<Color32> {
        if self.label.is_some() && !self.dirty {
            Some(COLOR_COMPLETE)
        } else {
            None
        }
    }
}

/// Container for all 5 pipeline stage statuses with cascade logic.
#[derive(Clone, Default)]
pub struct Stages {
    pub score: StageStatus,
    pub align: StageStatus,
    pub stack: StageStatus,
    pub sharpen: StageStatus,
    pub filter: StageStatus,
}

impl Stages {
    /// Mark this stage and all downstream stages as dirty.
    pub fn mark_dirty_from(&mut self, stage: PipelineStage) {
        match stage {
            PipelineStage::QualityAssessment => {
                self.score.mark_dirty();
                self.align.mark_dirty();
                self.stack.mark_dirty();
                self.sharpen.mark_dirty();
                self.filter.mark_dirty();
            }
            PipelineStage::Alignment => {
                self.align.mark_dirty();
                self.stack.mark_dirty();
                self.sharpen.mark_dirty();
                self.filter.mark_dirty();
            }
            PipelineStage::Stacking => {
                self.stack.mark_dirty();
                self.sharpen.mark_dirty();
                self.filter.mark_dirty();
            }
            PipelineStage::Sharpening => {
                self.sharpen.mark_dirty();
                self.filter.mark_dirty();
            }
            PipelineStage::Filtering => {
                self.filter.mark_dirty();
            }
            _ => {}
        }
    }

    /// Clear completion for stages strictly after this one.
    pub fn clear_downstream(&mut self, stage: PipelineStage) {
        match stage {
            PipelineStage::QualityAssessment => {
                self.align.clear();
                self.stack.clear();
                self.sharpen.clear();
                self.filter.clear();
            }
            PipelineStage::Alignment => {
                self.stack.clear();
                self.sharpen.clear();
                self.filter.clear();
            }
            PipelineStage::Stacking => {
                self.sharpen.clear();
                self.filter.clear();
            }
            PipelineStage::Sharpening => {
                self.filter.clear();
            }
            PipelineStage::Filtering => {}
            _ => {}
        }
    }

    /// Clear everything (for file-load).
    pub fn reset_all(&mut self) {
        self.score.reset();
        self.align.reset();
        self.stack.reset();
        self.sharpen.reset();
        self.filter.reset();
    }
}
