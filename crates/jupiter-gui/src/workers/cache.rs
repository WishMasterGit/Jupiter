use std::path::PathBuf;

use jupiter_core::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame, QualityScore};
use jupiter_core::io::disk_cache::{DiskColorFrame, DiskFrame, DiskFrameStore};
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
    /// Disk-backed frame store for streaming mode (RAII: cleaned up on drop).
    pub(crate) disk_store: Option<DiskFrameStore>,
    /// Stacked result persisted to disk (streaming mode).
    pub(crate) stacked_disk: Option<DiskFrameResult>,
    /// Sharpened result persisted to disk (streaming mode).
    pub(crate) sharpened_disk: Option<DiskFrameResult>,
    /// Filtered result persisted to disk (streaming mode).
    pub(crate) filtered_disk: Option<DiskFrameResult>,
}

/// Either a mono or color disk-backed frame result.
pub(crate) enum DiskFrameResult {
    Mono(DiskFrame),
    Color(DiskColorFrame),
}

impl DiskFrameResult {
    /// Read the disk-backed result into an in-memory [`PipelineOutput`].
    pub(crate) fn read(&self) -> jupiter_core::error::Result<PipelineOutput> {
        match self {
            DiskFrameResult::Mono(df) => Ok(PipelineOutput::Mono(df.read()?)),
            DiskFrameResult::Color(dcf) => Ok(PipelineOutput::Color(dcf.read()?)),
        }
    }
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
            disk_store: None,
            stacked_disk: None,
            sharpened_disk: None,
            filtered_disk: None,
        }
    }

    /// Ensure a disk store exists (created lazily on first use).
    pub(crate) fn ensure_disk_store(&mut self) -> jupiter_core::error::Result<&DiskFrameStore> {
        if self.disk_store.is_none() {
            self.disk_store = Some(DiskFrameStore::new()?);
        }
        Ok(self.disk_store.as_ref().unwrap())
    }

    /// Latest available output for display/saving.
    ///
    /// In streaming mode, loads from disk on demand (one frame at a time in RAM).
    pub(crate) fn latest_output(&self) -> Option<PipelineOutput> {
        // Prefer in-memory first
        if let Some(ref f) = self.filtered {
            return Some(f.clone());
        }
        if let Some(ref f) = self.filtered_disk {
            return f.read().ok();
        }
        if let Some(ref s) = self.sharpened {
            return Some(s.clone());
        }
        if let Some(ref s) = self.sharpened_disk {
            return s.read().ok();
        }
        if let Some(ref s) = self.stacked {
            return Some(s.clone());
        }
        if let Some(ref s) = self.stacked_disk {
            return s.read().ok();
        }
        None
    }

    /// Set the stacked result and clear downstream stages.
    pub(crate) fn set_stacked(&mut self, output: PipelineOutput) {
        self.stacked = Some(output);
        self.stacked_disk = None;
        self.clear_downstream_of_stack();
    }

    /// Set the stacked result on disk (streaming mode) and clear downstream.
    pub(crate) fn set_stacked_disk(&mut self, result: DiskFrameResult) {
        self.stacked = None;
        self.stacked_disk = Some(result);
        self.clear_downstream_of_stack();
    }

    /// Get the stacked output, loading from disk if needed.
    pub(crate) fn get_stacked(&self) -> Option<PipelineOutput> {
        if let Some(ref s) = self.stacked {
            return Some(s.clone());
        }
        if let Some(ref s) = self.stacked_disk {
            return s.read().ok();
        }
        None
    }

    /// Get the sharpened output, loading from disk if needed.
    pub(crate) fn get_sharpened(&self) -> Option<PipelineOutput> {
        if let Some(ref s) = self.sharpened {
            return Some(s.clone());
        }
        if let Some(ref s) = self.sharpened_disk {
            return s.read().ok();
        }
        None
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
        self.stacked_disk = None;
        self.clear_downstream_of_stack();
    }

    fn clear_downstream_of_stack(&mut self) {
        self.sharpened = None;
        self.sharpened_disk = None;
        self.filtered = None;
        self.filtered_disk = None;
    }

    /// Store a pipeline output to disk via the cache's disk store.
    ///
    /// Returns `None` if the disk store could not be initialized.
    pub(crate) fn store_output_to_disk(
        &mut self,
        output: &PipelineOutput,
    ) -> Option<DiskFrameResult> {
        let store = self.ensure_disk_store().ok()?;
        match output {
            PipelineOutput::Mono(frame) => store.store_frame(frame).ok().map(DiskFrameResult::Mono),
            PipelineOutput::Color(cf) => {
                store.store_color_frame(cf).ok().map(DiskFrameResult::Color)
            }
        }
    }
}
