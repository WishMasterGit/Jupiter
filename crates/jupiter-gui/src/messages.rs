use std::path::PathBuf;
use std::time::Duration;

use jupiter_core::compute::DevicePreference;
use jupiter_core::frame::{Frame, SourceInfo};
use jupiter_core::pipeline::config::{
    FilterStep, PipelineConfig, QualityMetric, SharpeningConfig, StackMethod,
};
use jupiter_core::pipeline::PipelineStage;

/// Commands sent from UI thread to worker thread.
pub enum WorkerCommand {
    /// Open file and show metadata (lightweight, no frame loading).
    LoadFileInfo { path: PathBuf },

    /// Preview a single raw frame by index.
    PreviewFrame { path: PathBuf, frame_index: usize },

    /// Stage 1: Read all frames + score quality. Caches frames + ranked list.
    LoadAndScore { path: PathBuf, metric: QualityMetric },

    /// Stage 2: Select best frames, align, and stack.
    Stack {
        select_percentage: f32,
        method: StackMethod,
        device: DevicePreference,
    },

    /// Stage 3: Apply deconvolution + wavelet sharpening to cached stacked frame.
    Sharpen {
        config: SharpeningConfig,
        device: DevicePreference,
    },

    /// Stage 4: Apply filter chain to cached sharpened (or stacked) frame.
    ApplyFilters { filters: Vec<FilterStep> },

    /// Run all stages in sequence.
    RunAll { config: PipelineConfig },

    /// Save the currently displayed frame to disk.
    SaveImage { path: PathBuf },
}

/// Results sent from worker thread back to UI thread.
pub enum WorkerResult {
    FileInfo {
        path: PathBuf,
        info: SourceInfo,
    },
    FramePreview {
        frame: Frame,
        index: usize,
    },

    /// Stage 1 complete: frames loaded and scored.
    LoadAndScoreComplete {
        frame_count: usize,
        ranked_preview: Vec<(usize, f64)>,
    },

    /// Stage 2 complete: stacked result ready for preview.
    StackComplete {
        result: Frame,
        elapsed: Duration,
    },

    /// Stage 3 complete: sharpened result ready for preview.
    SharpenComplete {
        result: Frame,
        elapsed: Duration,
    },

    /// Stage 4 complete: filtered result ready for preview.
    FilterComplete {
        result: Frame,
        elapsed: Duration,
    },

    /// Full pipeline complete (RunAll mode).
    PipelineComplete {
        result: Frame,
        elapsed: Duration,
    },

    /// Progress update during any stage.
    Progress {
        #[allow(dead_code)]
        stage: PipelineStage,
        items_done: Option<usize>,
        items_total: Option<usize>,
    },

    ImageSaved {
        path: PathBuf,
    },
    Error {
        message: String,
    },
    Log {
        message: String,
    },
}
