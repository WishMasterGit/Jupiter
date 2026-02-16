use std::path::PathBuf;
use std::time::Duration;

use jupiter_core::compute::DevicePreference;
use jupiter_core::frame::SourceInfo;
use jupiter_core::io::crop::CropRect;
use jupiter_core::pipeline::config::{
    AlignmentConfig, DebayerConfig, FilterStep, PipelineConfig, QualityMetric, SharpeningConfig,
    StackMethod,
};
use jupiter_core::pipeline::{PipelineOutput, PipelineStage};

/// Commands sent from UI thread to worker thread.
pub enum WorkerCommand {
    /// Open file and show metadata (lightweight, no frame loading).
    LoadFileInfo { path: PathBuf },

    /// Preview a single raw frame by index.
    PreviewFrame { path: PathBuf, frame_index: usize },

    /// Stage 1: Read all frames + score quality. Caches frames + ranked list.
    LoadAndScore {
        path: PathBuf,
        metric: QualityMetric,
        debayer: Option<DebayerConfig>,
    },

    /// Stage 2: Select best frames and compute alignment offsets.
    Align {
        select_percentage: f32,
        alignment: AlignmentConfig,
        device: DevicePreference,
    },

    /// Stage 3: Stack using cached aligned frames.
    Stack {
        method: StackMethod,
    },

    /// Stage 4: Apply deconvolution + wavelet sharpening to cached stacked frame.
    Sharpen {
        config: SharpeningConfig,
        device: DevicePreference,
    },

    /// Stage 5: Apply filter chain to cached sharpened (or stacked) frame.
    ApplyFilters { filters: Vec<FilterStep> },

    /// Run all stages in sequence.
    RunAll { config: PipelineConfig },

    /// Save the currently displayed frame to disk.
    SaveImage { path: PathBuf },

    /// Crop a SER file and save to a new file.
    CropAndSave {
        source_path: PathBuf,
        output_path: PathBuf,
        crop: CropRect,
    },
}

/// Results sent from worker thread back to UI thread.
pub enum WorkerResult {
    FileInfo {
        path: PathBuf,
        info: SourceInfo,
    },
    FramePreview {
        output: PipelineOutput,
        index: usize,
    },

    /// Stage 1 complete: frames loaded and scored.
    LoadAndScoreComplete {
        frame_count: usize,
        ranked_preview: Vec<(usize, f64)>,
    },

    /// Stage 2 complete: frames selected and alignment offsets computed.
    AlignComplete {
        frame_count: usize,
        elapsed: Duration,
    },

    /// Stage 3 complete: stacked result ready for preview.
    StackComplete {
        result: PipelineOutput,
        elapsed: Duration,
    },

    /// Stage 3 complete: sharpened result ready for preview.
    SharpenComplete {
        result: PipelineOutput,
        elapsed: Duration,
    },

    /// Stage 4 complete: filtered result ready for preview.
    FilterComplete {
        result: PipelineOutput,
        elapsed: Duration,
    },

    /// Full pipeline complete (RunAll mode).
    PipelineComplete {
        result: PipelineOutput,
        elapsed: Duration,
    },

    /// Progress update during any stage.
    Progress {
        stage: PipelineStage,
        items_done: Option<usize>,
        items_total: Option<usize>,
    },

    CropComplete {
        output_path: PathBuf,
        elapsed: Duration,
    },
    ImageSaved {
        path: PathBuf,
    },
    Error {
        message: String,
    },
    ConfigImported {
        config: jupiter_core::pipeline::config::PipelineConfig,
    },
    Log {
        message: String,
    },
}
