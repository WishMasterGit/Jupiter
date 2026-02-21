/// Minimum pixel count (h*w) to use row-level Rayon parallelism.
pub const PARALLEL_PIXEL_THRESHOLD: usize = 65_536;

/// Minimum frame count to use frame-level Rayon parallelism.
pub const PARALLEL_FRAME_THRESHOLD: usize = 4;

/// B3 spline 1D kernel coefficients: [1, 4, 6, 4, 1] / 16.
pub const B3_KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Small epsilon to avoid division by zero in floating-point comparisons.
pub const EPSILON: f32 = 1e-10;

/// ITU-R BT.601 luminance coefficient for the red channel.
pub const LUMINANCE_R: f32 = 0.299;

/// ITU-R BT.601 luminance coefficient for the green channel.
pub const LUMINANCE_G: f32 = 0.587;

/// ITU-R BT.601 luminance coefficient for the blue channel.
pub const LUMINANCE_B: f32 = 0.114;

/// Decoded frame data size (in bytes) above which the pipeline switches to
/// low-memory streaming mode. Default: 1 GiB.
pub const LOW_MEMORY_THRESHOLD_BYTES: usize = 1_073_741_824;

/// Number of frames decoded simultaneously during streaming quality scoring.
/// Balances memory usage vs. parallelism. At 4096x4096 f32, 8 frames = 512 MB.
pub const STREAMING_BATCH_SIZE: usize = 8;

/// Number of channels in a color frame (R, G, B).
pub const COLOR_CHANNEL_COUNT: usize = 3;

/// Default upsampling factor for enhanced phase correlation (Guizar-Sicairos).
/// 20 gives ~0.05 px accuracy; 100 gives ~0.01 px accuracy.
pub const DEFAULT_ENHANCED_PHASE_UPSAMPLE: usize = 20;

/// Default intensity threshold for centroid alignment (fraction of max brightness).
pub const DEFAULT_CENTROID_THRESHOLD: f32 = 0.1;

/// Default number of Gaussian pyramid levels for coarse-to-fine alignment.
pub const DEFAULT_PYRAMID_LEVELS: usize = 3;

/// Search window (in pixels) around the coarse peak for enhanced phase
/// correlation upsampled DFT refinement.
pub const ENHANCED_PHASE_SEARCH_WINDOW: f64 = 1.5;

/// Gaussian blur sigma used for building the pyramid in coarse-to-fine alignment.
pub const PYRAMID_BLUR_SIGMA: f32 = 1.0;

/// Default number of frames to sample for auto-crop planet detection.
pub const DEFAULT_AUTOCROP_SAMPLE_COUNT: usize = 30;

/// Default padding fraction around the detected planet for auto-crop (15%).
pub const DEFAULT_AUTOCROP_PADDING_FRACTION: f32 = 0.15;

/// Default sigma multiplier for MeanPlusSigma thresholding in auto-crop.
pub const DEFAULT_AUTOCROP_SIGMA_MULTIPLIER: f32 = 2.0;

/// Number of histogram bins for Otsu's thresholding.
pub const OTSU_HISTOGRAM_BINS: usize = 256;

/// Default Gaussian blur sigma for noise suppression before planet detection.
pub const DEFAULT_AUTOCROP_BLUR_SIGMA: f32 = 2.5;

/// Minimum connected component area (pixels) to be considered a planet candidate.
pub const DEFAULT_AUTOCROP_MIN_AREA: usize = 100;

/// Sigma threshold for outlier centroid rejection during temporal filtering.
pub const AUTOCROP_SIGMA_CLIP_THRESHOLD: f64 = 2.5;

/// Number of sigma-clipping iterations for centroid outlier rejection.
pub const AUTOCROP_SIGMA_CLIP_ITERATIONS: usize = 3;

/// Minimum number of valid detections for multi-frame analysis to proceed.
pub const AUTOCROP_MIN_VALID_DETECTIONS: usize = 3;

/// Crop size alignment multiple for FFT efficiency.
pub const AUTOCROP_SIZE_ALIGNMENT: u32 = 32;

/// Width of the border strip (pixels) used for background level estimation.
pub const AUTOCROP_BORDER_STRIP_WIDTH: usize = 10;

/// Number of center frames to median-combine for fallback detection.
pub const AUTOCROP_FALLBACK_FRAME_COUNT: usize = 5;

/// Divisor to derive AP size from planet diameter: `ap_size = diameter / divisor`.
pub const AUTO_AP_DIVISOR: usize = 8;

/// Minimum auto-computed AP size in pixels.
pub const AUTO_AP_SIZE_MIN: usize = 32;

/// Maximum auto-computed AP size in pixels.
pub const AUTO_AP_SIZE_MAX: usize = 256;

/// Auto AP size is rounded down to this alignment (pixels).
pub const AUTO_AP_SIZE_ALIGN: usize = 8;
