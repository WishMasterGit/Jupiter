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
pub const DEFAULT_AUTOCROP_SAMPLE_COUNT: usize = 1;

/// Default padding fraction around the detected planet for auto-crop (10%).
pub const DEFAULT_AUTOCROP_PADDING_FRACTION: f32 = 0.1;

/// Default sigma multiplier for MeanPlusSigma thresholding in auto-crop.
pub const DEFAULT_AUTOCROP_SIGMA_MULTIPLIER: f32 = 2.0;

/// Number of histogram bins for Otsu's thresholding.
pub const OTSU_HISTOGRAM_BINS: usize = 256;
