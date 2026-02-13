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
