/// Minimum pixel count (h*w) to use row-level Rayon parallelism.
pub const PARALLEL_PIXEL_THRESHOLD: usize = 65_536;

/// Minimum frame count to use frame-level Rayon parallelism.
pub const PARALLEL_FRAME_THRESHOLD: usize = 4;

/// B3 spline 1D kernel coefficients: [1, 4, 6, 4, 1] / 16.
pub const B3_KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Small epsilon to avoid division by zero in floating-point comparisons.
pub const EPSILON: f32 = 1e-10;
