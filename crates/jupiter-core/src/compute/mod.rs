use std::fmt;
use std::sync::Arc;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod wgpu_backend;

/// Device preference for compute operations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum DevicePreference {
    #[default]
    Auto,
    Cpu,
    Gpu,
    Cuda,
}

impl fmt::Display for DevicePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "Auto"),
            Self::Cpu => write!(f, "CPU"),
            Self::Gpu => write!(f, "GPU"),
            Self::Cuda => write!(f, "CUDA"),
        }
    }
}

/// Opaque buffer that may reside on CPU or GPU memory.
///
/// Carries logical `(height, width)` metadata. For complex FFT data stored
/// as interleaved `[re, im, re, im, ...]`, `width` is the original real width,
/// while the underlying storage has `width * 2` columns.
pub struct GpuBuffer {
    pub(crate) inner: BufferInner,
    /// Logical height of the data.
    pub height: usize,
    /// Logical width (original real-valued width, not storage width).
    pub width: usize,
}

#[allow(dead_code)]
pub(crate) enum BufferInner {
    Cpu(Array2<f32>),
    #[cfg(feature = "gpu")]
    Wgpu {
        buffer: wgpu::Buffer,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    },
}

impl GpuBuffer {
    /// Wrap an existing CPU array in a GpuBuffer (zero-cost).
    pub fn from_array(data: Array2<f32>) -> Self {
        let (h, w) = data.dim();
        Self {
            inner: BufferInner::Cpu(data),
            height: h,
            width: w,
        }
    }

    /// Get a reference to the underlying CPU array, if this buffer is on CPU.
    pub fn as_array(&self) -> Option<&Array2<f32>> {
        match &self.inner {
            BufferInner::Cpu(arr) => Some(arr),
            #[cfg(feature = "gpu")]
            _ => None,
        }
    }
}

/// Object-safe trait for GPU-accelerable compute operations.
///
/// Held as `Arc<dyn ComputeBackend>`. The pipeline creates one backend at
/// startup and threads it through.
pub trait ComputeBackend: Send + Sync {
    /// Human-readable name (e.g. "CPU/Rayon", "wgpu/Metal").
    fn name(&self) -> &str;

    /// Whether this backend benefits from GPU-accelerated code paths.
    /// CPU backends return false to avoid format-conversion overhead.
    fn is_gpu(&self) -> bool {
        false
    }

    // --- FFT ---

    /// Forward 2D FFT. Input is real `(h, w)`, output is complex interleaved
    /// `(h, w*2)` with logical width `w`.
    fn fft2d(&self, input: &GpuBuffer) -> GpuBuffer;

    /// Inverse 2D FFT returning real part only, cropped to `(height, width)`.
    fn ifft2d_real(&self, input: &GpuBuffer, height: usize, width: usize) -> GpuBuffer;

    /// Normalized cross-power spectrum of two complex buffers.
    fn cross_power_spectrum(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer;

    /// Apply Hann window to a real buffer.
    fn hann_window(&self, input: &GpuBuffer) -> GpuBuffer;

    /// Find peak `(row, col, value)` in a real buffer.
    fn find_peak(&self, input: &GpuBuffer) -> (usize, usize, f64);

    // --- Bilinear shift ---

    /// Shift image by `(dx, dy)` using bilinear interpolation.
    fn shift_bilinear(&self, input: &GpuBuffer, dx: f64, dy: f64) -> GpuBuffer;

    // --- Convolutions ---

    /// Separable convolution with a 1D kernel (row pass then column pass).
    fn convolve_separable(&self, input: &GpuBuffer, kernel: &[f32]) -> GpuBuffer;

    /// A-trous (dilated) B3 spline convolution at the given wavelet scale.
    fn atrous_convolve(&self, input: &GpuBuffer, scale: usize) -> GpuBuffer;

    // --- Element-wise ---

    /// Element-wise complex multiplication of two interleaved complex buffers.
    fn complex_mul(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer;

    /// Element-wise real division: out[i] = a[i] / (b[i] + epsilon).
    fn divide_real(&self, a: &GpuBuffer, b: &GpuBuffer, epsilon: f32) -> GpuBuffer;

    /// Element-wise real multiplication: out[i] = a[i] * b[i].
    fn multiply_real(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer;

    // --- Transfer ---

    /// Upload a CPU array to this backend's preferred memory.
    fn upload(&self, data: &Array2<f32>) -> GpuBuffer;

    /// Download buffer contents to a CPU array.
    fn download(&self, buf: &GpuBuffer) -> Array2<f32>;
}

/// Create a compute backend based on the user's preference.
pub fn create_backend(pref: &DevicePreference) -> Arc<dyn ComputeBackend> {
    match pref {
        DevicePreference::Cpu => Arc::new(cpu::CpuBackend),
        DevicePreference::Auto => {
            #[cfg(feature = "gpu")]
            {
                match wgpu_backend::WgpuBackend::new() {
                    Ok(backend) => return Arc::new(backend),
                    Err(e) => {
                        tracing::warn!("GPU not available, falling back to CPU: {e}");
                    }
                }
            }
            Arc::new(cpu::CpuBackend)
        }
        DevicePreference::Gpu => {
            #[cfg(feature = "gpu")]
            {
                match wgpu_backend::WgpuBackend::new() {
                    Ok(backend) => return Arc::new(backend),
                    Err(e) => {
                        tracing::warn!("GPU not available, falling back to CPU: {e}");
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                tracing::warn!("GPU feature not enabled, using CPU");
            }
            Arc::new(cpu::CpuBackend)
        }
        DevicePreference::Cuda => {
            tracing::warn!("CUDA not yet implemented, falling back to CPU");
            Arc::new(cpu::CpuBackend)
        }
    }
}
