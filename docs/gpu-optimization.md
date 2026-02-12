# GPU Acceleration in Jupiter

Jupiter's lucky imaging pipeline is compute-heavy. Frame alignment requires 2D FFTs on every frame, deconvolution runs 20-50 iterations of FFT-based Richardson-Lucy, and per-pixel operations touch millions of elements. These workloads are embarrassingly parallel and map naturally to GPU hardware.

Jupiter offloads these parallel workloads to Metal, Vulkan, or DX12 via [wgpu](https://wgpu.rs/). GPU support is feature-flagged: build with `cargo build --features gpu` to enable it, or leave it off for a zero-cost CPU-only binary.

## Architecture Overview

The GPU system is built around a single trait abstraction that hides the difference between CPU and GPU execution. The key types live in `crates/jupiter-core/src/compute/mod.rs`:

- **`ComputeBackend` trait** -- defines all compute operations (FFT, convolution, element-wise math, etc.)
- **`GpuBuffer`** -- opaque data container that may live in CPU or GPU memory
- **`BufferInner`** -- enum discriminating between `Cpu(Array2<f32>)` and `Wgpu { buffer, device, queue }`
- **`DevicePreference`** -- user's choice: `Auto`, `Cpu`, `Gpu`, or `Cuda`
- **`create_backend()`** -- factory that returns `Arc<dyn ComputeBackend>`

The backend is injected into the pipeline as `Arc<dyn ComputeBackend>`. Algorithms call trait methods without knowing whether they're running on CPU or GPU. When a code path benefits from GPU acceleration, it checks `backend.is_gpu()` to choose between GPU-specific and CPU-specific implementations.

```
                          ┌──────────────────────┐
                          │   ComputeBackend     │
                          │       (trait)         │
                          └──────┬───────┬───────┘
                                 │       │
                    ┌────────────┘       └────────────┐
                    │                                  │
             ┌──────┴──────┐                  ┌───────┴───────┐
             │  CpuBackend │                  │  WgpuBackend  │
             │ rustfft +   │                  │ Metal/Vulkan/ │
             │ ndarray +   │                  │ DX12 via wgpu │
             │ rayon       │                  └───────────────┘
             └─────────────┘
```

### GpuBuffer

`GpuBuffer` carries logical `(height, width)` metadata alongside its storage. For complex FFT data stored as interleaved `[re, im, re, im, ...]`, `width` is the original real width while the underlying storage has `width * 2` columns.

```rust
pub struct GpuBuffer {
    pub(crate) inner: BufferInner,
    pub height: usize,
    pub width: usize,
}

pub(crate) enum BufferInner {
    Cpu(Array2<f32>),
    #[cfg(feature = "gpu")]
    Wgpu {
        buffer: wgpu::Buffer,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    },
}
```

CPU buffers wrap an `Array2<f32>` at zero cost via `GpuBuffer::from_array()`. GPU buffers hold a wgpu `Buffer` plus shared references to the device and queue for later readback.

### Device Selection

`create_backend()` maps the user's `DevicePreference` to a concrete backend:

| Preference | Behavior |
|------------|----------|
| `Auto` | Try wgpu GPU, fall back to CPU on failure |
| `Cpu` | Always use CPU backend |
| `Gpu` | Try wgpu GPU, warn and fall back to CPU if unavailable |
| `Cuda` | Not yet implemented, falls back to CPU with warning |

When the `gpu` feature is not compiled in, GPU/Auto requests silently fall back to CPU.

## The ComputeBackend Trait

The trait defines every operation the pipeline needs. Algorithms compose these primitives without caring about the backend.

```rust
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn is_gpu(&self) -> bool { false }

    // FFT
    fn fft2d(&self, input: &GpuBuffer) -> GpuBuffer;
    fn ifft2d_real(&self, input: &GpuBuffer, height: usize, width: usize) -> GpuBuffer;
    fn cross_power_spectrum(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer;
    fn hann_window(&self, input: &GpuBuffer) -> GpuBuffer;
    fn find_peak(&self, input: &GpuBuffer) -> (usize, usize, f64);

    // Spatial
    fn shift_bilinear(&self, input: &GpuBuffer, dx: f64, dy: f64) -> GpuBuffer;
    fn convolve_separable(&self, input: &GpuBuffer, kernel: &[f32]) -> GpuBuffer;
    fn atrous_convolve(&self, input: &GpuBuffer, scale: usize) -> GpuBuffer;

    // Element-wise
    fn complex_mul(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer;
    fn divide_real(&self, a: &GpuBuffer, b: &GpuBuffer, epsilon: f32) -> GpuBuffer;
    fn multiply_real(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer;

    // Transfer
    fn upload(&self, data: &Array2<f32>) -> GpuBuffer;
    fn download(&self, buf: &GpuBuffer) -> Array2<f32>;
}
```

The trait is `Send + Sync`, allowing shared access from Rayon parallel iterators. `is_gpu()` defaults to `false` so CPU backends don't need to override it.

## CPU Backend

`CpuBackend` in `compute/cpu.rs` is the reference implementation. It uses:

- **rustfft** for FFT/IFFT
- **ndarray** for array operations
- **Rayon** for parallelism on large images

Complex data is stored as interleaved f32 pairs `[re, im, re, im, ...]` to match the GPU's data format, keeping upload/download trivial.

The CPU backend respects a parallelism threshold: `PARALLEL_PIXEL_THRESHOLD = 65,536` (256x256). Images below this size run single-threaded to avoid Rayon overhead on small frames.

## wgpu Backend

`WgpuBackend` in `compute/wgpu_backend.rs` is the GPU implementation. It targets Metal (macOS), Vulkan (Linux/Windows), and DX12 (Windows) through wgpu's hardware abstraction.

### Initialization

Initialization is synchronous (via `pollster::block_on`) and performs four steps:

1. **Create wgpu instance** with default backend selection
2. **Request adapter** with `HighPerformance` power preference
3. **Request device and queue** with default limits
4. **Compile all shader modules and create 15 compute pipelines**

```rust
pub fn new() -> Result<Self, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "No suitable GPU adapter found".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(/* ... */))
        .map_err(|e| format!("Failed to create GPU device: {e}"))?;

    // Compile 13 shader modules, create 15 compute pipelines...
}
```

All 15 pipelines are compiled upfront so there's no shader compilation stall during processing.

### The 15 Compute Pipelines

| Pipeline | Shader Source | Purpose |
|----------|--------------|---------|
| `hann_pipeline` | hann_window.wgsl | 2D Hann window |
| `cross_power_pipeline` | cross_power.wgsl | Normalized cross-power spectrum |
| `shift_pipeline` | bilinear_shift.wgsl | Sub-pixel image shift |
| `peak_local_pipeline` | find_peak.wgsl | Per-workgroup max reduction |
| `peak_global_pipeline` | find_peak.wgsl | Global max reduction |
| `fft_pipeline` | fft_stockham.wgsl | Stockham radix-2 FFT stage |
| `transpose_complex_pipeline` | inline WGSL | Complex matrix transpose |
| `pad_r2c_pipeline` | inline WGSL | Real-to-complex zero-padding |
| `extract_real_pipeline` | inline WGSL | Extract real part + scale + crop |
| `convolve_rows_pipeline` | convolve_separable.wgsl | Horizontal separable convolution |
| `convolve_cols_pipeline` | convolve_separable.wgsl | Vertical separable convolution |
| `complex_mul_pipeline` | elementwise.wgsl | Complex multiplication |
| `normalize_pipeline` | inline WGSL | Scale array by constant |
| `divide_real_pipeline` | inline WGSL | Element-wise division |
| `multiply_real_pipeline` | inline WGSL | Element-wise multiplication |

### Dispatch Pattern

All single-pass operations share a common dispatch helper:

```rust
fn dispatch(
    &self,
    pipeline: &wgpu::ComputePipeline,
    entries: &[wgpu::BindGroupEntry],
    workgroups: (u32, u32, u32),
) {
    let layout = pipeline.get_bind_group_layout(0);
    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries,
    });
    let mut enc = self.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    self.queue.submit(std::iter::once(enc.finish()));
}
```

The pattern is: create bind group from pipeline layout -> encode compute pass -> dispatch -> submit. Multi-pass operations (FFT, peak finding) build custom command encoders to batch multiple dispatches into a single submission.

## GPU FFT: Stockham Radix-2

The FFT is the most complex GPU operation. Jupiter implements a 2D FFT using the four-step approach: row FFTs, transpose, column FFTs, transpose.

### Forward FFT Pipeline

```
Input (h x w, real)
    │
    ▼
Zero-pad to power-of-2 (ph x pw)
    │
    ▼
Real-to-complex (pad_r2c shader)
    │
    ▼
Row-wise FFT (batch of ph FFTs, length pw)
    │
    ▼
Transpose (ph x pw) → (pw x ph)
    │
    ▼
Column-wise FFT (batch of pw FFTs, length ph)
    │
    ▼
Transpose (pw x ph) → (ph x pw)
    │
    ▼
Output (ph x pw, complex interleaved)
```

### Stockham Algorithm

Each 1D FFT uses the Stockham radix-2 algorithm, which has two key advantages over Cooley-Tukey:

1. **Auto-sorted output** -- no bit-reversal permutation needed
2. **Out-of-place** -- naturally maps to ping-pong GPU buffers

The algorithm runs `log2(n)` stages. Each stage performs butterfly operations that read from one buffer and write to another, alternating between two buffers:

```rust
fn fft_1d_batch(&self, input: &wgpu::Buffer, total_complex: u32,
                n: u32, batch_count: u32, batch_stride: u32,
                direction: f32) -> wgpu::Buffer
{
    let num_stages = n.trailing_zeros();

    // Pre-create uniform buffers for each stage
    let stage_uniforms: Vec<wgpu::Buffer> = (0..num_stages)
        .map(|s| self.create_uniform(&FftParams {
            n, stage: s, direction, batch_count, batch_stride
        }))
        .collect();

    // Encode all stages in one command buffer
    let mut enc = self.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(input, 0, &buf_a, 0, byte_size);

    for stage in 0..num_stages {
        let (src, dst) = if stage % 2 == 0 {
            (&buf_a, &buf_b)
        } else {
            (&buf_b, &buf_a)
        };
        // Dispatch fft_pipeline with src → dst
    }

    self.queue.submit(std::iter::once(enc.finish()));
    if num_stages % 2 == 0 { buf_a } else { buf_b }
}
```

All stages are encoded into a single command buffer and submitted once, minimizing CPU-GPU synchronization.

### Inverse FFT

IFFT uses the same pipeline with `direction = -1.0`. The shader computes twiddle factors as `exp(j * TAU * k / N)` instead of `exp(-j * TAU * k / N)`. After the inverse transform, the `extract_real_scaled` shader extracts the real part, scales by `1/(h*w)`, and crops back to the original dimensions.

## WGSL Compute Shaders

Jupiter uses 7 external WGSL shader files plus 6 inline shaders embedded in `wgpu_backend.rs`. (The `shaders/` directory also contains `transpose.wgsl` and `real_to_complex.wgsl`, which are superseded by inline versions.)

### External Shaders (`compute/shaders/`)

**hann_window.wgsl** (workgroup 16x16) -- Applies a 2D Hann (raised-cosine) window. Each thread computes `out[r,c] = in[r,c] * 0.5*(1 - cos(TAU*r/h)) * 0.5*(1 - cos(TAU*c/w))`. Used before FFT to reduce spectral leakage in phase correlation.

**cross_power.wgsl** (workgroup 256) -- Computes the normalized cross-power spectrum: `(A * conj(B)) / |A * conj(B)|` for each complex element. This is the core of phase correlation -- the peak in its inverse FFT gives the translation offset.

**bilinear_shift.wgsl** (workgroup 16x16) -- Shifts an image by fractional `(dx, dy)` using bilinear interpolation. Out-of-bounds pixels return 0.0. Used to align frames after computing offsets.

**find_peak.wgsl** (workgroup 256) -- Two-pass parallel reduction to find the maximum value and its location. `find_peak_local` reduces within each workgroup, `find_peak_global` reduces across workgroups. Output is `[flat_index, 0, value_bits]` as u32.

**fft_stockham.wgsl** (workgroup 256) -- Single-stage radix-2 Stockham butterfly. Parameters: FFT length `n`, stage index, direction (forward/inverse), batch count, and batch stride. Each thread computes one butterfly with twiddle factors calculated on-the-fly.

**convolve_separable.wgsl** (workgroup 256) -- Two entry points: `convolve_rows` and `convolve_cols`. Supports dilated (a-trous) convolution via a `step` parameter. Uses mirror reflection at boundaries. Used for wavelet decomposition and Gaussian blur.

**elementwise.wgsl** (workgroup 256) -- Complex multiplication: `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`. Used in FFT-based convolution (multiply spectra in frequency domain).

### Inline Shaders

These are small enough to embed as string constants in `wgpu_backend.rs`:

- **normalize** (workgroup 256) -- Scale every element by a constant factor
- **transpose_complex** (workgroup 16x16) -- Transpose a complex matrix using shared-memory tiling (16x17 tiles to avoid bank conflicts)
- **pad_real_to_complex** (workgroup 16x16) -- Zero-pad a real image to power-of-2 dimensions and convert to interleaved complex format
- **extract_real_scaled** (workgroup 16x16) -- Extract the real part from complex data, scale by `1/(h*w)`, crop to original dimensions
- **divide_real** (workgroup 256) -- Element-wise `a[i] / (b[i] + epsilon)`, used in Richardson-Lucy ratio step
- **multiply_real** (workgroup 256) -- Element-wise `a[i] * b[i]`, used in Richardson-Lucy multiplicative update

## GPU-Accelerated Alignment

Phase correlation alignment is the first major GPU-accelerated operation. The implementation lives in `crates/jupiter-core/src/align/phase_correlation.rs`.

### Single-Frame Offset Computation

`compute_offset_gpu()` computes the translation offset between a reference frame and a target frame:

```
Reference ──► upload ──► Hann window ──► FFT ──┐
                                                ├──► cross-power ──► IFFT ──► find_peak
Target    ──► upload ──► Hann window ──► FFT ──┘                       │
                                                                       ▼
                                                              download for
                                                          subpixel refinement
                                                               (CPU)
```

The flow:

1. **Upload** reference and target frames to GPU
2. **Apply Hann window** to both (reduces spectral leakage at edges)
3. **Forward 2D FFT** on both frames (with zero-padding to power-of-2)
4. **Cross-power spectrum** -- normalized product of one spectrum with the conjugate of the other
5. **Inverse FFT** to get the correlation surface
6. **Find peak** on GPU to get integer-pixel offset
7. **Download** the correlation surface for subpixel refinement via paraboloid fit on CPU

Wrap-around handling converts peak positions past the halfway point to negative offsets:

```rust
let dy = if peak_row > ph / 2 { peak_row as f64 - ph as f64 } else { peak_row as f64 };
let dx = if peak_col > pw / 2 { peak_col as f64 - pw as f64 } else { peak_col as f64 };
```

### Multi-Frame Alignment

`align_frames_gpu_with_progress()` aligns multiple frames concurrently. When there are 4 or more frames (the `PARALLEL_FRAME_THRESHOLD`), it uses Rayon's parallel iterator with the shared `Arc<dyn ComputeBackend>`:

```rust
frames.par_iter().enumerate().map(|(i, frame)| {
    let offset = compute_offset_gpu(&reference.data, &frame.data, backend.as_ref())?;
    let shifted_buf = backend.shift_bilinear(
        &backend.upload(&frame.data), offset.dx, offset.dy,
    );
    let shifted_data = backend.download(&shifted_buf);
    Ok(Frame::new(shifted_data, frame.original_bit_depth))
})
```

Each frame gets its offset computed and is shifted using the GPU's bilinear interpolation shader. Progress is reported via an `AtomicUsize` counter.

## GPU-Accelerated Deconvolution

Richardson-Lucy deconvolution is where GPU acceleration has the biggest impact. Each iteration requires 4 FFT/IFFT operations, and a typical run uses 20-50 iterations. That's 80-200 FFTs that stay entirely on the GPU.

### Richardson-Lucy on GPU

The implementation in `sharpen/deconvolution.rs` keeps almost everything on-device:

```
┌─────────────── ONE-TIME SETUP (3 CPU ops) ───────────────┐
│ 1. Generate PSF on CPU                                    │
│ 2. Upload PSF, compute FFT(PSF) — stays on GPU            │
│ 3. Flip PSF, upload, compute FFT(PSF_flipped) — on GPU    │
└──────────────────────────────────────────────────────────┘

┌─────────────── PER ITERATION (all GPU) ──────────────────┐
│ blurred     = IFFT( FFT(estimate) * H )                   │
│ ratio       = observed / (blurred + epsilon)               │
│ correction  = IFFT( FFT(ratio) * H_flipped )              │
│ estimate    = estimate * correction                        │
└──────────────────────────────────────────────────────────┘

┌─────────────── FINAL (1 CPU op) ─────────────────────────┐
│ Download result, clamp to [0.0, 1.0]                      │
└──────────────────────────────────────────────────────────┘
```

The PSF FFTs are computed once and reused across all iterations. The per-iteration loop is:

```rust
for _iter in 0..iterations {
    // Forward model: blurred = IFFT(FFT(estimate) * H)
    let est_fft = backend.fft2d(&estimate);
    let blurred_fft = backend.complex_mul(&est_fft, &h_fft);
    let blurred = backend.ifft2d_real(&blurred_fft, h, w);

    // Ratio = observed / (blurred + epsilon)
    let ratio = backend.divide_real(&observed, &blurred, 1e-10);

    // Correction = IFFT(FFT(ratio) * H_flipped)
    let ratio_fft = backend.fft2d(&ratio);
    let corr_fft = backend.complex_mul(&ratio_fft, &h_flip_fft);
    let correction = backend.ifft2d_real(&corr_fft, h, w);

    // Multiplicative update
    estimate = backend.multiply_real(&estimate, &correction);
}
```

Only the final result is downloaded and clamped -- no round-trips during iteration.

### Wiener Filter

The Wiener filter currently falls back to CPU. It requires complex conjugate and magnitude-squared operations (`conj(H)`, `|H|^2`) that aren't yet in the `ComputeBackend` trait. Adding these two operations would enable a fully GPU-accelerated Wiener filter.

## Pipeline Integration

The pipeline orchestrator in `pipeline/mod.rs` accepts the backend and dispatches to GPU or CPU paths based on `backend.is_gpu()`.

### Backend Injection

Both pipeline entry points accept the backend as `Arc<dyn ComputeBackend>`:

```rust
pub fn run_pipeline<F>(
    config: &PipelineConfig,
    backend: Arc<dyn ComputeBackend>,
    mut on_progress: F,
) -> Result<Frame>

pub fn run_pipeline_reported(
    config: &PipelineConfig,
    backend: Arc<dyn ComputeBackend>,
    reporter: Arc<dyn ProgressReporter>,
) -> Result<Frame>
```

### Conditional Dispatch

The pipeline checks `backend.is_gpu()` at two points:

**Alignment:**
```rust
if backend.is_gpu() {
    align_frames_gpu_with_progress(&selected_frames, 0, backend.clone(), progress_fn)?
} else {
    align_frames_with_progress(&selected_frames, 0, progress_fn)?
}
```

**Deconvolution:**
```rust
if backend.is_gpu() {
    sharpened = deconvolve_gpu(&sharpened, deconv_config, &*backend);
} else {
    sharpened = deconvolve(&sharpened, deconv_config);
}
```

### What Stays on CPU

Not every operation benefits from GPU acceleration. These remain CPU-only:

- **Wavelet sharpening** -- uses small (5-tap) B3 spline kernels; GPU dispatch overhead would dominate
- **Gaussian blur** -- separable 1D convolution with small kernels
- **Post-processing filters** -- histogram stretch, levels, unsharp mask
- **Stacking** -- mean/median/sigma-clip operate across frames, not spatially

## CLI Usage

The CLI exposes device selection via the `--device` flag:

```bash
# Default: try GPU, fall back to CPU
jupiter run input.ser

# Explicit GPU
jupiter run input.ser --device gpu

# Force CPU (useful for benchmarking or debugging)
jupiter run input.ser --device cpu

# Auto (same as default)
jupiter run input.ser --device auto
```

The selected backend name is printed in the pipeline summary:

```
Pipeline summary:
  Device: wgpu/Apple M1 Max
  Input:  saturn_2024.ser
  ...
```

Progress is reported through `MultiProgressReporter` using `indicatif::MultiProgress` with a stage bar and a detail bar.

## Feature Flags and Build

### Cargo.toml Configuration

In `crates/jupiter-core/Cargo.toml`:

```toml
[features]
default = []
gpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]
```

The `gpu` feature gates three dependencies:

| Crate | Purpose |
|-------|---------|
| `wgpu` | GPU compute API (Metal/Vulkan/DX12) |
| `pollster` | Blocks on wgpu async operations |
| `bytemuck` | Safe transmute for GPU buffer data |

### Conditional Compilation

The `#[cfg(feature = "gpu")]` attribute gates:

- The `wgpu_backend` module in `compute/mod.rs`
- The `Wgpu` variant of `BufferInner`
- GPU backend creation in `create_backend()`

When the `gpu` feature is disabled, the `Wgpu` buffer variant doesn't exist, the wgpu module isn't compiled, and all device preferences silently resolve to `CpuBackend`.

### Build Commands

```bash
# CPU-only (default)
cargo build --release

# With GPU acceleration
cargo build --release --features gpu

# Run tests (both modes)
cargo test
cargo test --features gpu
```

## Source File Reference

| File | Role |
|------|------|
| `crates/jupiter-core/src/compute/mod.rs` | `ComputeBackend` trait, `GpuBuffer`, `DevicePreference`, `create_backend()` |
| `crates/jupiter-core/src/compute/cpu.rs` | CPU reference backend (rustfft + ndarray + Rayon) |
| `crates/jupiter-core/src/compute/wgpu_backend.rs` | wgpu GPU backend, 15 pipelines, FFT, dispatch |
| `crates/jupiter-core/src/compute/shaders/*.wgsl` | 7 external WGSL compute shaders |
| `crates/jupiter-core/src/align/phase_correlation.rs` | `compute_offset_gpu()`, `align_frames_gpu_with_progress()` |
| `crates/jupiter-core/src/sharpen/deconvolution.rs` | `deconvolve_gpu()`, Richardson-Lucy on GPU |
| `crates/jupiter-core/src/pipeline/mod.rs` | Pipeline orchestration with `Arc<dyn ComputeBackend>` |
| `crates/jupiter-core/src/pipeline/config.rs` | `PipelineConfig` with `device: DevicePreference` |
| `crates/jupiter-cli/src/commands/pipeline.rs` | CLI `--device` flag handling |
| `crates/jupiter-core/Cargo.toml` | Feature flag definitions |
