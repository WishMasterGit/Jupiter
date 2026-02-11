# Phase 3: Performance — CPU Parallelism + GPU Acceleration

## Context

Every pixel-level operation in jupiter-core runs sequentially in nested `for row..for col` loops. The only parallelism is two `rayon::par_iter` calls over *frames* (not pixels) in quality scoring. FFT plans are recreated on every call. Stacking (median/sigma-clip) scales quadratically with frame count. On a typical 1000-frame, 1024x1024 pipeline, this is leaving 10-25x performance on the table.

This plan adds CPU parallelism (rayon for pixel-level work) and optional GPU acceleration (wgpu compute shaders behind a `gpu` feature flag) with automatic CPU fallback.

---

## Phase A: CPU-Only Quick Wins

### A1. Rayon pixel-level parallelism in stacking (HIGHEST IMPACT)

**Files:** `stack/median.rs`, `stack/sigma_clip.rs`

Each pixel's median/sigma-clip is independent — parallelize the outer row loop with `(0..h).into_par_iter()`. Each rayon thread gets its own scratch `Vec<f32>` for pixel values. Write results through `result.as_slice_mut()` indexed as `row * w + col`.

Pre-transpose frame data for cache locality before the stacking loop: store all frame values for each pixel contiguously in a single `Vec<f32>` of shape `[pixel_index * n_frames + frame_index]`. This turns N random reads across N separate `Array2` allocations into a sequential read of N consecutive f32s.

**Expected speedup:** 6-12x on stacking (4-8x from rayon + 1.5-2x from cache layout).

### A2. FFT Plan Caching

**File:** `align/phase_correlation.rs`

Create an `FftContext` struct owning a `FftPlanner<f64>`. Pass it into `fft2d`/`ifft2d`/`compute_offset` instead of recreating `FftPlanner::new()` on every call. The planner caches plans internally for repeated same-size transforms.

Update callers: `align_frames()`, `pipeline/mod.rs`, `filters/rgb_align.rs`.

**Expected speedup:** 10-20% on alignment phase.

### A3. Rayon parallel alignment loop

**File:** `pipeline/mod.rs` (lines 93-110)

Each frame's alignment (compute_offset + shift_frame) is independent of the others. Use `par_iter` over frames with thread-local `FftContext` via `rayon::iter::ThreadLocal` or `std::thread_local!`.

**Complication:** The `on_progress` closure is `FnMut`, incompatible with parallel access. Change to use `AtomicUsize` counter for progress, and accept `Fn(PipelineStage, f32) + Send + Sync` instead.

**Expected speedup:** 2-4x on alignment phase.

### A4. Rayon in convolution and per-pixel ops

**Files:** `sharpen/wavelet.rs` (convolve_rows/convolve_cols), `filters/gaussian_blur.rs` (convolve_rows/convolve_cols), `align/phase_correlation.rs` (shift_frame, apply_hann)

Same pattern everywhere: parallelize outer row loop. For quality metrics (`laplacian.rs`, `gradient.rs`), the frame-level `par_iter` already exists — only add pixel-level parallelism when frame count < CPU count to avoid oversubscription.

**Expected speedup:** 2-4x on sharpening/filtering stages.

### A5. Avoid unnecessary frame cloning in pipeline

**File:** `pipeline/mod.rs`

Currently `selected_frames` clones each selected `Frame` (including the `Array2<f32>` data). For 250 frames at 1024x1024, that is ~1GB of unnecessary allocation. Instead, consume the original `frames` vec and move selected frames out, or pass indices and reference the original.

**Expected speedup:** 5-15% overall from reduced allocation.

---

## Phase B: GPU Infrastructure (behind `gpu` feature flag)

### B1. Add wgpu as optional dependency

**Files:** `Cargo.toml` (workspace), `jupiter-core/Cargo.toml`

```toml
[features]
default = []
gpu = ["dep:wgpu", "dep:bytemuck", "dep:pollster"]
```

Add `wgpu`, `bytemuck` (zero-copy GPU buffer casting), `pollster` (lightweight async block_on).

### B2. GPU context + buffer management

**New files:** `gpu/mod.rs`, `gpu/context.rs`, `gpu/buffer.rs`

`GpuContext` struct owns `wgpu::Device`, `Queue`, `Adapter`. Created once per pipeline run (~100-200ms init). `GpuImage` struct wraps a `wgpu::Buffer` with width/height metadata. Helpers for upload (`Array2<f32>` → GPU buffer) and download (GPU → `Array2<f32>`).

### B3. Compute backend enum

**New file:** `gpu/backend.rs`

```rust
pub enum ComputeBackend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(GpuContext),
}
```

Each algorithm function dispatches on the backend. The pipeline creates the backend once and passes it through. `PipelineConfig` gains `use_gpu: bool` field. CLI gains `--gpu` flag.

Graceful fallback: if GPU init fails, warn and fall back to CPU.

---

## Phase C: GPU Compute Shaders (WGSL)

### C1. GPU Stacking (highest GPU impact)

**New files:** `gpu/shaders/mean_stack.wgsl`, `gpu/shaders/median_stack.wgsl`, `gpu/stacking.rs`

Upload all frames as a single contiguous buffer. Dispatch one thread per pixel.
- **Mean:** Sum N values, divide. Trivial.
- **Median:** Fixed-size local array (max 64 frames), insertion sort, pick middle. WGSL doesn't support runtime-sized local arrays, but 64 is plenty for planetary imaging.
- **Sigma-clip:** Keep CPU-only initially (iterative nature complicates GPU).

**Expected speedup:** 10-50x over sequential CPU, 2-5x over rayon CPU.

### C2. GPU Separable Convolution

**New files:** `gpu/shaders/convolve_row.wgsl`, `gpu/shaders/convolve_col.wgsl`, `gpu/convolution.rs`

Single shader pair handles Gaussian blur, unsharp mask, and wavelet a-trous (via `step` dilation parameter). The full wavelet decompose→reconstruct stays on GPU: upload once, run 12 convolution dispatches + detail subtraction + reconstruction, download once.

**Expected speedup:** 10-30x on sharpening stage; main value is avoiding 12 CPU↔GPU round-trips by keeping data on GPU.

### C3. GPU Frame Shifting (bilinear interpolation)

**New files:** `gpu/shaders/shift_frame.wgsl`, `gpu/alignment.rs`

Each pixel computes its bilinear-interpolated source independently. Keep FFT on CPU (with plan caching from A2) — hybrid approach: CPU FFT to find offsets, GPU shift to apply them. One dispatch per frame.

**Expected speedup:** 5-10x per frame shift.

### C4. GPU Element-wise Filters

**New files:** `gpu/shaders/elementwise.wgsl`, `gpu/filters.rs`

Gamma, brightness/contrast, histogram stretch — trivial shaders. Main value: keeps data on GPU between stages (stacking → sharpening → filters → output) to avoid transfer overhead.

---

## Implementation Order

```
Phase A (CPU-only):
  A1 + A5 (stacking rayon + data layout)  ─ Highest impact, do first
  A2 (FFT plan caching)                   ─ Quick, standalone
  A3 (parallel alignment)                 ─ Depends on A2
  A4 (pixel-level rayon in convolution)   ─ Standalone

Phase B (GPU infra):
  B1 → B2 → B3                            ─ Sequential setup

Phase C (GPU kernels):
  C1 (GPU stacking)                        ─ First GPU kernel, prove the pipeline
  C2 (GPU convolution)                     ─ Second, wavelet stays on GPU
  C3 (GPU shifting)                        ─ Third
  C4 (GPU element-wise)                    ─ Last, lowest impact
```

## Key Files to Modify

| File | Changes |
|------|---------|
| `stack/median.rs` | A1: rayon row parallelism + data pre-transpose |
| `stack/sigma_clip.rs` | A1: rayon row parallelism + data pre-transpose |
| `align/phase_correlation.rs` | A2: FftContext struct, A4: rayon in shift_frame/apply_hann |
| `sharpen/wavelet.rs` | A4: rayon in convolve_rows/convolve_cols |
| `filters/gaussian_blur.rs` | A4: rayon in convolve_rows/convolve_cols |
| `pipeline/mod.rs` | A3: parallel alignment, A5: avoid clones, B3: backend dispatch |
| `pipeline/config.rs` | B3: `use_gpu` field |
| `jupiter-core/Cargo.toml` | B1: wgpu/bytemuck/pollster optional deps |
| CLI `main.rs`, `pipeline.rs` | B3: `--gpu` flag |

## New Files to Create

| File | Purpose |
|------|---------|
| `gpu/mod.rs` | Module root (cfg-gated) |
| `gpu/context.rs` | GpuContext (device/queue/adapter) |
| `gpu/buffer.rs` | GpuImage upload/download helpers |
| `gpu/backend.rs` | ComputeBackend enum |
| `gpu/stacking.rs` | GPU stacking dispatch |
| `gpu/convolution.rs` | GPU convolution dispatch |
| `gpu/alignment.rs` | GPU frame shifting dispatch |
| `gpu/filters.rs` | GPU element-wise ops |
| `gpu/shaders/*.wgsl` | WGSL compute shaders |

## Verification

1. `cargo build` — compiles without `gpu` feature (CPU-only path unaffected)
2. `cargo build --features gpu` — compiles with GPU support
3. Run pipeline on synthetic SER with and without `--gpu`, verify identical results (within floating-point tolerance)
4. Benchmark each phase: time the full pipeline before/after each optimization step
5. Test GPU fallback: request GPU on a machine without one, verify graceful fallback message + CPU execution
