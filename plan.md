# Jupiter: Planetary Image Processing Library + CLI

## Context

Earth-based planetary imaging suffers from atmospheric turbulence ("seeing") that limits resolution far below telescope diffraction limits. The standard workflow captures thousands of short-exposure video frames, selects the sharpest ones ("lucky imaging"), aligns and stacks them to boost signal-to-noise ratio, then applies wavelet sharpening and deconvolution to reveal fine detail.

**Existing tools** (AutoStakkert, RegiStax, AstroSurface) are mostly Windows-only, closed-source, or abandonware. Open-source alternatives (Siril, PlanetarySystemStacker) exist but are C or Python. **No Rust-based tool exists** — there's an opportunity for a fast, cross-platform, modern implementation.

**Goal**: Build a Rust library (`jupiter-core`) with a thin CLI (`jupiter-cli`) to test each processing stage independently and run the full pipeline end-to-end.

---

## Market Overview

| Tool | Lang | Open Source | Platform | Strengths |
|------|------|------------|----------|-----------|
| AutoStakkert! | ? | No (freeware) | Windows | Best alignment/stacking |
| RegiStax | ? | No (free) | Windows | Best wavelet sharpening |
| Siril | C | Yes (GPL) | All | Complete pipeline, active dev |
| PlanetarySystemStacker | Python | Yes | All | Comparable to AutoStakkert |
| ImPPG | C++ | Yes (GPL) | Windows | GPU deconvolution |
| AstroSurface | ? | No (free) | Windows | All-in-one |

---

## Project Structure

```
jupiter/
├── Cargo.toml                  # workspace root
├── .gitignore
├── crates/
│   ├── jupiter-core/           # library crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── error.rs        # JupiterError, Result<T>
│   │       ├── frame.rs        # Frame, QualityScore, AlignmentOffset, ColorMode
│   │       ├── io/
│   │       │   ├── mod.rs
│   │       │   ├── ser.rs      # SER video format reader (mmap-based)
│   │       │   └── image_io.rs # PNG/TIFF output via `image` crate
│   │       ├── quality/
│   │       │   ├── mod.rs
│   │       │   └── laplacian.rs  # Laplacian variance metric
│   │       ├── align/
│   │       │   ├── mod.rs
│   │       │   ├── phase_correlation.rs  # FFT-based shift detection
│   │       │   └── subpixel.rs           # Parabola fitting refinement
│   │       ├── stack/
│   │       │   ├── mod.rs
│   │       │   └── mean.rs     # Mean stacking
│   │       ├── sharpen/
│   │       │   ├── mod.rs
│   │       │   └── wavelet.rs  # A trous wavelet decomposition
│   │       └── pipeline/
│   │           ├── mod.rs      # Pipeline executor
│   │           └── config.rs   # Serializable pipeline config
│   └── jupiter-cli/            # binary crate
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           └── commands/
│               ├── mod.rs
│               ├── info.rs     # Show SER/image metadata
│               ├── quality.rs  # Score & rank frames
│               ├── stack.rs    # Align + stack
│               ├── sharpen.rs  # Wavelet sharpen
│               └── pipeline.rs # Full pipeline
```

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `ndarray` | Core 2D array type for pixel data |
| `rayon` | Parallel frame scoring/alignment |
| `rustfft` + `num-complex` | FFT for phase correlation alignment |
| `image` | PNG/TIFF I/O |
| `byteorder` + `memmap2` | SER file parsing with zero-copy mmap |
| `thiserror` / `anyhow` | Error handling (lib / cli) |
| `clap` (derive) | CLI argument parsing |
| `indicatif` | Progress bars |
| `tracing` | Structured logging |
| `serde` + `toml` | Pipeline config serialization |

## Core Design Decisions

- **Canonical pixel type**: `ndarray::Array2<f32>` normalized to `[0.0, 1.0]`. Conversion from u8/u16 happens once at read time, back at write time. Avoids type-parameterizing every algorithm.
- **Mono-first**: Process grayscale frames. Color (RGB) = three separate mono channels processed independently, merged at output. Keeps algorithms simple.
- **Memory-mapped SER reading**: SER files can be gigabytes. Use `memmap2` for zero-copy frame access without loading everything into RAM.

## Algorithm Details

### 1. SER File Reader (`io/ser.rs`)
- Parse 178-byte header: magic "LUCAM-RECORDER", dimensions, bit depth, color mode, frame count
- Memory-map the file, compute frame offsets as `178 + index * frame_byte_size`
- Convert raw pixels (u8 or u16 LE/BE) to `Array2<f32>` in `[0.0, 1.0]`
- Parse optional timestamp trailer (8 bytes per frame after all frame data)

### 2. Frame Quality Assessment (`quality/laplacian.rs`)
- Convolve frame with 3x3 Laplacian kernel `[[0,1,0],[1,-4,1],[0,1,0]]`
- Compute variance of result — higher variance = sharper frame
- Use `rayon::par_iter` to score all frames concurrently
- Return sorted ranking with indices

### 3. Phase Correlation Alignment (`align/phase_correlation.rs`)
- Apply Hann window to both reference and target frames (reduces edge spectral leakage)
- 2D FFT via row-wise then column-wise 1D `rustfft` calls
- Normalized cross-power spectrum: `(F_ref * conj(F_target)) / |F_ref * conj(F_target)|`
- Inverse 2D FFT, find peak = integer-pixel shift
- Subpixel: fit paraboloid to 3x3 neighborhood around peak

### 4. Mean Stacking (`stack/mean.rs`)
- Shift each aligned frame by its offset (bilinear interpolation for subpixel shifts)
- Accumulate pixel-wise sum, divide by count
- SNR improves by `sqrt(N)`

### 5. Wavelet Sharpening (`sharpen/wavelet.rs`)
- **A trous** wavelet decomposition with B3 spline kernel `[1, 4, 6, 4, 1] / 16`
- 2D convolution applied separably (rows then columns)
- At scale `j`, kernel is dilated: read input pixels at intervals of `2^j`
- Detail layer `w_j = c_{j-1} - c_j` (difference between successive smoothings)
- Reconstruction: `result = k_1*w_1 + k_2*w_2 + ... + k_n*w_n + residual`
- User controls `k_i` per layer (>1.0 sharpens, <1.0 suppresses)
- Default 6 layers with coefficients `[1.5, 1.3, 1.2, 1.1, 1.0, 1.0]`

## CLI Commands

```
jupiter info <FILE>              # Show SER/image metadata
jupiter quality <FILE>           # Score and rank all frames
jupiter stack <FILE>             # Select best frames, align, stack
  --select <N%>                  #   top N% of frames (default: 25)
  --method mean                  #   stacking method
  -o result.tiff                 #   output file
jupiter sharpen <IMAGE>          # Wavelet sharpen a stacked image
  --layers 6                     #   number of wavelet layers
  --coefficients 1.5,1.3,1.2,1.1,1.0,1.0
  -o sharpened.tiff
jupiter run <FILE>               # Full pipeline
  [--config pipeline.toml]       #   or inline params
  -o final.tiff
```

## Implementation Order

Files created in dependency order, each compiling and passing tests before moving on:

1. **Workspace scaffolding** — `Cargo.toml` (root), `.gitignore`, both crate `Cargo.toml`s, empty `lib.rs`/`main.rs`
2. **`error.rs`** — `JupiterError` enum, `Result<T>` alias
3. **`frame.rs`** — `Frame`, `QualityScore`, `AlignmentOffset`, `ColorMode`, `SourceInfo`
4. **`io/ser.rs`** — SER header parsing, mmap, frame reading + unit tests with synthetic SER data
5. **`io/image_io.rs`** — Save `Frame` as 16-bit TIFF / 8-bit PNG
6. **`quality/laplacian.rs`** — Laplacian variance metric + parallel scoring
7. **`align/phase_correlation.rs`** + **`align/subpixel.rs`** — FFT alignment with subpixel refinement
8. **`stack/mean.rs`** — Mean stacking with subpixel shift interpolation
9. **`sharpen/wavelet.rs`** — A trous decomposition, per-layer coefficients, reconstruction
10. **`pipeline/`** — Wire stages together, `PipelineConfig` serialization
11. **CLI commands** — `info`, `quality`, `stack`, `sharpen`, `run`
12. **Integration tests** — End-to-end with synthetic SER file

## Future Phases (not in this plan)

- **Phase 2**: Median/sigma-clip stacking, Tenengrad metric, FITS I/O, debayering, Richardson-Lucy deconvolution, `extract` command
- **Phase 3**: Wiener deconvolution, AVI input, GPU acceleration (wgpu), multi-point alignment, drizzle super-resolution, Python bindings (PyO3)

## License

MIT

## Verification

1. `cargo build` — workspace compiles with no errors
2. `cargo test` — all unit tests pass
3. `cargo run --bin jupiter -- info <real-file.ser>` — displays correct metadata
4. `cargo run --bin jupiter -- quality <real-file.ser>` — scores and ranks frames
5. `cargo run --bin jupiter -- run <real-file.ser> -o result.tiff` — produces a sharpened output
6. Visual comparison against the same SER processed in AutoStakkert/RegiStax
