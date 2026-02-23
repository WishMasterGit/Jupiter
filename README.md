# Jupiter

**Planetary image processing for lucky imaging pipelines**

[![CI](https://github.com/WishMasterGit/Jupiter/actions/workflows/ci.yml/badge.svg)](https://github.com/WishMasterGit/Jupiter/actions)
[![Latest Release](https://img.shields.io/github/v/release/WishMasterGit/Jupiter)](https://github.com/WishMasterGit/Jupiter/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

Jupiter turns raw planetary video captures (SER files) into sharp, detail-rich images using the lucky imaging technique:

1. **Score** every frame by sharpness
2. **Select** the best percentage of frames
3. **Align** frames to a common reference
4. **Stack** selected frames to reduce noise
5. **Sharpen** with wavelet decomposition and optional deconvolution
6. **Filter** with histogram stretch, gamma, etc.
7. **Save** to TIFF or PNG

Both a headless CLI (`jupiter`) and an interactive GUI (`jupiter-gui`) are provided.

---

## Features

- **5 alignment methods**: Phase Correlation, Enhanced Phase (sub-pixel), Centroid, Gradient Correlation, Gaussian Pyramid
- **6 stacking methods**: Mean, Median, Sigma Clip, Multi-Point (AutoStakkert-style), Drizzle super-resolution, Surface Warp
- **Wavelet sharpening**: A trous B3-spline decomposition with per-layer coefficients and denoise thresholds
- **Deconvolution**: Richardson-Lucy and Wiener filter with Gaussian, Kolmogorov, and Airy PSF models
- **GPU acceleration**: Metal (macOS), Vulkan (Linux), DX12 (Windows) via wgpu — enabled with `--features gpu`
- **Low-memory streaming**: Process giant SER files without loading everything into RAM
- **Debayering**: Bilinear and Malvar-He-Cutler (MHC) demosaicing for Bayer-pattern cameras
- **Planet auto-crop**: Detect the planet and trim all frames to a tight bounding box
- **TOML config files**: Save and load full pipeline configurations

---

## Installation

### Prerequisites

- **Rust** (stable toolchain) — install from [rustup.rs](https://rustup.rs)
- **macOS / Windows**: no extra system libraries needed
- **Linux**: install X11/Wayland headers:

```bash
sudo apt install \
  libwayland-dev libx11-dev libxcb1-dev \
  libxrandr-dev libxinerama-dev libxi-dev \
  pkg-config libxkbcommon-dev
```

### Build (CPU only)

```bash
cargo build --release --workspace
```

### Build with GPU support

```bash
cargo build --release --workspace --features gpu
```

Binaries land in `target/release/`:
- `target/release/jupiter` — headless CLI
- `target/release/jupiter-gui` — desktop GUI

---

## Quick Start

```bash
# Run the full pipeline on a SER file
jupiter run input.ser -o result.tiff

# Launch the interactive GUI
jupiter-gui
```

---

## CLI Reference

All subcommands accept `--verbose` / `-v` for debug logging.

### `jupiter info`

Print SER file metadata.

```
jupiter info <file>
```

Displays: frame count, dimensions, bit depth, color mode, observer, telescope, instrument, total data size.

---

### `jupiter quality`

Score and rank frames by sharpness.

```
jupiter quality <file> [OPTIONS]

Options:
  --top <N>         Show top N frames [default: 20]
  --metric <m>      Quality metric: laplacian | gradient [default: laplacian]
```

---

### `jupiter stack`

Align and stack the best frames (standalone — no sharpening).

```
jupiter stack <file> [OPTIONS]

Options:
  --select <pct>        Percentage of best frames to keep [default: 25]
  --method <m>          mean | median | sigma-clip | multi-point | drizzle | surface-warp
                        [default: mean]
  --sigma <v>           Sigma threshold for sigma-clip [default: 2.5]
  --ap-size <px>        Alignment point size in pixels [default: 64]
  --search-radius <px>  Local search radius per AP [default: 16]
  --min-brightness <v>  Min mean brightness to place an AP [default: 0.05]
  --drizzle-scale <v>   Drizzle output scale factor [default: 2.0]
  --pixfrac <v>         Drizzle drop size 0.0–1.0 [default: 0.7]
  -o, --output <file>   Output file [default: stacked.tiff]
```

---

### `jupiter sharpen`

Apply wavelet sharpening (and optional deconvolution) to an existing image.

```
jupiter sharpen <file> [OPTIONS]

Options:
  --layers <n>           Number of wavelet layers [default: 6]
  --coefficients <list>  Comma-separated per-layer boost factors
                         (e.g. 1.5,1.3,1.2,1.1,1.0,1.0)
  --denoise <list>       Comma-separated per-layer denoise thresholds
                         (e.g. 3.0,2.0,1.0,0,0,0)
  --deconv <method>      Deconvolution: rl | wiener
  --psf <model>          PSF model: gaussian | kolmogorov | airy [default: gaussian]
  --psf-sigma <v>        Gaussian PSF sigma in pixels [default: 2.0]
  --seeing <v>           Kolmogorov seeing FWHM in pixels [default: 3.0]
  --airy-radius <v>      Airy first dark ring radius in pixels [default: 2.5]
  --rl-iterations <n>    Richardson-Lucy iterations [default: 20]
  --noise-ratio <v>      Wiener noise-to-signal ratio [default: 0.001]
  -o, --output <file>    Output file [default: sharpened.tiff]
```

---

### `jupiter filter`

Apply post-processing filters to an existing image.

```
jupiter filter <file> [OPTIONS]

Options:
  --stretch <spec>      Histogram stretch: "auto" or "black,white" (e.g. "0.01,0.99")
  --gamma <v>           Gamma correction (e.g. 1.2)
  --brightness <v>      Brightness adjustment -1.0 to 1.0
  --contrast <v>        Contrast adjustment (1.0 = unchanged)
  --unsharp-mask <spec> "radius,amount,threshold" (e.g. "2.0,0.5,0.01")
  --blur <sigma>        Gaussian blur sigma
  -o, --output <file>   Output file [default: filtered.tiff]
```

---

### `jupiter run`

Run the complete pipeline in one shot.

```
jupiter run <file> [OPTIONS]
jupiter run --config pipeline.toml [<file>] [-o <output>]

Input/Output:
  <file>                Input SER file
  --config <toml>       Load settings from a TOML config file (CLI flags override it)
  -o, --output <file>   Output file [default: result.tiff]
  --save-config <file>  Save effective config as TOML and exit without processing

Device & Memory:
  --device <d>          auto | cpu | gpu | cuda [default: auto]
  --memory <m>          auto | eager | low-memory [default: low-memory]

Color:
  --debayer <method>    bilinear | mhc  (force debayering of Bayer SER files)
  --mono                Force mono processing even for Bayer/RGB files

Frame Selection:
  --select <pct>        Percentage of best frames to keep [default: 25]

Alignment:
  --align-method <m>    phase | enhanced-phase | centroid | gradient | pyramid
                        [default: phase]
  --upsample-factor <n> Upsampling factor for enhanced-phase [default: 20]
  --centroid-threshold <v>  Intensity threshold for centroid [default: 0.1]
  --pyramid-levels <n>  Pyramid levels for coarse-to-fine [default: 3]

Stacking:
  --method <m>          mean | median | sigma-clip | multi-point | drizzle | surface-warp
                        [default: multi-point]
  --sigma <v>           Sigma threshold for sigma-clip [default: 2.5]
  --ap-size <px>        Alignment point size in pixels [default: 64]
  --search-radius <px>  Local search radius per AP [default: 16]
  --min-brightness <v>  Min mean brightness to place an AP [default: 0.05]
  --drizzle-scale <v>   Drizzle output scale factor [default: 2.0]
  --pixfrac <v>         Drizzle drop size 0.0–1.0 [default: 0.7]

Sharpening:
  --sharpen <list>      Comma-separated wavelet boost coefficients per layer
  --denoise <list>      Comma-separated wavelet denoise thresholds per layer
  --no-sharpen          Disable sharpening entirely
  --deconv <method>     rl | wiener
  --psf <model>         gaussian | kolmogorov | airy [default: gaussian]
  --psf-sigma <v>       Gaussian PSF sigma in pixels [default: 2.0]
  --seeing <v>          Kolmogorov seeing FWHM in pixels [default: 3.0]
  --airy-radius <v>     Airy first dark ring radius in pixels [default: 2.5]
  --rl-iterations <n>   Richardson-Lucy iterations [default: 20]
  --noise-ratio <v>     Wiener noise-to-signal ratio [default: 0.001]

Post-processing Filters:
  --auto-stretch        Auto histogram stretch after stacking
  --gamma <v>           Gamma correction after stacking
```

---

### `jupiter config`

Print (or save) a default pipeline config as TOML.

```
jupiter config [-o config.toml]
```

Use this to generate a starting point you can then edit and pass to `jupiter run --config`.

---

### `jupiter auto-crop`

Detect the planet and crop every frame in a SER file to a tight bounding box.

```
jupiter auto-crop <file> [OPTIONS]

Options:
  -o, --output <file>   Output SER file (auto-named if omitted)
  --padding <frac>      Padding around planet as fraction of diameter [default: 0.15]
  --samples <n>         Frames to sample for detection [default: 30]
  --threshold <m>       auto | otsu [default: auto]
  --sigma <v>           Sigma multiplier for auto threshold [default: 2.0]
  --fixed-threshold <v> Fixed threshold in [0.0, 1.0] (overrides --threshold)
  --blur-sigma <v>      Pre-detection blur sigma [default: 2.5]
  --min-area <px>       Minimum component area for planet detection [default: 100]
```

---

## Pipeline Config (TOML)

Generate a default config with:

```bash
jupiter config -o config.toml
```

Then run with:

```bash
jupiter run input.ser --config config.toml -o result.tiff
```

Example annotated `config.toml`:

```toml
input  = "input.ser"
output = "result.tiff"

# Compute device: "Auto" | "Cpu" | "Gpu" | "Cuda"
device = "Auto"

# Memory strategy: "Auto" | "Eager" | "LowMemory"
memory = "Auto"

# Debayering — omit to auto-detect from SER header
# [debayer]
# method = "MalvarHeCutler"   # or "Bilinear"

# Force mono even for Bayer/RGB sources
force_mono = false

[frame_selection]
select_percentage = 0.25        # Keep best 25% of frames
metric = "Laplacian"            # "Laplacian" | "Gradient"

[alignment]
# method = "PhaseCorrelation"   # default
# method = { EnhancedPhaseCorrelation = { upsample_factor = 20 } }
# method = { Centroid = { threshold = 0.1 } }
# method = { Pyramid = { levels = 3 } }
# method = "GradientCorrelation"

[stacking]
# method = "Mean"
# method = "Median"
# method = { SigmaClip = { sigma = 2.5, iterations = 5 } }
method = { MultiPoint = { ap_size = 64, search_radius = 16, min_brightness = 0.05, select_percentage = 0.25 } }
# method = { Drizzle = { scale = 2.0, pixfrac = 0.7, quality_weighted = true } }
# method = { SurfaceWarp = { ap_size = 64, search_radius = 16, min_brightness = 0.05, select_percentage = 0.25 } }

[sharpening]
[sharpening.wavelet]
num_layers   = 6
coefficients = [1.5, 1.3, 1.2, 1.1, 1.0, 1.0]
denoise      = []

# Optional deconvolution (applied before wavelet sharpening)
# [sharpening.deconvolution]
# method = { RichardsonLucy = { iterations = 20 } }
# psf    = { Gaussian = { sigma = 2.0 } }
# psf    = { Kolmogorov = { seeing = 3.0 } }
# psf    = { Airy = { radius = 2.5 } }

# Post-processing filter chain (applied in order)
# [[filters]]
# AutoStretch = { low_percentile = 0.001, high_percentile = 0.999 }

# [[filters]]
# Gamma = 1.1

# [[filters]]
# HistogramStretch = { black_point = 0.01, white_point = 0.99 }
```

---

## Stacking Methods

| Method | When to use |
|---|---|
| **Mean** | Many high-quality frames, minimal noise |
| **Median** | Robust against hot pixels and cosmic rays |
| **Sigma Clip** | Like median but preserves more detail; good default for planetary |
| **Multi-Point** | Best general-purpose choice; handles atmospheric distortion across the disk |
| **Drizzle** | Super-resolve fine detail; needs many frames (50+) at sub-Nyquist sampling |
| **Surface Warp** | Smooth per-pixel warping for severe atmospheric distortion |

**Multi-Point** is the default for `jupiter run` because it corrects local atmospheric distortions that global alignment cannot handle.

**Drizzle** produces output at `--drizzle-scale` × the input resolution. A scale of `2.0` doubles linear resolution. Use `--pixfrac 0.5`–`0.7` for best sharpness.

---

## Alignment Methods

| Method | Accuracy | Speed | Best for |
|---|---|---|---|
| **Phase Correlation** | ~0.5 px | Fast | Default — works well for most cases |
| **Enhanced Phase** | ~0.01 px | Medium | Maximum sub-pixel precision |
| **Centroid** | ~1–2 px | Very fast | Bright planetary disk, simple scenes |
| **Gradient Correlation** | ~0.5 px | Medium | Noisy or low-contrast frames |
| **Pyramid** | ~0.5 px | Slow | Large displacements, wide-field |

Multi-point local alignment always uses Phase Correlation internally, regardless of the global alignment setting.

---

## Sharpening

Sharpening runs in two stages:

1. **Deconvolution** (optional, first): reverses blur caused by the atmosphere or optics.
   - *Richardson-Lucy* — iterative, non-linear, good for Poisson noise. Use 10–30 iterations.
   - *Wiener* — linear, faster. `--noise-ratio` controls the regularization strength.

2. **Wavelet sharpening** (always, unless `--no-sharpen`): amplifies fine-detail wavelet layers.
   - Default: 6 layers with coefficients `[1.5, 1.3, 1.2, 1.1, 1.0, 1.0]`
   - Tune with `--sharpen 2.0,1.8,1.5,1.2,1.0,1.0`
   - Add per-layer noise rejection: `--denoise 4.0,3.0,2.0,0,0,0`

Example — aggressive sharpening with RL deconvolution:

```bash
jupiter run input.ser \
  --deconv rl --rl-iterations 25 --psf gaussian --psf-sigma 1.5 \
  --sharpen 2.0,1.8,1.5,1.2,1.0,1.0 \
  --denoise 3.0,2.0,1.0,0,0,0 \
  -o result.tiff
```

---

## GPU Acceleration

Build with the `gpu` feature to unlock GPU-accelerated alignment and deconvolution:

```bash
cargo build --release --workspace --features gpu
```

Then select the GPU at runtime:

```bash
jupiter run input.ser --device gpu -o result.tiff
```

| Stage | GPU accelerated |
|---|---|
| Alignment (phase correlation) | Yes |
| Richardson-Lucy deconvolution | Yes |
| Wavelet sharpening | No (CPU, small kernels) |
| Stacking | No |

`--device auto` (the default) picks the GPU when available, falling back to CPU.

---

## Memory Modes

| Mode | Behaviour | Use when |
|---|---|---|
| `auto` | Streams when decoded data > 1 GiB, otherwise eager | Safe default |
| `eager` | Load all frames at once | Fastest; fine for small SER files |
| `low-memory` | Stream frames on demand, O(1) peak memory | Large SER files on memory-limited systems |

The CLI default for `jupiter run` is `low-memory`, which re-reads frames from disk as needed.

Mean and Drizzle stacking are fully streaming (~192 MB peak for a 94-frame 4096×4096 sequence). Median and Sigma Clip are semi-streaming: offsets are computed on-the-fly but the selected frames are held in RAM for stacking.

---

## GUI Guide

Launch with:

```bash
jupiter-gui
```

### Menu Bar

- **File → Open SER** (`Cmd/Ctrl+O`): load a SER file
- **File → Save Config** (`Cmd/Ctrl+S`): export current settings as TOML
- **File → Open Config**: import a TOML config
- **File → Quit** (`Cmd/Ctrl+Q`)

### Left Panel — Controls

The controls panel is divided into pipeline stages. Each stage has a **Run** button that re-runs only that stage and everything downstream.

**Score**
- Quality metric (Laplacian / Gradient)
- Frame selection percentage
- Alignment method and method-specific parameters

**Stack**
- Stacking method selector
- Method-specific parameters (AP size, search radius, Drizzle scale/pixfrac, etc.)

**Sharpen**
- Wavelet: number of layers, per-layer coefficients, denoise thresholds
- Deconvolution: method (RL / Wiener), PSF model and parameters

**Filters**
- Add / remove / reorder filter steps
- Supported filters: Auto Stretch, Histogram Stretch, Gamma, Brightness/Contrast, Unsharp Mask, Gaussian Blur

**Run All** button at the bottom executes the complete pipeline.

### Viewport

- **Scroll wheel**: zoom in/out
- **Middle drag** or **Ctrl+drag**: pan
- **Double-click**: fit image to viewport

Stage buttons use color indicators: green = result is current, orange = parameters have changed and re-running is recommended.

---

## File Formats

### Input

| Format | Notes |
|---|---|
| **SER** | Primary input format. Supports mono, Bayer (RGGB/BGGR/GRBG/GBRG), and RGB color modes. |
| **TIFF** | Accepted by `sharpen` and `filter` subcommands |
| **PNG** | Accepted by `sharpen` and `filter` subcommands |

### Output

| Format | Notes |
|---|---|
| **TIFF** | Default output, 32-bit float (lossless) |
| **PNG** | 16-bit, selected by file extension |

Output format is inferred from the output file extension.
