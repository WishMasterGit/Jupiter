# Jupiter CLI Reference

Jupiter is a planetary image processing tool for producing sharp results from earth-based telescope captures. It implements a lucky imaging pipeline: read SER video, score frame quality, select the best frames, align, stack, sharpen, and apply filters.

## Global Flags

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Enable verbose output (debug-level logging) |
| `--version` | | Print version |
| `--help` | `-h` | Print help |

---

## Commands

### `jupiter info`

Display metadata for a SER video file.

```
jupiter info <FILE>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE` | Yes | Path to a SER file |

**Output fields:** filename, frame count, dimensions, bit depth, color mode, observer, telescope, instrument, data size.

**Example:**

```bash
jupiter info captures/jupiter_2024.ser
```

---

### `jupiter quality`

Score and rank all frames by sharpness. Use this to preview frame quality before stacking.

```
jupiter quality <FILE> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE` | Yes | Path to a SER file |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--top <N>` | `20` | Show top N frames only |
| `--metric <METRIC>` | `laplacian` | Quality metric: `laplacian` or `gradient` |

**Metrics:**
- **laplacian** - Laplacian variance. Good general-purpose sharpness measure.
- **gradient** - Sobel gradient magnitude. Better for low-contrast targets.

**Examples:**

```bash
# Show top 20 frames ranked by Laplacian variance
jupiter quality jupiter.ser

# Show top 50 frames using gradient metric
jupiter quality jupiter.ser --top 50 --metric gradient
```

---

### `jupiter stack`

Align and stack the best frames from a SER video. This is the core stacking step without sharpening or filters.

```
jupiter stack <FILE> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE` | Yes | Path to a SER file |

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--select <N>` | | `25` | Percentage of best frames to keep (1-100) |
| `--method <METHOD>` | | `mean` | Stacking method (see below) |
| `--output <PATH>` | `-o` | `stacked.tiff` | Output file path |

**Stacking methods:** `mean`, `median`, `sigma-clip`, `multi-point`, `drizzle`

**Method-specific options:**

| Option | Default | Applies to | Description |
|--------|---------|------------|-------------|
| `--sigma <F>` | `2.5` | `sigma-clip` | Sigma rejection threshold |
| `--ap-size <N>` | `64` | `multi-point` | Alignment point size in pixels |
| `--search-radius <N>` | `16` | `multi-point` | Search radius around each AP |
| `--min-brightness <F>` | `0.05` | `multi-point` | Minimum mean brightness to place an AP |
| `--drizzle-scale <F>` | `2.0` | `drizzle` | Output scale factor (e.g. 2.0 = 2x resolution) |
| `--pixfrac <F>` | `0.7` | `drizzle` | Pixel drop fraction (0.0-1.0) |

**Examples:**

```bash
# Basic mean stack of top 25% frames
jupiter stack jupiter.ser

# Median stack of top 10%
jupiter stack jupiter.ser --select 10 --method median -o jupiter_stacked.tiff

# Sigma-clip with custom threshold
jupiter stack jupiter.ser --method sigma-clip --sigma 2.0

# Multi-point (AutoStakkert-style) stacking
jupiter stack jupiter.ser --method multi-point --ap-size 48 --search-radius 12

# Drizzle 2x super-resolution
jupiter stack jupiter.ser --method drizzle --drizzle-scale 2.0 --pixfrac 0.7
```

---

### `jupiter sharpen`

Apply wavelet sharpening and optional deconvolution to a stacked image. Input is a TIFF or PNG.

```
jupiter sharpen <FILE> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE` | Yes | Input image file (TIFF or PNG) |

**Wavelet options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--layers <N>` | `6` | Number of wavelet decomposition layers |
| `--coefficients <LIST>` | `1.5,1.3,1.2,1.1,1.0,1.0` | Comma-separated sharpening coefficient per layer |
| `--denoise <LIST>` | *(none)* | Comma-separated denoise threshold per layer |
| `--output <PATH>` / `-o` | `sharpened.tiff` | Output file path |

Higher coefficients boost detail at that spatial scale. Layer 1 = finest detail, layer N = coarsest. Coefficients of 1.0 leave that layer unchanged. Denoise thresholds suppress wavelet coefficients below the threshold (sigma-based).

**Deconvolution options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--deconv <METHOD>` | *(none)* | Deconvolution method: `rl` (Richardson-Lucy) or `wiener` |
| `--psf <MODEL>` | `gaussian` | PSF model: `gaussian`, `kolmogorov`, or `airy` |
| `--psf-sigma <F>` | `2.0` | Gaussian PSF sigma in pixels |
| `--seeing <F>` | `3.0` | Kolmogorov seeing FWHM in pixels |
| `--airy-radius <F>` | `2.5` | Airy first dark ring radius in pixels |
| `--rl-iterations <N>` | `20` | Richardson-Lucy iteration count |
| `--noise-ratio <F>` | `0.001` | Wiener noise-to-signal ratio |

Deconvolution runs before wavelet sharpening when both are enabled. Only the PSF parameter matching the chosen model is used.

**Examples:**

```bash
# Default wavelet sharpening
jupiter sharpen stacked.tiff

# Aggressive fine-detail sharpening
jupiter sharpen stacked.tiff --coefficients 2.0,1.8,1.5,1.2,1.0,1.0

# Sharpening with denoise on fine layers
jupiter sharpen stacked.tiff --coefficients 1.8,1.5,1.2,1.0,1.0,1.0 \
  --denoise 3.0,2.0,1.0,0,0,0

# Richardson-Lucy deconvolution + wavelet sharpening
jupiter sharpen stacked.tiff --deconv rl --psf gaussian --psf-sigma 1.5 --rl-iterations 30

# Wiener deconvolution with Kolmogorov PSF
jupiter sharpen stacked.tiff --deconv wiener --psf kolmogorov --seeing 2.5 --noise-ratio 0.002
```

---

### `jupiter filter`

Apply post-processing filters to an image. Filters are applied in order: stretch, gamma, brightness/contrast, unsharp mask, blur.

```
jupiter filter <FILE> [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE` | Yes | Input image file (TIFF or PNG) |

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--stretch <SPEC>` | | *(none)* | `auto` or `black,white` (e.g. `0.01,0.99`) |
| `--gamma <F>` | | *(none)* | Gamma correction (e.g. 1.2) |
| `--brightness <F>` | | *(none)* | Brightness adjustment (-1.0 to 1.0) |
| `--contrast <F>` | | *(none)* | Contrast multiplier (1.0 = no change) |
| `--unsharp-mask <SPEC>` | | *(none)* | `radius,amount,threshold` (e.g. `2.0,0.5,0.01`) |
| `--blur <F>` | | *(none)* | Gaussian blur sigma |
| `--output <PATH>` | `-o` | `filtered.tiff` | Output file path |

All filter options are optional. Only the specified filters are applied. Auto stretch uses 0.1% / 99.9% percentiles.

**Examples:**

```bash
# Auto histogram stretch
jupiter filter sharpened.tiff --stretch auto

# Manual stretch + gamma correction
jupiter filter sharpened.tiff --stretch 0.01,0.99 --gamma 1.3

# Brightness/contrast adjustment
jupiter filter sharpened.tiff --brightness 0.05 --contrast 1.2

# Unsharp mask for extra edge detail
jupiter filter sharpened.tiff --unsharp-mask 2.0,0.5,0.01

# Mild blur to reduce noise
jupiter filter sharpened.tiff --blur 0.8 -o final.tiff
```

---

### `jupiter run`

Run the full processing pipeline in one step: frame selection, stacking, sharpening, and filtering. This combines all the individual commands.

```
jupiter run [FILE] [OPTIONS]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `FILE` | Yes* | Path to a SER file (*not required if `--config` is set) |

**Input/output options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config <PATH>` | | *(none)* | Pipeline config file (TOML). Replaces all CLI args |
| `--output <PATH>` | `-o` | `result.tiff` | Output file path |
| `--save-config <PATH>` | | *(none)* | Save effective config as TOML and exit |
| `--device <DEVICE>` | | `auto` | Compute device: `auto`, `cpu`, `gpu`, `cuda` |

**Frame selection:**

| Option | Default | Description |
|--------|---------|-------------|
| `--select <N>` | `25` | Percentage of best frames to keep (1-100) |

**Stacking options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--method <METHOD>` | `multi-point` | Stacking method |
| `--sigma <F>` | `2.5` | Sigma threshold (sigma-clip) |
| `--ap-size <N>` | `64` | AP size in pixels (multi-point) |
| `--search-radius <N>` | `16` | Search radius (multi-point) |
| `--min-brightness <F>` | `0.05` | Min brightness for AP placement (multi-point) |
| `--drizzle-scale <F>` | `2.0` | Output scale factor (drizzle) |
| `--pixfrac <F>` | `0.7` | Pixel drop fraction (drizzle) |

> **Note:** The `run` command defaults to `multi-point` stacking, while the standalone `stack` command defaults to `mean`.

**Sharpening options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--no-sharpen` | *(off)* | Disable sharpening entirely |
| `--sharpen <LIST>` | `1.5,1.3,1.2,1.1,1.0,1.0` | Comma-separated wavelet coefficients |
| `--denoise <LIST>` | *(none)* | Comma-separated denoise thresholds |
| `--deconv <METHOD>` | *(none)* | `rl` or `wiener` |
| `--psf <MODEL>` | `gaussian` | `gaussian`, `kolmogorov`, or `airy` |
| `--psf-sigma <F>` | `2.0` | Gaussian PSF sigma |
| `--seeing <F>` | `3.0` | Kolmogorov seeing FWHM |
| `--airy-radius <F>` | `2.5` | Airy first dark ring radius |
| `--rl-iterations <N>` | `20` | Richardson-Lucy iterations |
| `--noise-ratio <F>` | `0.001` | Wiener noise-to-signal ratio |

**Filter options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--auto-stretch` | *(off)* | Auto histogram stretch (0.1%/99.9%) |
| `--gamma <F>` | *(none)* | Gamma correction |

**Examples:**

```bash
# Full pipeline with defaults (multi-point stacking + wavelet sharpening)
jupiter run jupiter.ser

# GPU-accelerated pipeline
jupiter run jupiter.ser --device gpu

# Custom stacking + sharpening
jupiter run jupiter.ser --method drizzle --drizzle-scale 2.0 \
  --sharpen 2.0,1.5,1.2,1.0,1.0,1.0 --auto-stretch -o jupiter_final.tiff

# Mean stack, no sharpening, with gamma
jupiter run jupiter.ser --method mean --no-sharpen --gamma 1.2

# Save config for later use
jupiter run jupiter.ser --method multi-point --save-config my_pipeline.toml

# Run from config file
jupiter run --config my_pipeline.toml

# Config file + CLI overrides (CLI wins)
jupiter run jupiter_2024.ser --config base.toml -o output_2024.tiff
```

---

### `jupiter config`

Print or save a default pipeline configuration as TOML. Useful as a starting point for custom config files.

```
jupiter config [OPTIONS]
```

**Options:**

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output <PATH>` | `-o` | *(stdout)* | Write config to a file |

**Examples:**

```bash
# Print default config to terminal
jupiter config

# Save to file
jupiter config -o my_config.toml
```

---

## TOML Configuration

The `jupiter run --config` command accepts a TOML file. Use `jupiter config` to generate a template.

### Annotated example

```toml
input = "captures/jupiter.ser"
output = "result.tiff"
device = "Auto"  # "Auto", "Cpu", "Gpu", "Cuda"

[frame_selection]
select_percentage = 0.25      # 0.0-1.0 (25%)
metric = "Laplacian"          # "Laplacian" or "Gradient"

[stacking]
method = "Mean"               # Simple variant — no sub-table needed

# --- OR for sigma-clip: ---
# [stacking.method.SigmaClip]
# sigma = 2.5
# iterations = 2

# --- OR for multi-point: ---
# [stacking.method.MultiPoint]
# ap_size = 64
# search_radius = 16
# select_percentage = 0.25
# min_brightness = 0.05
# quality_metric = "Laplacian"
# local_stack_method = "Mean"

# --- OR for drizzle: ---
# [stacking.method.Drizzle]
# scale = 2.0
# pixfrac = 0.7
# quality_weighted = true
# kernel = "Square"

[sharpening.wavelet]
num_layers = 6
coefficients = [1.5, 1.3, 1.2, 1.1, 1.0, 1.0]
denoise = []                  # e.g. [3.0, 2.0, 1.0, 0.0, 0.0, 0.0]

# Deconvolution (optional — omit entire section to skip)
# [sharpening.deconvolution.method.RichardsonLucy]
# iterations = 20
# [sharpening.deconvolution.psf.Gaussian]
# sigma = 2.0
#
# --- OR Wiener + Kolmogorov: ---
# [sharpening.deconvolution.method.Wiener]
# noise_ratio = 0.001
# [sharpening.deconvolution.psf.Kolmogorov]
# seeing = 3.0

# Filter steps are applied in order.
# [[filters]]
# HistogramStretch = { black_point = 0.01, white_point = 0.99 }
#
# [[filters]]
# AutoStretch = { low_percentile = 0.001, high_percentile = 0.999 }
#
# [[filters]]
# Gamma = 1.2
#
# [[filters]]
# BrightnessContrast = { brightness = 0.0, contrast = 1.2 }
#
# [[filters]]
# UnsharpMask = { radius = 2.0, amount = 0.5, threshold = 0.01 }
#
# [[filters]]
# GaussianBlur = { sigma = 1.5 }
```

### Config + CLI precedence

When using `--config` with a positional file or `--output`, CLI arguments override values in the TOML file. This lets you keep a base config and vary input/output per session.

---

## Typical Workflows

### Step-by-step (recommended for first-time use)

```bash
# 1. Inspect the capture
jupiter info jupiter.ser

# 2. Check frame quality
jupiter quality jupiter.ser --top 30

# 3. Stack the best 20%
jupiter stack jupiter.ser --select 20 --method multi-point -o stacked.tiff

# 4. Sharpen
jupiter sharpen stacked.tiff --coefficients 1.8,1.5,1.2,1.0,1.0,1.0 -o sharp.tiff

# 5. Final adjustments
jupiter filter sharp.tiff --stretch auto --gamma 1.15 -o jupiter_final.tiff
```

### One-liner full pipeline

```bash
# Jupiter — multi-point + deconvolution + auto stretch
jupiter run jupiter.ser --method multi-point --ap-size 48 \
  --sharpen 1.8,1.5,1.3,1.1,1.0,1.0 --deconv rl --rl-iterations 25 \
  --auto-stretch -o jupiter_final.tiff

# Saturn — drizzle for ring detail
jupiter run saturn.ser --method drizzle --drizzle-scale 2.0 --pixfrac 0.5 \
  --sharpen 1.5,1.3,1.1,1.0,1.0,1.0 --auto-stretch -o saturn_final.tiff

# Moon — large FOV, mean stack is fast and effective
jupiter run moon.ser --method mean --select 50 \
  --sharpen 2.0,1.5,1.2,1.0,1.0,1.0 --auto-stretch -o moon_final.tiff
```

---

## Choosing a Stacking Method

| Method | Best for | Notes |
|--------|----------|-------|
| **mean** | Quick results, large frame counts | Fast. Averages out noise but also blurs fine detail slightly |
| **median** | Removing transient artifacts (satellites, cosmic rays) | Robust outlier rejection |
| **sigma-clip** | Noisy data with outliers | Iterative rejection (`--sigma` controls aggressiveness) |
| **multi-point** | Planetary imaging (Jupiter, Saturn, Mars) | AutoStakkert-style local alignment corrects atmospheric distortion per-region |
| **drizzle** | Undersampled data, super-resolution | Recovers sub-pixel detail. Use `--pixfrac < 1.0` for sharper output at the cost of noise |

### Tips

- **multi-point** is the default for `jupiter run` because planetary targets benefit most from local alignment.
- Use `--ap-size 32-48` for small planets and `--ap-size 64-96` for Jupiter's full disc.
- **drizzle** with `--drizzle-scale 1.5` is a good compromise between resolution gain and noise.
- Combine `--pixfrac 0.5` with higher frame counts for the best drizzle results.
- For the Moon and Sun, **mean** or **median** stacking with a high `--select` percentage often works well since seeing effects are less localized.

---

## GPU Acceleration

GPU acceleration requires building with the `gpu` feature flag:

```bash
cargo build --release --features gpu
```

Use `--device gpu` (or `--device auto` which auto-detects) to enable GPU-accelerated alignment and deconvolution. Supported backends: Metal (macOS), Vulkan (Linux/Windows), DX12 (Windows).

Operations accelerated on GPU:
- FFT-based phase correlation alignment
- Richardson-Lucy deconvolution

Operations that remain CPU-only (small kernels, GPU overhead not beneficial):
- Wavelet sharpening
- Gaussian blur
- Wiener deconvolution
- All filter steps
