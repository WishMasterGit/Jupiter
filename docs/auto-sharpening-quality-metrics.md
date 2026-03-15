# Auto-Sharpening Quality Metrics & Optimization

## Overview

This document covers the research behind Jupiter's auto-sharpening optimizer: how to
measure output image quality programmatically, detect sharpening artifacts, and search
for optimal wavelet parameters automatically.

---

## 1. No-Reference Image Quality Metrics

No-reference (NR) metrics assess image quality without a pristine original — essential
for astronomical imaging where there is no ground truth.

### 1.1 Laplacian Variance (Sharpness)

The Laplacian operator highlights edges and fine detail. The variance of the Laplacian
response is a reliable sharpness proxy:

```
L(x,y) = -4·I(x,y) + I(x-1,y) + I(x+1,y) + I(x,y-1) + I(x,y+1)
sharpness = Var(L) = E[L²] - E[L]²
```

- **Pros**: fast, well-understood, already implemented in Jupiter (`laplacian_variance_array`)
- **Cons**: sensitive to noise (noise masquerades as sharpness)
- **Use in Jupiter**: primary sharpness metric for both frame quality scoring and output evaluation

### 1.2 Frequency-Domain Energy Ratio

Compute the 2D FFT of the image and measure the fraction of power in high spatial
frequencies:

```
F = FFT2(I)
high_freq_energy = Σ|F(u,v)|² for r(u,v) > cutoff·R_max
                   ─────────────────────────────────────
                          Σ|F(u,v)|²
```

where `r(u,v)` is the distance from the DC component and `cutoff` is typically 0.3.

- **Pros**: captures global sharpness including textures missed by edge detectors
- **Cons**: slower than spatial-domain metrics
- **Use in Jupiter**: secondary sharpness metric; validates that sharpening actually pushes
  energy into high frequencies rather than just amplifying noise

### 1.3 BRISQUE / NIQE

Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) and Natural Image Quality
Evaluator (NIQE) fit statistical models to "natural" image statistics. These are powerful
for photographic images but less suitable for planetary imagery:

- Trained on natural scenes, not astronomical targets
- Planets have unusual statistics (uniform disk, limb darkening, atmospheric bands)
- Would require retraining on planetary image datasets

**Decision**: Not used in Jupiter. Laplacian variance + frequency energy is sufficient
and domain-appropriate.

---

## 2. Sharpening Artifact Detection

Over-sharpening produces characteristic artifacts that degrade image quality:

### 2.1 Ringing (Gibbs Phenomenon)

Appears as oscillating bright/dark fringes along high-contrast edges (e.g., planetary
limb). Caused by boosting wavelet coefficients too aggressively.

**Detection approach**: Find strong edges via Sobel gradient magnitude, then examine
perpendicular profiles for sign reversals (undershoot/overshoot oscillations).

Algorithm:
1. Compute Sobel gradient magnitude `G` and direction `θ`
2. Find edge pixels where `G > percentile_threshold` (e.g., 90th percentile)
3. For each edge pixel, sample intensity in the perpendicular direction (±3 pixels)
4. Count sign changes in the second derivative of the perpendicular profile
5. Ringing score = mean sign-change count across all edge pixels

### 2.2 Halo Artifacts

Bright halos around dark features (or dark halos around bright features). A specific
form of ringing where the first overshoot is dominant.

Detected by the same perpendicular-profile analysis as ringing — halos produce a
single large overshoot rather than oscillating fringes.

### 2.3 Noise Amplification

Wavelet sharpening amplifies the finest detail layer, which contains both real detail
and noise. Over-sharpening makes noise visually dominant.

**Detection approach**: Median Absolute Deviation (MAD) of the finest wavelet detail
layer, scaled by the Gaussian consistency factor:

```
σ_noise = MAD(detail_layer_0) / 0.6745
```

This robust estimator is standard in wavelet denoising (Donoho & Johnstone, 1994).

---

## 3. Existing Tool Approaches

### 3.1 AutoStakkert!3

- No auto-sharpening; user manually adjusts wavelet sliders
- Provides real-time preview of sharpening effect
- 6-layer wavelet with per-layer coefficient sliders

### 3.2 RegiStax 6

- Wavelet sharpening with 6 layers and per-layer sliders
- "Dyadic" mode links layer coefficients by powers of 2
- Includes per-layer denoise slider
- No automatic parameter search

### 3.3 Siril

- Wavelets with manual coefficient control
- Unsharp mask with manual radius/amount
- No automated optimization

### 3.4 PixInsight

- MultiscaleMedianTransform with per-layer controls
- Automatic noise estimation per layer
- "Chrominance noise reduction" automatic mode
- Closest to automated, but still requires user parameter selection

### 3.5 Summary

No existing planetary imaging tool offers fully automatic sharpening parameter
optimization. This is a differentiating feature for Jupiter.

---

## 4. Compound Quality Scoring

A single composite score combines all metrics, enabling automated comparison of
parameter candidates:

```
composite = sharpness_gain - β·artifact_score - γ·max(0, noise_gain - 1)
```

Where:
- `sharpness_gain = sharpness(output) / sharpness(input)` — relative improvement
- `artifact_score` — ringing/halo severity (0 = none, higher = worse)
- `noise_gain = noise(output) / noise(input)` — noise amplification factor
- `β = 0.5` (artifact penalty weight)
- `γ = 0.3` (noise penalty weight)

The `max(0, noise_gain - 1)` term means noise reduction is not rewarded, only noise
amplification is penalized. This prevents the optimizer from preferring blurry
(low-noise) results.

### Why gains instead of absolute values?

Different input images have vastly different baseline sharpness and noise levels.
Using gains (ratios relative to the unsharpened input) makes the scoring
input-independent, so the same weights work across different targets and seeing
conditions.

---

## 5. Search Strategies

### 5.1 Naive Grid Search (rejected)

Searching all 6 wavelet coefficients independently with 5 steps each = 5⁶ = 15,625
evaluations. At ~50ms per evaluation on a 2048×2048 image, that's ~13 minutes.
Impractical.

### 5.2 Profile Parameterization (chosen)

**Key insight**: Good wavelet sharpening profiles are monotonically decreasing — fine
layers get more boost, coarse layers get less. We parameterize this as:

```
coeff[i] = base × decay^i
```

This reduces the 6D search to a 2D search over `(base, decay)`:
- `base` ∈ [1.0, 3.0] — amplitude of finest-layer boost
- `decay` ∈ [0.5, 1.0] — how quickly boost falls off with scale

### 5.3 Coarse-to-Fine Search (default strategy)

**Phase 1 — Coarse grid**: Grid search over `(base, decay)` space
- `coarse_steps` = 8 → 64 evaluations
- Identifies promising region of parameter space

**Phase 2 — Refinement**: Take top 3 candidates from coarse phase
- Vary each of the first `layers_to_optimize` layers independently ±0.2 in 5 steps
- ~60-90 additional evaluations
- Allows breaking away from the profile constraint for fine-tuning

**Total budget**: ~150 evaluations ≈ 7.5 seconds for 2048×2048

### 5.4 Sequential Per-Layer (alternative strategy)

Optimize one layer at a time, from finest to coarsest:
- For each layer: grid search its coefficient while holding others fixed
- Greedy but fast: `layers × steps_per_layer` evaluations
- Risk: may miss interactions between layers

### 5.5 Denoise Integration

When `OptimizeTarget::WaveletAndDenoise`:
- After finding best wavelet coefficients, do a secondary search over denoise thresholds
- Only for the finest 1-2 layers (where noise lives)
- Adds ~20 evaluations

---

## 6. Implementation Notes

### Performance Budget

Each evaluation involves:
1. Wavelet decomposition + reconstruction: ~30ms for 2048×2048
2. Quality metric computation: ~20ms (Laplacian + FFT + noise estimation)
3. Total: ~50ms per evaluation

With 200 max evaluations: ~10 seconds total — acceptable for a "one-click optimize" UX.

### Determinism

The optimizer is deterministic for the same input: no random sampling, same grid
produces same candidates in same order. This makes results reproducible.

### Progress Reporting

The optimizer reports progress via a callback: `fn(evaluations_done, total_evaluations)`.
This feeds into the pipeline's existing `ProgressReporter` infrastructure.

---

## References

1. Pech-Pacheco, J.L. et al. (2000). "Diatom autofocusing in brightfield microscopy:
   a comparative study." *Pattern Recognition*, 33(9), pp.1315-1328.
   — Laplacian variance as focus/sharpness measure

2. Donoho, D.L. & Johnstone, I.M. (1994). "Ideal spatial adaptation by wavelet
   shrinkage." *Biometrika*, 81(3), pp.425-455.
   — MAD noise estimator for wavelet coefficients

3. Mallat, S. (2009). *A Wavelet Tour of Signal Processing*, 3rd ed. Academic Press.
   — À trous wavelet transform theory

4. Guizar-Sicairos, M. et al. (2008). "Efficient subpixel image registration
   algorithms." *Optics Letters*, 33(2), pp.156-158.
   — Frequency-domain image analysis techniques

5. Mittal, A. et al. (2012). "No-reference image quality assessment in the spatial
   domain." *IEEE TIP*, 21(12), pp.4695-4708.
   — BRISQUE (not used, but referenced for completeness)
