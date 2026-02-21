# Multi-Point (AP-Based) Stacking for Planetary Imaging

## Comprehensive Research & Implementation Guide

---

## Table of Contents

1. [Introduction and Physical Motivation](#1-introduction-and-physical-motivation)
2. [Atmospheric Turbulence Model](#2-atmospheric-turbulence-model)
3. [The Multi-Point Stacking Algorithm](#3-the-multi-point-stacking-algorithm)
4. [AutoStakkert's Approach](#4-autostakkerts-approach)
5. [PlanetarySystemStacker's Approach](#5-planetarysystemstackers-approach)
6. [Quality Metrics and Frame Selection](#6-quality-metrics-and-frame-selection)
7. [Local Alignment Strategies](#7-local-alignment-strategies)
8. [Blending and Reconstruction](#8-blending-and-reconstruction)
9. [Common Pitfalls and Failure Modes](#9-common-pitfalls-and-failure-modes)
10. [Best Practices from the Community](#10-best-practices-from-the-community)
11. [Alternative Approaches](#11-alternative-approaches)
12. [Recommendations for Our Implementation](#12-recommendations-for-our-implementation)
13. [Algorithm Pseudocode](#13-algorithm-pseudocode)
14. [References](#14-references)

---

## 1. Introduction and Physical Motivation

### 1.1 Why Single-Point Alignment is Insufficient

In traditional "shift-and-add" lucky imaging, each frame is aligned by computing a single global translation offset (typically via cross-correlation or phase correlation against a reference frame), then all selected frames are averaged together.

This approach assumes the atmospheric distortion is a **pure translation** -- that the entire image shifts uniformly. This assumption holds only within a small angular region called the **isoplanatic patch**. For planetary disks that subtend 30-50 arcseconds (Jupiter) or 15-25 arcseconds (Mars/Saturn), the disk may span **multiple isoplanatic patches**. Different parts of the planet experience different atmospheric shifts in the same instant.

The result of single-point alignment: the region around the alignment point is sharp, but distant regions appear smeared because their local shifts were different from the global correction.

### 1.2 The Multi-Point Solution

Multi-point (or multi-AP, multi-alignment-point) stacking solves this by dividing the image into a grid of small overlapping regions, each treated as an independent isoplanatic patch. Each region gets:

- Its own **local quality scoring** (best frames for one region may differ from another)
- Its own **local shift measurement** (independent sub-pixel alignment)
- Its own **local frame selection** (only the sharpest frames for that specific region)

The independently-stacked patches are then **blended** back into a single composite image. This is the fundamental principle behind AutoStakkert, PlanetarySystemStacker, and similar tools.

### 1.3 The Mathematical Model

Let `I_k(x, y)` be the k-th short-exposure frame. The atmospheric distortion at position `(x, y)` in frame `k` can be modeled as:

```
I_k(x, y) = O(x - d_k_x(x,y), y - d_k_y(x,y)) * PSF_k(x,y) + n_k(x,y)
```

Where:
- `O(x, y)` is the true object (planet surface)
- `d_k(x, y) = (d_k_x, d_k_y)` is the **spatially-varying displacement field** (tip-tilt)
- `PSF_k(x, y)` is the spatially-varying point spread function
- `n_k(x, y)` is noise
- `*` denotes convolution

Single-point alignment assumes `d_k(x, y) = d_k` (constant across the field). Multi-point stacking approximates the spatially-varying field as **piecewise-constant** over AP regions:

```
d_k(x, y) ~ d_k^(j)    for (x, y) in region R_j
```

where `j` indexes the alignment point and `d_k^(j)` is the local shift for AP `j` in frame `k`.

---

## 2. Atmospheric Turbulence Model

### 2.1 Kolmogorov Turbulence and the Fried Parameter

Atmospheric turbulence follows the Kolmogorov cascade model. The key parameter is the **Fried parameter** `r_0`, defined as the aperture diameter over which the root-mean-square wavefront error equals 1 radian:

```
r_0 = [ 0.423 * k^2 * integral(C_n^2(z) dz) ]^(-3/5)
```

where `k = 2*pi/lambda` is the wavenumber and `C_n^2(z)` is the refractive index structure constant at altitude `z`.

The **seeing** (FWHM of the long-exposure PSF) is:

```
seeing (arcsec) = 0.98 * lambda / r_0
```

For typical amateur conditions:
- Good seeing: r_0 ~ 15-20 cm at 500 nm (seeing ~ 0.7-1.0")
- Average seeing: r_0 ~ 8-12 cm (seeing ~ 1.0-1.5")
- Poor seeing: r_0 ~ 5-7 cm (seeing ~ 1.5-2.5")

### 2.2 Isoplanatic Angle

The **isoplanatic angle** `theta_0` defines the angular region over which the wavefront distortion is correlated:

```
theta_0 ~ 0.314 * r_0 / h_eff
```

where `h_eff` is the effective turbulence height. Typical values:

| Conditions | r_0 (cm) | h_eff (km) | theta_0 (arcsec) |
|-----------|----------|-----------|------------------|
| Excellent | 20 | 5 | ~2.5 |
| Good | 12 | 8 | ~1.0 |
| Average | 8 | 10 | ~0.5 |
| Poor | 5 | 10 | ~0.3 |

Law et al. (2006) measured isoplanatic patch radii of 17-30 arcseconds in their lucky imaging observations at the Nordic Optical Telescope, implying effective turbulence heights of 5-10 km.

### 2.3 Implications for Planetary Imaging

Jupiter at opposition subtends ~47 arcseconds. With a 2-arcsecond isoplanatic patch, the disk contains roughly `(47/2)^2 ~ 550` independent isoplanatic patches. Even with generous patch sizes, different parts of Jupiter experience genuinely different atmospheric distortions in each frame.

This means:
- Frame `k` might have excellent seeing over the Great Red Spot but poor seeing over the north polar region
- Frame `k+1` might have the opposite pattern
- Single-point alignment and global frame selection throw away this information

### 2.4 Lucky Imaging Probability

The probability of getting a "lucky" frame (wavefront variance < 1 rad^2) with aperture D:

```
P(lucky) ~ 5.6 * exp(-0.1557 * (D/r_0)^2)    for D/r_0 >= 3.5
```

For a 280 mm (11") telescope with r_0 = 10 cm: `D/r_0 = 28`, giving `P ~ 5.6 * exp(-122) ~ 0` for the full aperture. But for a sub-aperture the size of `r_0` (10 cm): `D/r_0 = 1`, giving near-unity probability. This is why **local** lucky frame selection per-AP dramatically outperforms global selection.

---

## 3. The Multi-Point Stacking Algorithm

### 3.1 High-Level Pipeline

```
1. GLOBAL ALIGNMENT
   - Select or compute the best frame as reference
   - Compute global (whole-image) translational offsets for all frames
   - Apply global offsets (or carry them forward)

2. AP GRID CONSTRUCTION
   - Overlay a grid of alignment points on the reference frame
   - Filter out APs in low-contrast or dark regions
   - APs should overlap by ~50% for seamless blending

3. PER-AP QUALITY SCORING
   - For each AP region, score every frame's local quality
   - Quality metric: local contrast, Laplacian variance, gradient magnitude
   - Rank frames independently per-AP

4. PER-AP FRAME SELECTION
   - For each AP, select the top N% of frames (typically 10-30%)
   - Different APs will generally select different subsets of frames

5. PER-AP LOCAL ALIGNMENT
   - For each selected frame at each AP:
     a. Extract the AP region (with padding for search)
     b. Compute local sub-pixel shift via cross-correlation
     c. Apply combined global + local shift

6. PER-AP STACKING
   - Stack the locally-aligned patches for each AP
   - Method: mean, median, or sigma-clipped mean

7. BLENDING / RECONSTRUCTION
   - Combine all per-AP stacked patches into the final image
   - Use distance-weighted or cosine-window blending
   - Overlapping regions ensure seamless transitions
```

### 3.2 Why Each Step Matters

**Global alignment** removes the dominant whole-frame shift, reducing the search range needed for local alignment. Without it, local alignment must search over much larger regions, increasing computation and the risk of false matches.

**AP grid construction** determines the spatial resolution of atmospheric correction. Too few APs and you miss local distortions; too many and the regions become too small for reliable correlation.

**Per-AP quality scoring** is the key differentiator from simple multi-point alignment. It allows selecting the best frames *independently* for each region, exploiting the fact that atmospheric quality varies spatially.

**Local alignment** corrects residual shifts after global alignment. These residual shifts represent the anisoplanatic distortion -- the difference between the local atmosphere and the global average.

**Blending** is critical for avoiding visible seams. Overlapping AP regions with smooth weight falloff ensure that the transition between independently-processed patches is imperceptible.

---

## 4. AutoStakkert's Approach

AutoStakkert (AS!2, AS!3, AS!4) by Emil Kraaikamp is the de facto standard for amateur planetary stacking. While the source code is not published, the algorithm is well-described through documentation and community discussion.

### 4.1 AP Placement

AutoStakkert supports several AP placement modes:

- **Manual placement**: User clicks to place individual APs
- **Automatic grid**: Regular grid placement with brightness/contrast filtering
- **Multi-scale**: Multiple AP sizes simultaneously (small APs for high-detail areas, larger APs for broader structure)

Key insight: AutoStakkert does NOT assign each AP a predetermined output region. Instead, after stacking, **only the best APs** contribute to the final image. APs that produced poor results (due to insufficient local contrast for reliable alignment) are rejected.

### 4.2 Quality Scoring

AutoStakkert offers two quality metrics:
- **Gradient**: Computes the gradient magnitude in x and y directions. Better for images with strong edges (limb, belt boundaries).
- **Local contrast**: Measures local intensity variance. Better for textured areas with subtle detail.

The quality is scored per-AP, per-frame. Each AP independently ranks all frames and selects its own best subset.

### 4.3 Local Alignment

For each AP in each selected frame:
1. Extract a search region centered on the AP location (adjusted by global offset)
2. Cross-correlate with the reference region to find the local shift
3. The search region is larger than the AP by a "search radius" (typically 16-32 pixels)
4. Sub-pixel accuracy is achieved through peak interpolation (parabolic or Gaussian fit to the correlation peak)

### 4.4 The "Surface Model" Concept

In AS!3 and AS!4, AutoStakkert constructs a deformation surface model by interpolating local shifts across all APs for each frame. Rather than treating each AP as an independent tile, the software:

1. Computes local shifts at all AP positions
2. Fits a smooth surface through these shifts (e.g., bilinear or thin-plate spline interpolation)
3. Uses this surface to warp the entire frame
4. The warped frame is then sampled at each output pixel position

This approach:
- Avoids seams entirely (the warping is continuous)
- Better handles APs that fail to correlate (they are outliers in the surface fit and can be rejected)
- Provides a physically more realistic model of the atmospheric distortion

### 4.5 Reconstruction

The final image is reconstructed by:
1. For each output pixel, determine which APs contributed reliably
2. Weight contributions by distance from each AP center and by the AP's quality/reliability score
3. Blend using distance-weighted averaging

The multi-scale feature allows large APs to provide the broad structure while small APs refine fine details, with the weighting scheme preferring the smallest reliable AP at each location.

---

## 5. PlanetarySystemStacker's Approach

PlanetarySystemStacker (PSS) by Rolf Hempel is open-source (Python) and provides insight into a concrete implementation.

### 5.1 Algorithm Overview

1. **Global frame ranking**: All frames ranked by overall quality
2. **Reference selection**: The best frame becomes the reference
3. **Global alignment**: Find a rectangular patch with the most pronounced structure; align all frames globally using this patch
4. **Mean reference**: Compute a mean image from the best N frames (this serves as a higher-SNR reference for AP alignment)
5. **AP mesh construction**: Regular grid covering the object, filtering points with insufficient brightness or contrast
6. **Per-AP local quality ranking**: At each AP, rank all frames by local contrast
7. **Per-AP frame selection**: Select top N frames per AP
8. **Local shift computation**: Cross-correlate each AP patch against the mean reference
9. **Per-AP stacking**: Average the locally-aligned patches
10. **Blending**: Blend stacked patches into the global image

### 5.2 Key Design Decision: Simplified Blending

Rolf Hempel initially attempted per-pixel shift interpolation (computing a unique shift for every pixel by interpolating between AP centers). He found that simple patch-based blending with overlapping regions produced **nearly identical quality** at dramatically lower computational cost.

This suggests that the atmospheric coherence scale is large enough that piecewise-constant correction (one shift per AP) is a good approximation, and the blending handles the transitions adequately.

### 5.3 Mean Reference Frame

PSS uses a mean of the best globally-ranked frames as the correlation reference, rather than a single frame. This is important because:
- A single frame has high noise, leading to noisy correlation peaks
- The mean has higher SNR, producing more reliable shift measurements
- The mean is less biased toward one particular distortion pattern

---

## 6. Quality Metrics and Frame Selection

### 6.1 Common Quality Metrics

**Laplacian Variance** (used in our implementation):
```
L(I) = Var(nabla^2 I)
```
where `nabla^2` is the Laplacian operator. High values indicate sharp features (high spatial frequency content). The Laplacian is computed via convolution with kernel:
```
[0  1  0]
[1 -4  1]
[0  1  0]
```

**Gradient Magnitude** (Sobel-based):
```
G(I) = mean(sqrt(G_x^2 + G_y^2))
```
where `G_x`, `G_y` are Sobel gradient estimates. Measures edge strength.

**Local Contrast**:
```
C(I) = std(I) / mean(I)
```
or simply `std(I)`. Measures intensity variation within the patch.

**Tenengrad** (gradient magnitude via Sobel, thresholded):
```
T(I) = sum(G(x,y)^2)    for G(x,y) > threshold
```

**Normalized Variance**:
```
NV(I) = Var(I) / mean(I)
```

### 6.2 Which Metric Works Best?

Research on image quality measures for solar imaging (Deng et al., 2015) compared six families of focus measures: gradient, Laplacian, wavelet, intensity statistics, DCT, and miscellaneous. Key findings:

- **Laplacian-based metrics** are most reliable for detecting local sharpness
- **Gradient-based metrics** are more robust to noise but slightly less sensitive to fine detail
- **Intensity variance** can be fooled by high-contrast but blurred features

For per-AP scoring, the metric must work on small patches (32-128 pixels). At these scales:
- Laplacian variance can be noisy (few pixels to compute variance over)
- Gradient magnitude is more stable
- **Recommendation**: Use gradient magnitude for AP sizes below 48px, Laplacian variance for 64px and above

### 6.3 Per-AP vs. Global Frame Selection

The power of multi-point stacking comes from **independent per-AP frame selection**. Consider:

- Frame A: great seeing on the left half, poor on the right
- Frame B: poor seeing on the left half, great on the right
- Frame C: mediocre everywhere

Global selection might keep frames A and B and discard C. But for an AP on the left side, only frame A should be selected. For an AP on the right, only frame B. This doubles the effective per-AP selection rate.

**Critical insight**: With per-AP selection, the effective number of usable frames increases dramatically. Even in a poor-seeing run, most frames have *some* region with acceptable quality.

### 6.4 Quality-Weighted Stacking

Rather than binary selection (keep/discard), quality scores can weight contributions:

```
stacked(x,y) = sum_k( w_k * I_k(x,y) ) / sum_k( w_k )
```

where `w_k = quality_score_k^alpha` for some exponent `alpha > 0`.

Benefits:
- Smoother transition at the selection boundary (avoids "cliff" where the N-th frame is kept but N+1 is identical quality and discarded)
- Better SNR (more frames contribute, higher-quality frames contribute more)
- Alpha = 0 gives equal weighting (mean); alpha -> infinity gives best-frame-only

---

## 7. Local Alignment Strategies

### 7.1 Cross-Correlation vs. Phase Correlation

**Cross-correlation** (spatial domain):
```
CC(dx, dy) = sum_{x,y} R(x,y) * T(x+dx, y+dy)
```
Pros: robust, handles large brightness variations
Cons: slow for large search regions, biased by mean intensity

**Phase correlation** (frequency domain):
```
PC = IFFT( FFT(R) * conj(FFT(T)) / |FFT(R) * conj(FFT(T))| )
```
Pros: fast (FFT-based), handles brightness changes, sharp peak
Cons: less robust on small regions, sensitive to noise

**Normalized cross-correlation**:
```
NCC(dx, dy) = CC(dx, dy) / (n * sigma_R * sigma_T)
```
Best robustness to brightness/contrast variations. Recommended for per-AP local alignment where different regions of the planet have very different brightness levels.

### 7.2 Sub-Pixel Accuracy Requirements

For planetary imaging, sub-pixel alignment accuracy is critical. At typical image scales (0.1-0.3 arcsec/pixel), a 1-pixel error corresponds to 0.1-0.3 arcseconds -- a significant fraction of the seeing disk.

Required accuracy for various use cases:
- **Basic stacking**: 0.5 px sufficient (little benefit beyond this for average seeing)
- **Good seeing stacking**: 0.1-0.2 px desirable
- **Drizzle/super-resolution**: 0.05-0.1 px needed

### 7.3 Sub-Pixel Methods

**Parabolic peak fitting**: Fit a parabola to the 3x3 neighborhood around the correlation peak.
```
dx_sub = (CC[y,x-1] - CC[y,x+1]) / (2 * (CC[y,x-1] + CC[y,x+1] - 2*CC[y,x]))
dy_sub = (CC[y-1,x] - CC[y+1,x]) / (2 * (CC[y-1,x] + CC[y+1,x] - 2*CC[y,x]))
```
Accuracy: ~0.1 px for well-sampled peaks

**Gaussian peak fitting**: Fit a Gaussian to the peak neighborhood.
Accuracy: ~0.05 px, more robust to noise

**Guizar-Sicairos matrix-multiply DFT** (used in our enhanced_phase implementation):
1. Coarse alignment via standard FFT phase correlation (integer pixel)
2. Refine by computing DFT only in a small neighborhood around the peak
3. Uses matrix multiplication instead of zero-padded FFT
4. Accuracy: up to 0.01 px with upsample factor of 100
5. Memory: O(N) where N is image size, not O(N * upsample_factor)

**Recommendation for per-AP alignment**: Parabolic or Gaussian peak fitting is sufficient (0.05-0.1 px). Full Guizar-Sicairos upsampling is overkill for the noise level in individual AP patches but valuable for global alignment.

### 7.4 Search Radius Considerations

The search radius for local alignment determines how large a local displacement can be corrected. It should be:

```
search_radius >= max_residual_shift_after_global_alignment
```

In practice:
- With good global alignment: 8-16 px search radius is sufficient
- With mediocre global alignment: 16-32 px may be needed
- **Larger search radius = larger FFT = slower + more false match risk**

The total extracted region for correlation is `AP_size + 2 * search_radius`. This region must be large enough for reliable correlation (minimum ~32x32 effective correlation area).

---

## 8. Blending and Reconstruction

### 8.1 The Blending Problem

After independently stacking each AP region, we have a set of overlapping patches. Each patch was stacked from potentially different frame subsets with different local alignment corrections. Simply tiling them creates visible seams at patch boundaries due to:

- Different frames selected -> different noise patterns and brightness levels
- Different local shifts applied -> slight geometric mismatches at boundaries
- Per-AP stacking averages a different number of frames per pixel

### 8.2 Distance-Weighted Blending

The simplest approach is inverse-distance weighting from each AP center:

```
output(x, y) = sum_j( w_j(x,y) * patch_j(x,y) ) / sum_j( w_j(x,y) )

w_j(x, y) = 1 / max(dist(x,y, AP_j_center), epsilon)
```

This is simple but produces artifacts near AP centers (weight singularities) and falls off slowly, meaning distant APs still contribute.

### 8.3 Raised Cosine (Hann Window) Blending

A better approach uses a **Hann (raised cosine) window** as the blending weight for each patch. With 50% overlap between adjacent patches, the Hann window forms a **partition of unity**:

```
For a patch of size S with center at origin:
  w(d) = 0.5 * (1 + cos(pi * d / (S/2)))    for |d| <= S/2
  w(d) = 0                                     for |d| > S/2
```

**2D weight**: `W(x, y) = w(x) * w(y)` (separable Hann window)

The partition of unity property means:
```
sum_j W_j(x, y) = 1    for all (x, y) covered by the AP grid
```

This guarantees:
- No seams (weights sum to exactly 1 everywhere)
- Smooth transitions (cosine taper)
- Each pixel's value is a weighted average of overlapping patches

**Critical requirement**: The overlap must be exactly 50% (stride = AP_size / 2) for the Hann partition of unity to hold. Other overlap fractions require different window functions.

### 8.4 Gaussian Blending

An alternative to Hann is a Gaussian weight centered on each AP:

```
W_j(x, y) = exp(-(dist(x,y, AP_j)^2) / (2 * sigma^2))
```

where `sigma ~ AP_size / 3`. The weight is then normalized by dividing by the sum of all overlapping Gaussian weights. This does not form an exact partition of unity but is smooth and forgiving of irregular AP placement.

### 8.5 Multi-Resolution (Laplacian Pyramid) Blending

The most sophisticated approach decomposes each patch into frequency bands and blends them differently:
- Low frequencies (smooth gradients): blend over wide regions (large overlap)
- High frequencies (fine details): blend over narrow regions (minimize duplicate edges)

This is the same principle used in panoramic image stitching (Burt & Adelson, 1983). It avoids the "ghosting" that can occur when two patches have slightly different alignment of sharp features.

**Implementation sketch**:
1. For each AP patch, compute Gaussian/Laplacian pyramid (3-4 levels)
2. Blend each level using an appropriate mask (wider for lower levels)
3. Reconstruct from the blended pyramid

### 8.6 Surface-Model Warping (AutoStakkert-Style)

Instead of blending independent patches, interpolate the local shifts into a smooth deformation field and warp each frame:

```
For each frame k:
  1. Compute local shift d_k^(j) at each AP j
  2. Interpolate shifts across the image: D_k(x,y) = interpolate({d_k^(j)})
  3. Warp: I_warped_k(x,y) = I_k(x + D_k_x(x,y), y + D_k_y(x,y))
  4. Stack warped frames (per-pixel quality weighting optional)
```

Interpolation methods:
- **Bilinear**: fast, works for regular grids, can produce faceted artifacts
- **Thin-plate spline (TPS)**: smooth, handles irregular AP placement, more expensive
- **Natural neighbor / Sibson**: good compromise of smoothness and speed

**Advantage**: No seams possible, since the warping is continuous.
**Disadvantage**: More complex; requires reliable shift estimates at all APs (one bad estimate warps a large region).

---

## 9. Common Pitfalls and Failure Modes

### 9.1 AP Size Too Large

**Symptom**: Multi-point stacking produces results barely better than single-point.
**Cause**: Large APs span multiple isoplanatic patches, so the local alignment is still averaging different distortions.
**Rule of thumb**: AP size should approximate the isoplanatic patch size in pixels.

```
AP_size_pixels ~ theta_0 / plate_scale

Example: theta_0 = 2", plate_scale = 0.15"/px -> AP ~ 13 px
         With safety margin: AP ~ 32-48 px
```

In practice, 32-64 pixels is typical for most setups. For very good seeing or small apertures, 48-96 pixels works. For poor seeing, 24-32 pixels.

### 9.2 AP Size Too Small

**Symptom**: Noisy result, "wavy" or distorted features, loss of structure.
**Cause**: Small patches contain insufficient structure for reliable cross-correlation. The correlation peak is broad and noisy, leading to incorrect shift estimates.
**Minimum**: APs should contain recognizable structure with contrast in both x and y directions. Absolute minimum is ~24 pixels for high-contrast targets.

### 9.3 Insufficient Search Radius

**Symptom**: Sharp core with blurred/ghosted edges; frame alignment failures.
**Cause**: Local shifts exceed the search radius, so the correlation peak falls outside the search area. The algorithm finds a spurious peak instead.
**Solution**: Increase search radius. Monitor the distribution of measured local shifts; if many are near the search boundary, it is too small.

### 9.4 Excessive Search Radius

**Symptom**: Artifacts, "grid pattern" visible, features appear duplicated or misaligned.
**Cause**: Too-large search regions increase the chance of false correlation matches (locking onto the wrong feature). This is especially problematic for periodic structures (Jupiter's belts).
**Solution**: Reduce search radius, improve global alignment (so residual shifts are small), or use normalized cross-correlation which is more discriminating.

### 9.5 Poor Quality Metric for Local Alignment

**Symptom**: Some AP regions appear blurred despite having many frames selected.
**Cause**: The quality metric selects frames based on a criterion that does not correlate with local sharpness. For example, pure intensity variance can select high-contrast frames that are actually blurred if a bright feature happens to be in the patch.
**Solution**: Use Laplacian variance or gradient-based metrics that specifically measure edge sharpness. Consider combining metrics.

### 9.6 Blending Seams

**Symptom**: Visible grid pattern, brightness discontinuities at AP boundaries, "patchwork" appearance.
**Causes**:
- Overlap insufficient (< 50% for Hann blending)
- Weight function does not form partition of unity
- Different patches selected very different frame subsets leading to different brightness levels
- Edge handling incorrect (APs near the image boundary)
**Solutions**:
- Ensure exactly 50% overlap for Hann blending
- Normalize weights properly (`output = sum(w*p) / sum(w)`)
- Apply brightness normalization before blending
- Special handling for boundary APs

### 9.7 Reference Frame Quality Issues

**Symptom**: Poor results everywhere, despite reasonable settings.
**Cause**: Using a very sharp single frame as reference can bias toward frames with similar distortion patterns. Using a poor frame means all correlations are noisy.
**Solution**: Use a **mean of top 10-20% frames** as the correlation reference. This averages out individual distortion patterns and has much higher SNR than any single frame.

### 9.8 Too Few Frames Selected Per AP

**Symptom**: Very noisy result despite apparent sharpness.
**Cause**: Selecting only top 5-10% per AP gives very few frames (e.g., 50 frames from 1000). The noise reduction is proportional to sqrt(N), so too few frames means high noise.
**Balance**: 15-30% is typical. More is better for SNR, fewer for sharpness. The optimum depends on seeing conditions.

### 9.9 Edge Artifacts from Out-of-Bounds Sampling

**Symptom**: Dark borders, bright/dark fringes at image edges, AP regions near borders show artifacts.
**Cause**: When extracting shifted AP regions, pixels outside the image boundary are sampled. If clamped (repeat edge pixel) or zeroed, this creates artifacts.
**Solution**: Reject AP regions where the shifted extraction would exceed boundaries. Or weight edge pixels to zero contribution. For boundary APs, use a validity mask.

---

## 10. Best Practices from the Community

### 10.1 Optimal AP Size

From extensive community testing (Cloudy Nights forums, planetary-astronomy-and-imaging.com):

| Target | Typical Diameter (px) | Recommended AP Size | Notes |
|--------|----------------------|--------------------|----|
| Jupiter | 200-400 | 48-80 | Depends on focal length, seeing |
| Saturn (disk) | 120-200 | 48-64 | |
| Mars | 80-200 | 32-48 | Less surface detail, smaller APs |
| Moon (hi-res) | Wide field | 64-128 | Large field = many APs |
| Sun (hi-res) | Wide field | 64-128 | Granulation provides good correlation |

**Kraaikamp (AutoStakkert developer) recommends**: 30-40 APs for Jupiter including multi-scale. No improvement beyond ~100 APs.

### 10.2 How Many APs

Practical experience shows:
- **Minimum**: 10-15 APs for meaningful multi-point improvement
- **Sweet spot**: 30-80 APs for planets
- **Diminishing returns**: beyond ~100 APs on a 300px planet
- **Multi-scale**: Using 2-3 AP sizes simultaneously (e.g., 32 + 64 + 128) provides the best results, with large APs for stable structure and small APs for fine detail

### 10.3 Frame Selection Percentage

- **Excellent seeing (< 1" FWHM)**: Keep 25-40% (most frames are usable)
- **Good seeing (1-1.5" FWHM)**: Keep 15-25%
- **Average seeing (1.5-2.5" FWHM)**: Keep 10-15%
- **Poor seeing (> 2.5" FWHM)**: Keep 5-10%

The per-AP selection rate can be **more aggressive** (keep fewer frames) than global selection because each AP independently gets the best frames for its region.

### 10.4 Pre-Processing Steps

Before multi-point stacking:
1. **Cropping**: Crop video to planet ROI + margin (PIPP or similar)
2. **Centering**: Center the planet in each frame (global alignment handles this but pre-centering improves reference selection)
3. **Flat field**: Apply flat frame if available (removes optical vignetting)
4. **Derotation (WinJUPOS)**: For videos longer than ~2 minutes for Jupiter (~4 minutes for Saturn), derotate frames before stacking to avoid rotational smear
5. **Debayering**: For color cameras, debayer before or during processing

### 10.5 Post-Processing

After multi-point stacking:
1. **Wavelet sharpening** (Registax, AstraImage): Enhances detail; the stacked image should show detail "potential" that sharpening reveals
2. **Deconvolution**: Richardson-Lucy or Wiener deconvolution with estimated PSF
3. **Color balancing**: Adjust R/G/B balance
4. **Histogram stretch**: Gentle curves adjustment

### 10.6 Diagnostic Indicators

How to know if multi-point stacking is working correctly:
- **Consistent limb**: The planet limb should be uniformly sharp, not sharper on one side
- **No grid pattern**: No visible regular pattern in the background
- **Detail distribution**: Fine details should appear across the disk, not just near one AP
- **Improved over single-point**: Direct comparison should show broader area of sharpness
- **Local shift distribution**: Plot the local shifts per AP; they should be smooth and continuous, not random. Large jumps indicate alignment failures.

---

## 11. Alternative Approaches

### 11.1 WinJUPOS Derotation

WinJUPOS maps each frame onto a cylindrical/spherical coordinate system based on the known planetary geometry and rotation rate. Multiple stacked results taken over time can be "derotated" to the same rotational phase and combined.

**Key advantage**: Allows using 10-30 minutes of data (vs. 2-3 minutes without derotation), dramatically increasing the number of stackable frames.

**Workflow integration**: Derotation complements multi-point stacking. The typical pipeline is:
1. Capture 3-5 videos of 2-3 minutes each
2. Stack each video separately with multi-point stacking
3. Derotate all stacked results to a common epoch with WinJUPOS
4. Combine derotated results

### 11.2 Dense Optical Flow

Instead of piecewise-constant shifts at AP positions, compute a full dense deformation field:

```
For each frame k:
  (u_k(x,y), v_k(x,y)) = optical_flow(reference, frame_k)
  warped_k(x,y) = frame_k(x + u_k, y + v_k)
```

Methods: Lucas-Kanade (local), Horn-Schunck (global), Farneback (dense polynomial).

**Advantages**:
- Per-pixel correction, no AP grid needed
- Continuous smooth deformation
- Can correct rotation/scaling in addition to translation

**Disadvantages**:
- Computationally expensive
- Prone to errors in low-contrast regions
- Can "hallucinate" corrections in featureless areas
- Noise in the flow field adds noise to the output

**Research direction**: HybridFlow and similar variational methods offer promising large-deformation estimation.

### 11.3 MAP (Maximum A Posteriori) Estimation

Frame the stacking problem as Bayesian inference:

```
O*(x,y) = argmax_O P(O | {I_k}) = argmax_O [ P({I_k} | O) * P(O) ]
```

where `O` is the true planetary surface and `{I_k}` are the observed frames. The likelihood incorporates the atmospheric model, and the prior can enforce smoothness or spectral constraints.

**In practice**: This is equivalent to iterative deconvolution applied to the stacked image, where the PSF is estimated from the data (blind deconvolution).

### 11.4 Speckle Imaging

Speckle imaging reconstructs the object's power spectrum from the ensemble of short-exposure images:

```
|O(f)|^2 = <|I_k(f)|^2> / <|PSF_k(f)|^2>
```

Combined with phase recovery (e.g., from bispectral analysis), this can achieve diffraction-limited resolution without frame selection.

**Limitation**: Requires many frames (thousands) and works best for point-like sources. Less suitable for extended objects like planets where the phase recovery is ambiguous.

---

## 12. Recommendations for Our Implementation

Based on this research, here are specific improvements for our multi-point stacking implementation.

### 12.1 Use a Mean Reference Frame

**Current**: We use frame 0 as the reference.
**Recommended**: After global alignment, compute a mean of the top 10-20% frames as the correlation reference. This dramatically improves correlation reliability.

```
Pseudocode:
  1. Global-align all frames vs. frame 0 (quick & dirty reference)
  2. Score all frames globally
  3. Compute mean of top 20% (with global alignment applied)
  4. Re-run global alignment using mean as reference (optional but improves accuracy)
  5. Use this mean as the AP correlation reference
```

### 12.2 Validate Hann Blending Implementation

**Current**: We use Hann window blending with 50% overlap.
**Issue to verify**: The Hann weight function must be:
- `w(d) = 0.5 * (1 - cos(2*pi*d/N))` where `d` ranges from 0 to N-1
- With stride = N/2 (50% overlap), this forms a partition of unity
- The 2D weight is the product of 1D weights: `W(r,c) = w(r) * w(c)`

The current `hann_weight` function uses `0.5 * (1 - cos(TAU * t))` where `t = pos/size`. This is correct for the Hann window, but we need to verify:
- That the patch-to-image coordinate mapping places patches correctly
- That stride exactly equals `ap_size / 2`
- That boundary handling does not create uncovered pixels

### 12.3 Add Quality-Weighted Stacking

Instead of simple mean stacking within each AP:

```rust
// Weight each frame's contribution by its quality score
let total_weight: f64 = selected_frames.iter().map(|(_, score)| score).sum();
for &(frame_idx, score) in selected_frames {
    let w = score / total_weight;
    stacked += w * patch;
}
```

### 12.4 Improve Local Alignment Robustness

Current issues with cross-correlation on small patches:
1. **Add windowing**: Apply a Hann window to both reference and target patches before correlation. This reduces spectral leakage and improves peak localization.
2. **Peak validation**: After finding the correlation peak, validate that it is significantly above the background. If `peak / mean_correlation < threshold` (e.g., < 3.0), reject this frame for this AP.
3. **Outlier rejection**: After computing local shifts for all frames at an AP, reject shifts that are statistical outliers (more than 2 sigma from median).

### 12.5 Add Multi-Scale AP Support

Implement 2-3 AP sizes simultaneously:
```
AP sizes: [ap_size/2, ap_size, ap_size*2]

For each scale:
  - Build AP grid
  - Score, select, align, stack

Blend multi-scale results:
  - Decompose each scale's stacked patch into frequency bands
  - Use large-AP results for low frequencies
  - Use small-AP results for high frequencies
  - Recombine
```

### 12.6 Reference Frame Selection Strategy

Instead of always using frame 0:
1. Score all frames globally (fast Laplacian or gradient on full frame)
2. Select the frame closest to the **median** quality (not the best!)
3. Or better: compute mean of top 20% frames as reference

Using the median avoids bias toward an extreme atmospheric state. Using the mean avoids noise.

### 12.7 Correlation Peak Quality Check

Add a confidence metric to each local alignment:

```
confidence = peak_value / mean(correlation_surface)
```

or

```
confidence = (peak - second_peak) / std(correlation_surface)
```

Use this to:
- Reject unreliable AP measurements (confidence below threshold)
- Weight AP contributions in the blend (higher confidence = more weight)

### 12.8 Surface Model Interpolation (Advanced)

For maximum quality, instead of patch-based blending, interpolate the shift field:

```
For each frame k:
  1. Compute local shifts at all AP positions: {d_k^(j)}
  2. Reject outlier shifts (> 2 sigma from neighbors)
  3. Interpolate: D_k(x,y) = bilinear_interp({d_k^(j)}, x, y)
  4. Warp frame: warped_k(x,y) = bilinear_sample(frame_k, x - D_k_x, y - D_k_y)

Stack all warped frames (with per-pixel quality weighting)
```

This eliminates blending seams entirely.

---

## 13. Algorithm Pseudocode

### 13.1 Complete Multi-Point Stacking Pipeline

```
function multi_point_stack(frames, config):
    N = len(frames)

    // === Phase 1: Global Alignment ===

    // Score all frames globally
    global_scores = []
    for each frame in frames:
        global_scores.append(laplacian_variance(frame))

    // Select reference: mean of top 20%
    top_indices = argsort(global_scores, descending)[:N * 0.2]

    // Quick global align to frame with best score
    best_frame = frames[top_indices[0]]
    global_offsets = [AlignmentOffset(0, 0)]  // best frame has zero offset
    for each frame (except best):
        offset = phase_correlate(best_frame, frame)
        global_offsets.append(offset)

    // Compute mean reference from top frames
    mean_ref = zeros(H, W)
    for idx in top_indices:
        mean_ref += shift(frames[idx], global_offsets[idx])
    mean_ref /= len(top_indices)

    // Optionally: re-align all frames vs. mean_ref for better offsets
    for i in 0..N:
        global_offsets[i] = phase_correlate(mean_ref, frames[i])

    // === Phase 2: AP Grid Construction ===

    grid = []
    stride = config.ap_size / 2  // 50% overlap
    for cy in range(stride, H - stride, stride):
        for cx in range(stride, W - stride, stride):
            region = extract(mean_ref, cy, cx, config.ap_size)
            if mean(region) >= config.min_brightness:
                if gradient_score(region) >= config.min_contrast:
                    grid.append(AP(cy, cx))

    // === Phase 3: Per-AP Quality Scoring ===

    // Per-AP quality matrix: quality[ap][frame] = score
    quality = zeros(len(grid), N)
    for k in 0..N:
        frame = shift(frames[k], global_offsets[k])  // or carry offset
        for j, ap in enumerate(grid):
            region = extract(frame, ap.cy, ap.cx, config.ap_size)
            quality[j][k] = laplacian_variance(region)

    // Per-AP frame selection
    keep_count = ceil(N * config.select_percentage)
    ap_selections = []
    for j in 0..len(grid):
        ranked = argsort(quality[j], descending)
        ap_selections.append(ranked[:keep_count])

    // === Phase 4: Per-AP Local Alignment + Stacking ===

    search_size = config.ap_size + 2 * config.search_radius
    ap_stacks = []

    for j, ap in enumerate(grid):  // parallelizable
        ref_region = extract(mean_ref, ap.cy, ap.cx, search_size)

        patches = []
        weights = []

        for k in ap_selections[j]:
            // Extract with global offset applied
            tgt_region = extract_shifted(frames[k], ap.cy, ap.cx,
                                         search_size, global_offsets[k])

            // Local sub-pixel alignment
            local_offset = phase_correlate(ref_region, tgt_region)

            // Validate alignment quality
            if correlation_confidence(ref_region, tgt_region) < config.min_confidence:
                continue

            // Extract final patch with combined offset
            combined_offset = global_offsets[k] + local_offset
            patch = extract_shifted(frames[k], ap.cy, ap.cx,
                                    config.ap_size, combined_offset)

            patches.append(patch)
            weights.append(quality[j][k])

        // Quality-weighted stack
        stacked = weighted_mean(patches, weights)
        ap_stacks.append((ap, stacked))

    // === Phase 5: Blending ===

    output = zeros(H, W)
    weight_sum = zeros(H, W)

    for (ap, patch) in ap_stacks:
        for dr in 0..config.ap_size:
            for dc in 0..config.ap_size:
                img_r = ap.cy - config.ap_size/2 + dr
                img_c = ap.cx - config.ap_size/2 + dc

                w = hann_2d(dr, dc, config.ap_size)
                output[img_r, img_c] += w * patch[dr, dc]
                weight_sum[img_r, img_c] += w

    // Normalize
    output /= weight_sum  // (where weight_sum > 0)

    return output


function hann_2d(row, col, size):
    wy = 0.5 * (1 - cos(2 * pi * row / size))
    wx = 0.5 * (1 - cos(2 * pi * col / size))
    return wy * wx
```

### 13.2 Surface-Model Warping Alternative

```
function surface_model_stack(frames, config):
    // Phases 1-3 same as above
    ...

    // Phase 4: Compute shift field per frame
    for k in 0..N:
        shift_field_x = zeros(H, W)
        shift_field_y = zeros(H, W)
        shift_weights = zeros(H, W)

        for j, ap in enumerate(grid):
            if k not in ap_selections[j]:
                continue

            local_offset = compute_local_offset(frames[k], mean_ref, ap,
                                                 global_offsets[k], config)

            // Spread this shift with Gaussian weight around the AP
            for dy in range(-spread, spread):
                for dx in range(-spread, spread):
                    r, c = ap.cy + dy, ap.cx + dx
                    w = gaussian(dy, dx, sigma=config.ap_size/2)
                    shift_field_x[r, c] += w * (global_offsets[k].dx + local_offset.dx)
                    shift_field_y[r, c] += w * (global_offsets[k].dy + local_offset.dy)
                    shift_weights[r, c] += w

        // Normalize shift field
        shift_field_x /= shift_weights
        shift_field_y /= shift_weights

        // Warp frame using interpolated shift field
        warped = zeros(H, W)
        for r in 0..H:
            for c in 0..W:
                src_r = r - shift_field_y[r, c]
                src_c = c - shift_field_x[r, c]
                warped[r, c] = bilinear_sample(frames[k], src_r, src_c)

        // Accumulate with quality weight
        output += quality_weight[k] * warped
        total_weight += quality_weight[k]

    return output / total_weight
```

---

## 14. References

### Scientific Papers

1. Law, N.M., Mackay, C.D., Baldwin, J.E. (2006). "Lucky Imaging: High Angular Resolution Imaging in the Visible from the Ground." *Astronomy & Astrophysics*, 446, 739-745. [A&A PDF](https://www.aanda.org/articles/aa/pdf/2006/05/aa3695-05.pdf)

2. Baldwin, J.E., et al. (2008). "The point spread function in Lucky Imaging and variations in seeing on short timescales." *Astronomy & Astrophysics*. [A&A PDF](https://www.aanda.org/articles/aa/pdf/2008/11/aa9214-07.pdf)

3. Guizar-Sicairos, M., Thurman, S.T., Fienup, J.R. (2008). "Efficient subpixel image registration algorithms." *Optics Letters*, 33(2), 156-158. [PubMed](https://pubmed.ncbi.nlm.nih.gov/18197224/)

4. Vorontsov, M.A., Carhart, G.W. (2001). "Anisoplanatic imaging through turbulent media: image recovery by local information fusion from a set of short-exposure images." *JOSA A*, 18(6), 1312-1324. [Optica](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-18-6-1312)

5. Deng, H., et al. (2015). "Review of Image Quality Measures for Solar Imaging." *Solar Physics*. [Springer](https://link.springer.com/article/10.1007/s11207-017-1211-3)

6. Fried, D.L. (1966). "Optical Resolution Through a Randomly Inhomogeneous Medium for Very Long and Very Short Exposures." *JOSA*, 56(10), 1372-1379.

7. Mackay, C.D., et al. (2013). "High-Efficiency Lucky Imaging." *Monthly Notices of the Royal Astronomical Society*, 432(1), 702-710. [arXiv:1303.5108](https://arxiv.org/abs/1303.5108)

8. Peck, C.L. et al. (2020). "Introducing Hann windows for reducing edge-effects in patch-based image segmentation." *PLOS ONE*. [arXiv:1910.07831](https://arxiv.org/abs/1910.07831)

9. Foroosh, H., Zerubia, J., Berthod, M. (2002). "Extension of Phase Correlation to Subpixel Registration." *IEEE Transactions on Image Processing*. [UCF PDF](https://www.cs.ucf.edu/~foroosh/subreg.pdf)

10. He, L., et al. (2020). "Iterative Phase Correlation Algorithm for High-precision Subpixel Image Registration." *Astrophysical Journal Supplement Series*, 247, 8. [ADS](https://ui.adsabs.harvard.edu/abs/2020ApJS..247....8H/abstract)

### Software and Tools

11. Kraaikamp, E. "AutoStakkert! Stacking Software." [autostakkert.com](https://www.autostakkert.com/)

12. Hempel, R. "PlanetarySystemStacker." [GitHub](https://github.com/Rolf-Hempel/PlanetarySystemStacker)

13. Kasteleijn, W. "LuckyStackWorker." [GitHub](https://github.com/wkasteleijn/luckystackworker)

14. Schmid, G. "WinJUPOS." Planetary measurement and derotation software.

15. PIPP - Planetary Imaging PreProcessor. [astrophotography-telescope.com](https://astrophotography-telescope.com/download-autostakkert-2-and-3-free-planetary-stacking-software/)

### Community Resources

16. [Autostakkert!2: comparing alignment point sizes for Jupiter](https://www.planetary-astronomy-and-imaging.com/en/autostakkert2-comparing-alignment-point-sizes-for-jupiter/)

17. [Cloudy Nights: Autostakkert alignment points question](https://www.cloudynights.com/forums/topic/562976-autostakkert-alignment-points-question/)

18. [Cloudy Nights: Enough alignment points?](https://www.cloudynights.com/topic/847985-enough-alignment-points/)

19. [Cloudy Nights: PlanetarySystemStacker discussion](https://www.cloudynights.com/topic/645890-new-stacking-software-project-planetarysystemstacker/)

20. [High-Resolution Planetary Imaging Guide, Part 4](https://rouzastro.com/high-resolution-planetary-imaging-guide-part-4-imaging-software-and-processing/)

### Atmospheric Optics

21. [Fried parameter - Wikipedia](https://en.wikipedia.org/wiki/Fried_parameter)

22. [Isoplanatic patch - Wikipedia](https://en.wikipedia.org/wiki/Isoplanatic_patch)

23. [AO Tutorial: Turbulence](http://www.ctio.noao.edu/~atokovin/tutorial/part1/turb.html)

24. [Introduction to Astronomical Seeing](https://www.innovationsforesight.com/education/astronomical-seeing-tutorial/)

### Image Blending

25. Burt, P.J., Adelson, E.H. (1983). "A Multiresolution Spline With Application to Image Mosaics." *ACM Transactions on Graphics*.

26. [Image Alignment and Stitching: A Tutorial](https://pages.cs.wisc.edu/~dyer/cs534/papers/szeliski-alignment-tutorial.pdf) - Richard Szeliski

27. [Stanford: Stitching and Blending Lecture](https://web.stanford.edu/class/cs231m/lectures/lecture-5-stitching-blending.pdf)
