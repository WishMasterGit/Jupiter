# Auto-Cropping Planetary Video: A Deep Research Survey

## Abstract

Auto-cropping — the automatic detection, centering, and tight framing of a planetary disc within video frames — is a critical preprocessing step in the lucky imaging pipeline. A good auto-crop dramatically reduces processing time in downstream stages (alignment, stacking, sharpening) by eliminating unnecessary sky background, and ensures the planet is consistently centered across all frames. This article surveys the techniques used by leading astrophotography software, academic approaches to disc detection and centroiding, the unique challenges of planetary video data, and synthesizes these findings into a recommended algorithm for a robust implementation.

---

## 1. Why the Current Implementation Fails

The v1 autocrop in Jupiter (`io/autocrop.rs`) follows a three-step approach:

1. **Sample frames** — read 1 frame (the middle frame) or average N evenly-spaced frames
2. **Threshold** — compute a brightness threshold (mean + 2*sigma, Otsu, or fixed)
3. **Bounding box** — find the min/max row/col of all pixels above the threshold, add padding

This approach has several fundamental problems that cause failures on real-world data:

### 1.1 No Noise Rejection

The bounding box is computed over **every pixel** exceeding the threshold. A single hot pixel, cosmic ray, or sensor noise spike in a corner of the frame expands the bounding box to include that corner, resulting in a crop that is barely smaller than the original frame — or even the full frame. Real planetary cameras (ZWO ASI, QHY) routinely produce hot pixels, especially in long captures.

### 1.2 No Morphological Processing

There is no erosion, dilation, opening, or closing applied to the binary mask. Without morphological opening (erosion followed by dilation), isolated noise pixels are treated identically to the planet disc. Professional implementations universally apply at least one round of morphological cleanup.

### 1.3 Single-Frame Dependence

The default samples only the middle frame. If that frame happens to have poor seeing, sensor artifacts, or the planet partially out of frame, detection fails entirely. Even when sampling multiple frames, they are **averaged** — this dilutes faint features and creates a blurred composite that may not represent any real frame well.

### 1.4 No Drift Awareness

Planetary video captures routinely span 2-5 minutes. During this time, the planet drifts across the sensor due to imperfect tracking. The v1 implementation analyzes a static snapshot and produces a single crop rectangle. If the planet drifts 50 pixels between the first and last frame, the crop from the middle frame will clip the planet in early or late frames.

### 1.5 Threshold Sensitivity

The `mean + 2*sigma` default threshold depends on the brightness distribution of the entire frame. For a small planet on a large dark background, the mean is very low and the stddev is dominated by the dark sky, producing a threshold that works. But for a large planet filling much of the frame, or for data with sky gradients (dawn/dusk captures), the threshold can be completely wrong.

### 1.6 No Detection Validation

There is no sanity checking on the detected region. The algorithm does not verify that:
- The detected region is roughly circular (as expected for a planet)
- The detected area is within a plausible size range
- The aspect ratio makes sense for the target (circular for most planets, wider for Saturn)
- The detection is consistent across multiple frames

---

## 2. How Existing Software Solves This Problem

### 2.1 AutoStakkert (AS!3 / AS!4)

AutoStakkert is the gold standard for planetary stacking and uses a **threshold-based center-of-gravity (COG)** approach:

- **Planet Mode (COG)**: Automatically determines a brightness threshold separating the planet from the background. Then computes the intensity-weighted centroid (center of gravity) of all bright pixels in each frame independently. The centroid gives a sub-pixel planet center position.
- **Dynamic Background**: An option that recomputes the background level per-frame, handling captures where sky brightness changes (twilight, passing clouds). This prevents the threshold from becoming invalid mid-capture.
- **Surface Mode**: Alternative for lunar/solar close-ups where the entire frame is bright surface. Uses feature cross-correlation instead of COG.
- **Integration**: AutoStakkert does NOT use a separate crop step — it integrates detection and stabilization into its analysis phase. The planet center is tracked continuously and alignment points are placed relative to it.

**Key insight**: AutoStakkert processes every frame independently for centroid computation, not just a sample. This provides a complete drift trajectory and allows outlier rejection.

### 2.2 PIPP (Planetary Imaging PreProcessor)

PIPP is the most directly relevant reference — it is specifically a preprocessing tool for planetary video:

1. **Object Detection**: Each frame is analyzed independently. An "Auto Object Detection Threshold" computes a brightness threshold from the histogram. The result is a binary mask.
2. **Size Filtering**: The detected object must meet a configurable minimum size (in pixels). Small noise blobs are rejected.
3. **Frame Rejection**: Frames where the detected object **touches the frame border** are automatically rejected. This handles partially-visible discs cleanly.
4. **Centroiding**: The centroid of the valid bright region gives the planet center.
5. **Centering**: Each frame is translated so the planet centroid is at the frame center.
6. **Fixed Crop**: The user specifies a crop size in pixels. Each frame is cropped to this fixed size around the centered planet.
7. **Quality Sorting**: Optionally frames are sorted by quality metric and worst frames discarded.
8. **Debug Mode**: An "Object Find Debug" mode draws bounding rectangles for troubleshooting detection failures.

**Key insight**: PIPP processes per-frame (not averaged) and combines size filtering with border rejection for robustness.

### 2.3 Siril

Siril offers limited planetary auto-cropping:

- **Manual ROI**: The user must manually draw a selection box around the planet. Siril does not automatically detect the planetary disc.
- **Cross-Correlation**: Within the user-defined ROI, Siril uses cross-correlation for alignment.
- **Crop Modes**: "Maximum (bounding box)" adds borders to prevent clipping; "Minimum (common area)" crops to the intersection of all frames.
- **Limitation**: No automatic detection — the user is the detection algorithm.

### 2.4 WinJUPOS

WinJUPOS focuses on derotation rather than preprocessing, but its detection is relevant:

- **Automatic Outline Detection**: Detects the planet's limb and overlays an alignment mask showing the disc outline, equator, and pole markers.
- **Limb Darkening Compensation**: A configurable LD parameter helps define the planet's edge more precisely. Gas giants (Jupiter, Saturn) have significant limb darkening that causes simple thresholds to underestimate the disc size.
- **Ephemeris-Guided**: Uses time metadata and planetary ephemeris data to predict expected disc size and orientation, constraining the detection problem.

**Key insight**: Limb darkening compensation is critical. The edge of a gas giant can be 30-50% dimmer than center — a simple threshold misses the limb.

### 2.5 FireCapture

FireCapture performs real-time planet detection during capture:

- **Scanline Limb Detection**: Detects "lines of pixels which are more than 10 grey levels brighter than the background" (100 grey levels in 16-bit mode). This is an edge-based approach, not threshold-area.
- **Auto-center ROI**: Continuously tracks the planet and repositions the camera's ROI if the planet drifts toward the edges.
- **Minimum Border Distance**: A configurable trigger zone. If the planet's limb enters this border zone, the ROI is repositioned.
- **CofG Locking**: Center-of-gravity tracking within the ROI for continuous monitoring.

**Key insight**: FireCapture uses edge/gradient detection rather than area thresholding, making it more robust to limb darkening and background gradients.

---

## 3. Technical Approaches to Planetary Disc Detection

### 3.1 Otsu Thresholding

Otsu's method finds the optimal global threshold that maximizes between-class variance, assuming a bimodal histogram (sky vs planet).

**Algorithm**:
```
For each candidate threshold t:
    weight_bg = fraction of pixels below t
    weight_fg = fraction of pixels above t
    between_variance = weight_bg * weight_fg * (mean_bg - mean_fg)^2
Select t that maximizes between_variance
```

**Strengths**: Parameter-free, fast (single histogram pass), works well for clear sky/planet separation.

**Weaknesses**: Assumes bimodal distribution — fails for crescent phases, very small planets (tiny foreground peak), or background gradients. The threshold can be too low when there's significant mid-tone structure (cloud bands on Jupiter add intermediate brightness values).

**Verdict**: Good as a starting point but insufficient alone. Should be combined with morphological post-processing.

### 3.2 Connected Component Analysis (CCA)

After thresholding, CCA labels contiguous regions of foreground pixels:

1. Threshold image to binary
2. Two-pass labeling assigns unique IDs to connected regions
3. For each component, compute area, bounding box, centroid, and aspect ratio
4. Select the component with the **largest area** as the planetary disc

**Strengths**: Naturally rejects isolated noise pixels (they form tiny components). The bounding box of the largest component gives a clean crop rectangle. Centroid of the component gives a robust center position.

**Weaknesses**: Requires a reasonable threshold as input. Can merge the planet with a nearby star or satellite if they're connected at the threshold level.

**Verdict**: This is the missing piece in the v1 implementation. CCA after thresholding would solve the hot-pixel/noise problem immediately.

### 3.3 Intensity-Weighted Centroid (IWC)

Image moments provide the mathematical center of a brightness distribution:

```
M_00 = sum(I(x,y))                 -- total intensity
cx = sum(x * I(x,y)) / M_00        -- centroid x
cy = sum(y * I(x,y)) / M_00        -- centroid y
```

Using squared intensity gives more weight to the brightest regions:
```
cx = sum(x * I(x,y)^2) / sum(I(x,y)^2)
```

**Strengths**: Extremely fast (O(h*w) single pass), naturally produces sub-pixel coordinates, immune to exact threshold choice (as long as background is subtracted).

**Weaknesses**: Sensitive to background — must subtract background first or apply threshold mask. Biased by overexposed regions. For crescent phases, finds the center of the bright crescent rather than the geometric disc center.

**Verdict**: The method used by AutoStakkert (COG mode) and FireCapture. Should be the primary centroiding method after mask creation.

### 3.4 Hough Circle Transform

Searches a 3D parameter space (cx, cy, r) for circular features in edge-detected images:

1. Apply Canny edge detection
2. For each edge pixel, vote in accumulator for all circles passing through it
3. Peaks in accumulator correspond to detected circles

**Strengths**: Robust to partial occlusion (works even with partial limb visibility). Directly recovers both center and radius. Handles noise well due to voting.

**Weaknesses**: Computationally expensive for large images with wide radius ranges. Assumes **circular** geometry — fails for Saturn's rings and oblate planets. The three-dimensional search space (cx, cy, r) can be slow.

**Verdict**: Overkill for a preprocessing crop (CCA is simpler and sufficient), but useful for precise disc geometry when needed.

### 3.5 Edge Detection + Ellipse Fitting

Used extensively in spacecraft optical navigation:

1. Apply Canny/Sobel/Prewitt edge detection to find limb edges
2. Remove false edges using geometric constraints
3. Fit an ellipse to remaining edge points (least-squares or RANSAC)
4. Ellipse center and semi-axes give disc position and size

**Sub-pixel variant (Prewitt-Zernike moments)**: Achieves ~0.14 pixel RMS edge accuracy versus ~0.3 for polynomial interpolation.

**Strengths**: Handles non-circular shapes (Saturn, oblate Jupiter). Works with partial discs. Most accurate method documented.

**Weaknesses**: Complex to implement. Sensitive to surface features being detected as edges (Jupiter's cloud bands). Requires careful edge filtering.

**Verdict**: Best approach for precision work (derotation, measurement), but over-engineered for a preprocessing crop.

### 3.6 Adaptive Thresholding

Computes local thresholds based on neighborhood statistics rather than a single global value:

- Mean of local window
- Gaussian-weighted local mean
- Sauvola method: `T(x,y) = mean * (1 + k * (stddev/R - 1))`

**Strengths**: Handles background gradients (dawn/dusk captures, light pollution, vignetting).

**Weaknesses**: More computationally expensive. Can produce noisy masks near the planet limb where brightness transitions.

**Verdict**: Useful as a fallback when global thresholding fails due to gradients, but adds complexity.

### 3.7 Gaussian Blur Pre-Processing

Before any thresholding, applying a Gaussian blur (sigma 2-5 pixels) serves dual purposes:

1. **Noise suppression**: Hot pixels and read noise are smoothed away
2. **Object consolidation**: Small gaps in the planet disc (dark surface features) are filled in

This is perhaps the single most impactful improvement that can be made to a thresholding pipeline. A 5x5 Gaussian blur before Otsu thresholding dramatically improves detection robustness.

---

## 4. Challenges Specific to Planetary Video

### 4.1 Atmospheric Seeing

Turbulence causes the planet to jitter 5-50 pixels frame-to-frame. The apparent disc size can also vary as seeing cells act as weak lenses. Detection must be per-frame, and the resulting centroid trajectory must be smoothed to separate genuine drift from seeing jitter.

### 4.2 Limb Darkening

Gas giants exhibit significant limb darkening — the disc edges can be 30-50% dimmer than center. A threshold set to detect the bright center will miss the limb, underestimating disc diameter by 10-20%. Solutions:
- Use a lower threshold (risks including background)
- Use edge detection instead of thresholding
- Apply a known limb-darkening model to predict the threshold falloff
- Use the centroid + known angular diameter to estimate disc size independently of threshold

### 4.3 Overexposed Regions

Common in planetary imaging (Jupiter's equatorial zone, lunar highlights). Overexposure clips pixel values, creating flat-topped intensity profiles that distort centroid calculations. Solutions:
- Ignore saturated pixels in centroid computation
- Use edge-based detection (limb is rarely overexposed)
- Weight by I^2 rather than I (reduces impact of clipped regions)

### 4.4 Saturn's Rings

Saturn requires special handling:
- The rings extend 2-2.5x beyond the disc diameter
- The ring system creates a horizontally elongated bounding box
- Ring tilt angle varies year to year — aspect ratio is not constant
- The Cassini Division and ring gaps can cause the bounding box to fragment if threshold is too high
- **Solution**: Use the bounding box of the largest connected component directly (which naturally encompasses rings) rather than assuming circular geometry

### 4.5 Crescent Phases (Venus, Mercury)

The illuminated area is a thin arc, not a disc:
- Center-of-gravity finds the centroid of the crescent, offset from the geometric center
- For cropping purposes, centering on the bright crescent is usually acceptable
- For precise geometry, limb arc fitting is required
- The bounding box of the crescent is much smaller than the full disc, potentially clipping the dark side

### 4.6 Drift Across the Video

Typical drift during a 3-minute capture can be 20-100+ pixels. A single crop rectangle from one frame's detection will clip the planet in other frames. Solutions:
- Compute centroid per-frame and use the trajectory envelope for crop sizing
- Fit a drift model (linear or quadratic) and size the crop to encompass the full trajectory plus margin
- Better: center each frame independently (PIPP approach) so the crop rectangle is constant

### 4.7 Background Gradients

Dawn/dusk captures, moonlit skies, and sensor vignetting create non-uniform backgrounds. Global thresholding fails because the "background" brightness varies across the frame. Solutions:
- Local background estimation (median filter with large kernel, then subtract)
- Adaptive thresholding
- Edge-based detection (gradient operations naturally handle slow gradients)

---

## 5. Best Practices from the Community

### 5.1 Multi-Frame Analysis

Processing many frames rather than a single sample is universally recommended:
- Compute centroid per-frame independently
- Use temporal median of centroids to reject outlier frames (failed detection)
- Fit a smooth drift trajectory to the cleaned centroid time series
- The envelope of centroid positions determines the required crop margin

### 5.2 Robust Statistics

For centroid trajectory analysis:
- **Median Absolute Deviation (MAD)**: More robust than standard deviation. `MAD = median(|x_i - median(x)|)`. The robust scale estimate is `1.4826 * MAD`.
- **Sigma clipping**: Iteratively reject centroids deviating more than 2.5-3 sigma from the median. 2-3 iterations usually converge.
- **Physical plausibility**: Flag frames where centroid jumps more than a physically reasonable distance from the previous frame (e.g., >20 pixels at 30fps).

### 5.3 Crop Size Strategy

Several approaches in practice:
- **Proportional padding**: 10-20% of detected disc diameter. Scales naturally with planet size.
- **Worst-case envelope**: Crop sized to encompass the planet across ALL frames plus margin. Guarantees no clipping.
- **Power-of-two**: Round up to nearest power of 2 for FFT efficiency in downstream stacking.
- **Fixed aspect ratio**: Square crops simplify downstream processing.

### 5.4 Frame Rejection

Reject frames before cropping if:
- No object detected above minimum size
- Detected object touches frame border (partial visibility)
- Centroid position is an outlier relative to drift trajectory
- Detected area is anomalously small or large (suggesting detection failure)

---

## 6. Recommended Algorithm for Jupiter v2

Based on this research, the following pipeline balances robustness with implementation simplicity. It draws primarily from the PIPP and AutoStakkert approaches, which are proven on real planetary data.

### Phase 1: Multi-Frame Sampling and Detection

```
1. Sample K frames evenly across the video (K = 20-50, configurable)
2. For each sampled frame:
   a. Apply Gaussian blur (sigma = 2-3 pixels) for noise suppression
   b. Compute Otsu threshold on the blurred frame
   c. Apply morphological opening (3x3 kernel) to remove hot pixels
   d. Connected component analysis:
      - Label all connected regions
      - For each region: compute area, bounding box, centroid
      - Select the largest region as candidate planet
   e. Reject frame if:
      - No region exceeds minimum area (e.g., 100 pixels)
      - Largest region touches frame border
   f. Compute intensity-weighted centroid of the candidate region
      on the original (unblurred) frame with background subtraction:
      bg = median of pixels in the ~10px border strip
      I_corrected = max(0, I - bg)
      cx = sum(x * I_corrected * mask) / sum(I_corrected * mask)
      cy = sum(y * I_corrected * mask) / sum(I_corrected * mask)
   g. Record: (frame_index, cx, cy, bbox_width, bbox_height, area)
```

### Phase 2: Temporal Filtering and Drift Model

```
1. Collect all per-frame centroids from Phase 1
2. Compute median centroid position (cx_med, cy_med)
3. Apply sigma-clipping (2.5-sigma, 3 iterations) to reject outlier centroids
4. From surviving centroids:
   a. If < 50% survived, warn user and fall back to median position
   b. Fit linear drift model: cx(t) = a*t + b, cy(t) = c*t + d
   c. Compute residuals — these represent seeing jitter
```

### Phase 3: Crop Size Determination

```
1. Compute disc diameter:
   - Take median of detected bbox dimensions across valid frames
   - Use max(median_width, median_height) as diameter
2. Compute drift envelope:
   - drift_range_x = max(cx) - min(cx) across valid frames
   - drift_range_y = max(cy) - min(cy) across valid frames
3. Required crop size:
   crop_w = diameter + drift_range_x + 2 * padding
   crop_h = diameter + drift_range_y + 2 * padding
   where padding = diameter * padding_fraction (default 15%)
4. Optionally round up to next multiple of 32 (or power of 2) for FFT efficiency
5. Make square: size = max(crop_w, crop_h) for simplicity
```

### Phase 4: Per-Frame Centering and Cropping

Two modes depending on the use case:

**Mode A — Static Crop (simpler, for subsequent alignment)**:
Use the median centroid as the crop center. Apply the same crop rectangle to every frame. The downstream alignment stage handles per-frame registration. This is the simpler approach and appropriate when the pipeline already has alignment.

**Mode B — Dynamic Crop (PIPP-style, for maximum cropping)**:
Use the drift model to predict the planet center in each frame. Translate each frame to center the planet, then apply a fixed-size crop. This produces tighter crops but requires per-frame translation.

### Implementation Notes

- **Gaussian blur**: Already have `gaussian_blur` in `filters/gaussian_blur.rs` — reuse it
- **Connected components**: Implement a simple two-pass labeling (flood-fill variant) — no external dependency needed, ~50 lines of code
- **Morphological opening**: Implement as erosion then dilation on binary mask — ~30 lines
- **IWC centroid**: Already have centroid computation in `align/centroid.rs` (`compute_centroid_array`) — adapt it
- **Background estimation**: Median of border pixels, straightforward
- **Sigma clipping**: Already have sigma-clip logic in `stack/sigma_clip.rs` — adapt for 1D centroid data

### Fallback Behavior

If the multi-frame approach fails (< 3 valid detections):
1. Fall back to single-frame detection on a median-combined image of 5 center frames
2. If that fails, try progressively lower thresholds (Otsu * 0.8, then * 0.6)
3. If all detection fails, report error with diagnostic info (histogram statistics, tried thresholds)

---

## 7. Comparison of Approaches

| Approach | Noise Robust | Handles Drift | Handles Gradients | Saturn Rings | Crescent | Complexity |
|---|---|---|---|---|---|---|
| **v1 (current)** | Poor | No | No | Partial | Yes | Low |
| **Threshold + CCA** | Good | No | No | Yes | Yes | Low |
| **+ Gaussian blur** | Very Good | No | No | Yes | Yes | Low |
| **+ Multi-frame** | Very Good | Yes | No | Yes | Yes | Medium |
| **+ Background sub** | Very Good | Yes | Yes | Yes | Yes | Medium |
| **Hough Circle** | Good | Per-frame | No | No | No | High |
| **Edge + Ellipse Fit** | Good | Per-frame | Yes | Partial | Partial | High |
| **Full pipeline (v2)** | Very Good | Yes | Yes | Yes | Yes | Medium |

The recommended v2 pipeline sits in the sweet spot of robustness vs complexity. It solves all the failure modes of v1 without requiring complex geometric fitting.

---

## 8. References

### Software
- [AutoStakkert](https://www.autostakkert.com/) — Planet/COG mode, dynamic background
- [PIPP (Planetary Imaging PreProcessor)](https://sites.google.com/site/astropipp/) — Dedicated planetary preprocessing
- [Siril](https://siril.readthedocs.io/en/latest/preprocessing/registration.html) — Manual ROI, cross-correlation alignment
- [WinJUPOS](https://grischa-hahn.hier-im-netz.de/astro/winjupos/) — Limb detection, derotation, LD compensation
- [FireCapture](https://www.firecapture.de/) — Real-time scanline limb detection, CofG tracking

### Academic
- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Trans. Sys. Man. Cyber.* 9(1):62-66
- Guio, P. & Achilleos, N. (2010). "A New Method for Determining the Geometry of Planetary Images Using a Voronoi-Based Segmentation." [arXiv:1010.1213](https://arxiv.org/abs/1010.1213)
- Christian, J. (2017). "Accurate Planetary Limb Localization for Image-Based Spacecraft Navigation." *AIAA Journal of Guidance, Control, and Dynamics*. [DOI:10.2514/1.A33692](https://arc.aiaa.org/doi/10.2514/1.A33692)
- Owen, W. (2011). "Image Processing Algorithms for Deep-Space Autonomous Optical Navigation." *Journal of Navigation*
- Ma, X. et al. "High-accuracy Extraction Algorithm of Planet Centroid Image in Deep-space Autonomous Optical Navigation." *Journal of Navigation*, Cambridge
- Ma, X. et al. "Robust Centroid Extraction Using the Hybrid Genetic Algorithm with Applications to Planetary Optical Navigation." *Journal of Navigation*, Cambridge
- Hollitt, C. (2012). "Feature Detection in Radio Astronomy using the Circle Hough Transform." [arXiv:1204.0382](https://arxiv.org/abs/1204.0382)

### Community
- [CloudyNights: To PIPP or not to PIPP](https://www.cloudynights.com/topic/791394-to-pipp-or-not-to-pipp-that-is-the-question/) — Community discussion on preprocessing necessity
- [CloudyNights: AutoStakkert Cropping](https://www.cloudynights.com/topic/450978-autostakkert2-cropping-a-video/) — AS! centering behavior
- [lost-infinity.com: Night Sky Image Processing](https://www.lost-infinity.com/night-sky-image-processing-part-2-image-binarization-using-the-otsu-thresholding-algorithm/) — Otsu implementation for astronomy
- [lost-infinity.com: Sub-pixel Centroid](https://www.lost-infinity.com/night-sky-image-processing-part-4-calculate-the-star-centroid-with-sub-pixel-accuracy/) — Iterative sub-pixel centroiding

### Algorithms
- [Connected Component Analysis (Purdue)](https://engineering.purdue.edu/~bouman/ece637/notes/pdf/ConnectComp.pdf)
- [Hough Circle Transform (OpenCV)](https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html)
- [Robust Sigma Clipping (Astropy)](https://docs.astropy.org/en/stable/stats/robust.html)
- [Spacecraft Optical Navigation (JPL Monograph)](https://descanso.jpl.nasa.gov/monograph/series15/Spacecraft-Optical-Navigation.pdf)
