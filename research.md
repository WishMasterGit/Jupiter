# Planetary Image Processing: Tools, Algorithms, and Techniques

A comprehensive survey of the current landscape for processing planetary imagery captured through earth-based telescopes, with a focus on eliminating atmospheric turbulence effects and achieving higher resolution.

---

## Table of Contents

1. [The Problem: Atmospheric Seeing](#the-problem-atmospheric-seeing)
2. [Existing Software Tools](#existing-software-tools)
3. [The Processing Pipeline](#the-processing-pipeline)
4. [Core Algorithms](#core-algorithms)
   - [Lucky Imaging](#lucky-imaging)
   - [Frame Quality Assessment](#frame-quality-assessment)
   - [Frame Alignment and Registration](#frame-alignment-and-registration)
   - [Image Stacking](#image-stacking)
   - [Wavelet Sharpening](#wavelet-sharpening)
   - [Deconvolution](#deconvolution)
   - [Super-Resolution](#super-resolution)
5. [Atmospheric Turbulence Theory](#atmospheric-turbulence-theory)
6. [Rust Ecosystem for Astronomy](#rust-ecosystem-for-astronomy)
7. [Sources and References](#sources-and-references)

---

## The Problem: Atmospheric Seeing

Earth's atmosphere is in constant turbulent motion. Pockets of air at different temperatures create regions with different refractive indices, bending light from celestial objects unpredictably. This phenomenon, known as **astronomical seeing**, limits the angular resolution of ground-based telescopes far below their theoretical diffraction limit.

For a typical amateur telescope with a 200mm aperture, the diffraction-limited resolution at 550nm wavelength is about 0.56 arcseconds. However, typical atmospheric seeing limits resolution to 1-3 arcseconds — meaning the atmosphere wastes most of the telescope's resolving power.

The key insight that enables high-resolution planetary imaging from the ground is that atmospheric turbulence is **not constant**. Over short timescales (milliseconds), there are brief moments when the air above the telescope is relatively stable. By capturing thousands of short-exposure frames and selecting only the sharpest ones, amateur astronomers routinely achieve results that approach the diffraction limit of their telescopes.

---

## Existing Software Tools

### AutoStakkert! — The Industry Standard for Stacking

AutoStakkert is the most widely used tool for planetary image stacking. It automatically analyzes, aligns, and stacks planetary video frames using lucky imaging techniques.

**Key capabilities:**
- Multi-point alignment system that tracks surface features across the entire planetary disk
- Quality assessment with automatic selection of best frames (typically top 10-40%)
- Handles SER, AVI, TIFF, FITS, BMP formats
- Multicore processing support

**Strengths:** Superior alignment algorithms, very fast processing, the de facto standard in the planetary imaging community.

**Limitations:** Windows-only, closed source (freeware), limited post-processing — users typically export to RegiStax for sharpening.

### RegiStax — Wavelet Sharpening Pioneer

RegiStax is legendary for its wavelet processing capabilities. While originally a complete stacking solution, it is now primarily used for its unmatched wavelet sharpening.

**Key capabilities:**
- 6-layer wavelet processing with independent control per layer
- Each layer can be sharpened or denoised independently
- Linked wavelet layers for alternative sharpening approaches

**Strengths:** Exceptional wavelet sharpening that reveals planetary detail invisible in the raw stack.

**Limitations:** No longer actively maintained (considered abandonware), Windows-only, closed source.

### Siril — The Open Source Contender

Siril is the most complete open-source astronomical image processing suite, actively developed and cross-platform.

**Key capabilities:**
- Native SER file format support
- Advanced registration methods adapted to content type
- Multiple stacking algorithms
- Can process 60,000+ frames using all CPU cores
- Version 1.4.1 released January 2026

**Strengths:** Actively developed, cross-platform (Linux, macOS, Windows), free and open source (GNU GPL).

**Limitations:** Steeper learning curve, wavelet sharpening not as refined as RegiStax.

### Planetary System Stacker (PSS) — Python Open Source

A complete lucky imaging workflow comparable to AutoStakkert, built in Python.

**Key capabilities:**
- Quality matching or exceeding AutoStakkert!3
- Modern GUI using Qt5
- Array operations via OpenCV and numpy for speed
- Optimized for extended objects (Moon, Sun)

**Strengths:** Open source, platform-independent, active development.

**Limitations:** Python performance overhead, smaller community.

### ImPPG — GPU-Accelerated Post-Processing

Open-source image post-processor specializing in deconvolution.

**Key capabilities:**
- Lucy-Richardson deconvolution with Gaussian kernel
- Unsharp masking, brightness normalization, tone curves
- GPU acceleration via OpenGL (several times faster than CPU)

**Strengths:** GPU acceleration, dedicated deconvolution, open source (GPL v3+).

### AstroSurface — All-in-One

A comprehensive planetary processing suite in a single package.

**Key capabilities:**
- Complete workflow: alignment, stacking, wavelet sharpening, HDR
- White balance and color corrections
- Supports SER, AVI, FITS, and common image formats
- Version 4 released January 2025

**Strengths:** One tool for the entire workflow.

**Limitations:** Windows-only, closed source (free).

### Capture Software

Before processing, frames must be captured:

- **FireCapture** — Multi-platform (Java), free, automatic planet tracking/guiding
- **SharpCap** — Wide camera support, frequent updates, easy-to-use (GBP 10/year)

Both save to **SER format** — uncompressed raw data with >8-bit depth support, which is the preferred format over AVI for planetary imaging.

---

## The Processing Pipeline

The standard planetary imaging workflow follows a well-established sequence. Each stage builds on the previous one:

```
Capture (SER video)
    |
    v
Pre-processing
  - Stabilize/center planet in frame
  - Crop to region of interest
  - Debayer (if color camera with Bayer filter)
    |
    v
Quality Assessment
  - Calculate sharpness metric for every frame
  - Rank frames from best to worst
    |
    v
Lucky Imaging Selection
  - Keep top 10-40% based on quality scores
  - Balance: fewer frames = sharper but noisier
    |
    v
Frame Alignment
  - Detect translation between each frame and reference
  - Subpixel accuracy via interpolation
    |
    v
Stacking
  - Average aligned frames to reduce noise
  - SNR improves by sqrt(N)
  - Reject outliers (satellites, hot pixels, cosmic rays)
    |
    v
Derotation (if needed)
  - Correct for planetary rotation during observation
  - Critical for long sequences (Jupiter rotates visibly in ~2 minutes)
    |
    v
Deconvolution
  - Reverse optical/atmospheric blur using PSF model
  - Richardson-Lucy or Wiener filter
    |
    v
Wavelet Sharpening
  - Multi-scale detail enhancement
  - Independently control enhancement at each spatial frequency
    |
    v
Color Processing (if applicable)
  - Align RGB channels (atmospheric dispersion correction)
  - Color balance and saturation
    |
    v
Final Output
```

---

## Core Algorithms

### Lucky Imaging

**Concept:** Atmospheric turbulence varies on timescales of milliseconds to seconds. By capturing at high frame rates (>30 fps) with short exposures (< atmospheric coherence time), individual frames "freeze" the turbulence. Some frames, by chance, are captured during moments of exceptionally good seeing.

**Implementation:**
1. Capture thousands of frames at high frame rate
2. Grade each frame with a sharpness metric
3. Select only the best frames (typically top 5-40%)
4. Align and stack the selected frames

**Key parameters:**
- **Exposure time:** Must be shorter than the atmospheric coherence time (typically a few milliseconds to tens of milliseconds)
- **Selection percentage:** Trade-off between sharpness (fewer frames) and signal-to-noise ratio (more frames). For planets, 10-40% is typical.

**Theoretical basis:** The probability of a diffraction-limited image decreases exponentially with the ratio (D/r0)^2, where D is aperture diameter and r0 is the Fried parameter. For amateur telescopes (D ~ 200-350mm) in average seeing (r0 ~ 70mm), there's a reasonable probability of near-diffraction-limited frames.

### Frame Quality Assessment

Several metrics exist for automatically scoring frame sharpness:

#### Laplacian Variance
The most widely used metric. Convolve the image with the 3x3 Laplacian kernel:

```
 0  1  0
 1 -4  1
 0  1  0
```

Then compute the variance of the result. Sharp images have strong edge responses (high variance), while blurry images have smooth transitions (low variance).

**Advantages:** Fast to compute, well-correlated with visual sharpness.

#### Tenengrad (Sobel + Variance)
Apply Sobel operators in X and Y directions:

```
Sobel X:          Sobel Y:
-1  0  1          -1 -2 -1
-2  0  2           0  0  0
-1  0  1           1  2  1
```

Compute gradient magnitude: `G = sqrt(Gx^2 + Gy^2)`, then take the mean of squared magnitudes (or variance). More robust than Laplacian for planetary images.

#### Frequency Domain (DCT/FFT)
Measure the proportion of high-frequency content. Sharp images contain more high-frequency components. More computationally intensive but can be more discriminating.

**Performance comparison:** Studies show Sobel+Variance and Laplacian Variance are the most effective for astronomical frame selection, offering the best balance of speed and discrimination.

### Frame Alignment and Registration

#### Phase Correlation (Primary Method)

Phase correlation is the standard approach for planetary frame alignment. It works in the frequency domain:

1. **Compute 2D FFT** of both reference frame R and target frame T
2. **Cross-power spectrum:** `CPS = (F(R) * conj(F(T))) / |F(R) * conj(F(T))|`
3. **Inverse FFT** of CPS yields a correlation surface
4. **Peak location** = translation offset (dx, dy)

**Why it works:** The Fourier shift theorem states that a spatial shift corresponds to a linear phase shift in the frequency domain. The normalization step removes the influence of amplitude, making it robust to brightness changes between frames.

**Subpixel accuracy:** Fit a paraboloid (or Gaussian) to the 3x3 neighborhood around the integer peak to achieve subpixel precision. Alternatively, the Guizar-Sicairos method uses matrix DFT upsampling around the peak for very high subpixel accuracy.

**Edge handling:** Apply a window function (Hann or Tukey) before FFT to reduce spectral leakage from image edges.

**Challenge:** Phase correlation can degrade on featureless or low-contrast planetary surfaces. Pre-processing with edge enhancement can help.

#### Cross-Correlation Methods (Alternative)

- **SAD** (Sum of Absolute Differences)
- **SSD** (Sum of Squared Differences)
- **NCC** (Normalized Cross-Correlation)

Generally slower than phase correlation but can handle rotation and scale changes better.

### Image Stacking

#### Mean Stacking
Element-wise average across aligned frames. The simplest and fastest method.

**SNR improvement:** SNR scales as `sqrt(N)` where N is the number of frames. Stacking 100 frames improves SNR by 10x.

**Weakness:** Susceptible to outliers (cosmic rays, satellites, hot pixels).

#### Median Stacking
Takes the median value at each pixel position. Robust to outliers since the median ignores extreme values.

**Trade-off:** More computationally intensive (requires sorting per pixel), and can lose faint details compared to mean stacking.

#### Sigma-Clipped Mean
The best general-purpose method. Iteratively:
1. Compute the mean and standard deviation at each pixel across the stack
2. Reject pixels more than N sigma from the mean
3. Recompute mean from remaining pixels
4. Repeat for a fixed number of iterations

**Variants:**
- **Standard:** Reject outliers completely
- **Winsorized:** Replace outliers with the clipping boundary value
- **Median sigma clipping:** Use median instead of mean as the central estimator

Excellent for large datasets (>50 frames). Combines the SNR benefits of mean stacking with the outlier rejection of median stacking.

#### Drizzle (Super-Resolution)
Exploits subpixel shifts between frames to reconstruct a higher-resolution output than any individual input frame. Each input pixel is "drizzled" onto a finer output grid using the known subpixel offset.

**Requirements:** Frames must have genuine subpixel offsets (which atmospheric turbulence naturally provides). Cannot be combined with median/sigma-clip stacking.

**Limitation for planetary:** Most effective for point sources. For extended objects like planets, drizzle improves sampling but cannot recover information above the diffraction limit.

### Wavelet Sharpening

Wavelet sharpening is the technique that transforms a decent stack into a stunning planetary image. It decomposes the image into multiple frequency layers, allowing selective enhancement at each scale.

#### A Trous (With Holes) Algorithm

The standard wavelet for astronomical sharpening uses the **B3 spline scaling function**:

**1D kernel:** `[1/16, 4/16, 6/16, 4/16, 1/16]` or equivalently `[1, 4, 6, 4, 1] / 16`

**2D application:** Applied separably — convolve rows first, then columns.

**Multi-scale decomposition:**

Starting with the original image `c_0`:

For each scale `j = 0, 1, 2, ..., n-1`:
1. Construct the dilated kernel by inserting `2^j - 1` zeros between each coefficient
2. Convolve `c_j` with the dilated kernel to get `c_{j+1}` (smoothed approximation)
3. Compute the detail layer: `w_{j+1} = c_j - c_{j+1}`

The result is a set of detail layers `{w_1, w_2, ..., w_n}` and a residual `c_n`, where:
- `w_1` contains the finest details (1-2 pixel features)
- `w_2` contains slightly larger features
- Each subsequent layer captures progressively larger structures
- `c_n` is the smooth background

**Reconstruction:** `original = w_1 + w_2 + ... + w_n + c_n`

**Sharpening:** Multiply each detail layer by a coefficient before summing:

`sharpened = k_1*w_1 + k_2*w_2 + ... + k_n*w_n + c_n`

Where:
- `k_i > 1.0` enhances detail at that scale
- `k_i < 1.0` suppresses detail (useful for noise reduction)
- `k_i = 1.0` leaves that scale unchanged

**Typical settings for planetary work (6 layers):**
- Layers 1-2: Fine atmospheric features, surface detail — enhance by 1.5-4.0x
- Layers 3-4: Mid-scale structures — moderate enhancement (1.2-2.0x)
- Layers 5-6: Large-scale gradients — leave at 1.0 or slightly suppress

**Why it works so well:** Unlike simple unsharp masking or Laplacian sharpening, wavelets allow enhancing fine detail without amplifying noise (suppress noisy layers) and without creating halos around bright features (each scale is processed independently).

### Deconvolution

Deconvolution attempts to reverse the blurring caused by the telescope optics and residual atmospheric effects.

#### Richardson-Lucy Algorithm

An iterative maximum-likelihood method based on the Poisson noise model:

```
image_{n+1} = image_n * ((observed / (PSF * image_n)) ** PSF_flipped)
```

Where `*` is convolution and `/` is element-wise division.

**Properties:**
- Preserves total flux (photon counts) and non-negativity
- Typically 10-50 iterations for planetary imaging
- More iterations = sharper but noisier (must be regularized)
- Well-suited for photon-counting statistics

#### Wiener Filter

A single-step frequency domain method:

```
H_wiener(f) = H*(f) / (|H(f)|^2 + NSR)
```

Where `H` is the optical transfer function (FT of PSF) and `NSR` is the noise-to-signal ratio.

**Properties:**
- Faster than Richardson-Lucy (single operation, no iterations)
- Based on Gaussian noise assumption (appropriate for stacked images)
- Less prone to ringing artifacts
- Requires noise power spectrum estimate

#### Point Spread Function (PSF) Modeling

Both deconvolution methods require a PSF model:

- **Theoretical (Airy disk):** Calculated from aperture diameter, central obstruction, wavelength. Clean but doesn't account for optical aberrations.
- **Empirical:** Derived from star images captured during the same session. Captures real-world aberrations.
- **Atmospheric (Kolmogorov):** Based on turbulence theory with the Fried parameter r0.

**Application order:** Deconvolution is typically applied AFTER stacking (when SNR is high enough) and BEFORE or alongside wavelet sharpening.

### Super-Resolution

#### Multi-Image Super-Resolution (MISR)

Exploits the natural subpixel shifts between frames (caused by atmospheric turbulence and tracking imprecision) to reconstruct a higher-resolution image. Each frame samples the scene at slightly different subpixel positions.

**Key insight:** While each frame is sampled at the camera's pixel resolution, the stack of frames collectively samples the scene at finer resolution. MISR algorithms reconstruct this finer sampling.

#### Deep Learning Approaches (Emerging)

Recent work (2025) applies neural networks to astronomical super-resolution:
- Wavelet multi-scale decomposition combined with multi-branch CNNs
- Training on different frequency bands independently
- SwinIR-based architectures with wavelet decomposition

**Important limitation:** For extended objects like planets, super-resolution cannot recover spatial frequencies above the diffraction limit — that information is physically lost. However, subpixel sampling via drizzle/MISR can still improve effective resolution up to the diffraction limit.

---

## Atmospheric Turbulence Theory

### Kolmogorov Model

Atmospheric turbulence is described as a cascade of eddies at different scales, following Kolmogorov's theory:

- **Power spectral density:** `phi(kappa) ~ kappa^(-11/3)` where kappa is spatial frequency
- **Theoretical exponent:** beta = 5/3 for pure Kolmogorov turbulence
- Models refractive index fluctuations caused by temperature variations in the atmosphere

### Fried Parameter (r0)

The **coherence length** — the diameter over which the root-mean-square wavefront error is approximately 1 radian:

- **Typical values:** 5-10 cm (average ~7 cm at visible wavelengths)
- **Excellent sites** (Paranal, Chile): up to 40 cm
- **Wavelength dependent:** `r0 ~ lambda^(6/5)` — seeing improves at longer wavelengths
- **Practical meaning:** A telescope with aperture D behaves as if it were a collection of independent sub-apertures of diameter r0

**Seeing relationship:** `FWHM = lambda / r0` (in radians). For r0 = 7 cm at 550nm, seeing FWHM = 1.6 arcseconds.

### Coherence Time (t0)

The time interval over which the wavefront remains coherent:

- **Typical values:** Few milliseconds to tens of milliseconds
- **Determines:** Maximum useful exposure time for lucky imaging
- **Related to:** Wind velocity and r0: `t0 = 0.314 * r0 / v_wind`

For effective lucky imaging, exposure times must be shorter than t0 to "freeze" the turbulence.

### Practical Implications

| Seeing Condition | r0 (cm) | Seeing FWHM (") | Lucky Imaging Potential |
|-----------------|---------|-----------------|----------------------|
| Excellent | >15 | <0.7 | Outstanding — near-diffraction-limited results |
| Good | 10-15 | 0.7-1.1 | Very good — fine planetary detail visible |
| Average | 7-10 | 1.1-1.6 | Good — standard high-resolution imaging |
| Poor | 4-7 | 1.6-2.8 | Limited — coarse detail only |
| Very poor | <4 | >2.8 | Minimal — stacking helps but resolution severely limited |

---

## Rust Ecosystem for Astronomy

The Rust ecosystem for astronomical image processing is still emerging:

### Existing Crates

- **sciimg** — Basic planetary imaging library (MarsRaw/sciimg on GitHub). Focused on planetary image processing but limited scope.
- **rubbl** — Astrophysical data processing, focused on CASA radio astronomy table format.
- **rustronomy** — Standardized astronomy toolset with Python interoperability. Includes Python bindings.
- **astro-rust** — Astronomical algorithms (coordinate transforms, ephemerides, etc.).

### General Image Processing Crates

- **image** — De facto standard for image I/O (PNG, TIFF, JPEG, etc.)
- **imageproc** — Image processing operations (convolution, filters, etc.)
- **ndarray** — N-dimensional arrays, the Rust equivalent of numpy
- **rustfft** — Pure Rust FFT implementation
- **rayon** — Data parallelism for multi-core processing

### Gaps and Opportunities

- No complete planetary processing pipeline exists in Rust
- Most tools are in Python (OpenCV/numpy), C++, or Java
- Rust's performance characteristics (zero-cost abstractions, no GC) are ideal for processing thousands of frames
- Memory safety is valuable for processing large datasets where buffer overflows could corrupt results
- Built-in parallelism (rayon) maps naturally to frame-level parallelism

---

## Sources and References

### Software

- [AutoStakkert! Official Site](https://www.autostakkert.com/)
- [RegiStax Wavelet Documentation](https://www.astronomie.be/registax/linkedwavelets.html)
- [Siril Official Site](https://siril.org/) and [Documentation](https://siril.readthedocs.io/)
- [PlanetarySystemStacker on GitHub](https://github.com/Rolf-Hempel/PlanetarySystemStacker)
- [ImPPG on GitHub](https://github.com/GreatAttractor/imppg)
- [AstroSurface Official Site](https://astrosurface.com/pageuk.html)
- [FireCapture Official Site](https://www.firecapture.de/)
- [SharpCap Official Site](https://www.sharpcap.co.uk/)

### Algorithms and Theory

- [Lucky Imaging — Wikipedia](https://en.wikipedia.org/wiki/Lucky_imaging)
- [Phase Correlation — Wikipedia](https://en.wikipedia.org/wiki/Phase_correlation)
- [Richardson-Lucy Deconvolution — Wikipedia](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)
- [Fried Parameter — Wikipedia](https://en.wikipedia.org/wiki/Fried_parameter)
- [Astronomical Seeing — Wikipedia](https://en.wikipedia.org/wiki/Astronomical_seeing)
- [An Introduction to Lucky Imaging — Sky & Telescope](https://skyandtelescope.org/astronomy-blogs/imaging-foundations-richard-wright/lucky-imaging/)
- [Solar Image Quality Assessment Using Variance of Laplacian — arXiv](https://arxiv.org/html/2405.11490v1)
- [Review of Image Quality Measures for Solar Imaging — Solar Physics](https://link.springer.com/article/10.1007/s11207-017-1211-3)
- [Autofocus Using OpenCV: Focus Measures Comparative Study](https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/)
- [Planetary Image Live Stacking via Phase Correlation — IEEE](https://ieeexplore.ieee.org/document/7830782/)
- [Image Registration — FreeAstro](https://free-astro.org/index.php?title=Image_registration)
- [Stacking Methods Compared — Clark Vision](https://clarkvision.com/articles/image-stacking-methods/)
- [Efficient Deconvolution Methods for Astronomical Imaging — A&A](https://www.aanda.org/articles/aa/pdf/2012/03/aa18681-11.pdf)
- [Super Resolved Imaging with Adaptive Optics — arXiv](https://arxiv.org/html/2508.04648v1)
- [Effect of Atmospheric Turbulence on Telescope Image](https://www.telescope-optics.net/induced.htm)
- [Introduction to Astronomical Seeing — Innovations Foresight](https://www.innovationsforesight.com/education/astronomical-seeing-tutorial/)

### Tutorials and Guides

- [High-Resolution Planetary Imaging Guide — Rouz Astro](https://rouzastro.com/high-resolution-planetary-imaging-guide-part-4-imaging-software-and-processing/)
- [Wavelets in RegiStax Guide — BBC Sky at Night](https://www.skyatnightmagazine.com/astrophotography/astrophoto-tips/wavelets-registax-guide)
- [Planetary Processing with PixInsight — Deep Sky Workflows](https://deepskyworkflows.com/planetary-processing-with-pixinsight/)
- [How to Process Planetary Images — Sky & Telescope](https://skyandtelescope.org/astronomy-resources/how-to-process-planetary-images/)
- [When Drizzle Can Bring Better Resolution — Astroshop EU](https://www.astroshop.eu/magazine/practical-tips/weigand-s-technical-tips/when-the-drizzle-technique-can-bring-you-better-resolution/i,1534)

### SER File Format

- [SER File Format Specification — FreeAstro](https://free-astro.org/index.php/SER)
- [ser-player C++ Implementation — GitHub](https://github.com/cgarry/ser-player)
