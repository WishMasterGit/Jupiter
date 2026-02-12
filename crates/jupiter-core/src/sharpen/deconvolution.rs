use ndarray::Array2;
use num_complex::Complex;

use crate::compute::cpu::{fft2d_forward, ifft2d_inverse};
use crate::compute::ComputeBackend;
use crate::frame::Frame;
use crate::pipeline::config::{DeconvolutionConfig, DeconvolutionMethod, PsfModel};

/// Generate PSF and dispatch to the appropriate deconvolution algorithm.
pub fn deconvolve(frame: &Frame, config: &DeconvolutionConfig) -> Frame {
    let (h, w) = frame.data.dim();
    let psf = generate_psf(&config.psf, h, w);

    match &config.method {
        DeconvolutionMethod::RichardsonLucy { iterations } => {
            richardson_lucy(frame, &psf, *iterations)
        }
        DeconvolutionMethod::Wiener { noise_ratio } => wiener_filter(frame, &psf, *noise_ratio),
    }
}

// ---------------------------------------------------------------------------
// PSF generation
// ---------------------------------------------------------------------------

pub fn generate_psf(model: &PsfModel, height: usize, width: usize) -> Array2<f32> {
    match model {
        PsfModel::Gaussian { sigma } => generate_gaussian_psf(*sigma, height, width),
        PsfModel::Kolmogorov { seeing } => generate_kolmogorov_psf(*seeing, height, width),
        PsfModel::Airy { radius } => generate_airy_psf(*radius, height, width),
    }
}

/// Gaussian PSF centered at (0,0) with wrap-around (FFT-ready layout), normalized to sum=1.
fn generate_gaussian_psf(sigma: f32, h: usize, w: usize) -> Array2<f32> {
    let mut psf = Array2::<f32>::zeros((h, w));
    let sigma2 = 2.0 * (sigma as f64) * (sigma as f64);
    let mut sum = 0.0f64;

    for row in 0..h {
        let y = if row <= h / 2 {
            row as f64
        } else {
            row as f64 - h as f64
        };
        for col in 0..w {
            let x = if col <= w / 2 {
                col as f64
            } else {
                col as f64 - w as f64
            };
            let val = (-(x * x + y * y) / sigma2).exp();
            psf[[row, col]] = val as f32;
            sum += val;
        }
    }

    if sum > 0.0 {
        let inv = 1.0 / sum as f32;
        psf.mapv_inplace(|v| v * inv);
    }

    psf
}

/// Kolmogorov PSF via OTF in frequency domain.
/// Long-exposure OTF: exp(-3.44 * (f / f0)^(5/3)) where f0 = 0.98 / seeing_fwhm.
fn generate_kolmogorov_psf(seeing: f32, h: usize, w: usize) -> Array2<f32> {
    let f0 = 0.98 / seeing as f64;
    let mut otf = Array2::<Complex<f64>>::zeros((h, w));

    for row in 0..h {
        let fy = if row <= h / 2 {
            row as f64 / h as f64
        } else {
            (row as f64 - h as f64) / h as f64
        };
        for col in 0..w {
            let fx = if col <= w / 2 {
                col as f64 / w as f64
            } else {
                (col as f64 - w as f64) / w as f64
            };
            let f = (fx * fx + fy * fy).sqrt();
            let ratio = f / f0;
            let val = (-3.44 * ratio.powf(5.0 / 3.0)).exp();
            otf[[row, col]] = Complex::new(val, 0.0);
        }
    }

    // IFFT to spatial domain
    let spatial = ifft2d_inverse(&otf);

    // Convert to f32, clamp negatives, normalize
    let mut psf = Array2::<f32>::zeros((h, w));
    let mut sum = 0.0f64;
    for row in 0..h {
        for col in 0..w {
            let val = spatial[[row, col]].max(0.0);
            psf[[row, col]] = val as f32;
            sum += val;
        }
    }

    if sum > 0.0 {
        let inv = 1.0 / sum as f32;
        psf.mapv_inplace(|v| v * inv);
    }

    psf
}

/// Airy PSF: (2 * J1(pi*r/R) / (pi*r/R))^2.
/// `radius` = first dark ring radius in pixels.
fn generate_airy_psf(radius: f32, h: usize, w: usize) -> Array2<f32> {
    let mut psf = Array2::<f32>::zeros((h, w));
    let r_param = radius as f64;
    let mut sum = 0.0f64;

    for row in 0..h {
        let y = if row <= h / 2 {
            row as f64
        } else {
            row as f64 - h as f64
        };
        for col in 0..w {
            let x = if col <= w / 2 {
                col as f64
            } else {
                col as f64 - w as f64
            };
            let r = (x * x + y * y).sqrt();
            let val = if r < 1e-12 {
                1.0 // lim (2*J1(x)/x)^2 as x->0 = 1
            } else {
                let arg = std::f64::consts::PI * r / r_param;
                let jinc = 2.0 * bessel_j1(arg) / arg;
                jinc * jinc
            };
            psf[[row, col]] = val as f32;
            sum += val;
        }
    }

    if sum > 0.0 {
        let inv = 1.0 / sum as f32;
        psf.mapv_inplace(|v| v * inv);
    }

    psf
}

// ---------------------------------------------------------------------------
// Bessel J1 — Abramowitz & Stegun rational polynomial approximation
// ---------------------------------------------------------------------------

pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 8.0 {
        let y = x * x;
        let r1 = x
            * (72362614232.0
                + y * (-7895059235.0
                    + y * (242396853.1
                        + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let r2 = 144725228442.0
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        r1 / r2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        // Phase shift for Bessel J1 large-argument approximation (3pi/4)
        const FRAC_3PI_4: f64 = 3.0 * std::f64::consts::FRAC_PI_4;
        let xx = ax - FRAC_3PI_4;
        let p0 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let q0 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * (0.105787412e-6))));
        let ans = (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * p0 - z * xx.sin() * q0);
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    }
}

// ---------------------------------------------------------------------------
// Richardson-Lucy deconvolution
// ---------------------------------------------------------------------------

fn richardson_lucy(frame: &Frame, psf: &Array2<f32>, iterations: usize) -> Frame {
    let (h, w) = frame.data.dim();
    let epsilon: f32 = 1e-10;

    // Pre-compute H = FFT(psf) and H_flip = FFT(psf_flipped)
    let h_fft = fft2d_forward(psf);

    // Flipped PSF: psf_flipped[r,c] = psf[-r,-c] = psf[h-r, w-c] (with wrap at 0)
    let mut psf_flipped = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let src_row = if row == 0 { 0 } else { h - row };
            let src_col = if col == 0 { 0 } else { w - col };
            psf_flipped[[row, col]] = psf[[src_row, src_col]];
        }
    }
    let h_flip_fft = fft2d_forward(&psf_flipped);

    // Initial estimate = observed image
    let mut estimate = frame.data.clone();

    for _iter in 0..iterations {
        // Forward model: blurred = IFFT(FFT(estimate) * H)
        let est_fft = fft2d_forward(&estimate);
        let mut blurred_fft = Array2::<Complex<f64>>::zeros((h, w));
        for row in 0..h {
            for col in 0..w {
                blurred_fft[[row, col]] = est_fft[[row, col]] * h_fft[[row, col]];
            }
        }
        let blurred = ifft2d_inverse(&blurred_fft);

        // ratio = observed / (blurred + epsilon)
        let mut ratio = Array2::<f32>::zeros((h, w));
        for row in 0..h {
            for col in 0..w {
                ratio[[row, col]] = frame.data[[row, col]] / (blurred[[row, col]] as f32 + epsilon);
            }
        }

        // correction = IFFT(FFT(ratio) * H_flip)
        let ratio_fft = fft2d_forward(&ratio);
        let mut corr_fft = Array2::<Complex<f64>>::zeros((h, w));
        for row in 0..h {
            for col in 0..w {
                corr_fft[[row, col]] = ratio_fft[[row, col]] * h_flip_fft[[row, col]];
            }
        }
        let correction = ifft2d_inverse(&corr_fft);

        // Multiplicative update
        for row in 0..h {
            for col in 0..w {
                estimate[[row, col]] *= correction[[row, col]] as f32;
            }
        }
    }

    // Clamp to [0.0, 1.0]
    estimate.mapv_inplace(|v| v.clamp(0.0, 1.0));
    Frame::new(estimate, frame.original_bit_depth)
}

// ---------------------------------------------------------------------------
// Wiener filter
// ---------------------------------------------------------------------------

fn wiener_filter(frame: &Frame, psf: &Array2<f32>, noise_ratio: f32) -> Frame {
    let (h, w) = frame.data.dim();
    let k = noise_ratio as f64;

    let f_obs = fft2d_forward(&frame.data);
    let h_fft = fft2d_forward(psf);

    // F_restored = F_obs * conj(H) / (|H|^2 + K)
    let mut f_restored = Array2::<Complex<f64>>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let h_val = h_fft[[row, col]];
            let h_conj = h_val.conj();
            let h_mag_sq = h_val.norm_sqr();
            f_restored[[row, col]] = f_obs[[row, col]] * h_conj / (h_mag_sq + k);
        }
    }

    let restored = ifft2d_inverse(&f_restored);
    let result = restored.mapv(|v| (v as f32).clamp(0.0, 1.0));
    Frame::new(result, frame.original_bit_depth)
}

// ---------------------------------------------------------------------------
// GPU-accelerated deconvolution
// ---------------------------------------------------------------------------

/// GPU-accelerated deconvolution dispatch.
pub fn deconvolve_gpu(frame: &Frame, config: &DeconvolutionConfig, backend: &dyn ComputeBackend) -> Frame {
    let (h, w) = frame.data.dim();
    let psf = generate_psf(&config.psf, h, w);

    match &config.method {
        DeconvolutionMethod::RichardsonLucy { iterations } => {
            richardson_lucy_gpu(frame, &psf, *iterations, backend)
        }
        DeconvolutionMethod::Wiener { noise_ratio } => {
            wiener_filter_gpu(frame, &psf, *noise_ratio, backend)
        }
    }
}

/// GPU Richardson-Lucy: FFT/IFFT and complex_mul stay on GPU,
/// element-wise real operations use backend divide_real/multiply_real.
fn richardson_lucy_gpu(
    frame: &Frame,
    psf: &Array2<f32>,
    iterations: usize,
    backend: &dyn ComputeBackend,
) -> Frame {
    let (h, w) = frame.data.dim();

    // Upload observed image
    let observed = backend.upload(&frame.data);

    // FFT(psf) — stays on GPU
    let psf_gpu = backend.upload(psf);
    let h_fft = backend.fft2d(&psf_gpu);

    // Flipped PSF: psf_flipped[r,c] = psf[h-r, w-c] (wrap at 0)
    let mut psf_flipped = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let src_row = if row == 0 { 0 } else { h - row };
            let src_col = if col == 0 { 0 } else { w - col };
            psf_flipped[[row, col]] = psf[[src_row, src_col]];
        }
    }
    let psf_flip_gpu = backend.upload(&psf_flipped);
    let h_flip_fft = backend.fft2d(&psf_flip_gpu);

    // Initial estimate = observed image (on GPU)
    let mut estimate = backend.upload(&frame.data);

    for _iter in 0..iterations {
        // Forward model: blurred = IFFT(FFT(estimate) * H)
        let est_fft = backend.fft2d(&estimate);
        let blurred_fft = backend.complex_mul(&est_fft, &h_fft);
        let blurred = backend.ifft2d_real(&blurred_fft, h, w);

        // ratio = observed / (blurred + epsilon)
        let ratio = backend.divide_real(&observed, &blurred, 1e-10);

        // correction = IFFT(FFT(ratio) * H_flip)
        let ratio_fft = backend.fft2d(&ratio);
        let corr_fft = backend.complex_mul(&ratio_fft, &h_flip_fft);
        let correction = backend.ifft2d_real(&corr_fft, h, w);

        // Multiplicative update: estimate *= correction
        estimate = backend.multiply_real(&estimate, &correction);
    }

    // Download and clamp
    let mut result = backend.download(&estimate);
    result.mapv_inplace(|v| v.clamp(0.0, 1.0));
    Frame::new(result, frame.original_bit_depth)
}

/// GPU Wiener filter: single-pass FFT-based deconvolution.
fn wiener_filter_gpu(
    frame: &Frame,
    psf: &Array2<f32>,
    noise_ratio: f32,
    _backend: &dyn ComputeBackend,
) -> Frame {
    // Wiener filter requires conj(H) / (|H|^2 + K) which is complex arithmetic
    // not directly available in the current trait. Delegate to CPU for now.
    // The FFT heavy-lifting benefits come from Richardson-Lucy.
    wiener_filter(frame, psf, noise_ratio)
}
