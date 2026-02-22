use ndarray::Array2;

use jupiter_core::frame::Frame;
use jupiter_core::pipeline::config::{
    DeconvolutionConfig, DeconvolutionMethod, PsfModel,
};
use jupiter_core::sharpen::deconvolution::{bessel_j1, deconvolve, generate_psf, rewrap_psf_padded};

// ---------------------------------------------------------------------------
// Helper: create a simple test frame from an Array2
// ---------------------------------------------------------------------------

fn make_frame(data: Array2<f32>) -> Frame {
    Frame::new(data, 8)
}

fn flat_frame(h: usize, w: usize, value: f32) -> Frame {
    make_frame(Array2::from_elem((h, w), value))
}

// ---------------------------------------------------------------------------
// PSF generation tests
// ---------------------------------------------------------------------------

#[test]
fn gaussian_psf_sums_to_one() {
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 2.0 }, 64, 64);
    let sum: f32 = psf.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "Gaussian PSF sum = {sum}, expected ~1.0"
    );
}

#[test]
fn kolmogorov_psf_sums_to_one() {
    let psf = generate_psf(&PsfModel::Kolmogorov { seeing: 3.0 }, 64, 64);
    let sum: f32 = psf.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "Kolmogorov PSF sum = {sum}, expected ~1.0"
    );
}

#[test]
fn airy_psf_sums_to_one() {
    let psf = generate_psf(&PsfModel::Airy { radius: 2.5 }, 64, 64);
    let sum: f32 = psf.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "Airy PSF sum = {sum}, expected ~1.0"
    );
}

#[test]
fn gaussian_psf_peak_at_origin() {
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 2.0 }, 64, 64);
    let max_val = *psf
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    assert!(
        (psf[[0, 0]] - max_val).abs() < 1e-6,
        "Peak should be at [0,0], got max={max_val} vs origin={}",
        psf[[0, 0]]
    );
}

#[test]
fn airy_psf_peak_at_origin() {
    let psf = generate_psf(&PsfModel::Airy { radius: 3.0 }, 64, 64);
    let max_val = *psf
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    assert!(
        (psf[[0, 0]] - max_val).abs() < 1e-6,
        "Airy peak should be at origin"
    );
}

#[test]
fn gaussian_psf_is_symmetric() {
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 1.5 }, 32, 32);
    // psf[r, c] should equal psf[h-r, c] and psf[r, w-c] (wrap-around symmetry)
    for r in 1..16 {
        let mirror_r = 32 - r;
        for c in 1..16 {
            let mirror_c = 32 - c;
            let diff = (psf[[r, c]] - psf[[mirror_r, mirror_c]]).abs();
            assert!(
                diff < 1e-6,
                "Symmetry broken at [{r},{c}] vs [{mirror_r},{mirror_c}]: {} vs {}",
                psf[[r, c]],
                psf[[mirror_r, mirror_c]]
            );
        }
    }
}

#[test]
fn gaussian_psf_all_nonnegative() {
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 1.0 }, 32, 32);
    assert!(
        psf.iter().all(|&v| v >= 0.0),
        "Gaussian PSF should have no negative values"
    );
}

#[test]
fn kolmogorov_psf_all_nonnegative() {
    let psf = generate_psf(&PsfModel::Kolmogorov { seeing: 2.0 }, 32, 32);
    assert!(
        psf.iter().all(|&v| v >= 0.0),
        "Kolmogorov PSF should have no negative values"
    );
}

#[test]
fn airy_psf_all_nonnegative() {
    let psf = generate_psf(&PsfModel::Airy { radius: 2.0 }, 32, 32);
    assert!(
        psf.iter().all(|&v| v >= 0.0),
        "Airy PSF should have no negative values"
    );
}

#[test]
fn psf_different_sizes() {
    // Ensure PSF generation works for non-square and various sizes
    for &(h, w) in &[(32, 64), (64, 32), (17, 17), (128, 128)] {
        let psf = generate_psf(&PsfModel::Gaussian { sigma: 2.0 }, h, w);
        assert_eq!(psf.dim(), (h, w));
        let sum: f32 = psf.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "PSF sum for {h}x{w} = {sum}"
        );
    }
}

// ---------------------------------------------------------------------------
// Bessel J1 tests
// ---------------------------------------------------------------------------

#[test]
fn bessel_j1_at_zero() {
    // J1(0) = 0
    assert!(
        bessel_j1(0.0).abs() < 1e-10,
        "J1(0) should be 0, got {}",
        bessel_j1(0.0)
    );
}

#[test]
fn bessel_j1_known_values() {
    // J1(1.0) ≈ 0.44005058574
    let val = bessel_j1(1.0);
    assert!(
        (val - 0.44005058574).abs() < 1e-5,
        "J1(1.0) = {val}, expected ~0.4401"
    );

    // J1(3.8317) ≈ 0 (first zero after origin)
    let val = bessel_j1(3.8317);
    assert!(
        val.abs() < 0.01,
        "J1(3.8317) = {val}, expected ~0"
    );
}

#[test]
fn bessel_j1_odd_symmetry() {
    // J1 is an odd function: J1(-x) = -J1(x)
    for &x in &[0.5, 1.0, 2.0, 5.0, 10.0, 15.0] {
        let pos = bessel_j1(x);
        let neg = bessel_j1(-x);
        assert!(
            (pos + neg).abs() < 1e-6,
            "J1({x}) = {pos}, J1(-{x}) = {neg}, should be negatives"
        );
    }
}

#[test]
fn bessel_j1_large_argument() {
    // J1(10) ≈ 0.04347274617
    let val = bessel_j1(10.0);
    assert!(
        (val - 0.04347274617).abs() < 1e-4,
        "J1(10) = {val}, expected ~0.0435"
    );
}

// ---------------------------------------------------------------------------
// Flat image stability tests
// ---------------------------------------------------------------------------

#[test]
fn rl_flat_image_unchanged() {
    let frame = flat_frame(32, 32, 0.5);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 10 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve(&frame, &config);

    for r in 0..32 {
        for c in 0..32 {
            assert!(
                (result.data[[r, c]] - 0.5).abs() < 0.05,
                "RL on flat image: [{r},{c}] = {}, expected ~0.5",
                result.data[[r, c]]
            );
        }
    }
}

#[test]
fn wiener_flat_image_unchanged() {
    let frame = flat_frame(32, 32, 0.5);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.001 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve(&frame, &config);

    for r in 0..32 {
        for c in 0..32 {
            assert!(
                (result.data[[r, c]] - 0.5).abs() < 0.05,
                "Wiener on flat image: [{r},{c}] = {}, expected ~0.5",
                result.data[[r, c]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Output range tests (clamping to [0, 1])
// ---------------------------------------------------------------------------

#[test]
fn rl_output_in_valid_range() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 10..22 {
        for c in 10..22 {
            data[[r, c]] = 1.0;
        }
    }
    let frame = make_frame(data);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 20 },
        psf: PsfModel::Gaussian { sigma: 1.5 },
    };
    let result = deconvolve(&frame, &config);

    assert!(
        result.data.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "RL output should be clamped to [0, 1]"
    );
}

#[test]
fn wiener_output_in_valid_range() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 10..22 {
        for c in 10..22 {
            data[[r, c]] = 1.0;
        }
    }
    let frame = make_frame(data);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.01 },
        psf: PsfModel::Gaussian { sigma: 1.5 },
    };
    let result = deconvolve(&frame, &config);

    assert!(
        result.data.iter().all(|&v| (0.0..=1.0).contains(&v)),
        "Wiener output should be clamped to [0, 1]"
    );
}

// ---------------------------------------------------------------------------
// Zero-iteration RL returns (approximately) the original
// ---------------------------------------------------------------------------

#[test]
fn rl_zero_iterations_returns_input() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 8..24 {
        for c in 8..24 {
            data[[r, c]] = 0.7;
        }
    }
    let frame = make_frame(data.clone());
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 0 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve(&frame, &config);

    for r in 0..32 {
        for c in 0..32 {
            assert!(
                (result.data[[r, c]] - data[[r, c]]).abs() < 1e-5,
                "0-iteration RL should return input, [{r},{c}]: {} vs {}",
                result.data[[r, c]],
                data[[r, c]]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Preserves bit depth metadata
// ---------------------------------------------------------------------------

#[test]
fn deconvolve_preserves_bit_depth() {
    let frame = Frame::new(Array2::from_elem((16, 16), 0.5f32), 16);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 5 },
        psf: PsfModel::Gaussian { sigma: 1.0 },
    };
    let result = deconvolve(&frame, &config);
    assert_eq!(result.original_bit_depth, 16);
}

// ---------------------------------------------------------------------------
// Preserves frame dimensions
// ---------------------------------------------------------------------------

#[test]
fn deconvolve_preserves_dimensions() {
    let frame = make_frame(Array2::from_elem((48, 64), 0.3f32));
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.01 },
        psf: PsfModel::Kolmogorov { seeing: 3.0 },
    };
    let result = deconvolve(&frame, &config);
    assert_eq!(result.data.dim(), (48, 64));
}

// ---------------------------------------------------------------------------
// All PSF models work through deconvolve dispatch
// ---------------------------------------------------------------------------

#[test]
fn deconvolve_dispatch_all_psf_models() {
    let frame = flat_frame(32, 32, 0.4);

    let psf_models = vec![
        PsfModel::Gaussian { sigma: 1.5 },
        PsfModel::Kolmogorov { seeing: 2.0 },
        PsfModel::Airy { radius: 2.0 },
    ];

    for psf in psf_models {
        let config = DeconvolutionConfig {
            method: DeconvolutionMethod::RichardsonLucy { iterations: 3 },
            psf: psf.clone(),
        };
        let result = deconvolve(&frame, &config);
        assert_eq!(result.data.dim(), (32, 32), "Failed for PSF {:?}", psf);
    }
}

#[test]
fn deconvolve_dispatch_both_methods() {
    let frame = flat_frame(32, 32, 0.4);

    let methods = vec![
        DeconvolutionMethod::RichardsonLucy { iterations: 3 },
        DeconvolutionMethod::Wiener { noise_ratio: 0.01 },
    ];

    for method in methods {
        let config = DeconvolutionConfig {
            method: method.clone(),
            psf: PsfModel::Gaussian { sigma: 1.5 },
        };
        let result = deconvolve(&frame, &config);
        assert_eq!(result.data.dim(), (32, 32), "Failed for method {:?}", method);
    }
}

// ---------------------------------------------------------------------------
// RL sharpening effect: convolve a sharp image with Gaussian, then RL
// should partially recover the sharpness (edge gradient should increase).
// ---------------------------------------------------------------------------

#[test]
fn rl_recovers_sharpness_from_blurred_image() {
    let size = 64;
    // Sharp image: bright square on dark background
    let mut sharp = Array2::<f32>::zeros((size, size));
    for r in 20..44 {
        for c in 20..44 {
            sharp[[r, c]] = 0.8;
        }
    }

    // Manually blur with Gaussian: convolve in spatial domain
    let sigma = 2.0f32;
    let mut blurred = Array2::<f32>::zeros((size, size));
    let kernel_rad = 6i32;
    for r in 0..size {
        for c in 0..size {
            let mut sum = 0.0f32;
            let mut wt = 0.0f32;
            for kr in -kernel_rad..=kernel_rad {
                for kc in -kernel_rad..=kernel_rad {
                    let sr = (r as i32 + kr).clamp(0, size as i32 - 1) as usize;
                    let sc = (c as i32 + kc).clamp(0, size as i32 - 1) as usize;
                    let g =
                        (-(kr * kr + kc * kc) as f32 / (2.0 * sigma * sigma)).exp();
                    sum += sharp[[sr, sc]] * g;
                    wt += g;
                }
            }
            blurred[[r, c]] = sum / wt;
        }
    }

    // Measure edge gradient of blurred image (horizontal gradient at row 32, across left edge)
    let blurred_grad = (blurred[[32, 21]] - blurred[[32, 19]]).abs();

    // Deconvolve
    let frame = make_frame(blurred);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 15 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve(&frame, &config);

    let restored_grad = (result.data[[32, 21]] - result.data[[32, 19]]).abs();

    assert!(
        restored_grad > blurred_grad,
        "RL should sharpen edges: restored gradient {restored_grad} should exceed blurred gradient {blurred_grad}"
    );
}

// ---------------------------------------------------------------------------
// Wiener sharpening effect (same logic as RL test above)
// ---------------------------------------------------------------------------

#[test]
fn wiener_recovers_sharpness_from_blurred_image() {
    let size = 64;
    let mut sharp = Array2::<f32>::zeros((size, size));
    for r in 20..44 {
        for c in 20..44 {
            sharp[[r, c]] = 0.8;
        }
    }

    let sigma = 2.0f32;
    let mut blurred = Array2::<f32>::zeros((size, size));
    let kernel_rad = 6i32;
    for r in 0..size {
        for c in 0..size {
            let mut sum = 0.0f32;
            let mut wt = 0.0f32;
            for kr in -kernel_rad..=kernel_rad {
                for kc in -kernel_rad..=kernel_rad {
                    let sr = (r as i32 + kr).clamp(0, size as i32 - 1) as usize;
                    let sc = (c as i32 + kc).clamp(0, size as i32 - 1) as usize;
                    let g =
                        (-(kr * kr + kc * kc) as f32 / (2.0 * sigma * sigma)).exp();
                    sum += sharp[[sr, sc]] * g;
                    wt += g;
                }
            }
            blurred[[r, c]] = sum / wt;
        }
    }

    let blurred_grad = (blurred[[32, 21]] - blurred[[32, 19]]).abs();

    let frame = make_frame(blurred);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.001 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve(&frame, &config);

    let restored_grad = (result.data[[32, 21]] - result.data[[32, 19]]).abs();

    assert!(
        restored_grad > blurred_grad,
        "Wiener should sharpen edges: restored gradient {restored_grad} should exceed blurred gradient {blurred_grad}"
    );
}

// ---------------------------------------------------------------------------
// Black image stays black
// ---------------------------------------------------------------------------

#[test]
fn rl_black_image_stays_black() {
    let frame = flat_frame(32, 32, 0.0);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 10 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve(&frame, &config);

    let max_val = *result
        .data
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    assert!(
        max_val < 1e-5,
        "Black image should stay black after RL, got max={max_val}"
    );
}

#[test]
fn wiener_black_image_stays_black() {
    let frame = flat_frame(32, 32, 0.0);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.01 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve(&frame, &config);

    let max_val = *result
        .data
        .iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap();
    assert!(
        max_val < 1e-4,
        "Black image should stay black after Wiener, got max={max_val}"
    );
}

// ---------------------------------------------------------------------------
// Config serde round-trip
// ---------------------------------------------------------------------------

#[test]
fn deconvolution_config_serde_roundtrip() {
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 25 },
        psf: PsfModel::Gaussian { sigma: 1.8 },
    };
    let json = serde_json::to_string(&config).unwrap();
    let restored: DeconvolutionConfig = serde_json::from_str(&json).unwrap();
    // Verify via Debug representation
    assert_eq!(format!("{:?}", config), format!("{:?}", restored));
}

#[test]
fn deconvolution_config_serde_wiener_kolmogorov() {
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.005 },
        psf: PsfModel::Kolmogorov { seeing: 3.5 },
    };
    let json = serde_json::to_string(&config).unwrap();
    let restored: DeconvolutionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{:?}", config), format!("{:?}", restored));
}

#[test]
fn deconvolution_config_serde_airy() {
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 10 },
        psf: PsfModel::Airy { radius: 2.5 },
    };
    let json = serde_json::to_string(&config).unwrap();
    let restored: DeconvolutionConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(format!("{:?}", config), format!("{:?}", restored));
}

// ---------------------------------------------------------------------------
// RL more iterations = closer to sharp (convergence)
// ---------------------------------------------------------------------------

#[test]
fn rl_more_iterations_sharper() {
    let size = 64;
    let mut data = Array2::<f32>::zeros((size, size));
    for r in 20..44 {
        for c in 20..44 {
            data[[r, c]] = 0.8;
        }
    }
    let frame = make_frame(data);

    let config_few = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 3 },
        psf: PsfModel::Gaussian { sigma: 1.5 },
    };
    let config_many = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 20 },
        psf: PsfModel::Gaussian { sigma: 1.5 },
    };

    let result_few = deconvolve(&frame, &config_few);
    let result_many = deconvolve(&frame, &config_many);

    // More iterations should produce sharper edges (higher gradient at boundary)
    let grad_few = (result_few.data[[32, 21]] - result_few.data[[32, 19]]).abs();
    let grad_many = (result_many.data[[32, 21]] - result_many.data[[32, 19]]).abs();

    assert!(
        grad_many >= grad_few,
        "More iterations should produce >= edge gradient: {grad_many} vs {grad_few}"
    );
}

// ---------------------------------------------------------------------------
// Wiener: higher noise_ratio = smoother result
// ---------------------------------------------------------------------------

#[test]
fn wiener_higher_noise_ratio_smoother() {
    let size = 64;
    let mut data = Array2::<f32>::zeros((size, size));
    for r in 20..44 {
        for c in 20..44 {
            data[[r, c]] = 0.8;
        }
    }
    let frame = make_frame(data);

    let config_low = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.0001 },
        psf: PsfModel::Gaussian { sigma: 1.5 },
    };
    let config_high = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.1 },
        psf: PsfModel::Gaussian { sigma: 1.5 },
    };

    let result_low = deconvolve(&frame, &config_low);
    let result_high = deconvolve(&frame, &config_high);

    // Higher noise_ratio suppresses deconvolution, so edge gradient should be less
    let grad_low = (result_low.data[[32, 21]] - result_low.data[[32, 19]]).abs();
    let grad_high = (result_high.data[[32, 21]] - result_high.data[[32, 19]]).abs();

    assert!(
        grad_low >= grad_high,
        "Lower noise_ratio should produce >= edge gradient: {grad_low} vs {grad_high}"
    );
}

// ---------------------------------------------------------------------------
// Kolmogorov and Airy PSFs through full deconvolution
// ---------------------------------------------------------------------------

#[test]
fn rl_with_kolmogorov_psf() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 10..22 {
        for c in 10..22 {
            data[[r, c]] = 0.7;
        }
    }
    let frame = make_frame(data);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 10 },
        psf: PsfModel::Kolmogorov { seeing: 3.0 },
    };
    let result = deconvolve(&frame, &config);
    assert!(result.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn wiener_with_airy_psf() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 10..22 {
        for c in 10..22 {
            data[[r, c]] = 0.7;
        }
    }
    let frame = make_frame(data);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::Wiener { noise_ratio: 0.01 },
        psf: PsfModel::Airy { radius: 2.5 },
    };
    let result = deconvolve(&frame, &config);
    assert!(result.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

// ---------------------------------------------------------------------------
// PSF re-wrapping tests (for GPU FFT power-of-2 padding)
// ---------------------------------------------------------------------------

#[test]
fn rewrap_psf_padded_identity_when_power_of_two() {
    // 32x32 is already a power of 2 — rewrap to same dims is a no-op.
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 2.0 }, 32, 32);
    let rewrapped = rewrap_psf_padded(&psf, 32, 32);
    for r in 0..32 {
        for c in 0..32 {
            assert!(
                (psf[[r, c]] - rewrapped[[r, c]]).abs() < 1e-10,
                "Identity rewrap should match at [{r},{c}]"
            );
        }
    }
}

#[test]
fn rewrap_psf_padded_preserves_sum() {
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 2.0 }, 48, 48);
    let rewrapped = rewrap_psf_padded(&psf, 64, 64);
    let sum_orig: f32 = psf.iter().sum();
    let sum_rewrapped: f32 = rewrapped.iter().sum();
    assert!(
        (sum_orig - sum_rewrapped).abs() < 1e-6,
        "Rewrap should preserve total energy: {sum_orig} vs {sum_rewrapped}"
    );
}

#[test]
fn rewrap_psf_padded_origin_preserved() {
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 2.0 }, 48, 48);
    let rewrapped = rewrap_psf_padded(&psf, 64, 64);
    // Origin [0,0] should be preserved (peak of the PSF)
    assert!(
        (psf[[0, 0]] - rewrapped[[0, 0]]).abs() < 1e-10,
        "Origin should be preserved"
    );
    // Values near origin in positive quadrant should be preserved
    assert!(
        (psf[[1, 1]] - rewrapped[[1, 1]]).abs() < 1e-10,
        "Near-origin positive quadrant should be preserved"
    );
}

#[test]
fn rewrap_psf_padded_wraparound_quadrant() {
    let psf = generate_psf(&PsfModel::Gaussian { sigma: 2.0 }, 48, 48);
    let rewrapped = rewrap_psf_padded(&psf, 64, 64);
    // PSF[47, 47] (row=47, col=47 in 48x48) represents spatial offset (-1, -1).
    // In 64x64 layout, that should be at [63, 63].
    assert!(
        (psf[[47, 47]] - rewrapped[[63, 63]]).abs() < 1e-10,
        "Wrap-around corner should map correctly: psf[47,47]={} vs rewrapped[63,63]={}",
        psf[[47, 47]],
        rewrapped[[63, 63]]
    );
    // PSF[47, 0] (row=47, col=0) represents offset (-1, 0).
    // In 64x64 layout, that should be at [63, 0].
    assert!(
        (psf[[47, 0]] - rewrapped[[63, 0]]).abs() < 1e-10,
        "Wrap-around row should map correctly"
    );
}

// ---------------------------------------------------------------------------
// GPU Richardson-Lucy tests (require `gpu` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[test]
fn gpu_rl_matches_cpu_rl_non_power_of_two() {
    use jupiter_core::compute::{create_backend, DevicePreference};
    use jupiter_core::sharpen::deconvolution::deconvolve_gpu;

    let backend = create_backend(&DevicePreference::Gpu);
    if !backend.is_gpu() {
        return; // skip if no GPU available
    }

    // Non-power-of-2: 48x48 → padded to 64x64 by GPU FFT
    let size = 48;
    let mut data = Array2::<f32>::zeros((size, size));
    for r in 14..34 {
        for c in 14..34 {
            data[[r, c]] = 0.7;
        }
    }
    let frame = make_frame(data);

    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 5 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };

    let cpu_result = deconvolve(&frame, &config);
    let gpu_result = deconvolve_gpu(&frame, &config, &*backend);

    // GPU uses f32, CPU uses f64 internally — allow some tolerance
    let max_diff = cpu_result
        .data
        .iter()
        .zip(gpu_result.data.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 0.05,
        "GPU RL should approximately match CPU RL on non-power-of-2: max_diff={max_diff}"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_rl_flat_image_stable_non_power_of_two() {
    use jupiter_core::compute::{create_backend, DevicePreference};
    use jupiter_core::sharpen::deconvolution::deconvolve_gpu;

    let backend = create_backend(&DevicePreference::Gpu);
    if !backend.is_gpu() {
        return;
    }

    // 48x48 (non-power-of-2) — zero-padding to 64x64 causes border artifacts,
    // so we check only interior pixels (margin of 8 pixels = 4x sigma).
    let frame = flat_frame(48, 48, 0.5);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 10 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve_gpu(&frame, &config, &*backend);

    let margin = 8;
    for r in margin..48 - margin {
        for c in margin..48 - margin {
            assert!(
                (result.data[[r, c]] - 0.5).abs() < 0.05,
                "GPU RL on flat image interior: [{r},{c}] = {}, expected ~0.5",
                result.data[[r, c]]
            );
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_rl_recovers_sharpness_non_power_of_two() {
    use jupiter_core::compute::{create_backend, DevicePreference};
    use jupiter_core::sharpen::deconvolution::deconvolve_gpu;

    let backend = create_backend(&DevicePreference::Gpu);
    if !backend.is_gpu() {
        return;
    }

    let size = 48;
    let mut sharp = Array2::<f32>::zeros((size, size));
    for r in 14..34 {
        for c in 14..34 {
            sharp[[r, c]] = 0.8;
        }
    }

    // Manually blur
    let sigma = 2.0f32;
    let mut blurred = Array2::<f32>::zeros((size, size));
    let kernel_rad = 6i32;
    for r in 0..size {
        for c in 0..size {
            let mut sum = 0.0f32;
            let mut wt = 0.0f32;
            for kr in -kernel_rad..=kernel_rad {
                for kc in -kernel_rad..=kernel_rad {
                    let sr = (r as i32 + kr).clamp(0, size as i32 - 1) as usize;
                    let sc = (c as i32 + kc).clamp(0, size as i32 - 1) as usize;
                    let g = (-(kr * kr + kc * kc) as f32 / (2.0 * sigma * sigma)).exp();
                    sum += sharp[[sr, sc]] * g;
                    wt += g;
                }
            }
            blurred[[r, c]] = sum / wt;
        }
    }

    let blurred_grad = (blurred[[24, 15]] - blurred[[24, 13]]).abs();

    let frame = make_frame(blurred);
    let config = DeconvolutionConfig {
        method: DeconvolutionMethod::RichardsonLucy { iterations: 15 },
        psf: PsfModel::Gaussian { sigma: 2.0 },
    };
    let result = deconvolve_gpu(&frame, &config, &*backend);

    let restored_grad = (result.data[[24, 15]] - result.data[[24, 13]]).abs();

    assert!(
        restored_grad > blurred_grad,
        "GPU RL should sharpen edges: restored gradient {restored_grad} > blurred gradient {blurred_grad}"
    );
}
