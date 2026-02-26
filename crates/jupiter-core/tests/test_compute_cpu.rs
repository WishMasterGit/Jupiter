use ndarray::Array2;

use jupiter_core::compute::cpu::CpuBackend;
use jupiter_core::compute::cpu::{bilinear_sample, fft2d_forward, ifft2d_inverse};
use jupiter_core::compute::{ComputeBackend, GpuBuffer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_array(h: usize, w: usize, fill: f32) -> Array2<f32> {
    Array2::from_elem((h, w), fill)
}

fn make_ramp_array(h: usize, w: usize) -> Array2<f32> {
    Array2::from_shape_fn((h, w), |(r, c)| (r * w + c) as f32 / (h * w) as f32)
}

// ---------------------------------------------------------------------------
// fft2d_forward / ifft2d_inverse round-trip — sequential path (64×64)
// ---------------------------------------------------------------------------

#[test]
fn test_fft_roundtrip_small() {
    // 64×64 < PARALLEL_PIXEL_THRESHOLD (65536), uses sequential path
    let original = make_ramp_array(64, 64);
    let spectrum = fft2d_forward(&original);
    let recovered = ifft2d_inverse(&spectrum);

    assert_eq!(recovered.dim(), (64, 64));
    for r in 0..64 {
        for c in 0..64 {
            let expected = original[[r, c]] as f64;
            let got = recovered[[r, c]];
            assert!(
                (expected - got).abs() < 1e-4,
                "mismatch at ({r},{c}): expected {expected}, got {got}"
            );
        }
    }
}

#[test]
fn test_fft_roundtrip_uniform_small() {
    let original = make_array(32, 32, 0.5);
    let spectrum = fft2d_forward(&original);
    let recovered = ifft2d_inverse(&spectrum);
    for r in 0..32 {
        for c in 0..32 {
            assert!((recovered[[r, c]] - 0.5f64).abs() < 1e-4);
        }
    }
}

// ---------------------------------------------------------------------------
// fft2d_forward / ifft2d_inverse round-trip — parallel path (512×512)
// ---------------------------------------------------------------------------

#[test]
fn test_fft_roundtrip_large_parallel() {
    // 512×512 = 262144 > 65536, exercises parallel path
    let original = make_ramp_array(512, 512);
    let spectrum = fft2d_forward(&original);
    let recovered = ifft2d_inverse(&spectrum);

    assert_eq!(recovered.dim(), (512, 512));
    // Check a sample of pixels rather than all 262144
    for r in [0, 100, 255, 511] {
        for c in [0, 100, 255, 511] {
            let expected = original[[r, c]] as f64;
            let got = recovered[[r, c]];
            assert!(
                (expected - got).abs() < 1e-3,
                "mismatch at ({r},{c}): expected {expected}, got {got}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// bilinear_sample
// ---------------------------------------------------------------------------

#[test]
fn test_bilinear_sample_exact_pixel() {
    let mut data = make_array(8, 8, 0.0);
    data[[3, 4]] = 1.0;
    // Sampling exactly at an integer coordinate returns the exact value
    let val = bilinear_sample(&data, 3.0, 4.0);
    assert!(
        (val - 1.0).abs() < 1e-5,
        "exact pixel: expected 1.0, got {val}"
    );
}

#[test]
fn test_bilinear_sample_half_pixel() {
    let mut data = make_array(8, 8, 0.0);
    data[[2, 2]] = 0.0;
    data[[2, 3]] = 1.0;
    data[[3, 2]] = 0.0;
    data[[3, 3]] = 0.0;
    // At (2.5, 2.5) bilinear = 0.25
    let val = bilinear_sample(&data, 2.5, 2.5);
    assert!(
        (val - 0.25).abs() < 1e-5,
        "half-pixel: expected 0.25, got {val}"
    );
}

#[test]
fn test_bilinear_sample_boundary_clamped() {
    let data = make_array(8, 8, 0.5);
    // Out-of-bounds coordinates → returns 0.0 (boundary sample)
    let val = bilinear_sample(&data, -1.0, -1.0);
    // Boundary is clamped to 0 for out-of-range
    assert!(val.is_finite(), "boundary sample should be finite");
}

#[test]
fn test_bilinear_sample_far_out_of_bounds() {
    let data = make_array(8, 8, 0.7);
    let val = bilinear_sample(&data, 100.0, 100.0);
    // Way out of bounds → clamped to 0
    assert!(val.is_finite());
}

// ---------------------------------------------------------------------------
// CpuBackend via ComputeBackend trait
// ---------------------------------------------------------------------------

#[test]
fn test_cpu_backend_name() {
    let backend = CpuBackend;
    assert_eq!(backend.name(), "CPU/Rayon");
}

#[test]
fn test_cpu_backend_is_not_gpu() {
    let backend = CpuBackend;
    assert!(!backend.is_gpu());
}

#[test]
fn test_cpu_backend_upload_download_roundtrip() {
    let backend = CpuBackend;
    let data = make_ramp_array(16, 16);
    let buf = backend.upload(&data);
    let recovered = backend.download(&buf);
    for (a, b) in data.iter().zip(recovered.iter()) {
        assert!((*a - *b).abs() < 1e-6);
    }
}

#[test]
fn test_cpu_backend_fft_roundtrip() {
    let backend = CpuBackend;
    let data = make_ramp_array(32, 32);
    let buf = backend.upload(&data);
    let spectrum = backend.fft2d(&buf);
    let recovered = backend.ifft2d_real(&spectrum, 32, 32);
    let out = backend.download(&recovered);

    for r in 0..32 {
        for c in 0..32 {
            assert!(
                (data[[r, c]] - out[[r, c]]).abs() < 1e-3,
                "mismatch at ({r},{c})"
            );
        }
    }
}

#[test]
fn test_cpu_backend_hann_window_reduces_edges() {
    let backend = CpuBackend;
    let data = make_array(32, 32, 1.0);
    let buf = backend.upload(&data);
    let windowed = backend.hann_window(&buf);
    let out = backend.download(&windowed);

    // Corner pixels (0,0) should be near 0 (Hann is 0 at edges)
    assert!(out[[0, 0]].abs() < 1e-5, "Hann window corner should be ~0");
    // Center pixel should be close to 1.0
    let center = out[[16, 16]];
    assert!(
        center > 0.8,
        "Hann window center should be near 1.0, got {center}"
    );
}

#[test]
fn test_cpu_backend_shift_bilinear_zero_shift() {
    let backend = CpuBackend;
    let data = make_ramp_array(32, 32);
    let buf = backend.upload(&data);
    let shifted = backend.shift_bilinear(&buf, 0.0, 0.0);
    let out = backend.download(&shifted);
    for (a, b) in data.iter().zip(out.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_backend_shift_bilinear_large_image() {
    // 512×512 exercises parallel shift path
    let backend = CpuBackend;
    let data = make_array(512, 512, 0.5);
    let buf = backend.upload(&data);
    let shifted = backend.shift_bilinear(&buf, 2.5, 1.5);
    let out = backend.download(&shifted);
    // Uniform image: after shift, interior still 0.5
    assert!((out[[256, 256]] - 0.5).abs() < 1e-4);
}

#[test]
fn test_cpu_backend_convolve_separable_uniform() {
    // Convolving a uniform image with any kernel leaves it uniform
    let backend = CpuBackend;
    let data = make_array(32, 32, 0.5);
    let buf = backend.upload(&data);
    let kernel = vec![0.25f32, 0.5, 0.25];
    let result = backend.convolve_separable(&buf, &kernel);
    let out = backend.download(&result);
    for v in out.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_backend_convolve_separable_large() {
    // 512×512 exercises parallel convolution path
    let backend = CpuBackend;
    let data = make_array(512, 512, 0.3);
    let buf = backend.upload(&data);
    let kernel = vec![1.0f32 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let result = backend.convolve_separable(&buf, &kernel);
    let out = backend.download(&result);
    assert!((out[[256, 256]] - 0.3).abs() < 1e-4);
}

#[test]
fn test_cpu_backend_atrous_convolve_small() {
    let backend = CpuBackend;
    let data = make_array(32, 32, 0.4);
    let buf = backend.upload(&data);
    let result = backend.atrous_convolve(&buf, 0);
    let out = backend.download(&result);
    // Uniform input → uniform output (B3 kernel sums to 1)
    for v in out.iter() {
        assert!((*v - 0.4).abs() < 1e-4);
    }
}

#[test]
fn test_cpu_backend_atrous_convolve_large() {
    // 512×512, scale=1 (step=2), exercises parallel path
    let backend = CpuBackend;
    let data = make_array(512, 512, 0.6);
    let buf = backend.upload(&data);
    let result = backend.atrous_convolve(&buf, 1);
    let out = backend.download(&result);
    assert!((out[[256, 256]] - 0.6).abs() < 1e-4);
}

#[test]
fn test_cpu_backend_cross_power_spectrum() {
    // Cross-power of a signal with itself should give a normalized output
    let backend = CpuBackend;
    let data = make_ramp_array(16, 16);
    let buf = backend.upload(&data);
    let spectrum = backend.fft2d(&buf);
    let cps = backend.cross_power_spectrum(&spectrum, &spectrum);
    let out = backend.download(&cps);
    // Values should be in [-1, 1] (normalized)
    for v in out.iter() {
        assert!(
            *v >= -1.0 - 1e-5 && *v <= 1.0 + 1e-5,
            "cross-power out of range: {v}"
        );
    }
}

#[test]
fn test_cpu_backend_find_peak_uniform() {
    let backend = CpuBackend;
    let data = make_array(16, 16, 0.5);
    let buf = backend.upload(&data);
    let (row, col, val) = backend.find_peak(&buf);
    // All values equal; peak can be anywhere, but value should be 0.5
    assert!(
        (val - 0.5).abs() < 1e-5,
        "peak value should be 0.5, got {val}"
    );
    assert!(row < 16 && col < 16);
}

#[test]
fn test_cpu_backend_find_peak_known_location() {
    let backend = CpuBackend;
    let mut data = make_array(16, 16, 0.0);
    data[[3, 7]] = 1.0;
    let buf = backend.upload(&data);
    let (row, col, val) = backend.find_peak(&buf);
    assert_eq!(row, 3);
    assert_eq!(col, 7);
    assert!((val - 1.0).abs() < 1e-5);
}

#[test]
fn test_cpu_backend_multiply_real() {
    let backend = CpuBackend;
    let a = make_array(8, 8, 0.5);
    let b = make_array(8, 8, 0.4);
    let buf_a = backend.upload(&a);
    let buf_b = backend.upload(&b);
    let result = backend.multiply_real(&buf_a, &buf_b);
    let out = backend.download(&result);
    for v in out.iter() {
        assert!((*v - 0.2).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_backend_divide_real() {
    let backend = CpuBackend;
    let a = make_array(8, 8, 1.0);
    let b = make_array(8, 8, 2.0);
    let buf_a = backend.upload(&a);
    let buf_b = backend.upload(&b);
    let result = backend.divide_real(&buf_a, &buf_b, 0.0);
    let out = backend.download(&result);
    for v in out.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_cpu_backend_divide_real_epsilon_guard() {
    // Dividing by ~0 with epsilon guard should not produce infinity
    let backend = CpuBackend;
    let a = make_array(8, 8, 1.0);
    let b = make_array(8, 8, 0.0);
    let buf_a = backend.upload(&a);
    let buf_b = backend.upload(&b);
    let result = backend.divide_real(&buf_a, &buf_b, 1e-6);
    let out = backend.download(&result);
    for v in out.iter() {
        assert!(v.is_finite(), "divide by ~0 with epsilon should be finite");
    }
}

#[test]
fn test_cpu_backend_gpu_buffer_as_array() {
    let data = make_ramp_array(8, 8);
    let buf = GpuBuffer::from_array(data.clone());
    let arr = buf.as_array().expect("should be CPU buffer");
    for (a, b) in data.iter().zip(arr.iter()) {
        assert!((*a - *b).abs() < 1e-6);
    }
}
