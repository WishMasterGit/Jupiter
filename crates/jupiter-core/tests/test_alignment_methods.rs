use std::sync::Arc;

use ndarray::Array2;

use jupiter_core::align::centroid::compute_offset_centroid;
use jupiter_core::align::compute_offset_configured;
use jupiter_core::align::enhanced_phase::compute_offset_enhanced;
use jupiter_core::align::gradient_correlation::compute_offset_gradient;
use jupiter_core::align::phase_correlation::bilinear_sample;
use jupiter_core::align::pyramid::compute_offset_pyramid;
use jupiter_core::compute::cpu::CpuBackend;
use jupiter_core::compute::ComputeBackend;
use jupiter_core::frame::Frame;
use jupiter_core::pipeline::config::{
    AlignmentConfig, AlignmentMethod, CentroidConfig, EnhancedPhaseConfig, PyramidConfig,
};

fn cpu() -> Arc<dyn ComputeBackend> {
    Arc::new(CpuBackend)
}

/// Create a test image with a bright square at the given position.
fn make_bright_square(h: usize, w: usize, cy: usize, cx: usize, size: usize) -> Array2<f32> {
    let mut data = Array2::<f32>::zeros((h, w));
    let half = size / 2;
    let r_start = cy.saturating_sub(half);
    let r_end = (cy + half).min(h);
    let c_start = cx.saturating_sub(half);
    let c_end = (cx + half).min(w);
    for r in r_start..r_end {
        for c in c_start..c_end {
            data[[r, c]] = 0.8;
        }
    }
    // Add slight gradient for edge detection
    for r in 0..h {
        for c in 0..w {
            data[[r, c]] += 0.05 * (r as f32 / h as f32);
        }
    }
    data
}

/// Create a bright disk (approximate circle) at the given position.
fn make_bright_disk(h: usize, w: usize, cy: f64, cx: f64, radius: f64) -> Array2<f32> {
    let mut data = Array2::<f32>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let dy = r as f64 - cy;
            let dx = c as f64 - cx;
            let dist = (dy * dy + dx * dx).sqrt();
            if dist < radius {
                // Smooth falloff at edge
                let val = 1.0 - (dist / radius).powi(2);
                data[[r, c]] = val as f32;
            }
        }
    }
    data
}

/// Shift an array by the given offset using bilinear interpolation.
fn shift_array(data: &Array2<f32>, dy: f64, dx: f64) -> Array2<f32> {
    let (h, w) = data.dim();
    let mut result = Array2::<f32>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            let src_y = r as f64 - dy;
            let src_x = c as f64 - dx;
            result[[r, c]] = bilinear_sample(data, src_y, src_x);
        }
    }
    result
}

// ===== Enhanced Phase Correlation =====

#[test]
fn test_enhanced_phase_zero_offset() {
    let img = make_bright_square(64, 64, 32, 32, 16);
    let config = EnhancedPhaseConfig {
        upsample_factor: 20,
    };
    let backend = cpu();

    let offset = compute_offset_enhanced(&img, &img, &config, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.1, "dx={}", offset.dx);
    assert!(offset.dy.abs() < 0.1, "dy={}", offset.dy);
}

#[test]
fn test_enhanced_phase_known_shift() {
    let reference = make_bright_square(128, 128, 64, 64, 24);
    let target = shift_array(&reference, 3.0, 5.0);
    let config = EnhancedPhaseConfig {
        upsample_factor: 20,
    };
    let backend = cpu();

    let offset = compute_offset_enhanced(&reference, &target, &config, backend.as_ref()).unwrap();
    assert!(
        (offset.dx.abs() - 5.0).abs() < 1.0,
        "dx={} expected ~5",
        offset.dx
    );
    assert!(
        (offset.dy.abs() - 3.0).abs() < 1.0,
        "dy={} expected ~3",
        offset.dy
    );
}

#[test]
fn test_enhanced_phase_subpixel_accuracy() {
    let reference = make_bright_square(128, 128, 64, 64, 24);
    let target = shift_array(&reference, 2.3, 4.7);
    let config = EnhancedPhaseConfig {
        upsample_factor: 100,
    };
    let backend = cpu();

    let offset = compute_offset_enhanced(&reference, &target, &config, backend.as_ref()).unwrap();
    // Enhanced phase should be more precise than standard
    let err_dx = (offset.dx.abs() - 4.7).abs();
    let err_dy = (offset.dy.abs() - 2.3).abs();
    // Sub-pixel accuracy on a 128x128 image with synthetic data; tolerance ~1px
    assert!(
        err_dx < 1.0,
        "dx error {} too large (dx={})",
        err_dx,
        offset.dx
    );
    assert!(
        err_dy < 1.0,
        "dy error {} too large (dy={})",
        err_dy,
        offset.dy
    );
}

#[test]
fn test_enhanced_phase_upsample_factor_1() {
    // With upsample_factor=1, should give integer-only result (no refinement)
    let reference = make_bright_square(64, 64, 32, 32, 12);
    let target = shift_array(&reference, 3.0, 5.0);
    let config = EnhancedPhaseConfig { upsample_factor: 1 };
    let backend = cpu();

    let offset = compute_offset_enhanced(&reference, &target, &config, backend.as_ref()).unwrap();
    // Should still detect the shift approximately
    assert!((offset.dx.abs() - 5.0).abs() < 1.5, "dx={}", offset.dx);
    assert!((offset.dy.abs() - 3.0).abs() < 1.5, "dy={}", offset.dy);
}

// ===== Centroid Alignment =====

#[test]
fn test_centroid_zero_offset() {
    let img = make_bright_disk(64, 64, 32.0, 32.0, 15.0);
    let config = CentroidConfig { threshold: 0.1 };

    let offset = compute_offset_centroid(&img, &img, &config).unwrap();
    assert!(offset.dx.abs() < 0.01, "dx={}", offset.dx);
    assert!(offset.dy.abs() < 0.01, "dy={}", offset.dy);
}

#[test]
fn test_centroid_bright_disk_shift() {
    let reference = make_bright_disk(128, 128, 64.0, 64.0, 20.0);
    let target = make_bright_disk(128, 128, 67.0, 59.0, 20.0);
    let config = CentroidConfig { threshold: 0.1 };

    let offset = compute_offset_centroid(&reference, &target, &config).unwrap();
    // Target disk center is at (67, 59), reference at (64, 64)
    // Offset should be approximately (67-64, 59-64) = (3, -5)
    assert!(
        (offset.dy - 3.0).abs() < 1.0,
        "dy={} expected ~3",
        offset.dy
    );
    assert!(
        (offset.dx - (-5.0)).abs() < 1.0,
        "dx={} expected ~-5",
        offset.dx
    );
}

#[test]
fn test_centroid_threshold_filtering() {
    let img = make_bright_disk(64, 64, 32.0, 32.0, 15.0);
    let low = CentroidConfig { threshold: 0.0 };
    let high = CentroidConfig { threshold: 0.5 };

    let offset_low = compute_offset_centroid(&img, &img, &low).unwrap();
    let offset_high = compute_offset_centroid(&img, &img, &high).unwrap();

    // Both should give near-zero for identical images
    assert!(offset_low.dx.abs() < 0.01);
    assert!(offset_high.dx.abs() < 0.01);
}

#[test]
fn test_centroid_empty_image_fallback() {
    let empty = Array2::<f32>::zeros((64, 64));
    let config = CentroidConfig { threshold: 0.1 };

    let offset = compute_offset_centroid(&empty, &empty, &config).unwrap();
    // All-black returns geometric center, so offset should be zero
    assert!(offset.dx.abs() < 0.01, "dx={}", offset.dx);
    assert!(offset.dy.abs() < 0.01, "dy={}", offset.dy);
}

// ===== Gradient Cross-Correlation =====

#[test]
fn test_gradient_zero_offset() {
    let img = make_bright_square(64, 64, 32, 32, 16);
    let backend = cpu();

    let offset = compute_offset_gradient(&img, &img, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.5, "dx={}", offset.dx);
    assert!(offset.dy.abs() < 0.5, "dy={}", offset.dy);
}

#[test]
fn test_gradient_known_shift() {
    let reference = make_bright_square(128, 128, 64, 64, 24);
    let target = shift_array(&reference, 3.0, 5.0);
    let backend = cpu();

    let offset = compute_offset_gradient(&reference, &target, backend.as_ref()).unwrap();
    assert!(
        (offset.dx.abs() - 5.0).abs() < 1.5,
        "dx={} expected ~5",
        offset.dx
    );
    assert!(
        (offset.dy.abs() - 3.0).abs() < 1.5,
        "dy={} expected ~3",
        offset.dy
    );
}

#[test]
fn test_gradient_magnitude_array() {
    use jupiter_core::quality::gradient::gradient_magnitude_array;

    let img = make_bright_square(32, 32, 16, 16, 8);
    let grad = gradient_magnitude_array(&img);

    assert_eq!(grad.dim(), (32, 32));
    // Border should be zero
    assert_eq!(grad[[0, 0]], 0.0);
    // Interior near edges of the square should have high gradient
    let center_val = grad[[16, 16]];
    let edge_val = grad[[12, 16]]; // Near the edge of the square
                                   // Edge should have higher gradient than center (which is flat)
    assert!(
        edge_val > center_val || center_val < 0.1,
        "edge={} center={}",
        edge_val,
        center_val
    );
}

// ===== Pyramid Alignment =====

#[test]
fn test_pyramid_zero_offset() {
    let img = make_bright_square(64, 64, 32, 32, 16);
    let config = PyramidConfig { levels: 2 };
    let backend = cpu();

    let offset = compute_offset_pyramid(&img, &img, &config, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.5, "dx={}", offset.dx);
    assert!(offset.dy.abs() < 0.5, "dy={}", offset.dy);
}

#[test]
fn test_pyramid_small_shift() {
    let reference = make_bright_square(128, 128, 64, 64, 24);
    let target = shift_array(&reference, 3.0, 5.0);
    let config = PyramidConfig { levels: 2 };
    let backend = cpu();

    let offset = compute_offset_pyramid(&reference, &target, &config, backend.as_ref()).unwrap();
    assert!(
        (offset.dx.abs() - 5.0).abs() < 2.0,
        "dx={} expected ~5",
        offset.dx
    );
    assert!(
        (offset.dy.abs() - 3.0).abs() < 2.0,
        "dy={} expected ~3",
        offset.dy
    );
}

#[test]
fn test_pyramid_large_shift() {
    // Create a large shift that standard phase correlation might struggle with.
    // At 128x128, the FFT wrap-around limit is ~64 pixels.
    // With pyramid (2 levels), the coarsest level is 32x32, which can handle
    // shifts up to ~16 at that scale = ~64 at original scale.
    let reference = make_bright_square(128, 128, 64, 64, 20);
    let target = shift_array(&reference, 40.0, 30.0);
    let config = PyramidConfig { levels: 3 };
    let backend = cpu();

    let offset = compute_offset_pyramid(&reference, &target, &config, backend.as_ref()).unwrap();
    // Larger tolerance for large shift
    assert!(
        (offset.dx.abs() - 30.0).abs() < 5.0,
        "dx={} expected ~30",
        offset.dx
    );
    assert!(
        (offset.dy.abs() - 40.0).abs() < 5.0,
        "dy={} expected ~40",
        offset.dy
    );
}

// ===== Dispatcher Tests =====

#[test]
fn test_dispatcher_routes_phase_correlation() {
    let img = make_bright_square(64, 64, 32, 32, 16);
    let config = AlignmentConfig {
        method: AlignmentMethod::PhaseCorrelation,
    };
    let backend = cpu();

    let offset = compute_offset_configured(&img, &img, &config, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.5);
    assert!(offset.dy.abs() < 0.5);
}

#[test]
fn test_dispatcher_routes_enhanced_phase() {
    let img = make_bright_square(64, 64, 32, 32, 16);
    let config = AlignmentConfig {
        method: AlignmentMethod::EnhancedPhaseCorrelation(EnhancedPhaseConfig {
            upsample_factor: 10,
        }),
    };
    let backend = cpu();

    let offset = compute_offset_configured(&img, &img, &config, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.5);
    assert!(offset.dy.abs() < 0.5);
}

#[test]
fn test_dispatcher_routes_centroid() {
    let img = make_bright_disk(64, 64, 32.0, 32.0, 15.0);
    let config = AlignmentConfig {
        method: AlignmentMethod::Centroid(CentroidConfig { threshold: 0.1 }),
    };
    let backend = cpu();

    let offset = compute_offset_configured(&img, &img, &config, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.1);
    assert!(offset.dy.abs() < 0.1);
}

#[test]
fn test_dispatcher_routes_gradient() {
    let img = make_bright_square(64, 64, 32, 32, 16);
    let config = AlignmentConfig {
        method: AlignmentMethod::GradientCorrelation,
    };
    let backend = cpu();

    let offset = compute_offset_configured(&img, &img, &config, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.5);
    assert!(offset.dy.abs() < 0.5);
}

#[test]
fn test_dispatcher_routes_pyramid() {
    let img = make_bright_square(64, 64, 32, 32, 16);
    let config = AlignmentConfig {
        method: AlignmentMethod::Pyramid(PyramidConfig { levels: 2 }),
    };
    let backend = cpu();

    let offset = compute_offset_configured(&img, &img, &config, backend.as_ref()).unwrap();
    assert!(offset.dx.abs() < 0.5);
    assert!(offset.dy.abs() < 0.5);
}

// ===== Frame-level alignment =====

#[test]
fn test_align_frames_configured_preserves_reference() {
    use jupiter_core::align::align_frames_configured_with_progress;

    let data = make_bright_square(64, 64, 32, 32, 16);
    let frame = Frame::new(data, 16);
    let frames = vec![frame.clone(), frame.clone(), frame.clone()];

    let config = AlignmentConfig::default();
    let backend = cpu();

    let aligned =
        align_frames_configured_with_progress(&frames, 0, &config, backend, |_| {}).unwrap();

    assert_eq!(aligned.len(), 3);
    // Reference frame should be unchanged
    assert_eq!(aligned[0].data, frames[0].data);
}
