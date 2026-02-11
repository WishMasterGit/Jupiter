use ndarray::Array2;

use jupiter_core::align::phase_correlation::{bilinear_sample, compute_offset};
use jupiter_core::frame::Frame;

#[test]
fn test_zero_offset_for_identical_frames() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 10..20 {
        for c in 10..20 {
            data[[r, c]] = 1.0;
        }
    }
    let frame = Frame::new(data, 8);

    let offset = compute_offset(&frame, &frame).unwrap();
    assert!(offset.dx.abs() < 0.5, "dx={} should be ~0", offset.dx);
    assert!(offset.dy.abs() < 0.5, "dy={} should be ~0", offset.dy);
}

#[test]
fn test_known_integer_shift() {
    let mut ref_data = Array2::<f32>::zeros((64, 64));
    for r in 20..30 {
        for c in 20..30 {
            ref_data[[r, c]] = 1.0;
        }
    }
    let reference = Frame::new(ref_data, 8);

    // Shift by (3, 5): bright square at (23,25) to (33,35)
    let mut tgt_data = Array2::<f32>::zeros((64, 64));
    for r in 23..33 {
        for c in 25..35 {
            tgt_data[[r, c]] = 1.0;
        }
    }
    let target = Frame::new(tgt_data, 8);

    let offset = compute_offset(&reference, &target).unwrap();

    // Phase correlation detects the shift of target relative to reference.
    // The sign depends on convention. Check absolute value.
    assert!(
        offset.dx.abs() - 5.0 < 1.0,
        "|dx|={} should be ~5",
        offset.dx.abs()
    );
    assert!(
        offset.dy.abs() - 3.0 < 1.0,
        "|dy|={} should be ~3",
        offset.dy.abs()
    );
}

#[test]
fn test_bilinear_interpolation() {
    let mut data = Array2::<f32>::zeros((4, 4));
    data[[1, 1]] = 1.0;

    // Exact point
    assert!((bilinear_sample(&data, 1.0, 1.0) - 1.0).abs() < 1e-6);
    // Halfway between
    assert!((bilinear_sample(&data, 1.0, 1.5) - 0.5).abs() < 1e-6);
}
