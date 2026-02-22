use jupiter_core::frame::{AlignmentOffset, Frame};
use jupiter_core::stack::drizzle::{drizzle_stack, DrizzleConfig};
use ndarray::Array2;

#[test]
fn test_drizzle_single_frame_scale2() {
    let input = Array2::from_shape_vec((2, 2), vec![0.25, 0.5, 0.75, 1.0]).unwrap();
    let frame = Frame::new(input, 8);
    let offset = AlignmentOffset::default();

    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 1.0,
        quality_weighted: false,
        ..Default::default()
    };

    let result = drizzle_stack(&[frame], &[offset], &config, None).unwrap();
    assert_eq!(result.height(), 4);
    assert_eq!(result.width(), 4);
    // All output pixels should have values in [0, 1] (full coverage with pixfrac=1.0).
    assert!(result.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
}

#[test]
fn test_drizzle_output_dimensions_scale3() {
    let frame = Frame::new(Array2::ones((10, 15)), 8);
    let offset = AlignmentOffset::default();
    let config = DrizzleConfig {
        scale: 3.0,
        pixfrac: 0.7,
        quality_weighted: false,
        ..Default::default()
    };

    let result = drizzle_stack(&[frame], &[offset], &config, None).unwrap();
    assert_eq!(result.height(), 30);
    assert_eq!(result.width(), 45);
}

#[test]
fn test_drizzle_two_frames_subpixel() {
    let data = Array2::from_elem((4, 4), 0.8_f32);
    let f1 = Frame::new(data.clone(), 8);
    let f2 = Frame::new(data, 8);

    let offset1 = AlignmentOffset { dx: 0.0, dy: 0.0 };
    let offset2 = AlignmentOffset { dx: 0.5, dy: 0.5 };

    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.6,
        quality_weighted: false,
        ..Default::default()
    };

    let result = drizzle_stack(&[f1, f2], &[offset1, offset2], &config, None).unwrap();
    assert_eq!(result.height(), 8);
    assert_eq!(result.width(), 8);
    // Interior pixels should be close to 0.8 (both frames contribute).
    let center = result.data[[4, 4]];
    assert!(
        center > 0.5,
        "Center pixel should have contributions: {center}"
    );
}

#[test]
fn test_drizzle_quality_weighting() {
    let f1 = Frame::new(Array2::from_elem((2, 2), 1.0_f32), 8);
    let f2 = Frame::new(Array2::from_elem((2, 2), 0.0_f32), 8);

    let offsets = vec![AlignmentOffset::default(), AlignmentOffset::default()];
    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 1.0,
        quality_weighted: true,
        ..Default::default()
    };

    // First frame has much higher quality score.
    let scores = vec![10.0, 1.0];
    let result = drizzle_stack(&[f1, f2], &offsets, &config, Some(&scores)).unwrap();

    // Expected: (1.0*10.0 + 0.0*1.0) / (10.0 + 1.0) ~ 0.909
    let pixel = result.data[[0, 0]];
    assert!(
        pixel > 0.85,
        "Quality-weighted pixel should favor f1: {pixel}"
    );
}

#[test]
fn test_drizzle_empty_input() {
    let config = DrizzleConfig::default();
    let result = drizzle_stack(&[], &[], &config, None);
    assert!(result.is_err());
}

#[test]
fn test_drizzle_dimension_mismatch() {
    let frame = Frame::new(Array2::ones((4, 4)), 8);
    let config = DrizzleConfig::default();
    // 1 frame but 2 offsets.
    let result = drizzle_stack(
        &[frame],
        &[AlignmentOffset::default(), AlignmentOffset::default()],
        &config,
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_drizzle_invalid_pixfrac() {
    let frame = Frame::new(Array2::ones((4, 4)), 8);
    let config = DrizzleConfig {
        pixfrac: 0.0,
        ..Default::default()
    };
    let result = drizzle_stack(&[frame], &[AlignmentOffset::default()], &config, None);
    assert!(result.is_err());
}

#[test]
fn test_drizzle_uniform_image_preserves_value() {
    // A uniform image drizzled at scale 2 should produce a uniform output.
    let value = 0.6_f32;
    let frame = Frame::new(Array2::from_elem((8, 8), value), 8);
    let offset = AlignmentOffset::default();
    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 1.0,
        quality_weighted: false,
        ..Default::default()
    };

    let result = drizzle_stack(&[frame], &[offset], &config, None).unwrap();
    // Interior pixels (away from edges) should be close to the input value.
    let center = result.data[[8, 8]];
    assert!(
        (center - value).abs() < 0.01,
        "Uniform image should preserve value: expected {value}, got {center}"
    );
}

#[test]
fn test_drizzle_multiple_frames_converge() {
    // More frames with varied offsets should produce a cleaner result.
    let data = Array2::from_elem((8, 8), 0.5_f32);
    let offsets = vec![
        AlignmentOffset { dx: 0.0, dy: 0.0 },
        AlignmentOffset { dx: 0.3, dy: 0.1 },
        AlignmentOffset { dx: -0.2, dy: 0.4 },
        AlignmentOffset { dx: 0.5, dy: -0.3 },
        AlignmentOffset { dx: -0.1, dy: -0.2 },
    ];
    let frames: Vec<Frame> = (0..5).map(|_| Frame::new(data.clone(), 8)).collect();

    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.7,
        quality_weighted: false,
        ..Default::default()
    };

    let result = drizzle_stack(&frames, &offsets, &config, None).unwrap();
    assert_eq!(result.height(), 16);
    assert_eq!(result.width(), 16);
    // Interior pixel should be close to 0.5.
    let center = result.data[[8, 8]];
    assert!(
        (center - 0.5).abs() < 0.05,
        "Multiple frames should converge to input value: {center}"
    );
}
