use jupiter_core::frame::{AlignmentOffset, Frame};
use jupiter_core::stack::drizzle::{drizzle_output_dims, drizzle_single_frame, DrizzleConfig};
use jupiter_core::stack::mean::mean_stack;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::sigma_clip::{sigma_clip_stack, SigmaClipParams};
use ndarray::Array2;

#[test]
fn test_drizzle_single_frame_scale2() {
    let input = Array2::from_shape_vec((2, 2), vec![0.25, 0.5, 0.75, 1.0]).unwrap();
    let frame = Frame::new(input, 8);
    let offset = AlignmentOffset::default();

    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 1.0,
        ..Default::default()
    };

    let result = drizzle_single_frame(&frame, &offset, &config).unwrap();
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
        ..Default::default()
    };

    let result = drizzle_single_frame(&frame, &offset, &config).unwrap();
    assert_eq!(result.height(), 30);
    assert_eq!(result.width(), 45);
}

#[test]
fn test_drizzle_output_dims_helper() {
    assert_eq!(drizzle_output_dims(10, 15, 2.0), (20, 30));
    assert_eq!(drizzle_output_dims(10, 15, 3.0), (30, 45));
    assert_eq!(drizzle_output_dims(3, 3, 1.5), (5, 5)); // ceil(4.5)
}

#[test]
fn test_drizzle_then_mean_stack() {
    let data = Array2::from_elem((4, 4), 0.8_f32);
    let f1 = Frame::new(data.clone(), 8);
    let f2 = Frame::new(data, 8);

    let offset1 = AlignmentOffset { dx: 0.0, dy: 0.0 };
    let offset2 = AlignmentOffset { dx: 0.5, dy: 0.5 };

    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.6,
        ..Default::default()
    };

    let d1 = drizzle_single_frame(&f1, &offset1, &config).unwrap();
    let d2 = drizzle_single_frame(&f2, &offset2, &config).unwrap();
    assert_eq!(d1.height(), 8);
    assert_eq!(d1.width(), 8);

    let result = mean_stack(&[d1, d2]).unwrap();
    assert_eq!(result.height(), 8);
    assert_eq!(result.width(), 8);
    // Interior pixel should be close to 0.8 (both frames contribute).
    let center = result.data[[4, 4]];
    assert!(
        center > 0.5,
        "Center pixel should have contributions: {center}"
    );
}

#[test]
fn test_drizzle_then_median_stack() {
    let data = Array2::from_elem((4, 4), 0.6_f32);
    let offsets = vec![
        AlignmentOffset { dx: 0.0, dy: 0.0 },
        AlignmentOffset { dx: 0.3, dy: 0.1 },
        AlignmentOffset { dx: -0.2, dy: 0.4 },
    ];
    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.8,
        ..Default::default()
    };

    let drizzled: Vec<Frame> = (0..3)
        .map(|i| {
            let frame = Frame::new(data.clone(), 8);
            drizzle_single_frame(&frame, &offsets[i], &config).unwrap()
        })
        .collect();

    let result = median_stack(&drizzled).unwrap();
    assert_eq!(result.height(), 8);
    assert_eq!(result.width(), 8);
}

#[test]
fn test_drizzle_then_sigma_clip_stack() {
    let data = Array2::from_elem((4, 4), 0.5_f32);
    let offsets = vec![
        AlignmentOffset { dx: 0.0, dy: 0.0 },
        AlignmentOffset { dx: 0.2, dy: -0.1 },
        AlignmentOffset { dx: -0.3, dy: 0.2 },
        AlignmentOffset { dx: 0.1, dy: 0.3 },
    ];
    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.7,
        ..Default::default()
    };
    let params = SigmaClipParams {
        sigma: 2.5,
        iterations: 2,
    };

    let drizzled: Vec<Frame> = offsets
        .iter()
        .map(|offset| {
            let frame = Frame::new(data.clone(), 8);
            drizzle_single_frame(&frame, offset, &config).unwrap()
        })
        .collect();

    let result = sigma_clip_stack(&drizzled, &params).unwrap();
    assert_eq!(result.height(), 8);
    assert_eq!(result.width(), 8);
}

#[test]
fn test_drizzle_invalid_pixfrac() {
    let frame = Frame::new(Array2::ones((4, 4)), 8);
    let config = DrizzleConfig {
        pixfrac: 0.0,
        ..Default::default()
    };
    let result = drizzle_single_frame(&frame, &AlignmentOffset::default(), &config);
    assert!(result.is_err());
}

#[test]
fn test_drizzle_invalid_scale() {
    let frame = Frame::new(Array2::ones((4, 4)), 8);
    let config = DrizzleConfig {
        scale: 0.0,
        ..Default::default()
    };
    let result = drizzle_single_frame(&frame, &AlignmentOffset::default(), &config);
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
        ..Default::default()
    };

    let result = drizzle_single_frame(&frame, &offset, &config).unwrap();
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

    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.7,
        ..Default::default()
    };

    let drizzled: Vec<Frame> = offsets
        .iter()
        .map(|offset| {
            let frame = Frame::new(data.clone(), 8);
            drizzle_single_frame(&frame, offset, &config).unwrap()
        })
        .collect();

    let result = mean_stack(&drizzled).unwrap();
    assert_eq!(result.height(), 16);
    assert_eq!(result.width(), 16);
    // Interior pixel should be close to 0.5.
    let center = result.data[[8, 8]];
    assert!(
        (center - 0.5).abs() < 0.05,
        "Multiple frames should converge to input value: {center}"
    );
}

#[test]
fn test_drizzle_subpixel_offset() {
    // Verify drizzle handles sub-pixel offsets correctly
    let frame = Frame::new(Array2::from_elem((4, 4), 0.7_f32), 8);
    let offset = AlignmentOffset { dx: 0.5, dy: 0.5 };
    let config = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.7,
        ..Default::default()
    };

    let result = drizzle_single_frame(&frame, &offset, &config).unwrap();
    assert_eq!(result.height(), 8);
    assert_eq!(result.width(), 8);
    // Result should have valid pixel values
    assert!(result.data.iter().all(|&v| (0.0..=1.0).contains(&v)));
}
