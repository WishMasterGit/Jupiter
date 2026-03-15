use ndarray::Array2;

use jupiter_core::align::pre_center::{
    compute_centering_offsets, compute_common_overlap, crop_frame, detect_centroids,
    pre_center_color_frames, pre_center_frames,
};
use jupiter_core::detection::config::DetectionConfig;
use jupiter_core::frame::{AlignmentOffset, ColorFrame, Frame};
use jupiter_core::io::crop::CropRect;

/// Create a synthetic frame with a bright disk at (center_y, center_x).
fn make_disk_frame(h: usize, w: usize, center_y: f64, center_x: f64, radius: f64) -> Frame {
    let mut data = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let dy = row as f64 - center_y;
            let dx = col as f64 - center_x;
            if (dy * dy + dx * dx).sqrt() <= radius {
                data[[row, col]] = 0.9;
            }
        }
    }
    Frame::new(data, 16)
}

#[test]
fn test_detect_centroids_basic() {
    let frames = vec![
        make_disk_frame(128, 128, 40.0, 50.0, 15.0),
        make_disk_frame(128, 128, 70.0, 80.0, 15.0),
        make_disk_frame(128, 128, 64.0, 64.0, 15.0),
    ];
    let config = DetectionConfig::default();
    let centroids = detect_centroids(&frames, &config);

    assert_eq!(centroids.len(), 3);
    for c in &centroids {
        assert!(c.is_some(), "Expected detection to succeed");
    }

    // Check that detected centroids are approximately correct
    let (cy0, cx0) = centroids[0].unwrap();
    assert!((cy0 - 40.0).abs() < 3.0, "cy0={cy0}");
    assert!((cx0 - 50.0).abs() < 3.0, "cx0={cx0}");

    let (cy1, cx1) = centroids[1].unwrap();
    assert!((cy1 - 70.0).abs() < 3.0, "cy1={cy1}");
    assert!((cx1 - 80.0).abs() < 3.0, "cx1={cx1}");
}

#[test]
fn test_centering_offsets_basic() {
    // Planet at (40, 50) in a 128x128 frame → should shift to (64, 64)
    let centroids = vec![Some((40.0, 50.0)), Some((70.0, 80.0)), Some((64.0, 64.0))];
    let offsets = compute_centering_offsets(&centroids, 128, 128).unwrap();

    assert_eq!(offsets.len(), 3);
    // Frame 0: shift = center - centroid = (64-40, 64-50) = (24, 14)
    assert!((offsets[0].dy - 24.0).abs() < 0.01);
    assert!((offsets[0].dx - 14.0).abs() < 0.01);
    // Frame 2: already centered → near zero
    assert!(offsets[2].dx.abs() < 0.01);
    assert!(offsets[2].dy.abs() < 0.01);
}

#[test]
fn test_centering_offsets_failed_detection_uses_median() {
    // Two successful detections + one failure
    let centroids = vec![Some((40.0, 50.0)), None, Some((60.0, 70.0))];
    let offsets = compute_centering_offsets(&centroids, 128, 128).unwrap();

    // The failed detection (index 1) should use median of successful centroids
    // median cy = 40 or 60 (with 2 elements, picks index 1 = 60)
    // median cx = 50 or 70 (picks index 1 = 70)
    assert_eq!(offsets.len(), 3);
    // Just verify it produces a valid offset (not NaN)
    assert!(offsets[1].dx.is_finite());
    assert!(offsets[1].dy.is_finite());
}

#[test]
fn test_centering_offsets_majority_failure_returns_none() {
    // More than 50% failures → skip pre-centering
    let centroids = vec![None, None, Some((64.0, 64.0))];
    let result = compute_centering_offsets(&centroids, 128, 128);
    assert!(
        result.is_none(),
        "Should return None when majority of detections fail"
    );
}

#[test]
fn test_common_overlap_computation() {
    // Shift (+10, +5) and (-10, -5) — overlap should be reduced symmetrically
    let offsets = vec![
        AlignmentOffset { dx: 5.0, dy: 10.0 },
        AlignmentOffset {
            dx: -5.0,
            dy: -10.0,
        },
    ];
    let crop = compute_common_overlap(&offsets, 100, 100).unwrap();

    // Valid region: x=[5..95], y=[10..90]
    assert_eq!(crop.x, 5);
    assert_eq!(crop.y, 10);
    assert_eq!(crop.width, 90);
    assert_eq!(crop.height, 80);
}

#[test]
fn test_common_overlap_too_small_returns_none() {
    // Shifts so large that overlap is tiny
    let offsets = vec![
        AlignmentOffset { dx: 80.0, dy: 80.0 },
        AlignmentOffset {
            dx: -80.0,
            dy: -80.0,
        },
    ];
    let result = compute_common_overlap(&offsets, 100, 100);
    assert!(
        result.is_none(),
        "Should return None when overlap is too small"
    );
}

#[test]
fn test_crop_frame() {
    let mut data = Array2::<f32>::zeros((100, 100));
    data[[10, 20]] = 0.5;
    let frame = Frame::new(data, 16);

    let crop = CropRect {
        x: 10,
        y: 5,
        width: 30,
        height: 20,
    };
    let cropped = crop_frame(&frame, &crop);
    assert_eq!(cropped.width(), 30);
    assert_eq!(cropped.height(), 20);
    // The pixel at (10, 20) in original → (5, 10) in cropped
    assert!((cropped.data[[5, 10]] - 0.5).abs() < 1e-6);
}

#[test]
fn test_pre_center_frames_shifts_to_center() {
    // Create a frame with disk off-center, then verify it moves toward center
    let frame = make_disk_frame(128, 128, 40.0, 50.0, 15.0);
    let offsets = vec![AlignmentOffset { dx: 14.0, dy: 24.0 }]; // shift to center

    let centered = pre_center_frames(std::slice::from_ref(&frame), &offsets, None);
    assert_eq!(centered.len(), 1);
    assert_eq!(centered[0].width(), 128);
    assert_eq!(centered[0].height(), 128);

    // The peak brightness should now be near the center
    let (h, w) = centered[0].data.dim();
    let center_region = centered[0]
        .data
        .slice(ndarray::s![h / 2 - 5..h / 2 + 5, w / 2 - 5..w / 2 + 5]);
    let center_mean: f32 = center_region.iter().sum::<f32>() / center_region.len() as f32;
    assert!(center_mean > 0.5, "Center should be bright: {center_mean}");
}

#[test]
fn test_pre_center_color_frames() {
    let disk_r = make_disk_frame(64, 64, 20.0, 20.0, 10.0);
    let disk_g = make_disk_frame(64, 64, 20.0, 20.0, 10.0);
    let disk_b = make_disk_frame(64, 64, 20.0, 20.0, 10.0);
    let cf = ColorFrame {
        red: disk_r,
        green: disk_g,
        blue: disk_b,
    };

    let offsets = vec![AlignmentOffset { dx: 12.0, dy: 12.0 }];
    let centered = pre_center_color_frames(&[cf], &offsets, None);
    assert_eq!(centered.len(), 1);
    assert_eq!(centered[0].red.width(), 64);

    // All channels should have the same shift applied
    let r_center = centered[0].red.data[[32, 32]];
    let g_center = centered[0].green.data[[32, 32]];
    let b_center = centered[0].blue.data[[32, 32]];
    assert!(
        (r_center - g_center).abs() < 1e-6,
        "R and G should match at center"
    );
    assert!(
        (g_center - b_center).abs() < 1e-6,
        "G and B should match at center"
    );
}

#[test]
fn test_noop_when_planet_centered() {
    // Planet already at center → offsets should be near zero
    let centroids = vec![Some((64.0, 64.0)), Some((64.0, 64.0))];
    let offsets = compute_centering_offsets(&centroids, 128, 128).unwrap();

    for offset in &offsets {
        assert!(
            offset.dx.abs() < 0.01,
            "dx should be near zero: {}",
            offset.dx
        );
        assert!(
            offset.dy.abs() < 0.01,
            "dy should be near zero: {}",
            offset.dy
        );
    }
}

#[test]
fn test_pre_center_with_crop() {
    let frame = make_disk_frame(128, 128, 40.0, 50.0, 15.0);
    let offsets = vec![
        AlignmentOffset { dx: 14.0, dy: 24.0 },
        AlignmentOffset {
            dx: -10.0,
            dy: -20.0,
        },
    ];
    let crop = compute_common_overlap(&offsets, 128, 128).unwrap();

    let frames = vec![frame.clone(), make_disk_frame(128, 128, 84.0, 74.0, 15.0)];
    let centered = pre_center_frames(&frames, &offsets, Some(&crop));
    assert_eq!(centered.len(), 2);
    // Both frames should be cropped to the same size
    assert_eq!(centered[0].width(), centered[1].width());
    assert_eq!(centered[0].height(), centered[1].height());
    // Dimensions should match the crop rect
    assert_eq!(centered[0].width(), crop.width as usize);
    assert_eq!(centered[0].height(), crop.height as usize);
}

#[test]
fn test_end_to_end_detect_and_center() {
    // End-to-end: detect centroids → compute offsets → pre-center
    let frames = vec![
        make_disk_frame(128, 128, 40.0, 50.0, 15.0),
        make_disk_frame(128, 128, 80.0, 90.0, 15.0),
    ];

    let config = DetectionConfig::default();
    let centroids = detect_centroids(&frames, &config);
    let offsets = compute_centering_offsets(&centroids, 128, 128).unwrap();
    let crop = compute_common_overlap(&offsets, 128, 128);
    let centered = pre_center_frames(&frames, &offsets, crop.as_ref());

    // Both frames should now have their disk near the image center
    for (i, frame) in centered.iter().enumerate() {
        let (h, w) = frame.data.dim();
        let cy = h / 2;
        let cx = w / 2;
        // Check 5x5 region around center has significant brightness
        let region = frame
            .data
            .slice(ndarray::s![cy - 3..cy + 3, cx - 3..cx + 3]);
        let mean: f32 = region.iter().sum::<f32>() / region.len() as f32;
        assert!(
            mean > 0.3,
            "Frame {i}: center region should be bright after centering, got {mean}"
        );
    }
}
