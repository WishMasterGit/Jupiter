use std::io::Write;
use tempfile::NamedTempFile;

use ndarray::Array2;

use jupiter_core::io::autocrop::components::{connected_components, touches_border};
use jupiter_core::io::autocrop::detection::{detect_planet_in_frame, FrameDetection};
use jupiter_core::io::autocrop::morphology::morphological_opening;
use jupiter_core::io::autocrop::temporal::analyze_detections;
use jupiter_core::io::autocrop::{auto_detect_crop, AutoCropConfig, ThresholdMethod};
use jupiter_core::io::ser::SerReader;

const SER_HEADER_SIZE: usize = 178;

/// Build a minimal synthetic SER file with 8-bit mono frames.
fn build_synthetic_ser(width: u32, height: u32, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();

    buf.extend_from_slice(b"LUCAM-RECORDER");
    buf.extend_from_slice(&0i32.to_le_bytes()); // LuID
    buf.extend_from_slice(&0i32.to_le_bytes()); // ColorID = MONO
    buf.extend_from_slice(&0i32.to_le_bytes()); // LittleEndian
    buf.extend_from_slice(&(width as i32).to_le_bytes());
    buf.extend_from_slice(&(height as i32).to_le_bytes());
    buf.extend_from_slice(&8i32.to_le_bytes()); // PixelDepth = 8
    buf.extend_from_slice(&(frames.len() as i32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 40]); // Observer
    buf.extend_from_slice(&[0u8; 40]); // Instrument
    buf.extend_from_slice(&[0u8; 40]); // Telescope
    buf.extend_from_slice(&0u64.to_le_bytes()); // DateTime
    buf.extend_from_slice(&0u64.to_le_bytes()); // DateTimeUTC

    assert_eq!(buf.len(), SER_HEADER_SIZE);

    for frame in frames {
        buf.extend_from_slice(frame);
    }

    buf
}

/// Create a frame with a bright circle (planet) on a dark background.
fn make_planet_frame(width: u32, height: u32, cx: f32, cy: f32, radius: f32) -> Vec<u8> {
    let mut data = vec![0u8; (width * height) as usize];
    for row in 0..height {
        for col in 0..width {
            let dx = col as f32 - cx;
            let dy = row as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= radius {
                data[(row * width + col) as usize] = 200;
            }
        }
    }
    data
}

fn write_temp_ser(data: &[u8]) -> NamedTempFile {
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(data).unwrap();
    tmp
}

#[test]
fn test_autocrop_centered_disk() {
    let w = 64u32;
    let h = 64u32;
    let radius = 10.0;
    let frame = make_planet_frame(w, h, 32.0, 32.0, radius);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let config = AutoCropConfig::default();
    let crop = auto_detect_crop(&reader, &config).unwrap();

    // The crop must contain the entire planet disk.
    let planet_left = (32.0 - radius).floor() as u32;
    let planet_top = (32.0 - radius).floor() as u32;
    let planet_right = (32.0 + radius).ceil() as u32;
    let planet_bottom = (32.0 + radius).ceil() as u32;

    assert!(crop.x <= planet_left, "crop.x={} > planet_left={planet_left}", crop.x);
    assert!(crop.y <= planet_top, "crop.y={} > planet_top={planet_top}", crop.y);
    assert!(
        crop.x + crop.width >= planet_right,
        "crop right {} < planet_right {planet_right}",
        crop.x + crop.width
    );
    assert!(
        crop.y + crop.height >= planet_bottom,
        "crop bottom {} < planet_bottom {planet_bottom}",
        crop.y + crop.height
    );
}

#[test]
fn test_autocrop_offset_disk() {
    let w = 64u32;
    let h = 64u32;
    let cx = 15.0;
    let cy = 48.0;
    let radius = 8.0;
    let frame = make_planet_frame(w, h, cx, cy, radius);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let crop = auto_detect_crop(&reader, &AutoCropConfig::default()).unwrap();

    // Crop must contain the off-center planet.
    assert!(crop.x <= (cx - radius).floor() as u32);
    assert!(crop.y <= (cy - radius).floor() as u32);
    assert!(crop.x + crop.width >= (cx + radius).ceil() as u32);
    assert!(crop.y + crop.height >= (cy + radius).ceil() as u32);
}

#[test]
fn test_autocrop_near_border_rejected() {
    let w = 64u32;
    let h = 64u32;
    // Planet partially outside the frame (center at edge, touches border).
    // v2 correctly rejects border-touching planets, so detection should fail.
    let cx = 5.0;
    let cy = 5.0;
    let radius = 8.0;
    let frame = make_planet_frame(w, h, cx, cy, radius);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let result = auto_detect_crop(&reader, &AutoCropConfig::default());
    assert!(result.is_err(), "Border-touching planet should be rejected");
}

#[test]
fn test_autocrop_no_planet() {
    let w = 32u32;
    let h = 32u32;
    let frame = vec![0u8; (w * h) as usize]; // All black
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let result = auto_detect_crop(&reader, &AutoCropConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_autocrop_padding_increases_size() {
    let w = 64u32;
    let h = 64u32;
    let frame = make_planet_frame(w, h, 32.0, 32.0, 10.0);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let small_pad = auto_detect_crop(
        &reader,
        &AutoCropConfig {
            padding_fraction: 0.05,
            ..Default::default()
        },
    )
    .unwrap();

    let large_pad = auto_detect_crop(
        &reader,
        &AutoCropConfig {
            padding_fraction: 0.5,
            ..Default::default()
        },
    )
    .unwrap();

    let small_area = small_pad.width as u64 * small_pad.height as u64;
    let large_area = large_pad.width as u64 * large_pad.height as u64;
    assert!(large_area > small_area, "larger padding should produce bigger crop");
}

#[test]
fn test_autocrop_otsu() {
    let w = 64u32;
    let h = 64u32;
    let frame = make_planet_frame(w, h, 32.0, 32.0, 12.0);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let config = AutoCropConfig {
        threshold_method: ThresholdMethod::Otsu,
        ..Default::default()
    };
    let crop = auto_detect_crop(&reader, &config).unwrap();

    // Should still contain the planet.
    assert!(crop.x <= 20);
    assert!(crop.y <= 20);
    assert!(crop.x + crop.width >= 44);
    assert!(crop.y + crop.height >= 44);
}

#[test]
fn test_autocrop_fixed_threshold() {
    let w = 32u32;
    let h = 32u32;
    // Planet at brightness 200/255 ≈ 0.784
    let frame = make_planet_frame(w, h, 16.0, 16.0, 6.0);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let config = AutoCropConfig {
        threshold_method: ThresholdMethod::Fixed(0.5),
        ..Default::default()
    };
    let crop = auto_detect_crop(&reader, &config).unwrap();

    // Should detect the planet at 0.784 > 0.5 threshold.
    assert!(crop.x <= 10);
    assert!(crop.y <= 10);
    assert!(crop.x + crop.width >= 22);
    assert!(crop.y + crop.height >= 22);
}

// ============================================================
// v2 tests: morphology, CCA, detection, temporal, multi-frame
// ============================================================

/// Create an f32 array from a binary pattern for testing.
fn make_binary_mask(h: usize, w: usize, points: &[(usize, usize)]) -> Array2<bool> {
    let mut mask = Array2::from_elem((h, w), false);
    for &(r, c) in points {
        if r < h && c < w {
            mask[[r, c]] = true;
        }
    }
    mask
}

/// Create a frame with a planet and hot pixels as an f32 array.
fn make_planet_array_with_noise(
    width: usize,
    height: usize,
    cx: f32,
    cy: f32,
    radius: f32,
    hot_pixels: &[(usize, usize)],
) -> Array2<f32> {
    let mut data = Array2::zeros((height, width));
    for row in 0..height {
        for col in 0..width {
            let dx = col as f32 - cx;
            let dy = row as f32 - cy;
            if (dx * dx + dy * dy).sqrt() <= radius {
                data[[row, col]] = 200.0 / 255.0;
            }
        }
    }
    for &(r, c) in hot_pixels {
        if r < height && c < width {
            data[[r, c]] = 1.0; // Max brightness hot pixel.
        }
    }
    data
}

#[test]
fn test_morphological_opening_removes_noise() {
    // Create a 20x20 mask with a 5x5 solid block and 3 isolated pixels.
    let mut points = Vec::new();
    // 5x5 block at (8,8)
    for r in 8..13 {
        for c in 8..13 {
            points.push((r, c));
        }
    }
    // Isolated hot pixels
    points.push((1, 1));
    points.push((2, 18));
    points.push((17, 3));

    let mask = make_binary_mask(20, 20, &points);
    let opened = morphological_opening(&mask);

    // The isolated pixels should be removed.
    assert!(!opened[[1, 1]], "Isolated pixel at (1,1) should be removed");
    assert!(
        !opened[[2, 18]],
        "Isolated pixel at (2,18) should be removed"
    );
    assert!(
        !opened[[17, 3]],
        "Isolated pixel at (17,3) should be removed"
    );

    // The interior of the 5x5 block should be preserved (erosion shrinks
    // by 1 on each side, dilation restores — but corners may be lost).
    // The 3x3 interior at (9..12, 9..12) should survive erosion.
    assert!(opened[[10, 10]], "Center of block should survive opening");
}

#[test]
fn test_morphological_opening_preserves_large_region() {
    // A 7x7 solid block should survive opening with a 3x3 kernel.
    let mut points = Vec::new();
    for r in 5..12 {
        for c in 5..12 {
            points.push((r, c));
        }
    }
    let mask = make_binary_mask(20, 20, &points);
    let opened = morphological_opening(&mask);

    // The interior 5x5 core (6..11, 6..11) should definitely survive.
    for r in 6..11 {
        for c in 6..11 {
            assert!(
                opened[[r, c]],
                "Interior pixel ({r},{c}) should survive opening"
            );
        }
    }
}

#[test]
fn test_cca_single_component() {
    // A single 3x4 block.
    let mut points = Vec::new();
    for r in 2..5 {
        for c in 3..7 {
            points.push((r, c));
        }
    }
    let mask = make_binary_mask(10, 10, &points);
    let components = connected_components(&mask);

    assert_eq!(components.len(), 1, "Should find exactly one component");
    assert_eq!(components[0].area, 12, "Area should be 3*4=12");
    let bbox = components[0].bbox;
    assert_eq!(bbox, (2, 4, 3, 6));
}

#[test]
fn test_cca_multiple_components() {
    // Three separate blobs of different sizes.
    let mut points = Vec::new();
    // Blob 1: 3x3 = 9 pixels at top-left
    for r in 0..3 {
        for c in 0..3 {
            points.push((r, c));
        }
    }
    // Blob 2: 2x2 = 4 pixels at bottom-right
    for r in 8..10 {
        for c in 8..10 {
            points.push((r, c));
        }
    }
    // Blob 3: single pixel
    points.push((5, 5));

    let mask = make_binary_mask(10, 10, &points);
    let components = connected_components(&mask);

    assert_eq!(components.len(), 3, "Should find 3 components");
    // Sorted by area descending.
    assert_eq!(components[0].area, 9, "Largest should be 3x3=9");
    assert_eq!(components[1].area, 4, "Second should be 2x2=4");
    assert_eq!(components[2].area, 1, "Smallest should be 1");
}

#[test]
fn test_touches_border() {
    assert!(touches_border((0, 5, 2, 8), 10, 10)); // touches top
    assert!(touches_border((2, 9, 2, 8), 10, 10)); // touches bottom
    assert!(touches_border((2, 5, 0, 8), 10, 10)); // touches left
    assert!(touches_border((2, 5, 2, 9), 10, 10)); // touches right
    assert!(!touches_border((1, 5, 1, 8), 10, 10)); // fully interior
}

#[test]
fn test_detect_planet_with_hot_pixels() {
    let w = 64;
    let h = 64;
    let hot_pixels = vec![(0, 0), (0, 63), (63, 0), (63, 63), (10, 55)];
    let data = make_planet_array_with_noise(w, h, 32.0, 32.0, 10.0, &hot_pixels);

    let config = AutoCropConfig::default();
    let det = detect_planet_in_frame(&data, 0, &config);

    assert!(det.is_some(), "Should detect planet despite hot pixels");
    let det = det.unwrap();
    // Centroid should be near the planet center (32, 32), not pulled to corners.
    assert!(
        (det.cx - 32.0).abs() < 3.0,
        "cx={} should be near 32",
        det.cx
    );
    assert!(
        (det.cy - 32.0).abs() < 3.0,
        "cy={} should be near 32",
        det.cy
    );
}

#[test]
fn test_detect_planet_at_border_returns_none() {
    let w = 64;
    let h = 64;
    // Planet centered at (2, 2) with radius 8 — touches border.
    let data = make_planet_array_with_noise(w, h, 2.0, 2.0, 8.0, &[]);

    let config = AutoCropConfig::default();
    let det = detect_planet_in_frame(&data, 0, &config);
    assert!(det.is_none(), "Border-touching planet should be rejected");
}

#[test]
fn test_autocrop_v2_multi_frame_drift() {
    let w = 128u32;
    let h = 128u32;
    let radius = 12.0;

    // Create 10 frames with planet drifting from (50, 60) to (60, 70).
    let mut frames = Vec::new();
    for i in 0..10 {
        let cx = 50.0 + i as f32;
        let cy = 60.0 + i as f32;
        frames.push(make_planet_frame(w, h, cx, cy, radius));
    }

    let ser = build_synthetic_ser(w, h, &frames);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let config = AutoCropConfig {
        sample_count: 10,
        ..Default::default()
    };
    let crop = auto_detect_crop(&reader, &config).unwrap();

    // The crop must contain the planet at all positions.
    for i in 0..10 {
        let planet_left = (50.0 + i as f32 - radius).floor() as u32;
        let planet_top = (60.0 + i as f32 - radius).floor() as u32;
        let planet_right = (50.0 + i as f32 + radius).ceil() as u32;
        let planet_bottom = (60.0 + i as f32 + radius).ceil() as u32;

        assert!(
            crop.x <= planet_left,
            "Frame {i}: crop.x={} > planet_left={planet_left}",
            crop.x
        );
        assert!(
            crop.y <= planet_top,
            "Frame {i}: crop.y={} > planet_top={planet_top}",
            crop.y
        );
        assert!(
            crop.x + crop.width >= planet_right,
            "Frame {i}: crop right {} < planet_right {planet_right}",
            crop.x + crop.width
        );
        assert!(
            crop.y + crop.height >= planet_bottom,
            "Frame {i}: crop bottom {} < planet_bottom {planet_bottom}",
            crop.y + crop.height
        );
    }
}

#[test]
fn test_autocrop_v2_outlier_rejection() {
    let w = 128u32;
    let h = 128u32;
    let radius = 10.0;

    // 8 frames with planet at (64, 64), 2 frames with planet at (20, 20) — outliers.
    let mut frames = Vec::new();
    for _ in 0..8 {
        frames.push(make_planet_frame(w, h, 64.0, 64.0, radius));
    }
    for _ in 0..2 {
        frames.push(make_planet_frame(w, h, 20.0, 20.0, radius));
    }

    let ser = build_synthetic_ser(w, h, &frames);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let config = AutoCropConfig {
        sample_count: 10,
        ..Default::default()
    };
    let crop = auto_detect_crop(&reader, &config).unwrap();

    // The crop should be centered near (64, 64), not pulled toward (20, 20).
    let crop_center_x = crop.x as f64 + crop.width as f64 / 2.0;
    let crop_center_y = crop.y as f64 + crop.height as f64 / 2.0;
    assert!(
        (crop_center_x - 64.0).abs() < 10.0,
        "Crop center X={crop_center_x} should be near 64"
    );
    assert!(
        (crop_center_y - 64.0).abs() < 10.0,
        "Crop center Y={crop_center_y} should be near 64"
    );
}

#[test]
fn test_autocrop_v2_fallback_few_detections() {
    let w = 64u32;
    let h = 64u32;

    // 20 frames: only center frames (8-12) have the planet.
    // With sample_count=3, we sample frames 0, 9, 19 — only frame 9 has a planet,
    // giving 1 valid detection (< 3 min), so fallback kicks in.
    // Fallback reads 5 center frames (8-12), all of which have the planet,
    // so the median-combine produces a clear disc.
    let mut frames = Vec::new();
    for i in 0..20 {
        if (8..=12).contains(&i) {
            frames.push(make_planet_frame(w, h, 32.0, 32.0, 10.0));
        } else {
            frames.push(vec![0u8; (w * h) as usize]);
        }
    }

    let ser = build_synthetic_ser(w, h, &frames);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let config = AutoCropConfig {
        sample_count: 3,
        ..Default::default()
    };
    let crop = auto_detect_crop(&reader, &config).unwrap();

    // Should still produce a valid crop containing the planet.
    assert!(crop.x <= 22, "crop.x={} should be <= 22", crop.x);
    assert!(crop.y <= 22, "crop.y={} should be <= 22", crop.y);
    assert!(crop.x + crop.width >= 42);
    assert!(crop.y + crop.height >= 42);
}

#[test]
fn test_autocrop_v2_crop_aligned_to_32() {
    let w = 128u32;
    let h = 128u32;
    let frame = make_planet_frame(w, h, 64.0, 64.0, 15.0);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let config = AutoCropConfig {
        align_to_fft: true,
        ..Default::default()
    };
    let crop = auto_detect_crop(&reader, &config).unwrap();

    assert_eq!(
        crop.width % 32,
        0,
        "Width {} should be multiple of 32",
        crop.width
    );
    assert_eq!(
        crop.height % 32,
        0,
        "Height {} should be multiple of 32",
        crop.height
    );
}

#[test]
fn test_temporal_analysis_basic() {
    let detections = vec![
        FrameDetection {
            frame_index: 0,
            cx: 50.0,
            cy: 60.0,
            bbox_width: 20,
            bbox_height: 22,
            area: 314,
        },
        FrameDetection {
            frame_index: 1,
            cx: 52.0,
            cy: 61.0,
            bbox_width: 21,
            bbox_height: 21,
            area: 320,
        },
        FrameDetection {
            frame_index: 2,
            cx: 51.0,
            cy: 59.0,
            bbox_width: 20,
            bbox_height: 20,
            area: 310,
        },
    ];

    let analysis = analyze_detections(&detections);

    assert_eq!(analysis.valid_count, 3);
    assert!(
        (analysis.median_cx - 51.0).abs() < 0.5,
        "median_cx={}",
        analysis.median_cx
    );
    assert!(
        (analysis.median_cy - 60.0).abs() < 0.5,
        "median_cy={}",
        analysis.median_cy
    );
    assert!(analysis.drift_range_x >= 1.5, "drift_x={}", analysis.drift_range_x);
    assert!(analysis.drift_range_y >= 1.5, "drift_y={}", analysis.drift_range_y);
    assert!(analysis.median_diameter >= 20.0);
}
