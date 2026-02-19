use std::io::Write;
use tempfile::NamedTempFile;

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
fn test_autocrop_near_border() {
    let w = 64u32;
    let h = 64u32;
    // Planet partially outside the frame (center at edge).
    let cx = 5.0;
    let cy = 5.0;
    let radius = 8.0;
    let frame = make_planet_frame(w, h, cx, cy, radius);
    let ser = build_synthetic_ser(w, h, &[frame]);
    let tmp = write_temp_ser(&ser);
    let reader = SerReader::open(tmp.path()).unwrap();

    let crop = auto_detect_crop(&reader, &AutoCropConfig::default()).unwrap();

    // Crop must not exceed image bounds.
    assert!(crop.x + crop.width <= w);
    assert!(crop.y + crop.height <= h);
    // Crop must start at or near 0 since planet is at corner.
    assert!(crop.x <= 1);
    assert!(crop.y <= 1);
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
