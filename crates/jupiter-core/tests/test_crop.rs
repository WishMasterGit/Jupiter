use std::io::Write;
use tempfile::NamedTempFile;

use jupiter_core::frame::ColorMode;
use jupiter_core::io::crop::{crop_ser, CropRect};
use jupiter_core::io::ser::SerReader;

const SER_HEADER_SIZE: usize = 178;

/// Build a minimal synthetic SER file in memory.
fn build_synthetic_ser(
    width: u32,
    height: u32,
    bit_depth: u32,
    color_id: i32,
    frames: &[Vec<u8>],
    timestamps: Option<&[u64]>,
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic (14 bytes)
    buf.extend_from_slice(b"LUCAM-RECORDER");
    // LuID (4 bytes)
    buf.extend_from_slice(&0i32.to_le_bytes());
    // ColorID
    buf.extend_from_slice(&color_id.to_le_bytes());
    // LittleEndian = 0 (little-endian per Siril convention)
    buf.extend_from_slice(&0i32.to_le_bytes());
    // Width
    buf.extend_from_slice(&(width as i32).to_le_bytes());
    // Height
    buf.extend_from_slice(&(height as i32).to_le_bytes());
    // PixelDepth
    buf.extend_from_slice(&(bit_depth as i32).to_le_bytes());
    // FrameCount
    buf.extend_from_slice(&(frames.len() as i32).to_le_bytes());
    // Observer (40 bytes)
    let mut observer = [0u8; 40];
    observer[..4].copy_from_slice(b"Test");
    buf.extend_from_slice(&observer);
    // Instrument (40 bytes)
    buf.extend_from_slice(&[0u8; 40]);
    // Telescope (40 bytes)
    let mut telescope = [0u8; 40];
    telescope[..7].copy_from_slice(b"MyScope");
    buf.extend_from_slice(&telescope);
    // DateTime (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // DateTimeUTC (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());

    assert_eq!(buf.len(), SER_HEADER_SIZE);

    // Frame data
    for frame in frames {
        buf.extend_from_slice(frame);
    }

    // Optional timestamps
    if let Some(ts) = timestamps {
        for &t in ts {
            buf.extend_from_slice(&t.to_le_bytes());
        }
    }

    buf
}

fn write_temp(data: &[u8]) -> NamedTempFile {
    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(data).unwrap();
    tmpfile.flush().unwrap();
    tmpfile
}

#[test]
fn test_crop_8bit_mono() {
    // 4x4 image, crop center 2x2
    let w = 4u32;
    let h = 4u32;
    let mut frame_data = vec![0u8; 16];
    // Fill with row*4+col pattern
    for row in 0..4u8 {
        for col in 0..4u8 {
            frame_data[(row * 4 + col) as usize] = row * 16 + col;
        }
    }
    let ser_data = build_synthetic_ser(w, h, 8, 0, &[frame_data], None);
    let src = write_temp(&ser_data);

    let reader = SerReader::open(src.path()).unwrap();
    let crop = CropRect { x: 1, y: 1, width: 2, height: 2 };

    let dst = NamedTempFile::new().unwrap();
    crop_ser(&reader, dst.path(), &crop, |_, _| {}).unwrap();

    let cropped = SerReader::open(dst.path()).unwrap();
    assert_eq!(cropped.header.width, 2);
    assert_eq!(cropped.header.height, 2);
    assert_eq!(cropped.header.pixel_depth, 8);
    assert_eq!(cropped.frame_count(), 1);

    let raw = cropped.frame_raw(0).unwrap();
    // Row 1, col 1..3 of original => values 1*16+1=17, 1*16+2=18
    // Row 2, col 1..3 => 2*16+1=33, 2*16+2=34
    assert_eq!(raw, &[17, 18, 33, 34]);
}

#[test]
fn test_crop_16bit_mono() {
    // 4x3 image (16-bit), crop left 2x2
    let w = 4u32;
    let h = 3u32;
    let mut frame_data = Vec::new();
    for row in 0..3u16 {
        for col in 0..4u16 {
            let val = row * 1000 + col * 100;
            frame_data.extend_from_slice(&val.to_le_bytes());
        }
    }
    let ser_data = build_synthetic_ser(w, h, 16, 0, &[frame_data], None);
    let src = write_temp(&ser_data);

    let reader = SerReader::open(src.path()).unwrap();
    let crop = CropRect { x: 0, y: 0, width: 2, height: 2 };

    let dst = NamedTempFile::new().unwrap();
    crop_ser(&reader, dst.path(), &crop, |_, _| {}).unwrap();

    let cropped = SerReader::open(dst.path()).unwrap();
    assert_eq!(cropped.header.width, 2);
    assert_eq!(cropped.header.height, 2);
    assert_eq!(cropped.header.pixel_depth, 16);

    let raw = cropped.frame_raw(0).unwrap();
    // Row 0: 0, 100; Row 1: 1000, 1100 (all LE u16)
    let expected: Vec<u8> = [0u16, 100, 1000, 1100]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(raw, &expected[..]);
}

#[test]
fn test_crop_bayer_snaps_to_even() {
    // Bayer RGGB (color_id=8), 6x6 image
    let w = 6u32;
    let h = 6u32;
    let frame_data = vec![42u8; 36];
    let ser_data = build_synthetic_ser(w, h, 8, 8, &[frame_data], None);
    let src = write_temp(&ser_data);

    let reader = SerReader::open(src.path()).unwrap();
    assert_eq!(reader.header.color_mode(), ColorMode::BayerRGGB);

    // Request crop at odd x=1, y=1, 3x3 — should snap to 0,0, 2x2
    let crop = CropRect { x: 1, y: 1, width: 3, height: 3 };
    let validated = crop
        .validated(w, h, &reader.header.color_mode())
        .unwrap();
    assert_eq!(validated.x, 0);
    assert_eq!(validated.y, 0);
    assert_eq!(validated.width, 2);
    assert_eq!(validated.height, 2);
}

#[test]
fn test_crop_out_of_bounds_rejected() {
    let w = 4u32;
    let h = 4u32;
    let frame_data = vec![0u8; 16];
    let ser_data = build_synthetic_ser(w, h, 8, 0, &[frame_data], None);
    let src = write_temp(&ser_data);

    let reader = SerReader::open(src.path()).unwrap();

    // Entirely out of bounds
    let crop = CropRect { x: 5, y: 0, width: 2, height: 2 };
    assert!(crop.validated(w, h, &reader.header.color_mode()).is_err());

    // Partially out of bounds
    let crop = CropRect { x: 3, y: 3, width: 2, height: 2 };
    assert!(crop.validated(w, h, &reader.header.color_mode()).is_err());
}

#[test]
fn test_crop_multi_frame_with_timestamps() {
    let w = 4u32;
    let h = 2u32;

    let frame1: Vec<u8> = (0..8).collect();
    let frame2: Vec<u8> = (10..18).collect();
    let frame3: Vec<u8> = (20..28).collect();
    let timestamps = vec![100u64, 200, 300];

    let ser_data = build_synthetic_ser(
        w,
        h,
        8,
        0,
        &[frame1, frame2, frame3],
        Some(&timestamps),
    );
    let src = write_temp(&ser_data);

    let reader = SerReader::open(src.path()).unwrap();
    let crop = CropRect { x: 1, y: 0, width: 2, height: 2 };

    let dst = NamedTempFile::new().unwrap();
    let mut progress_calls = Vec::new();
    crop_ser(&reader, dst.path(), &crop, |done, total| {
        progress_calls.push((done, total));
    })
    .unwrap();

    // Check progress was called 3 times
    assert_eq!(progress_calls.len(), 3);
    assert_eq!(progress_calls[0], (1, 3));
    assert_eq!(progress_calls[2], (3, 3));

    let cropped = SerReader::open(dst.path()).unwrap();
    assert_eq!(cropped.header.width, 2);
    assert_eq!(cropped.header.height, 2);
    assert_eq!(cropped.frame_count(), 3);

    // Frame 0: original row0 cols 1..3 => [1,2], row1 cols 1..3 => [5,6]
    let raw0 = cropped.frame_raw(0).unwrap();
    assert_eq!(raw0, &[1, 2, 5, 6]);

    // Frame 1: original row0 cols 1..3 => [11,12], row1 cols 1..3 => [15,16]
    let raw1 = cropped.frame_raw(1).unwrap();
    assert_eq!(raw1, &[11, 12, 15, 16]);

    // Verify timestamps were preserved
    let f0 = cropped.read_frame(0).unwrap();
    assert_eq!(f0.metadata.timestamp_us, Some(100));
    let f2 = cropped.read_frame(2).unwrap();
    assert_eq!(f2.metadata.timestamp_us, Some(300));
}

#[test]
fn test_crop_round_trip() {
    // Create a 6x4 image, crop to 4x2, then re-read and verify pixel values
    let w = 6u32;
    let h = 4u32;
    let mut frame_data = Vec::new();
    for row in 0..4u8 {
        for col in 0..6u8 {
            frame_data.push(row * 10 + col);
        }
    }
    let ser_data = build_synthetic_ser(w, h, 8, 0, &[frame_data], None);
    let src = write_temp(&ser_data);

    let reader = SerReader::open(src.path()).unwrap();
    let crop = CropRect { x: 1, y: 1, width: 4, height: 2 };

    let dst = NamedTempFile::new().unwrap();
    crop_ser(&reader, dst.path(), &crop, |_, _| {}).unwrap();

    let cropped = SerReader::open(dst.path()).unwrap();
    assert_eq!(cropped.header.width, 4);
    assert_eq!(cropped.header.height, 2);

    // Re-read as f32 and verify values
    let frame = cropped.read_frame(0).unwrap();
    assert_eq!(frame.width(), 4);
    assert_eq!(frame.height(), 2);

    // Original row 1, cols 1..5: [11, 12, 13, 14]
    // Original row 2, cols 1..5: [21, 22, 23, 24]
    let raw = cropped.frame_raw(0).unwrap();
    assert_eq!(raw, &[11, 12, 13, 14, 21, 22, 23, 24]);
}

#[test]
fn test_crop_preserves_metadata() {
    let w = 4u32;
    let h = 4u32;
    let frame_data = vec![0u8; 16];
    let ser_data = build_synthetic_ser(w, h, 8, 0, &[frame_data], None);
    let src = write_temp(&ser_data);

    let reader = SerReader::open(src.path()).unwrap();
    let crop = CropRect { x: 0, y: 0, width: 2, height: 2 };

    let dst = NamedTempFile::new().unwrap();
    crop_ser(&reader, dst.path(), &crop, |_, _| {}).unwrap();

    let cropped = SerReader::open(dst.path()).unwrap();
    // Header metadata should be preserved
    assert_eq!(cropped.header.observer, "Test");
    assert_eq!(cropped.header.telescope, "MyScope");
    assert_eq!(cropped.header.pixel_depth, 8);
    assert_eq!(cropped.header.color_mode(), ColorMode::Mono);
}
