use std::io::Write;
use tempfile::NamedTempFile;

use jupiter_core::frame::ColorMode;
use jupiter_core::io::ser::SerReader;

const SER_HEADER_SIZE: usize = 178;

/// Build a minimal synthetic SER file in memory.
fn build_synthetic_ser(
    width: u32,
    height: u32,
    bit_depth: u32,
    frames: &[Vec<u8>],
) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic (14 bytes)
    buf.extend_from_slice(b"LUCAM-RECORDER");
    // LuID (4 bytes)
    buf.extend_from_slice(&0i32.to_le_bytes());
    // ColorID = MONO (4 bytes)
    buf.extend_from_slice(&0i32.to_le_bytes());
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

    buf
}

#[test]
fn test_parse_8bit_mono() {
    let w = 4u32;
    let h = 3u32;
    let frame_data: Vec<u8> = (0u8..12).collect();
    let ser_data = build_synthetic_ser(w, h, 8, &[frame_data]);

    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(&ser_data).unwrap();

    let reader = SerReader::open(tmpfile.path()).unwrap();
    assert_eq!(reader.frame_count(), 1);
    assert_eq!(reader.header.width, 4);
    assert_eq!(reader.header.height, 3);
    assert_eq!(reader.header.pixel_depth, 8);
    assert_eq!(reader.header.color_mode(), ColorMode::Mono);
    assert_eq!(reader.header.observer, "Test");

    let frame = reader.read_frame(0).unwrap();
    assert_eq!(frame.width(), 4);
    assert_eq!(frame.height(), 3);
    assert!((frame.data[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((frame.data[[0, 1]] - 1.0 / 255.0).abs() < 1e-4);
    assert!((frame.data[[2, 3]] - 11.0 / 255.0).abs() < 1e-4);
}

#[test]
fn test_parse_16bit_mono() {
    let w = 2u32;
    let h = 2u32;
    let values: [u16; 4] = [0, 1000, 32767, 65535];
    let mut frame_data = Vec::new();
    for v in &values {
        frame_data.extend_from_slice(&v.to_le_bytes());
    }
    let ser_data = build_synthetic_ser(w, h, 16, &[frame_data]);

    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(&ser_data).unwrap();

    let reader = SerReader::open(tmpfile.path()).unwrap();
    let frame = reader.read_frame(0).unwrap();

    assert!((frame.data[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((frame.data[[0, 1]] - 1000.0 / 65535.0).abs() < 1e-4);
    assert!((frame.data[[1, 1]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_multiple_frames() {
    let w = 2u32;
    let h = 2u32;
    let frame1: Vec<u8> = vec![0, 50, 100, 200];
    let frame2: Vec<u8> = vec![255, 200, 100, 50];
    let ser_data = build_synthetic_ser(w, h, 8, &[frame1, frame2]);

    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(&ser_data).unwrap();

    let reader = SerReader::open(tmpfile.path()).unwrap();
    assert_eq!(reader.frame_count(), 2);

    let f0 = reader.read_frame(0).unwrap();
    let f1 = reader.read_frame(1).unwrap();

    assert!((f0.data[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((f1.data[[0, 0]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_out_of_range() {
    let w = 2u32;
    let h = 2u32;
    let frame_data: Vec<u8> = vec![0, 0, 0, 0];
    let ser_data = build_synthetic_ser(w, h, 8, &[frame_data]);

    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(&ser_data).unwrap();

    let reader = SerReader::open(tmpfile.path()).unwrap();
    assert!(reader.read_frame(1).is_err());
}

#[test]
fn test_frames_iterator() {
    let w = 2u32;
    let h = 2u32;
    let frame1: Vec<u8> = vec![10, 20, 30, 40];
    let frame2: Vec<u8> = vec![50, 60, 70, 80];
    let frame3: Vec<u8> = vec![90, 100, 110, 120];
    let ser_data = build_synthetic_ser(w, h, 8, &[frame1, frame2, frame3]);

    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(&ser_data).unwrap();

    let reader = SerReader::open(tmpfile.path()).unwrap();
    let frames: Vec<_> = reader.frames().collect::<Result<_, _>>().unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0].metadata.frame_index, 0);
    assert_eq!(frames[2].metadata.frame_index, 2);
}
