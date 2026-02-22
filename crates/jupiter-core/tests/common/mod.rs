use jupiter_core::io::ser::SER_HEADER_SIZE;

/// Build a SER file header for mono 8-bit frames.
///
/// Returns a `Vec<u8>` containing just the 178-byte header.
/// Append frame pixel data after calling this function.
pub fn build_ser_header(width: u32, height: u32, num_frames: usize) -> Vec<u8> {
    build_ser_header_full(width, height, 8, num_frames, 0)
}

/// Build a SER file header with configurable bit depth and color mode.
///
/// `color_id`: 0=MONO, 8=BAYER_RGGB, 9=BAYER_GRBG, 10=BAYER_GBRG, 11=BAYER_BGGR,
///             100=RGB, 101=BGR
pub fn build_ser_header_full(
    width: u32,
    height: u32,
    bit_depth: u32,
    num_frames: usize,
    color_id: i32,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(SER_HEADER_SIZE);

    // Magic (14 bytes)
    buf.extend_from_slice(b"LUCAM-RECORDER");
    // LuID (4 bytes)
    buf.extend_from_slice(&0i32.to_le_bytes());
    // ColorID (4 bytes)
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
    buf.extend_from_slice(&(num_frames as i32).to_le_bytes());
    // Observer (40 bytes)
    buf.extend_from_slice(&[0u8; 40]);
    // Instrument (40 bytes)
    buf.extend_from_slice(&[0u8; 40]);
    // Telescope (40 bytes)
    buf.extend_from_slice(&[0u8; 40]);
    // DateTime (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());
    // DateTimeUTC (8 bytes)
    buf.extend_from_slice(&0u64.to_le_bytes());

    assert_eq!(buf.len(), SER_HEADER_SIZE);
    buf
}

/// Build a complete synthetic mono 8-bit SER file with the given frame data.
pub fn build_ser_with_frames(width: u32, height: u32, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = build_ser_header(width, height, frames.len());
    for frame in frames {
        buf.extend_from_slice(frame);
    }
    buf
}

/// Write a SER buffer to a temporary file and return the temp file handle.
///
/// The file stays alive as long as the returned `NamedTempFile` is not dropped.
pub fn write_test_ser(data: &[u8]) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::NamedTempFile::new().expect("create temp file");
    f.write_all(data).expect("write SER data");
    f.flush().expect("flush");
    f
}
