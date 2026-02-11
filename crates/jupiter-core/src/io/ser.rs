use std::fs::File;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use ndarray::Array2;

use crate::error::{JupiterError, Result};
use crate::frame::{ColorMode, Frame, FrameMetadata, SourceInfo};

const SER_HEADER_SIZE: usize = 178;
const SER_MAGIC: &[u8; 14] = b"LUCAM-RECORDER";

/// SER file header (178 bytes).
#[derive(Clone, Debug)]
pub struct SerHeader {
    pub color_id: i32,
    pub little_endian: bool,
    pub width: u32,
    pub height: u32,
    pub pixel_depth: u32,
    pub frame_count: u32,
    pub observer: String,
    pub instrument: String,
    pub telescope: String,
    pub date_time: u64,
    pub date_time_utc: u64,
}

impl SerHeader {
    /// Bytes per pixel plane (1 for 8-bit, 2 for 9-16 bit).
    pub fn bytes_per_pixel_plane(&self) -> usize {
        if self.pixel_depth <= 8 { 1 } else { 2 }
    }

    /// Number of planes per pixel (1 for mono/bayer, 3 for RGB/BGR).
    pub fn planes_per_pixel(&self) -> usize {
        match self.color_id {
            100 | 101 => 3,
            _ => 1,
        }
    }

    /// Total bytes per frame.
    pub fn frame_byte_size(&self) -> usize {
        let pixels = (self.width as usize)
            .checked_mul(self.height as usize)
            .expect("Image dimensions too large");
        let bytes_per_pixel = self.bytes_per_pixel_plane() * self.planes_per_pixel();
        pixels
            .checked_mul(bytes_per_pixel)
            .expect("Frame size calculation overflow")
    }

    pub fn color_mode(&self) -> ColorMode {
        match self.color_id {
            0 => ColorMode::Mono,
            8 => ColorMode::BayerRGGB,
            9 => ColorMode::BayerGRBG,
            10 => ColorMode::BayerGBRG,
            11 => ColorMode::BayerBGGR,
            100 => ColorMode::RGB,
            101 => ColorMode::BGR,
            _ => ColorMode::Mono,
        }
    }
}

/// Memory-mapped SER file reader.
pub struct SerReader {
    mmap: Mmap,
    pub header: SerHeader,
}

impl SerReader {
    /// Open a SER file and parse its header.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < SER_HEADER_SIZE {
            return Err(JupiterError::InvalidSer(
                "File too small for SER header".into(),
            ));
        }

        if &mmap[0..14] != SER_MAGIC {
            return Err(JupiterError::InvalidSer(
                "Missing LUCAM-RECORDER magic".into(),
            ));
        }

        let header = parse_header(&mmap[..SER_HEADER_SIZE])?;

        let expected_data_size =
            SER_HEADER_SIZE + header.frame_byte_size() * header.frame_count as usize;
        if mmap.len() < expected_data_size {
            return Err(JupiterError::InvalidSer(format!(
                "File truncated: expected at least {} bytes, got {}",
                expected_data_size,
                mmap.len()
            )));
        }

        Ok(Self { mmap, header })
    }

    pub fn frame_count(&self) -> usize {
        self.header.frame_count as usize
    }

    /// Get the raw bytes for a single frame (zero-copy from mmap).
    pub fn frame_raw(&self, index: usize) -> Result<&[u8]> {
        let count = self.frame_count();
        if index >= count {
            return Err(JupiterError::FrameIndexOutOfRange {
                index,
                total: count,
            });
        }
        let offset = SER_HEADER_SIZE + index * self.header.frame_byte_size();
        let end = offset + self.header.frame_byte_size();
        Ok(&self.mmap[offset..end])
    }

    /// Read a single frame, converting to f32 in [0.0, 1.0].
    pub fn read_frame(&self, index: usize) -> Result<Frame> {
        let raw = self.frame_raw(index)?;
        let h = self.header.height as usize;
        let w = self.header.width as usize;
        let bpp = self.header.bytes_per_pixel_plane();
        let planes = self.header.planes_per_pixel();

        // For mono/bayer: single plane. For RGB/BGR: average to luminance for now.
        let data = if planes == 1 {
            decode_mono_plane(raw, h, w, bpp, self.header.pixel_depth, self.header.little_endian)?
        } else {
            // RGB/BGR: extract green channel as luminance approximation
            decode_plane_from_interleaved(
                raw, h, w, bpp, planes, 1, // green channel index
                self.header.pixel_depth, self.header.little_endian,
            )?
        };

        let mut frame = Frame::new(data, bpp as u8 * 8);
        frame.metadata = FrameMetadata {
            frame_index: index,
            quality_score: None,
            timestamp_us: self.read_timestamp(index),
        };
        Ok(frame)
    }

    /// Read per-frame timestamp from the optional trailer.
    fn read_timestamp(&self, index: usize) -> Option<u64> {
        let trailer_offset =
            SER_HEADER_SIZE + self.header.frame_byte_size() * self.header.frame_count as usize;
        let ts_offset = trailer_offset + index * 8;
        if ts_offset + 8 <= self.mmap.len() {
            let bytes = &self.mmap[ts_offset..ts_offset + 8];
            Some(u64::from_le_bytes(bytes.try_into().ok()?))
        } else {
            None
        }
    }

    /// Build SourceInfo from the header.
    pub fn source_info(&self, path: &Path) -> SourceInfo {
        SourceInfo {
            filename: path.to_path_buf(),
            total_frames: self.frame_count(),
            width: self.header.width,
            height: self.header.height,
            bit_depth: self.header.pixel_depth as u8,
            color_mode: self.header.color_mode(),
            observer: non_empty(&self.header.observer),
            telescope: non_empty(&self.header.telescope),
            instrument: non_empty(&self.header.instrument),
        }
    }

    /// Iterator over all frames.
    pub fn frames(&self) -> impl Iterator<Item = Result<Frame>> + '_ {
        (0..self.frame_count()).map(move |i| self.read_frame(i))
    }
}

fn parse_header(buf: &[u8]) -> Result<SerHeader> {
    let mut cursor = std::io::Cursor::new(&buf[14..]); // skip magic

    let _lu_id = cursor.read_i32::<LittleEndian>()?;
    let color_id = cursor.read_i32::<LittleEndian>()?;
    let le_flag = cursor.read_i32::<LittleEndian>()?;
    let width = cursor.read_i32::<LittleEndian>()? as u32;
    let height = cursor.read_i32::<LittleEndian>()? as u32;
    let pixel_depth = cursor.read_i32::<LittleEndian>()? as u32;
    let frame_count = cursor.read_i32::<LittleEndian>()? as u32;

    let observer = read_fixed_string(&buf[42..82]);
    let instrument = read_fixed_string(&buf[82..122]);
    let telescope = read_fixed_string(&buf[122..162]);

    let mut cursor = std::io::Cursor::new(&buf[162..]);
    let date_time = cursor.read_u64::<LittleEndian>()?;
    let date_time_utc = cursor.read_u64::<LittleEndian>()?;

    if width == 0 || height == 0 {
        return Err(JupiterError::InvalidDimensions { width, height });
    }

    // SER spec: LittleEndian field = 0 means big-endian pixel data,
    // but many writers (including FireCapture) use 0 for little-endian.
    // Follow Siril's convention: treat 0 as little-endian.
    let little_endian = le_flag != 1;

    Ok(SerHeader {
        color_id,
        little_endian,
        width,
        height,
        pixel_depth,
        frame_count,
        observer,
        instrument,
        telescope,
        date_time,
        date_time_utc,
    })
}

fn read_fixed_string(buf: &[u8]) -> String {
    String::from_utf8_lossy(buf)
        .trim_end_matches('\0')
        .trim()
        .to_string()
}

fn non_empty(s: &str) -> Option<String> {
    if s.is_empty() { None } else { Some(s.to_string()) }
}

fn decode_mono_plane(
    raw: &[u8],
    height: usize,
    width: usize,
    bytes_per_sample: usize,
    bit_depth: u32,
    little_endian: bool,
) -> Result<Array2<f32>> {
    let max_val = ((1u32 << bit_depth) - 1) as f32;
    let mut data = Array2::<f32>::zeros((height, width));

    for row in 0..height {
        for col in 0..width {
            let idx = (row * width + col) * bytes_per_sample;
            let val = if bytes_per_sample == 1 {
                raw[idx] as f32
            } else {
                let pair = [raw[idx], raw[idx + 1]];
                if little_endian {
                    u16::from_le_bytes(pair) as f32
                } else {
                    u16::from_be_bytes(pair) as f32
                }
            };
            data[[row, col]] = val / max_val;
        }
    }

    Ok(data)
}

#[allow(clippy::too_many_arguments)]
fn decode_plane_from_interleaved(
    raw: &[u8],
    height: usize,
    width: usize,
    bytes_per_sample: usize,
    planes: usize,
    plane_index: usize,
    bit_depth: u32,
    little_endian: bool,
) -> Result<Array2<f32>> {
    let max_val = ((1u32 << bit_depth) - 1) as f32;
    let mut data = Array2::<f32>::zeros((height, width));

    for row in 0..height {
        for col in 0..width {
            let pixel_offset = (row * width + col) * planes * bytes_per_sample;
            let idx = pixel_offset + plane_index * bytes_per_sample;
            let val = if bytes_per_sample == 1 {
                raw[idx] as f32
            } else {
                let pair = [raw[idx], raw[idx + 1]];
                if little_endian {
                    u16::from_le_bytes(pair) as f32
                } else {
                    u16::from_be_bytes(pair) as f32
                }
            };
            data[[row, col]] = val / max_val;
        }
    }

    Ok(data)
}
