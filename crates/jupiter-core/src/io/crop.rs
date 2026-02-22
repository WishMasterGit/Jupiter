use std::path::Path;

use crate::color::debayer::is_bayer;
use crate::error::{JupiterError, Result};
use crate::frame::ColorMode;
use crate::io::ser::SerReader;
use crate::io::ser_writer::SerWriter;

/// A rectangle in image coordinates for cropping.
#[derive(Clone, Debug, PartialEq)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl CropRect {
    /// Validate and snap the crop rect to fit within source dimensions.
    /// Snaps x/y/width/height to even values for Bayer color modes.
    pub fn validated(&self, src_w: u32, src_h: u32, color_mode: &ColorMode) -> Result<CropRect> {
        let needs_even = is_bayer(color_mode);

        let mut x = self.x;
        let mut y = self.y;
        let mut w = self.width;
        let mut h = self.height;

        // Snap to even for Bayer
        if needs_even {
            x &= !1;
            y &= !1;
            w &= !1;
            h &= !1;
        }

        if w == 0 || h == 0 {
            return Err(JupiterError::InvalidCrop(
                "Crop width and height must be > 0".into(),
            ));
        }

        if x + w > src_w || y + h > src_h {
            return Err(JupiterError::InvalidCrop(format!(
                "Crop region ({x},{y} {}x{}) exceeds source dimensions ({src_w}x{src_h})",
                w, h
            )));
        }

        Ok(CropRect {
            x,
            y,
            width: w,
            height: h,
        })
    }
}

/// Crop a SER file to a new SER file, operating on raw bytes.
///
/// `progress` is called with `(frames_done, total_frames)`.
pub fn crop_ser(
    reader: &SerReader,
    output: &Path,
    crop: &CropRect,
    mut progress: impl FnMut(usize, usize),
) -> Result<()> {
    let src_header = &reader.header;
    let validated = crop.validated(
        src_header.width,
        src_header.height,
        &src_header.color_mode(),
    )?;

    let mut new_header = src_header.clone();
    new_header.width = validated.width;
    new_header.height = validated.height;

    let mut writer = SerWriter::create(output, &new_header)?;

    let bytes_per_pixel = src_header.bytes_per_pixel_plane() * src_header.planes_per_pixel();
    let src_row_stride = src_header.width as usize * bytes_per_pixel;
    let col_byte_offset = validated.x as usize * bytes_per_pixel;
    let crop_row_bytes = validated.width as usize * bytes_per_pixel;

    let total = reader.frame_count();
    let mut crop_buf = vec![0u8; new_header.frame_byte_size()];

    for i in 0..total {
        let raw = reader.frame_raw(i)?;

        // Extract cropped rows
        for row in 0..validated.height as usize {
            let src_row = validated.y as usize + row;
            let src_start = src_row * src_row_stride + col_byte_offset;
            let dst_start = row * crop_row_bytes;
            crop_buf[dst_start..dst_start + crop_row_bytes]
                .copy_from_slice(&raw[src_start..src_start + crop_row_bytes]);
        }

        writer.write_raw_frame(&crop_buf)?;
        progress(i + 1, total);
    }

    // Copy timestamps if present
    let mut timestamps = Vec::new();
    for i in 0..total {
        if let Some(ts) = read_timestamp(reader, i) {
            timestamps.push(ts);
        } else {
            break;
        }
    }
    if timestamps.len() == total {
        writer.write_timestamps(&timestamps)?;
    }

    writer.finalize()?;
    Ok(())
}

/// Read a timestamp from the SER trailer (mirrors SerReader::read_timestamp).
fn read_timestamp(reader: &SerReader, index: usize) -> Option<u64> {
    // We access the reader's public interface; the timestamp is read via read_frame's metadata.
    // But we need raw access. Use a frame read to get the timestamp.
    let frame = reader.read_frame(index).ok()?;
    frame.metadata.timestamp_us
}
