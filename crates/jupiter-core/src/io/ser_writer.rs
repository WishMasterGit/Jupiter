use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::Result;
use crate::io::ser::{SerHeader, SER_HEADER_SIZE, SER_MAGIC};

/// Writes a valid SER file at the raw byte level.
pub struct SerWriter {
    writer: BufWriter<File>,
    header: SerHeader,
    frames_written: u32,
}

impl SerWriter {
    /// Create a new SER file and write the header.
    pub fn create(path: &Path, header: &SerHeader) -> Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer, header)?;
        Ok(Self {
            writer,
            header: header.clone(),
            frames_written: 0,
        })
    }

    /// Write a single raw frame (bytes must match the header's frame_byte_size).
    pub fn write_raw_frame(&mut self, data: &[u8]) -> Result<()> {
        debug_assert_eq!(data.len(), self.header.frame_byte_size());
        self.writer.write_all(data)?;
        self.frames_written += 1;
        Ok(())
    }

    /// Write the optional timestamp trailer (one u64 per frame, little-endian).
    pub fn write_timestamps(&mut self, timestamps: &[u64]) -> Result<()> {
        for &ts in timestamps {
            self.writer.write_all(&ts.to_le_bytes())?;
        }
        Ok(())
    }

    /// Flush and finalize the file.
    pub fn finalize(mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

fn write_header(w: &mut impl Write, header: &SerHeader) -> Result<()> {
    // Magic (14 bytes)
    w.write_all(SER_MAGIC)?;
    // LuID (4 bytes)
    w.write_all(&0i32.to_le_bytes())?;
    // ColorID (4 bytes)
    w.write_all(&header.color_id.to_le_bytes())?;
    // LittleEndian flag: 0 = little-endian (Siril convention)
    let le_flag: i32 = if header.little_endian { 0 } else { 1 };
    w.write_all(&le_flag.to_le_bytes())?;
    // Width (4 bytes)
    w.write_all(&(header.width as i32).to_le_bytes())?;
    // Height (4 bytes)
    w.write_all(&(header.height as i32).to_le_bytes())?;
    // PixelDepth (4 bytes)
    w.write_all(&(header.pixel_depth as i32).to_le_bytes())?;
    // FrameCount (4 bytes)
    w.write_all(&(header.frame_count as i32).to_le_bytes())?;
    // Observer (40 bytes)
    write_fixed_string(w, &header.observer, 40)?;
    // Instrument (40 bytes)
    write_fixed_string(w, &header.instrument, 40)?;
    // Telescope (40 bytes)
    write_fixed_string(w, &header.telescope, 40)?;
    // DateTime (8 bytes)
    w.write_all(&header.date_time.to_le_bytes())?;
    // DateTimeUTC (8 bytes)
    w.write_all(&header.date_time_utc.to_le_bytes())?;

    debug_assert_eq!(
        14 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 40 + 40 + 40 + 8 + 8,
        SER_HEADER_SIZE
    );
    Ok(())
}

fn write_fixed_string(w: &mut impl Write, s: &str, len: usize) -> Result<()> {
    let bytes = s.as_bytes();
    let to_write = bytes.len().min(len);
    w.write_all(&bytes[..to_write])?;
    // Pad with zeros
    for _ in to_write..len {
        w.write_all(&[0u8])?;
    }
    Ok(())
}
