use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use ndarray::Array2;

use crate::consts::{DISK_FRAME_HEADER_SIZE, DISK_FRAME_MAGIC};
use crate::error::{JupiterError, Result};
use crate::frame::{ColorFrame, Frame};

/// RAII handle to a temporary file containing one serialized [`Frame`].
///
/// Binary format (16-byte header):
///   4-byte magic `JFRM`
///   4-byte height (u32 LE)
///   4-byte width  (u32 LE)
///   1-byte bit_depth
///   3-byte padding
/// Followed by `height * width` f32 values in row-major LE order.
pub struct DiskFrame {
    path: PathBuf,
    height: usize,
    width: usize,
}

impl DiskFrame {
    /// Deserialize the file back into an in-memory [`Frame`].
    pub fn read(&self) -> Result<Frame> {
        let data = std::fs::read(&self.path)?;
        Self::deserialize_frame(&data, self.height, self.width)
    }

    /// Memory-map the backing file for zero-copy pixel access.
    pub fn mmap(&self) -> Result<Mmap> {
        let file = std::fs::File::open(&self.path)?;
        // SAFETY: the file is exclusively ours (written by DiskFrameStore) and
        // immutable after creation.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(mmap)
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn deserialize_frame(data: &[u8], height: usize, width: usize) -> Result<Frame> {
        let expected = DISK_FRAME_HEADER_SIZE + height * width * std::mem::size_of::<f32>();
        if data.len() < expected {
            return Err(JupiterError::Pipeline(format!(
                "Truncated disk frame: expected {} bytes, got {}",
                expected,
                data.len()
            )));
        }

        // Validate header
        if &data[..4] != DISK_FRAME_MAGIC {
            return Err(JupiterError::Pipeline(
                "Invalid disk frame magic bytes".into(),
            ));
        }
        let bit_depth = data[8 + 4]; // offset 12 in header

        // Read pixel data
        let pixel_data = &data[DISK_FRAME_HEADER_SIZE..];
        let mut cursor = std::io::Cursor::new(pixel_data);
        let pixel_count = height * width;
        let mut pixels = Vec::with_capacity(pixel_count);
        for _ in 0..pixel_count {
            pixels.push(cursor.read_f32::<LittleEndian>()?);
        }

        let array = Array2::from_shape_vec((height, width), pixels).map_err(|e| {
            JupiterError::Pipeline(format!("Failed to reshape disk frame data: {e}"))
        })?;

        Ok(Frame::new(array, bit_depth))
    }
}

impl Drop for DiskFrame {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// RAII handle to three [`DiskFrame`] files (R, G, B channels).
pub struct DiskColorFrame {
    pub red: DiskFrame,
    pub green: DiskFrame,
    pub blue: DiskFrame,
}

impl DiskColorFrame {
    /// Read all three channels back into an in-memory [`ColorFrame`].
    pub fn read(&self) -> Result<ColorFrame> {
        Ok(ColorFrame {
            red: self.red.read()?,
            green: self.green.read()?,
            blue: self.blue.read()?,
        })
    }
}

/// Manages a temporary directory of cached frames with automatic cleanup.
///
/// Thread-safe: uses [`AtomicUsize`] for file naming so multiple threads can
/// call [`store_frame`] concurrently.
pub struct DiskFrameStore {
    dir: tempfile::TempDir,
    counter: AtomicUsize,
}

impl DiskFrameStore {
    /// Create a new store with a temporary directory.
    pub fn new() -> Result<Self> {
        let dir = tempfile::TempDir::new()?;
        Ok(Self {
            dir,
            counter: AtomicUsize::new(0),
        })
    }

    /// Create a new store under the specified parent directory.
    pub fn with_parent(parent: &Path) -> Result<Self> {
        let dir = tempfile::TempDir::new_in(parent)?;
        Ok(Self {
            dir,
            counter: AtomicUsize::new(0),
        })
    }

    /// Serialize a [`Frame`] to a file in the store and return an RAII handle.
    pub fn store_frame(&self, frame: &Frame) -> Result<DiskFrame> {
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        let path = self.dir.path().join(format!("frame_{id:06}.jfrm"));
        Self::serialize_frame(frame, &path)?;
        Ok(DiskFrame {
            path,
            height: frame.height(),
            width: frame.width(),
        })
    }

    /// Serialize a [`ColorFrame`] to three files in the store.
    pub fn store_color_frame(&self, cf: &ColorFrame) -> Result<DiskColorFrame> {
        let red = self.store_frame(&cf.red)?;
        let green = self.store_frame(&cf.green)?;
        let blue = self.store_frame(&cf.blue)?;
        Ok(DiskColorFrame { red, green, blue })
    }

    /// Path to the temporary directory (for diagnostics).
    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    fn serialize_frame(frame: &Frame, path: &Path) -> Result<()> {
        let h = frame.height() as u32;
        let w = frame.width() as u32;
        let pixel_count = frame.height() * frame.width();
        let total_size = DISK_FRAME_HEADER_SIZE + pixel_count * std::mem::size_of::<f32>();

        let mut buf = Vec::with_capacity(total_size);

        // Header: magic + height + width + bit_depth + 3 padding bytes
        buf.extend_from_slice(DISK_FRAME_MAGIC);
        buf.write_u32::<LittleEndian>(h)?;
        buf.write_u32::<LittleEndian>(w)?;
        buf.push(frame.original_bit_depth);
        buf.extend_from_slice(&[0u8; 3]); // padding

        // Pixel data in row-major order
        for &val in frame.data.iter() {
            buf.write_f32::<LittleEndian>(val)?;
        }

        std::fs::write(path, &buf)?;
        Ok(())
    }
}
