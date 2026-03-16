use memmap2::Mmap;

use crate::consts::{DISK_FRAME_HEADER_SIZE, EPSILON};
use crate::error::{JupiterError, Result};
use crate::frame::Frame;
use crate::io::disk_cache::DiskFrame;
use crate::utils::{compute_median, masked_mean_stddev, par_or_seq_rows};

/// Zero-copy pixel access over a memory-mapped [`DiskFrame`] file.
///
/// The OS page cache handles demand-paging: only the rows currently being
/// processed by Rayon workers need physical RAM.
pub struct MmapFrameSlice {
    mmap: Mmap,
    height: usize,
    width: usize,
}

impl MmapFrameSlice {
    /// Create a slice by memory-mapping the backing file of a [`DiskFrame`].
    pub fn from_disk_frame(disk_frame: &DiskFrame) -> Result<Self> {
        let mmap = disk_frame.mmap()?;
        let expected = DISK_FRAME_HEADER_SIZE
            + disk_frame.height() * disk_frame.width() * std::mem::size_of::<f32>();
        if mmap.len() < expected {
            return Err(JupiterError::Pipeline(format!(
                "Mmap too small: expected {} bytes, got {}",
                expected,
                mmap.len()
            )));
        }
        Ok(Self {
            mmap,
            height: disk_frame.height(),
            width: disk_frame.width(),
        })
    }

    /// Read a single pixel value from the mmap'd data.
    #[inline]
    pub fn pixel(&self, row: usize, col: usize) -> f32 {
        let offset = DISK_FRAME_HEADER_SIZE + (row * self.width + col) * std::mem::size_of::<f32>();
        let bytes = &self.mmap[offset..offset + 4];
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }
}

/// Stack frames from disk using median, reading pixel data via mmap.
///
/// Mirrors [`super::median::median_stack`] but reads from mmap'd files instead
/// of in-memory `Frame` arrays. Peak RAM: ~M × page_size × active_threads
/// instead of M × h × w × 4.
pub fn median_stack_disk(slices: &[MmapFrameSlice], bit_depth: u8) -> Result<Frame> {
    if slices.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let h = slices[0].height();
    let w = slices[0].width();
    let n = slices.len();

    let result = par_or_seq_rows(h, w, |row| {
        let mut pixel_values = vec![0.0f32; n];
        let mut row_result = vec![0.0f32; w];
        for (col, result) in row_result.iter_mut().enumerate() {
            for (i, slice) in slices.iter().enumerate() {
                pixel_values[i] = slice.pixel(row, col);
            }
            *result = compute_median(&mut pixel_values, n);
        }
        row_result
    });

    Ok(Frame::new(result, bit_depth))
}

/// Stack frames from disk using sigma-clipped mean, reading pixel data via mmap.
///
/// Mirrors [`super::sigma_clip::sigma_clip_stack`] but reads from mmap'd files.
pub fn sigma_clip_stack_disk(
    slices: &[MmapFrameSlice],
    sigma: f32,
    max_iter: usize,
    bit_depth: u8,
) -> Result<Frame> {
    if slices.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let h = slices[0].height();
    let w = slices[0].width();
    let n = slices.len();

    let result = par_or_seq_rows(h, w, |row| {
        let mut pixel_values = vec![0.0f32; n];
        let mut mask = vec![true; n];
        let mut row_result = vec![0.0f32; w];
        for (col, result) in row_result.iter_mut().enumerate() {
            for (i, slice) in slices.iter().enumerate() {
                pixel_values[i] = slice.pixel(row, col);
            }
            *result = sigma_clip_pixel_disk(&pixel_values, &mut mask, n, sigma, max_iter);
        }
        row_result
    });

    Ok(Frame::new(result, bit_depth))
}

fn sigma_clip_pixel_disk(
    pixel_values: &[f32],
    mask: &mut [bool],
    n: usize,
    sigma: f32,
    max_iter: usize,
) -> f32 {
    for m in mask.iter_mut() {
        *m = true;
    }

    for _ in 0..max_iter {
        let (mean, stddev) = masked_mean_stddev(pixel_values, mask);
        if stddev < EPSILON {
            break;
        }
        let lo = mean - sigma * stddev;
        let hi = mean + sigma * stddev;
        for i in 0..n {
            if mask[i] && (pixel_values[i] < lo || pixel_values[i] > hi) {
                mask[i] = false;
            }
        }
    }

    let mut sum = 0.0f32;
    let mut count = 0u32;
    for i in 0..n {
        if mask[i] {
            sum += pixel_values[i];
            count += 1;
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        pixel_values.iter().sum::<f32>() / n as f32
    }
}
