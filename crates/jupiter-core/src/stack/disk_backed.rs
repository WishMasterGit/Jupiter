use memmap2::Mmap;
use ndarray::Array2;
use rayon::prelude::*;

use crate::consts::{DISK_FRAME_HEADER_SIZE, EPSILON, PARALLEL_PIXEL_THRESHOLD};
use crate::error::{JupiterError, Result};
use crate::frame::Frame;
use crate::io::disk_cache::DiskFrame;

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

    if h * w >= PARALLEL_PIXEL_THRESHOLD && n > 1 {
        median_stack_disk_parallel(slices, h, w, n, bit_depth)
    } else {
        median_stack_disk_sequential(slices, h, w, n, bit_depth)
    }
}

fn median_stack_disk_parallel(
    slices: &[MmapFrameSlice],
    h: usize,
    w: usize,
    n: usize,
    bit_depth: u8,
) -> Result<Frame> {
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
            let mut pixel_values = vec![0.0f32; n];
            let mut row_result = vec![0.0f32; w];
            for (col, result) in row_result.iter_mut().enumerate() {
                for (i, slice) in slices.iter().enumerate() {
                    pixel_values[i] = slice.pixel(row, col);
                }
                *result = compute_median(&mut pixel_values, n);
            }
            row_result
        })
        .collect();

    let mut result = Array2::<f32>::zeros((h, w));
    for (row, row_data) in rows.into_iter().enumerate() {
        for (col, val) in row_data.into_iter().enumerate() {
            result[[row, col]] = val;
        }
    }
    Ok(Frame::new(result, bit_depth))
}

fn median_stack_disk_sequential(
    slices: &[MmapFrameSlice],
    h: usize,
    w: usize,
    n: usize,
    bit_depth: u8,
) -> Result<Frame> {
    let mut result = Array2::<f32>::zeros((h, w));
    let mut pixel_values = vec![0.0f32; n];

    for row in 0..h {
        for col in 0..w {
            for (i, slice) in slices.iter().enumerate() {
                pixel_values[i] = slice.pixel(row, col);
            }
            result[[row, col]] = compute_median(&mut pixel_values, n);
        }
    }
    Ok(Frame::new(result, bit_depth))
}

fn compute_median(pixel_values: &mut [f32], n: usize) -> f32 {
    if n == 1 {
        pixel_values[0]
    } else if n % 2 == 1 {
        let mid = n / 2;
        *pixel_values
            .select_nth_unstable_by(mid, |a, b| a.total_cmp(b))
            .1
    } else {
        let mid = n / 2;
        pixel_values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        pixel_values[..mid].select_nth_unstable_by(mid - 1, |a, b| a.total_cmp(b));
        (pixel_values[mid - 1] + pixel_values[mid]) / 2.0
    }
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

    if h * w >= PARALLEL_PIXEL_THRESHOLD && n > 1 {
        sigma_clip_stack_disk_parallel(slices, sigma, max_iter, h, w, n, bit_depth)
    } else {
        sigma_clip_stack_disk_sequential(slices, sigma, max_iter, h, w, n, bit_depth)
    }
}

fn sigma_clip_stack_disk_parallel(
    slices: &[MmapFrameSlice],
    sigma: f32,
    max_iter: usize,
    h: usize,
    w: usize,
    n: usize,
    bit_depth: u8,
) -> Result<Frame> {
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
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
        })
        .collect();

    let mut result = Array2::<f32>::zeros((h, w));
    for (row, row_data) in rows.into_iter().enumerate() {
        for (col, val) in row_data.into_iter().enumerate() {
            result[[row, col]] = val;
        }
    }
    Ok(Frame::new(result, bit_depth))
}

fn sigma_clip_stack_disk_sequential(
    slices: &[MmapFrameSlice],
    sigma: f32,
    max_iter: usize,
    h: usize,
    w: usize,
    n: usize,
    bit_depth: u8,
) -> Result<Frame> {
    let mut result = Array2::<f32>::zeros((h, w));
    let mut pixel_values = vec![0.0f32; n];
    let mut mask = vec![true; n];

    for row in 0..h {
        for col in 0..w {
            for (i, slice) in slices.iter().enumerate() {
                pixel_values[i] = slice.pixel(row, col);
            }
            result[[row, col]] =
                sigma_clip_pixel_disk(&pixel_values, &mut mask, n, sigma, max_iter);
        }
    }
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
        let (mean, stddev) = mean_stddev(pixel_values, mask);
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

fn mean_stddev(values: &[f32], mask: &[bool]) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for (i, &v) in values.iter().enumerate() {
        if mask[i] {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / count as f32;

    let mut var_sum = 0.0f32;
    for (i, &v) in values.iter().enumerate() {
        if mask[i] {
            let d = v - mean;
            var_sum += d * d;
        }
    }
    let stddev = (var_sum / count as f32).sqrt();
    (mean, stddev)
}
