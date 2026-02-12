use ndarray::Array2;
use rayon::prelude::*;

use crate::frame::Frame;

/// Minimum pixel count (h*w) to justify row-level parallelism.
const PARALLEL_PIXEL_THRESHOLD: usize = 65_536;

/// Apply Gaussian blur to a frame using separable 1D convolution.
pub fn gaussian_blur(frame: &Frame, sigma: f32) -> Frame {
    let blurred = gaussian_blur_array(&frame.data, sigma);
    Frame::new(blurred, frame.original_bit_depth)
}

/// Apply Gaussian blur to a raw array.
pub fn gaussian_blur_array(data: &Array2<f32>, sigma: f32) -> Array2<f32> {
    let kernel = make_gaussian_kernel(sigma);
    let row_pass = convolve_rows(data, &kernel);
    convolve_cols(&row_pass, &kernel)
}

fn make_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0f32; size];
    let s2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;

    for (i, k) in kernel.iter_mut().enumerate() {
        let x = i as f32 - radius as f32;
        *k = (-x * x / s2).exp();
        sum += *k;
    }

    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

fn convolve_rows(data: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (h, w) = data.dim();
    let radius = kernel.len() / 2;

    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        let rows: Vec<Vec<f32>> = (0..h)
            .into_par_iter()
            .map(|row| {
                (0..w)
                    .map(|col| {
                        let mut sum = 0.0f32;
                        for (ki, &kv) in kernel.iter().enumerate() {
                            let src_col = (col as isize + ki as isize - radius as isize)
                                .clamp(0, w as isize - 1) as usize;
                            sum += data[[row, src_col]] * kv;
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        let mut result = Array2::<f32>::zeros((h, w));
        for (row, row_data) in rows.into_iter().enumerate() {
            for (col, val) in row_data.into_iter().enumerate() {
                result[[row, col]] = val;
            }
        }
        result
    } else {
        let mut result = Array2::<f32>::zeros((h, w));
        for row in 0..h {
            for col in 0..w {
                let mut sum = 0.0f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_col = (col as isize + ki as isize - radius as isize)
                        .clamp(0, w as isize - 1) as usize;
                    sum += data[[row, src_col]] * kv;
                }
                result[[row, col]] = sum;
            }
        }
        result
    }
}

fn convolve_cols(data: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (h, w) = data.dim();
    let radius = kernel.len() / 2;

    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        let rows: Vec<Vec<f32>> = (0..h)
            .into_par_iter()
            .map(|row| {
                (0..w)
                    .map(|col| {
                        let mut sum = 0.0f32;
                        for (ki, &kv) in kernel.iter().enumerate() {
                            let src_row = (row as isize + ki as isize - radius as isize)
                                .clamp(0, h as isize - 1) as usize;
                            sum += data[[src_row, col]] * kv;
                        }
                        sum
                    })
                    .collect()
            })
            .collect();

        let mut result = Array2::<f32>::zeros((h, w));
        for (row, row_data) in rows.into_iter().enumerate() {
            for (col, val) in row_data.into_iter().enumerate() {
                result[[row, col]] = val;
            }
        }
        result
    } else {
        let mut result = Array2::<f32>::zeros((h, w));
        for row in 0..h {
            for col in 0..w {
                let mut sum = 0.0f32;
                for (ki, &kv) in kernel.iter().enumerate() {
                    let src_row = (row as isize + ki as isize - radius as isize)
                        .clamp(0, h as isize - 1) as usize;
                    sum += data[[src_row, col]] * kv;
                }
                result[[row, col]] = sum;
            }
        }
        result
    }
}
