use ndarray::Array2;

use crate::frame::Frame;
use crate::utils::par_or_seq_rows;

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
    par_or_seq_rows(h, w, |row| {
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
}

fn convolve_cols(data: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (h, w) = data.dim();
    let radius = kernel.len() / 2;
    par_or_seq_rows(h, w, |row| {
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
}
