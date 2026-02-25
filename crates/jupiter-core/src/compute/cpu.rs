use ndarray::Array2;
use num_complex::Complex;
use rayon::prelude::*;
use rustfft::FftPlanner;

use crate::consts::{B3_KERNEL, PARALLEL_PIXEL_THRESHOLD};

use super::{BufferInner, ComputeBackend, GpuBuffer};

/// CPU backend using Rayon for parallelism.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str {
        "CPU/Rayon"
    }

    fn fft2d(&self, input: &GpuBuffer) -> GpuBuffer {
        let data = cpu_array(input);
        let complex = fft2d_forward(data);
        complex_to_interleaved(&complex)
    }

    fn ifft2d_real(&self, input: &GpuBuffer, height: usize, width: usize) -> GpuBuffer {
        let data = cpu_array(input);
        let complex = interleaved_to_complex(data);
        let result = ifft2d_inverse(&complex);
        let (h, w) = result.dim();
        let out_h = height.min(h);
        let out_w = width.min(w);
        if out_h == h && out_w == w {
            GpuBuffer::from_array(result.mapv(|v| v as f32))
        } else {
            let cropped = result
                .slice(ndarray::s![..out_h, ..out_w])
                .mapv(|v| v as f32);
            GpuBuffer::from_array(cropped.to_owned())
        }
    }

    fn cross_power_spectrum(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let a_data = cpu_array(a);
        let b_data = cpu_array(b);
        let (h, w2) = a_data.dim();
        let w = w2 / 2;

        let mut result = Array2::<f32>::zeros((h, w2));
        for r in 0..h {
            for c in 0..w {
                let a_re = a_data[[r, c * 2]] as f64;
                let a_im = a_data[[r, c * 2 + 1]] as f64;
                let b_re = b_data[[r, c * 2]] as f64;
                let b_im = b_data[[r, c * 2 + 1]] as f64;

                // cross = a * conj(b)
                let cross_re = a_re * b_re + a_im * b_im;
                let cross_im = a_im * b_re - a_re * b_im;

                let mag = (cross_re * cross_re + cross_im * cross_im).sqrt();
                if mag > 1e-12 {
                    result[[r, c * 2]] = (cross_re / mag) as f32;
                    result[[r, c * 2 + 1]] = (cross_im / mag) as f32;
                }
            }
        }

        GpuBuffer {
            inner: BufferInner::Cpu(result),
            height: h,
            width: w,
        }
    }

    fn hann_window(&self, input: &GpuBuffer) -> GpuBuffer {
        let data = cpu_array(input);
        let (h, w) = data.dim();
        let mut result = Array2::<f32>::zeros((h, w));

        for row in 0..h {
            let wy = 0.5 * (1.0 - (std::f64::consts::TAU * row as f64 / h as f64).cos());
            for col in 0..w {
                let wx = 0.5 * (1.0 - (std::f64::consts::TAU * col as f64 / w as f64).cos());
                result[[row, col]] = data[[row, col]] * (wy * wx) as f32;
            }
        }

        GpuBuffer::from_array(result)
    }

    fn find_peak(&self, input: &GpuBuffer) -> (usize, usize, f64) {
        let data = cpu_array(input);
        find_peak_array(data)
    }

    fn shift_bilinear(&self, input: &GpuBuffer, dx: f64, dy: f64) -> GpuBuffer {
        let data = cpu_array(input);
        let (h, w) = data.dim();
        if h * w >= PARALLEL_PIXEL_THRESHOLD {
            shift_bilinear_parallel(data, dx, dy, h, w)
        } else {
            shift_bilinear_sequential(data, dx, dy, h, w)
        }
    }

    fn convolve_separable(&self, input: &GpuBuffer, kernel: &[f32]) -> GpuBuffer {
        let data = cpu_array(input);
        let after_rows = convolve_rows_clamped(data, kernel);
        let after_cols = convolve_cols_clamped(&after_rows, kernel);
        GpuBuffer::from_array(after_cols)
    }

    fn atrous_convolve(&self, input: &GpuBuffer, scale: usize) -> GpuBuffer {
        let data = cpu_array(input);
        let step = 1usize << scale;
        let after_rows = convolve_rows_atrous(data, &B3_KERNEL, step);
        let after_cols = convolve_cols_atrous(&after_rows, &B3_KERNEL, step);
        GpuBuffer::from_array(after_cols)
    }

    fn complex_mul(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let a_data = cpu_array(a);
        let b_data = cpu_array(b);
        let (h, w2) = a_data.dim();
        let w = w2 / 2;

        let mut result = Array2::<f32>::zeros((h, w2));
        for r in 0..h {
            for c in 0..w {
                let a_re = a_data[[r, c * 2]] as f64;
                let a_im = a_data[[r, c * 2 + 1]] as f64;
                let b_re = b_data[[r, c * 2]] as f64;
                let b_im = b_data[[r, c * 2 + 1]] as f64;

                result[[r, c * 2]] = (a_re * b_re - a_im * b_im) as f32;
                result[[r, c * 2 + 1]] = (a_re * b_im + a_im * b_re) as f32;
            }
        }

        GpuBuffer {
            inner: BufferInner::Cpu(result),
            height: h,
            width: w,
        }
    }

    fn divide_real(&self, a: &GpuBuffer, b: &GpuBuffer, epsilon: f32) -> GpuBuffer {
        let a_data = cpu_array(a);
        let b_data = cpu_array(b);
        let result = ndarray::Zip::from(a_data)
            .and(b_data)
            .map_collect(|&av, &bv| av / (bv + epsilon));
        GpuBuffer::from_array(result)
    }

    fn multiply_real(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let a_data = cpu_array(a);
        let b_data = cpu_array(b);
        let result = a_data * b_data;
        GpuBuffer::from_array(result)
    }

    fn upload(&self, data: &Array2<f32>) -> GpuBuffer {
        GpuBuffer::from_array(data.clone())
    }

    fn download(&self, buf: &GpuBuffer) -> Array2<f32> {
        cpu_array(buf).clone()
    }
}

// ---------------------------------------------------------------------------
// Helper: extract CPU array from buffer
// ---------------------------------------------------------------------------

fn cpu_array(buf: &GpuBuffer) -> &Array2<f32> {
    match &buf.inner {
        BufferInner::Cpu(arr) => arr,
        #[cfg(feature = "gpu")]
        _ => panic!("CpuBackend received non-CPU buffer"),
    }
}

// ---------------------------------------------------------------------------
// Shared FFT helpers (used by CpuBackend and existing alignment/deconvolution)
// ---------------------------------------------------------------------------

/// 2D forward FFT with parallel row/column processing for large images.
pub fn fft2d_forward(data: &Array2<f32>) -> Array2<Complex<f64>> {
    let (h, w) = data.dim();
    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_forward(w);
    let fft_col = planner.plan_fft_forward(h);

    let mut result = Array2::<Complex<f64>>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            result[[row, col]] = Complex::new(data[[row, col]] as f64, 0.0);
        }
    }

    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        fft2d_forward_parallel(&mut result, &fft_row, &fft_col, h, w);
    } else {
        fft2d_forward_sequential(&mut result, &fft_row, &fft_col, h, w);
    }

    result
}

fn fft2d_forward_parallel(
    result: &mut Array2<Complex<f64>>,
    fft_row: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    fft_col: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    h: usize,
    w: usize,
) {
    let processed_rows: Vec<Vec<Complex<f64>>> = (0..h)
        .into_par_iter()
        .map(|row| {
            let mut row_data: Vec<Complex<f64>> = (0..w).map(|c| result[[row, c]]).collect();
            fft_row.process(&mut row_data);
            row_data
        })
        .collect();
    for (row, row_data) in processed_rows.into_iter().enumerate() {
        for (col, val) in row_data.into_iter().enumerate() {
            result[[row, col]] = val;
        }
    }

    let processed_cols: Vec<Vec<Complex<f64>>> = (0..w)
        .into_par_iter()
        .map(|col| {
            let mut col_data: Vec<Complex<f64>> = (0..h).map(|r| result[[r, col]]).collect();
            fft_col.process(&mut col_data);
            col_data
        })
        .collect();
    for (col, col_data) in processed_cols.into_iter().enumerate() {
        for (row, val) in col_data.into_iter().enumerate() {
            result[[row, col]] = val;
        }
    }
}

fn fft2d_forward_sequential(
    result: &mut Array2<Complex<f64>>,
    fft_row: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    fft_col: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    h: usize,
    w: usize,
) {
    for row in 0..h {
        let mut row_data: Vec<Complex<f64>> = (0..w).map(|c| result[[row, c]]).collect();
        fft_row.process(&mut row_data);
        for col in 0..w {
            result[[row, col]] = row_data[col];
        }
    }
    for col in 0..w {
        let mut col_data: Vec<Complex<f64>> = (0..h).map(|r| result[[r, col]]).collect();
        fft_col.process(&mut col_data);
        for row in 0..h {
            result[[row, col]] = col_data[row];
        }
    }
}

/// 2D inverse FFT, returning real part normalized by `1/(h*w)`.
pub fn ifft2d_inverse(data: &Array2<Complex<f64>>) -> Array2<f64> {
    let (h, w) = data.dim();
    let mut planner = FftPlanner::new();
    let ifft_row = planner.plan_fft_inverse(w);
    let ifft_col = planner.plan_fft_inverse(h);

    let mut work = data.clone();

    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        ifft2d_inverse_parallel(&mut work, &ifft_row, &ifft_col, h, w);
    } else {
        ifft2d_inverse_sequential(&mut work, &ifft_row, &ifft_col, h, w);
    }

    let scale = 1.0 / (h * w) as f64;
    let mut result = Array2::<f64>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            result[[row, col]] = work[[row, col]].re * scale;
        }
    }

    result
}

fn ifft2d_inverse_parallel(
    work: &mut Array2<Complex<f64>>,
    ifft_row: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    ifft_col: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    h: usize,
    w: usize,
) {
    let processed_cols: Vec<Vec<Complex<f64>>> = (0..w)
        .into_par_iter()
        .map(|col| {
            let mut col_data: Vec<Complex<f64>> = (0..h).map(|r| work[[r, col]]).collect();
            ifft_col.process(&mut col_data);
            col_data
        })
        .collect();
    for (col, col_data) in processed_cols.into_iter().enumerate() {
        for (row, val) in col_data.into_iter().enumerate() {
            work[[row, col]] = val;
        }
    }

    let processed_rows: Vec<Vec<Complex<f64>>> = (0..h)
        .into_par_iter()
        .map(|row| {
            let mut row_data: Vec<Complex<f64>> = (0..w).map(|c| work[[row, c]]).collect();
            ifft_row.process(&mut row_data);
            row_data
        })
        .collect();
    for (row, row_data) in processed_rows.into_iter().enumerate() {
        for (col, val) in row_data.into_iter().enumerate() {
            work[[row, col]] = val;
        }
    }
}

fn ifft2d_inverse_sequential(
    work: &mut Array2<Complex<f64>>,
    ifft_row: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    ifft_col: &std::sync::Arc<dyn rustfft::Fft<f64>>,
    h: usize,
    w: usize,
) {
    for col in 0..w {
        let mut col_data: Vec<Complex<f64>> = (0..h).map(|r| work[[r, col]]).collect();
        ifft_col.process(&mut col_data);
        for row in 0..h {
            work[[row, col]] = col_data[row];
        }
    }
    for row in 0..h {
        let mut row_data: Vec<Complex<f64>> = (0..w).map(|c| work[[row, c]]).collect();
        ifft_row.process(&mut row_data);
        for col in 0..w {
            work[[row, col]] = row_data[col];
        }
    }
}

// ---------------------------------------------------------------------------
// Interleaved complex <-> Complex<f64> conversion
// ---------------------------------------------------------------------------

fn complex_to_interleaved(data: &Array2<Complex<f64>>) -> GpuBuffer {
    let (h, w) = data.dim();
    let mut interleaved = Array2::<f32>::zeros((h, w * 2));
    for r in 0..h {
        for c in 0..w {
            interleaved[[r, c * 2]] = data[[r, c]].re as f32;
            interleaved[[r, c * 2 + 1]] = data[[r, c]].im as f32;
        }
    }
    GpuBuffer {
        inner: BufferInner::Cpu(interleaved),
        height: h,
        width: w,
    }
}

fn interleaved_to_complex(data: &Array2<f32>) -> Array2<Complex<f64>> {
    let (h, w2) = data.dim();
    let w = w2 / 2;
    let mut complex = Array2::<Complex<f64>>::zeros((h, w));
    for r in 0..h {
        for c in 0..w {
            complex[[r, c]] = Complex::new(data[[r, c * 2]] as f64, data[[r, c * 2 + 1]] as f64);
        }
    }
    complex
}

// ---------------------------------------------------------------------------
// Peak finding
// ---------------------------------------------------------------------------

fn find_peak_array(data: &Array2<f32>) -> (usize, usize, f64) {
    let (h, w) = data.dim();
    let mut best_row = 0;
    let mut best_col = 0;
    let mut best_val = f64::NEG_INFINITY;

    for row in 0..h {
        for col in 0..w {
            let val = data[[row, col]] as f64;
            if val > best_val {
                best_val = val;
                best_row = row;
                best_col = col;
            }
        }
    }

    (best_row, best_col, best_val)
}

// ---------------------------------------------------------------------------
// Bilinear interpolation
// ---------------------------------------------------------------------------

pub fn bilinear_sample(data: &Array2<f32>, y: f64, x: f64) -> f32 {
    let (h, w) = data.dim();

    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;

    let sample = |r: i64, c: i64| -> f32 {
        if r >= 0 && r < h as i64 && c >= 0 && c < w as i64 {
            data[[r as usize, c as usize]]
        } else {
            0.0
        }
    };

    let v00 = sample(y0, x0);
    let v10 = sample(y0, x1);
    let v01 = sample(y1, x0);
    let v11 = sample(y1, x1);

    v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) + v01 * (1.0 - fx) * fy + v11 * fx * fy
}

fn shift_bilinear_parallel(
    data: &Array2<f32>,
    dx: f64,
    dy: f64,
    h: usize,
    w: usize,
) -> GpuBuffer {
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
            (0..w)
                .map(|col| {
                    let src_y = row as f64 - dy;
                    let src_x = col as f64 - dx;
                    bilinear_sample(data, src_y, src_x)
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
    GpuBuffer::from_array(result)
}

fn shift_bilinear_sequential(
    data: &Array2<f32>,
    dx: f64,
    dy: f64,
    h: usize,
    w: usize,
) -> GpuBuffer {
    let mut result = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let src_y = row as f64 - dy;
            let src_x = col as f64 - dx;
            result[[row, col]] = bilinear_sample(data, src_y, src_x);
        }
    }
    GpuBuffer::from_array(result)
}

// ---------------------------------------------------------------------------
// Separable convolution (clamped boundary, for gaussian blur)
// ---------------------------------------------------------------------------

fn convolve_rows_clamped(data: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (h, w) = data.dim();
    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        convolve_rows_clamped_parallel(data, kernel, h, w)
    } else {
        convolve_rows_clamped_sequential(data, kernel, h, w)
    }
}

fn convolve_rows_clamped_parallel(
    data: &Array2<f32>,
    kernel: &[f32],
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = kernel.len() / 2;
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
            (0..w)
                .map(|col| {
                    let mut sum = 0.0f32;
                    for (k, &kv) in kernel.iter().enumerate() {
                        let c = (col as isize + k as isize - radius as isize)
                            .max(0)
                            .min(w as isize - 1) as usize;
                        sum += data[[row, c]] * kv;
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
}

fn convolve_rows_clamped_sequential(
    data: &Array2<f32>,
    kernel: &[f32],
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = kernel.len() / 2;
    let mut result = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let c = (col as isize + k as isize - radius as isize)
                    .max(0)
                    .min(w as isize - 1) as usize;
                sum += data[[row, c]] * kv;
            }
            result[[row, col]] = sum;
        }
    }
    result
}

fn convolve_cols_clamped(data: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (h, w) = data.dim();
    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        convolve_cols_clamped_parallel(data, kernel, h, w)
    } else {
        convolve_cols_clamped_sequential(data, kernel, h, w)
    }
}

fn convolve_cols_clamped_parallel(
    data: &Array2<f32>,
    kernel: &[f32],
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = kernel.len() / 2;
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
            (0..w)
                .map(|col| {
                    let mut sum = 0.0f32;
                    for (k, &kv) in kernel.iter().enumerate() {
                        let r = (row as isize + k as isize - radius as isize)
                            .max(0)
                            .min(h as isize - 1) as usize;
                        sum += data[[r, col]] * kv;
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
}

fn convolve_cols_clamped_sequential(
    data: &Array2<f32>,
    kernel: &[f32],
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = kernel.len() / 2;
    let mut result = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let r = (row as isize + k as isize - radius as isize)
                    .max(0)
                    .min(h as isize - 1) as usize;
                sum += data[[r, col]] * kv;
            }
            result[[row, col]] = sum;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// A-trous convolution (mirror boundary, for wavelet decomposition)
// ---------------------------------------------------------------------------

fn mirror_index(idx: isize, size: usize) -> usize {
    if idx < 0 {
        let pos = (-idx) as usize;
        if pos < size {
            pos
        } else {
            0
        }
    } else if idx >= size as isize {
        let overshoot = idx as usize - size;
        if overshoot < size {
            size - 1 - overshoot
        } else {
            size - 1
        }
    } else {
        idx as usize
    }
}

fn convolve_rows_atrous(data: &Array2<f32>, kernel: &[f32; 5], step: usize) -> Array2<f32> {
    let (h, w) = data.dim();
    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        convolve_rows_atrous_parallel(data, kernel, step, h, w)
    } else {
        convolve_rows_atrous_sequential(data, kernel, step, h, w)
    }
}

fn convolve_rows_atrous_parallel(
    data: &Array2<f32>,
    kernel: &[f32; 5],
    step: usize,
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = 2isize;
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
            (0..w)
                .map(|col| {
                    let mut sum = 0.0f32;
                    for (k, &kval) in kernel.iter().enumerate() {
                        let offset = (k as isize - radius) * step as isize;
                        let c = mirror_index(col as isize + offset, w);
                        sum += data[[row, c]] * kval;
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
}

fn convolve_rows_atrous_sequential(
    data: &Array2<f32>,
    kernel: &[f32; 5],
    step: usize,
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = 2isize;
    let mut result = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0f32;
            for (k, &kval) in kernel.iter().enumerate() {
                let offset = (k as isize - radius) * step as isize;
                let c = mirror_index(col as isize + offset, w);
                sum += data[[row, c]] * kval;
            }
            result[[row, col]] = sum;
        }
    }
    result
}

fn convolve_cols_atrous(data: &Array2<f32>, kernel: &[f32; 5], step: usize) -> Array2<f32> {
    let (h, w) = data.dim();
    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        convolve_cols_atrous_parallel(data, kernel, step, h, w)
    } else {
        convolve_cols_atrous_sequential(data, kernel, step, h, w)
    }
}

fn convolve_cols_atrous_parallel(
    data: &Array2<f32>,
    kernel: &[f32; 5],
    step: usize,
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = 2isize;
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
            (0..w)
                .map(|col| {
                    let mut sum = 0.0f32;
                    for (k, &kval) in kernel.iter().enumerate() {
                        let offset = (k as isize - radius) * step as isize;
                        let r = mirror_index(row as isize + offset, h);
                        sum += data[[r, col]] * kval;
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
}

fn convolve_cols_atrous_sequential(
    data: &Array2<f32>,
    kernel: &[f32; 5],
    step: usize,
    h: usize,
    w: usize,
) -> Array2<f32> {
    let radius = 2isize;
    let mut result = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0f32;
            for (k, &kval) in kernel.iter().enumerate() {
                let offset = (k as isize - radius) * step as isize;
                let r = mirror_index(row as isize + offset, h);
                sum += data[[r, col]] * kval;
            }
            result[[row, col]] = sum;
        }
    }
    result
}
