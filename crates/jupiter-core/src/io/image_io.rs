use std::path::Path;

use image::{GrayImage, ImageFormat, Luma, Rgb};
use ndarray::Array2;

use crate::error::Result;
use crate::frame::{ColorFrame, Frame};

/// Save a frame as 16-bit grayscale TIFF.
pub fn save_tiff(frame: &Frame, path: &Path) -> Result<()> {
    let h = frame.height();
    let w = frame.width();

    let mut pixels: Vec<u16> = Vec::with_capacity(h * w);
    for row in 0..h {
        for col in 0..w {
            let val = (frame.data[[row, col]].clamp(0.0, 1.0) * 65535.0) as u16;
            pixels.push(val);
        }
    }

    let img = image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w as u32, h as u32, pixels)
        .expect("buffer size matches dimensions");
    img.save(path)?;
    Ok(())
}

/// Save a frame as 8-bit grayscale PNG.
pub fn save_png(frame: &Frame, path: &Path) -> Result<()> {
    let h = frame.height();
    let w = frame.width();

    let mut img = GrayImage::new(w as u32, h as u32);
    for row in 0..h {
        for col in 0..w {
            let val = (frame.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(col as u32, row as u32, Luma([val]));
        }
    }

    img.save_with_format(path, ImageFormat::Png)?;
    Ok(())
}

/// Save frame, choosing format from file extension.
pub fn save_image(frame: &Frame, path: &Path) -> Result<()> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("tiff" | "tif") => save_tiff(frame, path),
        Some("png") => save_png(frame, path),
        _ => save_tiff(frame, path),
    }
}

/// Save a ColorFrame as 16-bit RGB TIFF.
pub fn save_color_tiff(color: &ColorFrame, path: &Path) -> Result<()> {
    let h = color.red.height();
    let w = color.red.width();

    let mut pixels: Vec<u16> = Vec::with_capacity(h * w * 3);
    for row in 0..h {
        for col in 0..w {
            let r = (color.red.data[[row, col]].clamp(0.0, 1.0) * 65535.0) as u16;
            let g = (color.green.data[[row, col]].clamp(0.0, 1.0) * 65535.0) as u16;
            let b = (color.blue.data[[row, col]].clamp(0.0, 1.0) * 65535.0) as u16;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    let img = image::ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(w as u32, h as u32, pixels)
        .expect("buffer size matches dimensions");
    img.save(path)?;
    Ok(())
}

/// Save a ColorFrame as 8-bit RGB PNG.
pub fn save_color_png(color: &ColorFrame, path: &Path) -> Result<()> {
    let h = color.red.height();
    let w = color.red.width();

    let mut img = image::RgbImage::new(w as u32, h as u32);
    for row in 0..h {
        for col in 0..w {
            let r = (color.red.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (color.green.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (color.blue.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            img.put_pixel(col as u32, row as u32, Rgb([r, g, b]));
        }
    }

    img.save_with_format(path, ImageFormat::Png)?;
    Ok(())
}

/// Save a ColorFrame, choosing format from file extension.
pub fn save_color_image(color: &ColorFrame, path: &Path) -> Result<()> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("tiff" | "tif") => save_color_tiff(color, path),
        Some("png") => save_color_png(color, path),
        _ => save_color_tiff(color, path),
    }
}

/// Load a grayscale image file into a Frame.
pub fn load_image(path: &Path) -> Result<Frame> {
    let img = image::open(path)?;
    let gray = img.to_luma16();
    let (w, h) = gray.dimensions();
    let mut data = Array2::<f32>::zeros((h as usize, w as usize));

    for row in 0..h as usize {
        for col in 0..w as usize {
            let pixel = gray.get_pixel(col as u32, row as u32);
            data[[row, col]] = pixel.0[0] as f32 / 65535.0;
        }
    }

    Ok(Frame::new(data, 16))
}
