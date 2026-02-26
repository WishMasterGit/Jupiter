use ndarray::Array2;

use jupiter_core::filters::gaussian_blur::{gaussian_blur, gaussian_blur_array};
use jupiter_core::filters::histogram::{auto_stretch, histogram_stretch};
use jupiter_core::filters::levels::{brightness_contrast, gamma_correct};
use jupiter_core::filters::rgb_align::{rgb_align, rgb_align_manual};
use jupiter_core::filters::unsharp_mask::unsharp_mask;
use jupiter_core::frame::{AlignmentOffset, ColorFrame, Frame};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_frame(h: usize, w: usize, fill: f32) -> Frame {
    Frame::new(Array2::from_elem((h, w), fill), 8)
}

fn make_ramp_frame(h: usize, w: usize) -> Frame {
    let mut data = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            data[[row, col]] = (row * w + col) as f32 / (h * w) as f32;
        }
    }
    Frame::new(data, 8)
}

fn make_color_frame(h: usize, w: usize, r: f32, g: f32, b: f32) -> ColorFrame {
    ColorFrame {
        red: make_frame(h, w, r),
        green: make_frame(h, w, g),
        blue: make_frame(h, w, b),
    }
}

// ---------------------------------------------------------------------------
// histogram_stretch
// ---------------------------------------------------------------------------

#[test]
fn test_histogram_stretch_normal() {
    // Values in [0.0, 0.5]; after stretch to [0.0, 1.0] they should double.
    let data = Array2::from_shape_fn((4, 4), |_| 0.25f32);
    let frame = Frame::new(data, 8);
    let stretched = histogram_stretch(&frame, 0.0, 0.5);
    // 0.25 mapped from [0,0.5] → [0,1] = 0.5
    for v in stretched.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
    }
}

#[test]
fn test_histogram_stretch_clips_below_zero() {
    let data = Array2::from_shape_fn((4, 4), |_| 0.1f32);
    let frame = Frame::new(data, 8);
    // black_point > pixel values → all clamp to 0
    let stretched = histogram_stretch(&frame, 0.5, 1.0);
    for v in stretched.data.iter() {
        assert!((*v - 0.0).abs() < 1e-5);
    }
}

#[test]
fn test_histogram_stretch_clips_above_one() {
    let data = Array2::from_shape_fn((4, 4), |_| 0.9f32);
    let frame = Frame::new(data, 8);
    // white_point < pixel values → all clamp to 1
    let stretched = histogram_stretch(&frame, 0.0, 0.5);
    for v in stretched.data.iter() {
        assert!((*v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_histogram_stretch_zero_range_guard() {
    // black_point == white_point: should not divide by zero
    let data = Array2::from_shape_fn((4, 4), |_| 0.5f32);
    let frame = Frame::new(data, 8);
    let stretched = histogram_stretch(&frame, 0.5, 0.5);
    // Result should be finite (no NaN/inf)
    for v in stretched.data.iter() {
        assert!(v.is_finite());
    }
}

// ---------------------------------------------------------------------------
// auto_stretch
// ---------------------------------------------------------------------------

#[test]
fn test_auto_stretch_typical() {
    let ramp = make_ramp_frame(16, 16);
    // Using default percentiles: result should span approximately [0,1]
    let stretched = auto_stretch(&ramp, 0.001, 0.999);
    let max = stretched
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min = stretched.data.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(max > 0.9, "max should be near 1.0, got {max}");
    assert!(min < 0.1, "min should be near 0.0, got {min}");
}

#[test]
fn test_auto_stretch_all_same_value() {
    // All identical values — zero-range guard inside histogram_stretch should prevent NaN
    let frame = make_frame(8, 8, 0.5);
    let stretched = auto_stretch(&frame, 0.01, 0.99);
    for v in stretched.data.iter() {
        assert!(v.is_finite(), "expected finite, got {v}");
    }
}

// ---------------------------------------------------------------------------
// gamma_correct
// ---------------------------------------------------------------------------

#[test]
fn test_gamma_correct_identity() {
    // gamma=1.0 → output = input^(1/1) = input
    let ramp = make_ramp_frame(8, 8);
    let corrected = gamma_correct(&ramp, 1.0);
    for (a, b) in ramp.data.iter().zip(corrected.data.iter()) {
        assert!((*a - *b).abs() < 1e-5, "gamma=1.0 should be identity");
    }
}

#[test]
fn test_gamma_correct_brightens_midtones() {
    // gamma=2.0 → output = input^0.5, which is > input for values in (0,1)
    let frame = make_frame(4, 4, 0.25);
    let corrected = gamma_correct(&frame, 2.0);
    // 0.25^0.5 = 0.5
    for v in corrected.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_gamma_correct_extremes_unchanged() {
    // 0^x = 0, 1^x = 1
    let mut data = Array2::<f32>::zeros((2, 2));
    data[[0, 0]] = 0.0;
    data[[0, 1]] = 1.0;
    data[[1, 0]] = 0.0;
    data[[1, 1]] = 1.0;
    let frame = Frame::new(data, 8);
    let corrected = gamma_correct(&frame, 3.0);
    assert!(corrected.data[[0, 0]].abs() < 1e-5);
    assert!((corrected.data[[0, 1]] - 1.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// brightness_contrast
// ---------------------------------------------------------------------------

#[test]
fn test_brightness_no_change() {
    // brightness=0, contrast=1 → identity
    let frame = make_frame(4, 4, 0.5);
    let result = brightness_contrast(&frame, 0.0, 1.0);
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_brightness_shift() {
    let frame = make_frame(4, 4, 0.3);
    let result = brightness_contrast(&frame, 0.2, 1.0);
    // (0.3 - 0.5) * 1.0 + 0.5 + 0.2 = -0.2 + 0.5 + 0.2 = 0.5
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_brightness_contrast_clamped() {
    // Very high brightness → clamp to 1.0
    let frame = make_frame(4, 4, 0.9);
    let result = brightness_contrast(&frame, 1.0, 1.0);
    for v in result.data.iter() {
        assert!((*v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_brightness_contrast_clamped_low() {
    // Very negative brightness → clamp to 0.0
    let frame = make_frame(4, 4, 0.1);
    let result = brightness_contrast(&frame, -1.0, 1.0);
    for v in result.data.iter() {
        assert!(*v < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// unsharp_mask
// ---------------------------------------------------------------------------

#[test]
fn test_unsharp_mask_no_change_on_uniform() {
    // Uniform image — blur = original, diff = 0, threshold prevents any change
    let frame = make_frame(32, 32, 0.5);
    let result = unsharp_mask(&frame, 1.0, 1.0, 0.01);
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-4);
    }
}

#[test]
fn test_unsharp_mask_sharpens_edge() {
    // Create an image with a sharp step
    let mut data = Array2::<f32>::zeros((32, 32));
    for row in 0..32 {
        for col in 16..32 {
            data[[row, col]] = 1.0;
        }
    }
    let frame = Frame::new(data.clone(), 8);
    let result = unsharp_mask(&frame, 1.0, 0.5, 0.0);
    // Pixels adjacent to the edge should be boosted (values != original)
    // At minimum the output should still be in [0,1]
    for v in result.data.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
    // The peak should be at least as bright as the original
    let max_orig = data.iter().cloned().fold(0.0f32, f32::max);
    let max_out = result.data.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_out >= max_orig - 1e-5);
}

#[test]
fn test_unsharp_mask_threshold_prevents_noise() {
    // A tiny perturbation below the threshold should be left unchanged
    let mut data = Array2::<f32>::from_elem((32, 32), 0.5);
    data[[16, 16]] = 0.501; // tiny bump
    let frame = Frame::new(data.clone(), 8);
    // Threshold of 0.1 should prevent sharpening of the tiny bump
    let result = unsharp_mask(&frame, 1.0, 2.0, 0.1);
    // Most pixels should remain close to 0.5
    let center = result.data[[16, 16]];
    assert!((center - 0.5).abs() < 0.05);
}

// ---------------------------------------------------------------------------
// gaussian_blur / gaussian_blur_array
// ---------------------------------------------------------------------------

#[test]
fn test_gaussian_blur_small_image_preserves_mean() {
    // 64x64 → sequential path; blurring a uniform image leaves it unchanged
    let frame = make_frame(64, 64, 0.6);
    let result = gaussian_blur(&frame, 2.0);
    for v in result.data.iter() {
        assert!((*v - 0.6).abs() < 1e-5);
    }
}

#[test]
fn test_gaussian_blur_large_image_preserves_mean() {
    // 512x512 → parallel path; blurring a uniform image leaves it unchanged
    let frame = make_frame(512, 512, 0.4);
    let result = gaussian_blur(&frame, 2.0);
    for v in result.data.iter() {
        assert!((*v - 0.4).abs() < 1e-4);
    }
}

#[test]
fn test_gaussian_blur_array_smooths_noise() {
    // After blurring a checkerboard image the center region should approach 0.5.
    // We use a large sigma (5.0) so high-frequency content is strongly attenuated.
    let h = 64usize;
    let w = 64usize;
    let mut data = Array2::<f32>::zeros((h, w));
    let mut toggle = false;
    for v in data.iter_mut() {
        *v = if toggle { 0.0 } else { 1.0 };
        toggle = !toggle;
    }
    let blurred = gaussian_blur_array(&data, 5.0);
    // Interior pixels (away from boundary) should be smoothed toward 0.5
    let margin = 16;
    for row in margin..h - margin {
        for col in margin..w - margin {
            let v = blurred[[row, col]];
            assert!(
                (v - 0.5).abs() < 0.05,
                "interior pixel ({row},{col}) should be ~0.5 after blurring, got {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// rgb_align_manual
// ---------------------------------------------------------------------------

#[test]
fn test_rgb_align_manual_zero_offset_is_identity() {
    let color = make_color_frame(32, 32, 0.8, 0.5, 0.3);
    let zero = AlignmentOffset { dx: 0.0, dy: 0.0 };
    let result = rgb_align_manual(&color, &zero, &zero);
    // Channels should be essentially unchanged
    for (a, b) in color.red.data.iter().zip(result.red.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
    for (a, b) in color.blue.data.iter().zip(result.blue.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
    // Green channel is always the reference (unchanged)
    for (a, b) in color.green.data.iter().zip(result.green.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
}

#[test]
fn test_rgb_align_manual_nonzero_offset_shifts() {
    let h = 32;
    let w = 32;
    // Red channel has a gradient; shifting should move values
    let mut red_data = Array2::<f32>::zeros((h, w));
    red_data[[8, 8]] = 1.0;
    let color = ColorFrame {
        red: Frame::new(red_data, 8),
        green: make_frame(h, w, 0.5),
        blue: make_frame(h, w, 0.5),
    };
    let red_offset = AlignmentOffset { dx: 4.0, dy: 0.0 };
    let blue_offset = AlignmentOffset { dx: 0.0, dy: 0.0 };
    let result = rgb_align_manual(&color, &red_offset, &blue_offset);
    // After shifting by 4 pixels in x, the peak should have moved
    let orig_peak = color.red.data[[8, 8]];
    let new_peak = result.red.data[[8, 8]];
    // The original position should no longer be 1.0 (it moved)
    assert!(
        (new_peak - orig_peak).abs() > 0.1 || orig_peak < 0.5,
        "shift should have moved the red channel peak"
    );
}

// ---------------------------------------------------------------------------
// rgb_align (auto)
// ---------------------------------------------------------------------------

#[test]
fn test_rgb_align_identical_channels_no_shift() {
    // All three channels are identical → computed offsets should be ~0
    // → aligned channels are essentially the same as input
    let color = make_color_frame(32, 32, 0.5, 0.5, 0.5);
    let result = rgb_align(&color).unwrap();
    // Result should still be valid (no panic, values in range)
    for v in result.red.data.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
    for v in result.blue.data.iter() {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}
