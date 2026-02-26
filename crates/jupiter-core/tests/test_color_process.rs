use ndarray::Array2;

use jupiter_core::color::process::{
    from_channels, merge_rgb, process_color, process_color_parallel, split_rgb,
};
use jupiter_core::frame::{ColorFrame, Frame};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_frame(h: usize, w: usize, fill: f32) -> Frame {
    Frame::new(Array2::from_elem((h, w), fill), 8)
}

/// Build an interleaved RGB array (shape h × w*3) with uniform R/G/B values.
fn make_interleaved_rgb(h: usize, w: usize, r: f32, g: f32, b: f32) -> Array2<f32> {
    let mut data = Array2::<f32>::zeros((h, w * 3));
    for row in 0..h {
        for col in 0..w {
            data[[row, col * 3]] = r;
            data[[row, col * 3 + 1]] = g;
            data[[row, col * 3 + 2]] = b;
        }
    }
    data
}

// ---------------------------------------------------------------------------
// split_rgb
// ---------------------------------------------------------------------------

#[test]
fn test_split_rgb_correct_channels() {
    let h = 4;
    let w = 4;
    let data = make_interleaved_rgb(h, w, 0.1, 0.5, 0.9);
    let color = split_rgb(&data, 8);

    for v in color.red.data.iter() {
        assert!((*v - 0.1).abs() < 1e-5, "red channel wrong: {v}");
    }
    for v in color.green.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5, "green channel wrong: {v}");
    }
    for v in color.blue.data.iter() {
        assert!((*v - 0.9).abs() < 1e-5, "blue channel wrong: {v}");
    }
}

#[test]
fn test_split_rgb_output_dimensions() {
    let h = 6;
    let w = 8;
    let data = make_interleaved_rgb(h, w, 0.0, 0.0, 0.0);
    let color = split_rgb(&data, 16);
    assert_eq!(color.red.data.dim(), (h, w));
    assert_eq!(color.green.data.dim(), (h, w));
    assert_eq!(color.blue.data.dim(), (h, w));
    assert_eq!(color.red.original_bit_depth, 16);
}

// ---------------------------------------------------------------------------
// merge_rgb
// ---------------------------------------------------------------------------

#[test]
fn test_merge_rgb_correct_interleave() {
    let h = 4;
    let w = 4;
    let color = ColorFrame {
        red: make_frame(h, w, 0.2),
        green: make_frame(h, w, 0.5),
        blue: make_frame(h, w, 0.8),
    };
    let merged = merge_rgb(&color);
    assert_eq!(merged.dim(), (h, w * 3));
    for row in 0..h {
        for col in 0..w {
            assert!((merged[[row, col * 3]] - 0.2).abs() < 1e-5);
            assert!((merged[[row, col * 3 + 1]] - 0.5).abs() < 1e-5);
            assert!((merged[[row, col * 3 + 2]] - 0.8).abs() < 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------
// split_rgb + merge_rgb round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_split_merge_roundtrip() {
    let h = 8;
    let w = 8;
    let original = make_interleaved_rgb(h, w, 0.3, 0.6, 0.9);
    let color = split_rgb(&original, 8);
    let merged = merge_rgb(&color);

    for (a, b) in original.iter().zip(merged.iter()) {
        assert!((*a - *b).abs() < 1e-5, "round-trip mismatch: {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// process_color
// ---------------------------------------------------------------------------

#[test]
fn test_process_color_applies_to_each_channel() {
    let h = 4;
    let w = 4;
    let color = ColorFrame {
        red: make_frame(h, w, 0.2),
        green: make_frame(h, w, 0.4),
        blue: make_frame(h, w, 0.6),
    };
    // Double each pixel, clamp to 1.0
    let result = process_color(&color, |f| {
        Frame::new(f.data.mapv(|v| (v * 2.0).min(1.0)), f.original_bit_depth)
    });
    for v in result.red.data.iter() {
        assert!((*v - 0.4).abs() < 1e-5);
    }
    for v in result.green.data.iter() {
        assert!((*v - 0.8).abs() < 1e-5);
    }
    for v in result.blue.data.iter() {
        assert!((*v - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_process_color_identity() {
    let h = 4;
    let w = 4;
    let color = ColorFrame {
        red: make_frame(h, w, 0.3),
        green: make_frame(h, w, 0.5),
        blue: make_frame(h, w, 0.7),
    };
    let result = process_color(&color, |f| f.clone());
    for (a, b) in color.red.data.iter().zip(result.red.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
    for (a, b) in color.blue.data.iter().zip(result.blue.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// process_color_parallel
// ---------------------------------------------------------------------------

#[test]
fn test_process_color_parallel_same_as_sequential() {
    let h = 32;
    let w = 32;
    let color = ColorFrame {
        red: make_frame(h, w, 0.1),
        green: make_frame(h, w, 0.5),
        blue: make_frame(h, w, 0.9),
    };
    let seq_result = process_color(&color, |f| {
        Frame::new(f.data.mapv(|v| v * 2.0), f.original_bit_depth)
    });
    let par_result = process_color_parallel(&color, |f| {
        Frame::new(f.data.mapv(|v| v * 2.0), f.original_bit_depth)
    });
    for (a, b) in seq_result.red.data.iter().zip(par_result.red.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
    for (a, b) in seq_result
        .green
        .data
        .iter()
        .zip(par_result.green.data.iter())
    {
        assert!((*a - *b).abs() < 1e-5);
    }
    for (a, b) in seq_result.blue.data.iter().zip(par_result.blue.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// from_channels
// ---------------------------------------------------------------------------

#[test]
fn test_from_channels_correct_assignment() {
    let h = 4;
    let w = 4;
    let red = make_frame(h, w, 0.1);
    let green = make_frame(h, w, 0.5);
    let blue = make_frame(h, w, 0.9);
    let color = from_channels(red.clone(), green.clone(), blue.clone());

    for (a, b) in red.data.iter().zip(color.red.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
    for (a, b) in green.data.iter().zip(color.green.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
    for (a, b) in blue.data.iter().zip(color.blue.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
}
