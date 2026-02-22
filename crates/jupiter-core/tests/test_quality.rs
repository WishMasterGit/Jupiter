use ndarray::Array2;

use jupiter_core::frame::Frame;
use jupiter_core::quality::laplacian::{laplacian_variance, rank_frames, select_best};

#[test]
fn test_flat_image_has_zero_variance() {
    let data = Array2::<f32>::from_elem((10, 10), 0.5);
    let frame = Frame::new(data, 8);
    let lv = laplacian_variance(&frame);
    assert!(
        lv.abs() < 1e-10,
        "Flat image should have ~0 Laplacian variance"
    );
}

#[test]
fn test_sharp_beats_blurry() {
    // "Sharp" image: alternating pixels
    let mut sharp_data = Array2::<f32>::zeros((16, 16));
    for row in 0..16 {
        for col in 0..16 {
            sharp_data[[row, col]] = if (row + col) % 2 == 0 { 1.0 } else { 0.0 };
        }
    }
    let sharp = Frame::new(sharp_data, 8);

    // "Blurry" image: smooth gradient
    let mut blurry_data = Array2::<f32>::zeros((16, 16));
    for row in 0..16 {
        for col in 0..16 {
            blurry_data[[row, col]] = (row as f32 + col as f32) / 30.0;
        }
    }
    let blurry = Frame::new(blurry_data, 8);

    let sharp_score = laplacian_variance(&sharp);
    let blurry_score = laplacian_variance(&blurry);

    assert!(
        sharp_score > blurry_score,
        "Sharp image ({sharp_score}) should score higher than blurry ({blurry_score})"
    );
}

#[test]
fn test_rank_and_select() {
    let flat = Frame::new(Array2::from_elem((10, 10), 0.5), 8);

    let mut sharp_data = Array2::<f32>::zeros((10, 10));
    for r in 0..10 {
        for c in 0..10 {
            sharp_data[[r, c]] = if (r + c) % 2 == 0 { 1.0 } else { 0.0 };
        }
    }
    let sharp = Frame::new(sharp_data, 8);

    let frames = vec![flat.clone(), sharp, flat];
    let ranked = rank_frames(&frames);

    // Sharpest frame (index 1) should be first
    assert_eq!(ranked[0].0, 1);

    // 34% of 3 frames = ceil(1.02) = 2 frames
    let selected = select_best(&frames, 0.34);
    assert_eq!(selected.len(), 2);
    // Best frame (index 1) should be first in selection
    assert_eq!(selected[0], 1);
}
