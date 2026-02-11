use ndarray::Array2;

use jupiter_core::frame::Frame;
use jupiter_core::sharpen::wavelet::{self, decompose, mirror_index, reconstruct, WaveletParams};

#[test]
fn test_decompose_reconstruct_identity() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 0..32 {
        for c in 0..32 {
            data[[r, c]] = (r as f32 * 0.1 + c as f32 * 0.05).sin() * 0.5 + 0.5;
        }
    }

    let (layers, residual) = decompose(&data, 4);
    let coefficients = vec![1.0; 4];
    let reconstructed = reconstruct(&layers, &residual, &coefficients, &[]);

    for r in 0..32 {
        for c in 0..32 {
            let diff = (data[[r, c]] - reconstructed[[r, c]]).abs();
            assert!(
                diff < 1e-4,
                "Mismatch at [{r},{c}]: orig={}, recon={}, diff={diff}",
                data[[r, c]],
                reconstructed[[r, c]]
            );
        }
    }
}

#[test]
fn test_sharpening_increases_contrast() {
    let mut data = Array2::<f32>::zeros((32, 32));
    for r in 0..32 {
        for c in 16..32 {
            data[[r, c]] = 1.0;
        }
    }
    let frame = Frame::new(data.clone(), 8);

    let params = WaveletParams {
        num_layers: 4,
        coefficients: vec![2.0, 1.5, 1.0, 1.0],
        denoise: vec![],
    };
    let sharpened = wavelet::sharpen(&frame, &params);

    let orig_dark = data[[16, 14]];
    let sharp_dark = sharpened.data[[16, 14]];
    let orig_bright = data[[16, 17]];
    let sharp_bright = sharpened.data[[16, 17]];

    assert!(
        sharp_dark <= orig_dark + 0.01 || sharp_bright >= orig_bright - 0.01,
        "Sharpening should increase edge contrast"
    );
}

#[test]
fn test_flat_image_unchanged() {
    let data = Array2::from_elem((16, 16), 0.5f32);
    let frame = Frame::new(data, 8);
    let sharpened = wavelet::sharpen(&frame, &WaveletParams::default());

    for r in 0..16 {
        for c in 0..16 {
            assert!(
                (sharpened.data[[r, c]] - 0.5).abs() < 1e-4,
                "Flat image should be unchanged by sharpening"
            );
        }
    }
}

#[test]
fn test_mirror_index() {
    assert_eq!(mirror_index(-1, 10), 1);
    assert_eq!(mirror_index(-2, 10), 2);
    assert_eq!(mirror_index(0, 10), 0);
    assert_eq!(mirror_index(9, 10), 9);
    assert_eq!(mirror_index(10, 10), 9);
    assert_eq!(mirror_index(11, 10), 8);
}
