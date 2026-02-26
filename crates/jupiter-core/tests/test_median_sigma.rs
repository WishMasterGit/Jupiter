use ndarray::Array2;

use jupiter_core::frame::Frame;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::sigma_clip::{sigma_clip_stack, SigmaClipParams};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_frame(h: usize, w: usize, fill: f32) -> Frame {
    Frame::new(Array2::from_elem((h, w), fill), 8)
}

// ---------------------------------------------------------------------------
// median_stack — sequential path (small frames, 64×64)
// ---------------------------------------------------------------------------

#[test]
fn test_median_single_frame() {
    let frame = make_frame(8, 8, 0.7);
    let result = median_stack(&[frame]).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.7).abs() < 1e-5);
    }
}

#[test]
fn test_median_two_frames() {
    let f1 = make_frame(8, 8, 0.2);
    let f2 = make_frame(8, 8, 0.8);
    let result = median_stack(&[f1, f2]).unwrap();
    // Median of [0.2, 0.8] (even count) = (0.2 + 0.8) / 2 = 0.5
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_median_odd_count() {
    // Median of [0.1, 0.5, 0.9] = 0.5
    let f1 = make_frame(8, 8, 0.1);
    let f2 = make_frame(8, 8, 0.5);
    let f3 = make_frame(8, 8, 0.9);
    let result = median_stack(&[f1, f2, f3]).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_median_even_count() {
    // Median of [0.1, 0.3, 0.7, 0.9] = (0.3+0.7)/2 = 0.5
    let frames: Vec<Frame> = [0.1f32, 0.3, 0.7, 0.9]
        .iter()
        .map(|&v| make_frame(8, 8, v))
        .collect();
    let result = median_stack(&frames).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_median_empty_error() {
    let frames: Vec<Frame> = vec![];
    assert!(median_stack(&frames).is_err());
}

#[test]
fn test_median_rejects_outlier() {
    // 5 frames: four at 0.5 and one outlier at 0.0
    // Median should still be 0.5
    let mut frames: Vec<Frame> = (0..4).map(|_| make_frame(8, 8, 0.5)).collect();
    frames.push(make_frame(8, 8, 0.0));
    let result = median_stack(&frames).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// median_stack — parallel path (512×512 frames, >PARALLEL_PIXEL_THRESHOLD)
// ---------------------------------------------------------------------------

#[test]
fn test_median_large_frames_parallel() {
    // 512×512 = 262144 > 65536, uses parallel path
    let f1 = make_frame(512, 512, 0.3);
    let f2 = make_frame(512, 512, 0.5);
    let f3 = make_frame(512, 512, 0.7);
    let result = median_stack(&[f1, f2, f3]).unwrap();
    // Median of [0.3, 0.5, 0.7] = 0.5
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_median_large_frames_even_count() {
    // 512×512, 4 frames: median of [0.1, 0.4, 0.6, 0.9] = (0.4+0.6)/2 = 0.5
    let frames: Vec<Frame> = [0.1f32, 0.4, 0.6, 0.9]
        .iter()
        .map(|&v| make_frame(512, 512, v))
        .collect();
    let result = median_stack(&frames).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// SigmaClipParams::default
// ---------------------------------------------------------------------------

#[test]
fn test_sigma_clip_params_default() {
    let p = SigmaClipParams::default();
    assert_eq!(p.iterations, 2);
    assert!((p.sigma - 2.5).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// sigma_clip_stack — sequential path (small frames)
// ---------------------------------------------------------------------------

#[test]
fn test_sigma_clip_empty_error() {
    let frames: Vec<Frame> = vec![];
    let params = SigmaClipParams::default();
    assert!(sigma_clip_stack(&frames, &params).is_err());
}

#[test]
fn test_sigma_clip_identical_frames() {
    // All frames same value → stddev=0 → early break; result = that value
    let frames: Vec<Frame> = (0..5).map(|_| make_frame(8, 8, 0.5)).collect();
    let params = SigmaClipParams::default();
    let result = sigma_clip_stack(&frames, &params).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_sigma_clip_rejects_outlier() {
    // 5 frames at 0.5, one outlier at 1.0 → sigma clip should reject it
    let mut frames: Vec<Frame> = (0..5).map(|_| make_frame(8, 8, 0.5)).collect();
    frames.push(make_frame(8, 8, 1.0));
    let params = SigmaClipParams {
        iterations: 3,
        sigma: 2.0,
    };
    let result = sigma_clip_stack(&frames, &params).unwrap();
    // After clipping the outlier, mean should be close to 0.5
    for v in result.data.iter() {
        assert!(
            (*v - 0.5).abs() < 0.1,
            "expected ~0.5 after clipping, got {v}"
        );
    }
}

#[test]
fn test_sigma_clip_single_frame() {
    let frame = make_frame(8, 8, 0.3);
    let params = SigmaClipParams::default();
    let result = sigma_clip_stack(&[frame], &params).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.3).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// sigma_clip_stack — parallel path (512×512)
// ---------------------------------------------------------------------------

#[test]
fn test_sigma_clip_large_frames_parallel() {
    // 512×512 = 262144 > 65536, uses parallel path
    let frames: Vec<Frame> = (0..6).map(|_| make_frame(512, 512, 0.6)).collect();
    let params = SigmaClipParams::default();
    let result = sigma_clip_stack(&frames, &params).unwrap();
    for v in result.data.iter() {
        assert!((*v - 0.6).abs() < 1e-5);
    }
}

#[test]
fn test_sigma_clip_large_frames_outlier_rejected() {
    // Large image, 5 good frames + 1 outlier
    let mut frames: Vec<Frame> = (0..5).map(|_| make_frame(512, 512, 0.4)).collect();
    frames.push(make_frame(512, 512, 1.0));
    let params = SigmaClipParams {
        iterations: 2,
        sigma: 1.5,
    };
    let result = sigma_clip_stack(&frames, &params).unwrap();
    for v in result.data.iter() {
        assert!(
            (*v - 0.4).abs() < 0.1,
            "expected ~0.4 after clipping outlier, got {v}"
        );
    }
}
