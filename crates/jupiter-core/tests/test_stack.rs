use ndarray::Array2;

use jupiter_core::frame::Frame;
use jupiter_core::stack::mean::mean_stack;

#[test]
fn test_single_frame_stack() {
    let data = Array2::from_elem((4, 4), 0.5f32);
    let frame = Frame::new(data, 8);
    let result = mean_stack(&[frame]).unwrap();
    assert!((result.data[[0, 0]] - 0.5).abs() < 1e-6);
}

#[test]
fn test_identical_frames() {
    let data = Array2::from_elem((4, 4), 0.3f32);
    let frames: Vec<Frame> = (0..10).map(|_| Frame::new(data.clone(), 8)).collect();
    let result = mean_stack(&frames).unwrap();
    assert!((result.data[[2, 2]] - 0.3).abs() < 1e-5);
}

#[test]
fn test_mean_of_two() {
    let f1 = Frame::new(Array2::from_elem((4, 4), 0.0f32), 8);
    let f2 = Frame::new(Array2::from_elem((4, 4), 1.0f32), 8);
    let result = mean_stack(&[f1, f2]).unwrap();
    assert!((result.data[[0, 0]] - 0.5).abs() < 1e-6);
}

#[test]
fn test_empty_error() {
    let frames: Vec<Frame> = vec![];
    assert!(mean_stack(&frames).is_err());
}
