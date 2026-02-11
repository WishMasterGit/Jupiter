use ndarray::Array2;

use jupiter_core::frame::Frame;
use jupiter_core::io::image_io::{load_image, save_png, save_tiff};

#[test]
fn test_save_load_roundtrip_tiff() {
    let mut data = Array2::<f32>::zeros((4, 4));
    data[[0, 0]] = 0.0;
    data[[0, 1]] = 0.5;
    data[[1, 0]] = 1.0;
    data[[2, 3]] = 0.25;
    let frame = Frame::new(data, 16);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.tiff");

    save_tiff(&frame, &path).unwrap();
    let loaded = load_image(&path).unwrap();

    assert_eq!(loaded.width(), 4);
    assert_eq!(loaded.height(), 4);
    assert!((loaded.data[[0, 0]] - 0.0).abs() < 1e-4);
    assert!((loaded.data[[0, 1]] - 0.5).abs() < 1e-3);
    assert!((loaded.data[[1, 0]] - 1.0).abs() < 1e-4);
    assert!((loaded.data[[2, 3]] - 0.25).abs() < 1e-3);
}

#[test]
fn test_save_png() {
    let data = Array2::<f32>::from_elem((8, 8), 0.5);
    let frame = Frame::new(data, 8);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.png");

    save_png(&frame, &path).unwrap();
    assert!(path.exists());
}
