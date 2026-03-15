use ndarray::Array2;

use jupiter_core::consts::DISK_FRAME_HEADER_SIZE;
use jupiter_core::frame::{ColorFrame, Frame};
use jupiter_core::io::disk_cache::{DiskFrame, DiskFrameStore};
use jupiter_core::stack::disk_backed::{median_stack_disk, sigma_clip_stack_disk, MmapFrameSlice};
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::sigma_clip::{sigma_clip_stack, SigmaClipParams};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_frame(h: usize, w: usize, val: f32, bit_depth: u8) -> Frame {
    let data = Array2::from_elem((h, w), val);
    Frame::new(data, bit_depth)
}

fn make_gradient_frame(h: usize, w: usize, bit_depth: u8) -> Frame {
    let total = (h * w) as f32;
    let data = Array2::from_shape_fn((h, w), |(r, c)| (r * w + c) as f32 / total);
    Frame::new(data, bit_depth)
}

// ---------------------------------------------------------------------------
// Frame serialization round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_disk_frame_roundtrip_small() {
    let store = DiskFrameStore::new().unwrap();
    let original = make_frame(4, 8, 0.5, 16);

    let disk_frame = store.store_frame(&original).unwrap();
    let restored = disk_frame.read().unwrap();

    assert_eq!(restored.height(), 4);
    assert_eq!(restored.width(), 8);
    assert_eq!(restored.original_bit_depth, 16);
    assert_eq!(restored.data, original.data);
}

#[test]
fn test_disk_frame_roundtrip_8bit() {
    let store = DiskFrameStore::new().unwrap();
    let original = make_gradient_frame(16, 16, 8);

    let disk_frame = store.store_frame(&original).unwrap();
    let restored = disk_frame.read().unwrap();

    assert_eq!(restored.original_bit_depth, 8);
    for ((r, c), &val) in original.data.indexed_iter() {
        assert!(
            (restored.data[[r, c]] - val).abs() < 1e-7,
            "Pixel ({r},{c}) mismatch: {} vs {val}",
            restored.data[[r, c]]
        );
    }
}

#[test]
fn test_disk_frame_roundtrip_large() {
    let store = DiskFrameStore::new().unwrap();
    let original = make_gradient_frame(256, 256, 16);

    let disk_frame = store.store_frame(&original).unwrap();
    let restored = disk_frame.read().unwrap();

    assert_eq!(restored.data.dim(), (256, 256));
    assert_eq!(restored.data, original.data);
}

#[test]
fn test_disk_color_frame_roundtrip() {
    let store = DiskFrameStore::new().unwrap();
    let cf = ColorFrame {
        red: make_frame(8, 8, 0.1, 16),
        green: make_frame(8, 8, 0.5, 16),
        blue: make_frame(8, 8, 0.9, 16),
    };

    let disk_cf = store.store_color_frame(&cf).unwrap();
    let restored = disk_cf.read().unwrap();

    assert_eq!(restored.red.data, cf.red.data);
    assert_eq!(restored.green.data, cf.green.data);
    assert_eq!(restored.blue.data, cf.blue.data);
}

// ---------------------------------------------------------------------------
// DiskFrameStore: creation, concurrent writes, RAII cleanup
// ---------------------------------------------------------------------------

#[test]
fn test_disk_frame_store_creation() {
    let store = DiskFrameStore::new().unwrap();
    assert!(store.path().exists());
}

#[test]
fn test_disk_frame_store_with_parent() {
    let parent = tempfile::TempDir::new().unwrap();
    let store = DiskFrameStore::with_parent(parent.path()).unwrap();
    assert!(store.path().starts_with(parent.path()));
}

#[test]
fn test_disk_frame_store_concurrent_writes() {
    use rayon::prelude::*;

    let store = DiskFrameStore::new().unwrap();
    let frames: Vec<Frame> = (0..16)
        .map(|i| make_frame(8, 8, i as f32 / 16.0, 16))
        .collect();

    let disk_frames: Vec<DiskFrame> = frames
        .par_iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();

    assert_eq!(disk_frames.len(), 16);
    // Verify each roundtrips correctly
    for (original, disk) in frames.iter().zip(disk_frames.iter()) {
        let restored = disk.read().unwrap();
        assert_eq!(restored.data, original.data);
    }
}

#[test]
fn test_disk_frame_raii_cleanup() {
    let store = DiskFrameStore::new().unwrap();
    let frame = make_frame(4, 4, 0.3, 8);
    let disk_frame = store.store_frame(&frame).unwrap();
    let path = disk_frame.path().to_path_buf();
    assert!(path.exists());
    drop(disk_frame);
    assert!(!path.exists(), "File should be deleted on drop");
}

#[test]
fn test_disk_frame_store_raii_cleanup() {
    let store = DiskFrameStore::new().unwrap();
    let dir_path = store.path().to_path_buf();
    let frame = make_frame(4, 4, 0.3, 8);
    let _df = store.store_frame(&frame).unwrap();
    assert!(dir_path.exists());
    drop(_df);
    drop(store);
    assert!(
        !dir_path.exists(),
        "Temp directory should be deleted on store drop"
    );
}

// ---------------------------------------------------------------------------
// MmapFrameSlice pixel access
// ---------------------------------------------------------------------------

#[test]
fn test_mmap_frame_slice_pixel_access() {
    let store = DiskFrameStore::new().unwrap();
    let original = make_gradient_frame(16, 32, 16);
    let disk_frame = store.store_frame(&original).unwrap();

    let slice = MmapFrameSlice::from_disk_frame(&disk_frame).unwrap();
    assert_eq!(slice.height(), 16);
    assert_eq!(slice.width(), 32);

    for r in 0..16 {
        for c in 0..32 {
            let expected = original.data[[r, c]];
            let actual = slice.pixel(r, c);
            assert!(
                (actual - expected).abs() < 1e-7,
                "Mismatch at ({r},{c}): got {actual}, expected {expected}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Disk-backed median matches in-memory median
// ---------------------------------------------------------------------------

#[test]
fn test_median_stack_disk_matches_memory() {
    let store = DiskFrameStore::new().unwrap();
    let frames: Vec<Frame> = vec![
        make_frame(32, 32, 0.1, 16),
        make_frame(32, 32, 0.5, 16),
        make_frame(32, 32, 0.9, 16),
    ];

    let disk_frames: Vec<DiskFrame> = frames
        .iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();
    let slices: Vec<MmapFrameSlice> = disk_frames
        .iter()
        .map(|df| MmapFrameSlice::from_disk_frame(df).unwrap())
        .collect();

    let disk_result = median_stack_disk(&slices, 16).unwrap();
    let mem_result = median_stack(&frames).unwrap();

    assert_eq!(disk_result.data.dim(), mem_result.data.dim());
    for ((r, c), &expected) in mem_result.data.indexed_iter() {
        let actual = disk_result.data[[r, c]];
        assert!(
            (actual - expected).abs() < 1e-7,
            "Median mismatch at ({r},{c}): disk={actual}, mem={expected}"
        );
    }
}

#[test]
fn test_median_stack_disk_gradient_frames() {
    let store = DiskFrameStore::new().unwrap();
    let frames: Vec<Frame> = (0..5)
        .map(|i| {
            let data = Array2::from_shape_fn((64, 64), |(r, c)| {
                (r * 64 + c) as f32 / 4096.0 + i as f32 * 0.01
            });
            Frame::new(data, 16)
        })
        .collect();

    let disk_frames: Vec<DiskFrame> = frames
        .iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();
    let slices: Vec<MmapFrameSlice> = disk_frames
        .iter()
        .map(|df| MmapFrameSlice::from_disk_frame(df).unwrap())
        .collect();

    let disk_result = median_stack_disk(&slices, 16).unwrap();
    let mem_result = median_stack(&frames).unwrap();

    for ((r, c), &expected) in mem_result.data.indexed_iter() {
        let actual = disk_result.data[[r, c]];
        assert!(
            (actual - expected).abs() < 1e-6,
            "Gradient median mismatch at ({r},{c})"
        );
    }
}

// ---------------------------------------------------------------------------
// Disk-backed sigma-clip matches in-memory sigma-clip
// ---------------------------------------------------------------------------

#[test]
fn test_sigma_clip_stack_disk_matches_memory() {
    let store = DiskFrameStore::new().unwrap();
    let frames: Vec<Frame> = vec![
        make_frame(32, 32, 0.5, 16),
        make_frame(32, 32, 0.5, 16),
        make_frame(32, 32, 0.5, 16),
        make_frame(32, 32, 0.5, 16),
        make_frame(32, 32, 0.99, 16), // outlier
    ];

    let params = SigmaClipParams {
        sigma: 2.0,
        iterations: 3,
    };

    let disk_frames: Vec<DiskFrame> = frames
        .iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();
    let slices: Vec<MmapFrameSlice> = disk_frames
        .iter()
        .map(|df| MmapFrameSlice::from_disk_frame(df).unwrap())
        .collect();

    let disk_result = sigma_clip_stack_disk(&slices, params.sigma, params.iterations, 16).unwrap();
    let mem_result = sigma_clip_stack(&frames, &params).unwrap();

    assert_eq!(disk_result.data.dim(), mem_result.data.dim());
    for ((r, c), &expected) in mem_result.data.indexed_iter() {
        let actual = disk_result.data[[r, c]];
        assert!(
            (actual - expected).abs() < 1e-6,
            "Sigma-clip mismatch at ({r},{c}): disk={actual}, mem={expected}"
        );
    }
}

#[test]
fn test_sigma_clip_stack_disk_gradient_with_outliers() {
    let store = DiskFrameStore::new().unwrap();
    let mut frames: Vec<Frame> = (0..7)
        .map(|i| {
            let data = Array2::from_shape_fn((32, 32), |(r, c)| {
                (r * 32 + c) as f32 / 1024.0 + i as f32 * 0.005
            });
            Frame::new(data, 16)
        })
        .collect();
    // Add an outlier frame
    frames.push(Frame::new(Array2::from_elem((32, 32), 1.0), 16));

    let params = SigmaClipParams {
        sigma: 2.5,
        iterations: 2,
    };

    let disk_frames: Vec<DiskFrame> = frames
        .iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();
    let slices: Vec<MmapFrameSlice> = disk_frames
        .iter()
        .map(|df| MmapFrameSlice::from_disk_frame(df).unwrap())
        .collect();

    let disk_result = sigma_clip_stack_disk(&slices, params.sigma, params.iterations, 16).unwrap();
    let mem_result = sigma_clip_stack(&frames, &params).unwrap();

    for ((r, c), &expected) in mem_result.data.indexed_iter() {
        let actual = disk_result.data[[r, c]];
        assert!(
            (actual - expected).abs() < 1e-6,
            "Sigma-clip+outlier mismatch at ({r},{c})"
        );
    }
}

// ---------------------------------------------------------------------------
// Error handling: corrupt header, wrong magic, truncated file
// ---------------------------------------------------------------------------

#[test]
fn test_disk_frame_corrupt_magic() {
    let store = DiskFrameStore::new().unwrap();
    let frame = make_frame(4, 4, 0.5, 16);
    let disk_frame = store.store_frame(&frame).unwrap();

    // Corrupt the magic bytes
    let path = disk_frame.path().to_path_buf();
    let mut data = std::fs::read(&path).unwrap();
    data[0] = b'X';
    std::fs::write(&path, &data).unwrap();

    let err = disk_frame.read();
    assert!(err.is_err(), "Should fail with corrupt magic");
    let msg = format!("{}", err.unwrap_err());
    assert!(
        msg.contains("magic"),
        "Error should mention magic: got '{msg}'"
    );
}

#[test]
fn test_disk_frame_truncated_file() {
    let store = DiskFrameStore::new().unwrap();
    let frame = make_frame(8, 8, 0.5, 16);
    let disk_frame = store.store_frame(&frame).unwrap();

    // Truncate the file to just the header
    let path = disk_frame.path().to_path_buf();
    let data = std::fs::read(&path).unwrap();
    std::fs::write(&path, &data[..DISK_FRAME_HEADER_SIZE]).unwrap();

    let err = disk_frame.read();
    assert!(err.is_err(), "Should fail with truncated file");
}

#[test]
fn test_mmap_frame_slice_truncated() {
    let store = DiskFrameStore::new().unwrap();
    let frame = make_frame(8, 8, 0.5, 16);
    let disk_frame = store.store_frame(&frame).unwrap();

    // Truncate to half the expected size
    let path = disk_frame.path().to_path_buf();
    let data = std::fs::read(&path).unwrap();
    std::fs::write(&path, &data[..data.len() / 2]).unwrap();

    let err = MmapFrameSlice::from_disk_frame(&disk_frame);
    assert!(err.is_err(), "Should fail with truncated mmap");
}

// ---------------------------------------------------------------------------
// Edge cases: empty sequence, single frame
// ---------------------------------------------------------------------------

#[test]
fn test_median_stack_disk_empty() {
    let slices: Vec<MmapFrameSlice> = vec![];
    let err = median_stack_disk(&slices, 16);
    assert!(err.is_err());
}

#[test]
fn test_sigma_clip_stack_disk_empty() {
    let slices: Vec<MmapFrameSlice> = vec![];
    let err = sigma_clip_stack_disk(&slices, 2.5, 2, 16);
    assert!(err.is_err());
}

#[test]
fn test_median_stack_disk_single_frame() {
    let store = DiskFrameStore::new().unwrap();
    let frame = make_gradient_frame(16, 16, 16);
    let disk_frame = store.store_frame(&frame).unwrap();
    let slice = MmapFrameSlice::from_disk_frame(&disk_frame).unwrap();

    let result = median_stack_disk(&[slice], 16).unwrap();
    assert_eq!(result.data, frame.data);
}

#[test]
fn test_sigma_clip_stack_disk_single_frame() {
    let store = DiskFrameStore::new().unwrap();
    let frame = make_gradient_frame(16, 16, 16);
    let disk_frame = store.store_frame(&frame).unwrap();
    let slice = MmapFrameSlice::from_disk_frame(&disk_frame).unwrap();

    let result = sigma_clip_stack_disk(&[slice], 2.5, 2, 16).unwrap();
    for ((r, c), &expected) in frame.data.indexed_iter() {
        let actual = result.data[[r, c]];
        assert!(
            (actual - expected).abs() < 1e-7,
            "Single-frame sigma-clip mismatch at ({r},{c})"
        );
    }
}

// ---------------------------------------------------------------------------
// Two-frame median (even count)
// ---------------------------------------------------------------------------

#[test]
fn test_median_stack_disk_even_count() {
    let store = DiskFrameStore::new().unwrap();
    let frames = [make_frame(8, 8, 0.2, 16), make_frame(8, 8, 0.8, 16)];

    let disk_frames: Vec<DiskFrame> = frames
        .iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();
    let slices: Vec<MmapFrameSlice> = disk_frames
        .iter()
        .map(|df| MmapFrameSlice::from_disk_frame(df).unwrap())
        .collect();

    let result = median_stack_disk(&slices, 16).unwrap();
    // Even count median = average of two middle values = (0.2 + 0.8) / 2 = 0.5
    for &val in result.data.iter() {
        assert!((val - 0.5).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Parallel path tests (images >= 256x256 trigger row-level parallelism)
// ---------------------------------------------------------------------------

#[test]
fn test_median_stack_disk_parallel_path() {
    let store = DiskFrameStore::new().unwrap();
    // 256x256 = 65536 pixels, which matches PARALLEL_PIXEL_THRESHOLD
    let frames: Vec<Frame> = vec![
        make_frame(256, 256, 0.2, 16),
        make_frame(256, 256, 0.5, 16),
        make_frame(256, 256, 0.8, 16),
    ];

    let disk_frames: Vec<DiskFrame> = frames
        .iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();
    let slices: Vec<MmapFrameSlice> = disk_frames
        .iter()
        .map(|df| MmapFrameSlice::from_disk_frame(df).unwrap())
        .collect();

    let disk_result = median_stack_disk(&slices, 16).unwrap();
    let mem_result = median_stack(&frames).unwrap();

    assert_eq!(disk_result.data.dim(), mem_result.data.dim());
    // Spot-check corners and center
    for (r, c) in [(0, 0), (0, 255), (128, 128), (255, 0), (255, 255)] {
        assert!(
            (disk_result.data[[r, c]] - mem_result.data[[r, c]]).abs() < 1e-7,
            "Parallel median mismatch at ({r},{c})"
        );
    }
}

#[test]
fn test_sigma_clip_stack_disk_parallel_path() {
    let store = DiskFrameStore::new().unwrap();
    let mut frames: Vec<Frame> = (0..5)
        .map(|i| make_frame(256, 256, 0.5 + i as f32 * 0.01, 16))
        .collect();
    // Add outlier
    frames.push(make_frame(256, 256, 1.0, 16));

    let params = SigmaClipParams {
        sigma: 2.0,
        iterations: 3,
    };

    let disk_frames: Vec<DiskFrame> = frames
        .iter()
        .map(|f| store.store_frame(f).unwrap())
        .collect();
    let slices: Vec<MmapFrameSlice> = disk_frames
        .iter()
        .map(|df| MmapFrameSlice::from_disk_frame(df).unwrap())
        .collect();

    let disk_result = sigma_clip_stack_disk(&slices, params.sigma, params.iterations, 16).unwrap();
    let mem_result = sigma_clip_stack(&frames, &params).unwrap();

    assert_eq!(disk_result.data.dim(), mem_result.data.dim());
    for (r, c) in [(0, 0), (0, 255), (128, 128), (255, 0), (255, 255)] {
        assert!(
            (disk_result.data[[r, c]] - mem_result.data[[r, c]]).abs() < 1e-6,
            "Parallel sigma-clip mismatch at ({r},{c})"
        );
    }
}
