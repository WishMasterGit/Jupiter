#[allow(dead_code)]
mod common;

use jupiter_core::io::ser::SerReader;
use jupiter_core::stack::multi_point::{multi_point_stack, MultiPointConfig};
use jupiter_core::stack::surface_warp::{surface_warp_stack, SurfaceWarpConfig};

/// Build a SER with a bright disc that drifts across frames.
///
/// The disc shifts by `drift_per_frame` pixels each frame (both x and y).
/// With large drift, plain phase correlation can fail due to FFT wrap-around,
/// but pre-centering should handle it.
fn write_drifting_ser(
    width: u32,
    height: u32,
    num_frames: usize,
    drift_per_frame: isize,
) -> tempfile::NamedTempFile {
    let w = width as usize;
    let h = height as usize;
    let mut buf = common::build_ser_header(width, height, num_frames);

    let radius = 12isize;

    for frame_idx in 0..num_frames {
        let center_y = (h / 2) as isize + (frame_idx as isize) * drift_per_frame;
        let center_x = (w / 2) as isize + (frame_idx as isize) * drift_per_frame;

        let mut frame_data = vec![10u8; w * h];
        for row in 0..h {
            for col in 0..w {
                let dy = row as isize - center_y;
                let dx = col as isize - center_x;
                if dy * dy + dx * dx <= radius * radius {
                    frame_data[row * w + col] = 200;
                }
            }
        }
        buf.extend_from_slice(&frame_data);
    }

    common::write_test_ser(&buf)
}

#[test]
fn test_multi_point_pre_center_runs() {
    // Small drift — pre_center path should succeed
    let ser_file = write_drifting_ser(128, 128, 6, 2);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let config = MultiPointConfig {
        ap_size: 32,
        search_radius: 8,
        select_percentage: 0.5,
        min_brightness: 0.01,
        pre_center: true,
        ..Default::default()
    };

    let result = multi_point_stack(&reader, &config, |_| {});
    assert!(
        result.is_ok(),
        "multi_point_stack with pre_center failed: {:?}",
        result.err()
    );
    let frame = result.unwrap();
    assert_eq!(frame.data.dim(), (128, 128));
}

#[test]
fn test_surface_warp_pre_center_runs() {
    // Small drift — pre_center path should succeed
    let ser_file = write_drifting_ser(128, 128, 6, 2);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let config = SurfaceWarpConfig {
        ap_size: 32,
        search_radius: 8,
        select_percentage: 0.5,
        min_brightness: 0.01,
        pre_center: true,
        ..Default::default()
    };

    let result = surface_warp_stack(&reader, &config, |_| {});
    assert!(
        result.is_ok(),
        "surface_warp_stack with pre_center failed: {:?}",
        result.err()
    );
    let frame = result.unwrap();
    assert_eq!(frame.data.dim(), (128, 128));
}

#[test]
fn test_multi_point_pre_center_false_unchanged() {
    // With pre_center: false, should behave the same as before
    let ser_file = write_drifting_ser(128, 128, 6, 1);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let config = MultiPointConfig {
        ap_size: 32,
        search_radius: 8,
        select_percentage: 0.5,
        min_brightness: 0.01,
        pre_center: false,
        ..Default::default()
    };

    let result = multi_point_stack(&reader, &config, |_| {});
    assert!(result.is_ok());
}

#[test]
fn test_surface_warp_pre_center_false_unchanged() {
    // With pre_center: false, should behave the same as before
    let ser_file = write_drifting_ser(128, 128, 6, 1);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let config = SurfaceWarpConfig {
        ap_size: 32,
        search_radius: 8,
        select_percentage: 0.5,
        min_brightness: 0.01,
        pre_center: false,
        ..Default::default()
    };

    let result = surface_warp_stack(&reader, &config, |_| {});
    assert!(result.is_ok());
}

#[test]
fn test_pre_center_detection_failure_graceful() {
    // When planet detection fails (uniform dark frames), pre_center should
    // gracefully fall back to normal alignment (center_offsets = None).
    let w = 64u32;
    let h = 64u32;
    let num_frames = 4;
    let mut buf = common::build_ser_header(w, h, num_frames);

    // Uniform dark frames — detection will fail
    for _ in 0..num_frames {
        buf.extend_from_slice(&vec![5u8; (w * h) as usize]);
    }

    let ser_file = common::write_test_ser(&buf);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let config = MultiPointConfig {
        ap_size: 32,
        search_radius: 4,
        select_percentage: 1.0,
        min_brightness: 0.0,
        pre_center: true,
        ..Default::default()
    };

    // Should not error — just falls back to normal alignment
    let result = multi_point_stack(&reader, &config, |_| {});
    assert!(
        result.is_ok(),
        "pre_center with failed detection should not error: {:?}",
        result.err()
    );
}
