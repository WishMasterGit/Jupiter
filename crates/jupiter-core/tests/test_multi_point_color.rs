#[allow(dead_code)]
mod common;

use std::sync::Arc;

use tempfile::TempDir;

use jupiter_core::color::debayer::DebayerMethod;
use jupiter_core::compute::cpu::CpuBackend;
use jupiter_core::frame::ColorMode;
use jupiter_core::pipeline::config::{
    DebayerConfig, FrameSelectionConfig, PipelineConfig, StackMethod, StackingConfig,
};
use jupiter_core::pipeline::{run_pipeline, PipelineOutput};
use jupiter_core::stack::multi_point::{
    multi_point_stack_color, LocalStackMethod, MultiPointConfig,
};

/// Build a synthetic Bayer RGGB SER file with 8-bit pixels.
/// Each frame has a bright square that jitters slightly between frames.
fn build_bayer_ser(width: u32, height: u32, num_frames: usize) -> Vec<u8> {
    let mut buf = common::build_ser_header_full(width, height, 8, num_frames, 8);

    let w = width as usize;
    let h = height as usize;

    for frame_idx in 0..num_frames {
        let mut frame_data = vec![0u8; w * h];
        let center_y = h / 2;
        let center_x = w / 2;
        let jitter = frame_idx % 2;
        let square_size = 8;

        for row in 0..h {
            for col in 0..w {
                // RGGB pattern with distinct channel values
                let bg = match (row % 2, col % 2) {
                    (0, 0) => 180u8,          // R
                    (0, 1) | (1, 0) => 120u8, // G
                    _ => 60u8,                // B
                };
                frame_data[row * w + col] = bg;
            }
        }

        // Bright square
        let sy = center_y - square_size / 2 + jitter;
        let sx = center_x - square_size / 2;
        for row in sy..sy + square_size {
            for col in sx..sx + square_size {
                if row < h && col < w {
                    frame_data[row * w + col] = 240;
                }
            }
        }

        buf.extend_from_slice(&frame_data);
    }

    buf
}

fn write_bayer_ser(width: u32, height: u32, num_frames: usize) -> tempfile::NamedTempFile {
    let ser_data = build_bayer_ser(width, height, num_frames);
    common::write_test_ser(&ser_data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_multi_point_stack_color_basic() {
    let ser_file = write_bayer_ser(128, 128, 8);
    let reader = jupiter_core::io::ser::SerReader::open(ser_file.path()).unwrap();
    let config = MultiPointConfig {
        ap_size: 32,
        search_radius: 8,
        select_percentage: 0.5,
        min_brightness: 0.01,
        ..Default::default()
    };

    let result = multi_point_stack_color(
        &reader,
        &config,
        &ColorMode::BayerRGGB,
        &DebayerMethod::Bilinear,
        |_| {},
    );
    assert!(
        result.is_ok(),
        "multi_point_stack_color failed: {:?}",
        result.err()
    );

    let cf = result.unwrap();
    assert_eq!(cf.red.data.dim(), (128, 128));
    assert_eq!(cf.green.data.dim(), (128, 128));
    assert_eq!(cf.blue.data.dim(), (128, 128));

    // All pixels should be in [0, 1]
    for row in 0..128 {
        for col in 0..128 {
            assert!(cf.red.data[[row, col]] >= 0.0 && cf.red.data[[row, col]] <= 1.0);
            assert!(cf.green.data[[row, col]] >= 0.0 && cf.green.data[[row, col]] <= 1.0);
            assert!(cf.blue.data[[row, col]] >= 0.0 && cf.blue.data[[row, col]] <= 1.0);
        }
    }
}

#[test]
fn test_multi_point_stack_color_single_frame() {
    let ser_file = write_bayer_ser(64, 64, 1);
    let reader = jupiter_core::io::ser::SerReader::open(ser_file.path()).unwrap();
    let config = MultiPointConfig {
        ap_size: 32,
        search_radius: 4,
        select_percentage: 1.0,
        min_brightness: 0.01,
        ..Default::default()
    };

    let result = multi_point_stack_color(
        &reader,
        &config,
        &ColorMode::BayerRGGB,
        &DebayerMethod::Bilinear,
        |_| {},
    );
    assert!(
        result.is_ok(),
        "Single frame color multipoint failed: {:?}",
        result.err()
    );

    let cf = result.unwrap();
    assert_eq!(cf.red.data.dim(), (64, 64));
}

#[test]
fn test_multi_point_stack_color_channels_independent() {
    let ser_file = write_bayer_ser(128, 128, 8);
    let reader = jupiter_core::io::ser::SerReader::open(ser_file.path()).unwrap();
    let config = MultiPointConfig {
        ap_size: 32,
        search_radius: 8,
        select_percentage: 0.5,
        min_brightness: 0.01,
        ..Default::default()
    };

    let cf = multi_point_stack_color(
        &reader,
        &config,
        &ColorMode::BayerRGGB,
        &DebayerMethod::Bilinear,
        |_| {},
    )
    .unwrap();

    // The Bayer pattern has distinct R/G/B background values (180/120/60 on [0..255]),
    // so the stacked R, G, B channels should not be identical.
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let (h, w) = cf.red.data.dim();
    let n = (h * w) as f64;

    for row in 0..h {
        for col in 0..w {
            r_sum += cf.red.data[[row, col]] as f64;
            g_sum += cf.green.data[[row, col]] as f64;
            b_sum += cf.blue.data[[row, col]] as f64;
        }
    }

    let r_mean = r_sum / n;
    let g_mean = g_sum / n;
    let b_mean = b_sum / n;

    // R background ~0.706 (180/255), G ~0.471 (120/255), B ~0.235 (60/255)
    // They should differ from each other
    assert!(
        (r_mean - g_mean).abs() > 0.01,
        "R and G means should differ: r={r_mean:.4}, g={g_mean:.4}"
    );
    assert!(
        (g_mean - b_mean).abs() > 0.01,
        "G and B means should differ: g={g_mean:.4}, b={b_mean:.4}"
    );
}

#[test]
fn test_multi_point_stack_color_pipeline_integration() {
    let ser_file = write_bayer_ser(128, 128, 8);
    let out_dir = TempDir::new().unwrap();
    let output_path = out_dir.path().join("mp_color.tiff");

    let config = PipelineConfig {
        input: ser_file.path().to_path_buf(),
        output: output_path.clone(),
        device: Default::default(),
        memory: Default::default(),
        debayer: Some(DebayerConfig {
            method: DebayerMethod::Bilinear,
        }),
        force_mono: false,
        frame_selection: FrameSelectionConfig {
            select_percentage: 0.5,
            ..Default::default()
        },
        alignment: Default::default(),
        stacking: StackingConfig {
            method: StackMethod::MultiPoint(MultiPointConfig {
                ap_size: 32,
                search_radius: 8,
                min_brightness: 0.01,
                ..Default::default()
            }),
        },
        sharpening: None,
        filters: vec![],
    };

    let backend = Arc::new(CpuBackend);
    let result = run_pipeline(&config, backend, |_, _| {});
    assert!(result.is_ok(), "Pipeline failed: {:?}", result.err());

    match result.unwrap() {
        PipelineOutput::Color(cf) => {
            assert_eq!(cf.red.data.dim(), (128, 128));
            assert_eq!(cf.green.data.dim(), (128, 128));
            assert_eq!(cf.blue.data.dim(), (128, 128));
        }
        PipelineOutput::Mono(_) => {
            panic!("Expected Color output for Bayer + MultiPoint, got Mono")
        }
    }

    assert!(output_path.exists(), "Output file should exist");
}

#[test]
fn test_multi_point_stack_color_force_mono() {
    let ser_file = write_bayer_ser(128, 128, 8);
    let out_dir = TempDir::new().unwrap();
    let output_path = out_dir.path().join("mp_mono.tiff");

    let config = PipelineConfig {
        input: ser_file.path().to_path_buf(),
        output: output_path.clone(),
        device: Default::default(),
        memory: Default::default(),
        debayer: Some(DebayerConfig {
            method: DebayerMethod::Bilinear,
        }),
        force_mono: true,
        frame_selection: FrameSelectionConfig {
            select_percentage: 0.5,
            ..Default::default()
        },
        alignment: Default::default(),
        stacking: StackingConfig {
            method: StackMethod::MultiPoint(MultiPointConfig {
                ap_size: 32,
                search_radius: 8,
                min_brightness: 0.01,
                ..Default::default()
            }),
        },
        sharpening: None,
        filters: vec![],
    };

    let backend = Arc::new(CpuBackend);
    let result = run_pipeline(&config, backend, |_, _| {});
    assert!(result.is_ok(), "Pipeline failed: {:?}", result.err());

    match result.unwrap() {
        PipelineOutput::Mono(frame) => {
            assert_eq!(frame.width(), 128);
            assert_eq!(frame.height(), 128);
        }
        PipelineOutput::Color(_) => {
            panic!("Expected Mono output with force_mono=true, got Color")
        }
    }
}

#[test]
fn test_multi_point_stack_color_local_methods() {
    let ser_file = write_bayer_ser(128, 128, 8);

    let methods = [
        LocalStackMethod::Mean,
        LocalStackMethod::Median,
        LocalStackMethod::SigmaClip {
            sigma: 2.0,
            iterations: 2,
        },
    ];

    for method in &methods {
        let reader = jupiter_core::io::ser::SerReader::open(ser_file.path()).unwrap();
        let config = MultiPointConfig {
            ap_size: 32,
            search_radius: 8,
            select_percentage: 0.5,
            min_brightness: 0.01,
            local_stack_method: method.clone(),
            ..Default::default()
        };

        let result = multi_point_stack_color(
            &reader,
            &config,
            &ColorMode::BayerRGGB,
            &DebayerMethod::Bilinear,
            |_| {},
        );
        assert!(
            result.is_ok(),
            "Local method {:?} failed: {:?}",
            method,
            result.err()
        );

        let cf = result.unwrap();
        assert_eq!(cf.red.data.dim(), (128, 128));
        assert_eq!(cf.green.data.dim(), (128, 128));
        assert_eq!(cf.blue.data.dim(), (128, 128));
    }
}
