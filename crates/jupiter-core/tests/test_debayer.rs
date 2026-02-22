#[allow(dead_code)]
mod common;

use std::sync::Arc;

use ndarray::Array2;
use tempfile::TempDir;

use jupiter_core::color::debayer::{debayer, is_bayer, luminance, DebayerMethod};
use jupiter_core::compute::cpu::CpuBackend;
use jupiter_core::frame::{ColorFrame, ColorMode, Frame};
use jupiter_core::pipeline::config::{
    DebayerConfig, FrameSelectionConfig, PipelineConfig, StackingConfig,
};
use jupiter_core::pipeline::{run_pipeline, PipelineOutput};

// ---------------------------------------------------------------------------
// Helper: create synthetic Bayer mosaics
// ---------------------------------------------------------------------------

/// Build a uniform Bayer mosaic where every pixel has the same value.
fn uniform_bayer(h: usize, w: usize, value: f32) -> Array2<f32> {
    Array2::<f32>::from_elem((h, w), value)
}

/// Build an RGGB pattern where R=red_val, G=green_val, B=blue_val.
fn patterned_rggb(h: usize, w: usize, red_val: f32, green_val: f32, blue_val: f32) -> Array2<f32> {
    let mut raw = Array2::<f32>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            raw[[row, col]] = match (row % 2, col % 2) {
                (0, 0) => red_val,
                (0, 1) | (1, 0) => green_val,
                (1, 1) => blue_val,
                _ => unreachable!(),
            };
        }
    }
    raw
}

// ---------------------------------------------------------------------------
// Bilinear tests
// ---------------------------------------------------------------------------

#[test]
fn test_bilinear_uniform() {
    let raw = uniform_bayer(16, 16, 0.5);
    let result = debayer(&raw, &ColorMode::BayerRGGB, &DebayerMethod::Bilinear, 8);
    let cf = result.expect("should debayer RGGB");

    // Uniform input → all channels ≈ 0.5
    let (h, w) = cf.red.data.dim();
    assert_eq!(h, 16);
    assert_eq!(w, 16);

    for row in 1..h - 1 {
        for col in 1..w - 1 {
            assert!(
                (cf.red.data[[row, col]] - 0.5).abs() < 1e-4,
                "red[{row},{col}] = {}",
                cf.red.data[[row, col]]
            );
            assert!(
                (cf.green.data[[row, col]] - 0.5).abs() < 1e-4,
                "green[{row},{col}] = {}",
                cf.green.data[[row, col]]
            );
            assert!(
                (cf.blue.data[[row, col]] - 0.5).abs() < 1e-4,
                "blue[{row},{col}] = {}",
                cf.blue.data[[row, col]]
            );
        }
    }
}

#[test]
fn test_bilinear_rggb_known_pattern() {
    // 4x4 RGGB mosaic with distinct R/G/B values
    let raw = patterned_rggb(4, 4, 1.0, 0.5, 0.0);
    let cf = debayer(&raw, &ColorMode::BayerRGGB, &DebayerMethod::Bilinear, 8).unwrap();

    // At center-ish pixel (1,1) which is a blue pixel:
    // - Red should be interpolated from 4 diagonal neighbours (all 1.0) → ≈1.0
    // - Green should be interpolated from 4 cross neighbours (all 0.5) → ≈0.5
    // - Blue is native → 0.0
    assert!((cf.red.data[[1, 1]] - 1.0).abs() < 0.1, "red at blue pixel");
    assert!(
        (cf.green.data[[1, 1]] - 0.5).abs() < 0.1,
        "green at blue pixel"
    );
    assert!((cf.blue.data[[1, 1]] - 0.0).abs() < 0.01, "blue at blue pixel");
}

#[test]
fn test_bilinear_all_bayer_patterns() {
    let raw = uniform_bayer(8, 8, 0.4);
    let modes = [
        ColorMode::BayerRGGB,
        ColorMode::BayerGRBG,
        ColorMode::BayerGBRG,
        ColorMode::BayerBGGR,
    ];

    for mode in &modes {
        let result = debayer(&raw, mode, &DebayerMethod::Bilinear, 8);
        assert!(
            result.is_some(),
            "debayer should succeed for {:?}",
            mode
        );
        let cf = result.unwrap();
        // Uniform input → all channels should be uniform (≈0.4)
        for row in 1..7 {
            for col in 1..7 {
                assert!(
                    (cf.red.data[[row, col]] - 0.4).abs() < 1e-4,
                    "{:?} red[{row},{col}] = {}",
                    mode,
                    cf.red.data[[row, col]]
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MHC tests
// ---------------------------------------------------------------------------

#[test]
fn test_mhc_uniform() {
    let raw = uniform_bayer(16, 16, 0.6);
    let cf = debayer(&raw, &ColorMode::BayerRGGB, &DebayerMethod::MalvarHeCutler, 8).unwrap();

    for row in 2..14 {
        for col in 2..14 {
            assert!(
                (cf.red.data[[row, col]] - 0.6).abs() < 1e-3,
                "red[{row},{col}] = {}",
                cf.red.data[[row, col]]
            );
            assert!(
                (cf.green.data[[row, col]] - 0.6).abs() < 1e-3,
                "green[{row},{col}] = {}",
                cf.green.data[[row, col]]
            );
            assert!(
                (cf.blue.data[[row, col]] - 0.6).abs() < 1e-3,
                "blue[{row},{col}] = {}",
                cf.blue.data[[row, col]]
            );
        }
    }
}

#[test]
fn test_mhc_differs_from_bilinear() {
    // Create a Bayer mosaic with distinct per-color values creating color contrast.
    // MHC's gradient correction makes a difference when nearby pixels of different
    // Bayer phases have very different values (typical of color edges).
    let mut raw = Array2::<f32>::zeros((16, 16));
    for row in 0..16 {
        for col in 0..16 {
            // RGGB: red=high, green=medium, blue=low — creates inter-channel difference
            raw[[row, col]] = match (row % 2, col % 2) {
                (0, 0) => 0.9,  // R
                (0, 1) => 0.3,  // G on R row
                (1, 0) => 0.3,  // G on B row
                _ => 0.1,       // B
            };
        }
    }
    // Add a sharp step to the right half to create a spatial edge
    for row in 0..16 {
        for col in 8..16 {
            raw[[row, col]] *= 0.2;
        }
    }

    let bilinear = debayer(&raw, &ColorMode::BayerRGGB, &DebayerMethod::Bilinear, 8).unwrap();
    let mhc = debayer(&raw, &ColorMode::BayerRGGB, &DebayerMethod::MalvarHeCutler, 8).unwrap();

    // They should differ at or near the edge
    let mut any_diff = false;
    for row in 3..13 {
        for col in 3..13 {
            let diff = (bilinear.red.data[[row, col]] - mhc.red.data[[row, col]]).abs();
            if diff > 1e-6 {
                any_diff = true;
            }
        }
    }
    assert!(any_diff, "MHC should produce different results from bilinear near color edges");
}

// ---------------------------------------------------------------------------
// Edge cases and properties
// ---------------------------------------------------------------------------

#[test]
fn test_non_bayer_returns_none() {
    let raw = Array2::<f32>::zeros((4, 4));
    assert!(debayer(&raw, &ColorMode::Mono, &DebayerMethod::Bilinear, 8).is_none());
    assert!(debayer(&raw, &ColorMode::RGB, &DebayerMethod::Bilinear, 8).is_none());
    assert!(debayer(&raw, &ColorMode::BGR, &DebayerMethod::Bilinear, 8).is_none());
}

#[test]
fn test_output_range() {
    // Random-ish values in [0, 1]
    let mut raw = Array2::<f32>::zeros((32, 32));
    for row in 0..32 {
        for col in 0..32 {
            raw[[row, col]] = ((row * 7 + col * 13) % 100) as f32 / 100.0;
        }
    }

    for method in &[DebayerMethod::Bilinear, DebayerMethod::MalvarHeCutler] {
        let cf = debayer(&raw, &ColorMode::BayerRGGB, method, 8).unwrap();
        for row in 0..32 {
            for col in 0..32 {
                assert!(
                    cf.red.data[[row, col]] >= 0.0 && cf.red.data[[row, col]] <= 1.0,
                    "{:?} red[{row},{col}] = {} out of range",
                    method,
                    cf.red.data[[row, col]]
                );
                assert!(
                    cf.green.data[[row, col]] >= 0.0 && cf.green.data[[row, col]] <= 1.0,
                    "{:?} green out of range",
                    method,
                );
                assert!(
                    cf.blue.data[[row, col]] >= 0.0 && cf.blue.data[[row, col]] <= 1.0,
                    "{:?} blue out of range",
                    method,
                );
            }
        }
    }
}

#[test]
fn test_preserves_dimensions() {
    let raw = Array2::<f32>::zeros((17, 23)); // odd dimensions
    let cf = debayer(&raw, &ColorMode::BayerGRBG, &DebayerMethod::Bilinear, 8).unwrap();
    assert_eq!(cf.red.data.dim(), (17, 23));
    assert_eq!(cf.green.data.dim(), (17, 23));
    assert_eq!(cf.blue.data.dim(), (17, 23));
}

#[test]
fn test_is_bayer() {
    assert!(is_bayer(&ColorMode::BayerRGGB));
    assert!(is_bayer(&ColorMode::BayerGRBG));
    assert!(is_bayer(&ColorMode::BayerGBRG));
    assert!(is_bayer(&ColorMode::BayerBGGR));
    assert!(!is_bayer(&ColorMode::Mono));
    assert!(!is_bayer(&ColorMode::RGB));
    assert!(!is_bayer(&ColorMode::BGR));
}

// ---------------------------------------------------------------------------
// Luminance tests
// ---------------------------------------------------------------------------

#[test]
fn test_luminance_weights() {
    let red = Frame::new(Array2::<f32>::from_elem((4, 4), 1.0), 8);
    let green = Frame::new(Array2::<f32>::zeros((4, 4)), 8);
    let blue = Frame::new(Array2::<f32>::zeros((4, 4)), 8);
    let cf = ColorFrame { red, green, blue };

    let lum = luminance(&cf);
    // Pure red → luminance ≈ 0.299
    assert!(
        (lum.data[[1, 1]] - 0.299).abs() < 1e-3,
        "pure red lum = {}",
        lum.data[[1, 1]]
    );

    let red = Frame::new(Array2::<f32>::zeros((4, 4)), 8);
    let green = Frame::new(Array2::<f32>::from_elem((4, 4), 1.0), 8);
    let blue = Frame::new(Array2::<f32>::zeros((4, 4)), 8);
    let cf = ColorFrame { red, green, blue };
    let lum = luminance(&cf);
    // Pure green → luminance ≈ 0.587
    assert!(
        (lum.data[[1, 1]] - 0.587).abs() < 1e-3,
        "pure green lum = {}",
        lum.data[[1, 1]]
    );

    let red = Frame::new(Array2::<f32>::zeros((4, 4)), 8);
    let green = Frame::new(Array2::<f32>::zeros((4, 4)), 8);
    let blue = Frame::new(Array2::<f32>::from_elem((4, 4), 1.0), 8);
    let cf = ColorFrame { red, green, blue };
    let lum = luminance(&cf);
    // Pure blue → luminance ≈ 0.114
    assert!(
        (lum.data[[1, 1]] - 0.114).abs() < 1e-3,
        "pure blue lum = {}",
        lum.data[[1, 1]]
    );
}

#[test]
fn test_luminance_white_equals_one() {
    let red = Frame::new(Array2::<f32>::from_elem((4, 4), 1.0), 8);
    let green = Frame::new(Array2::<f32>::from_elem((4, 4), 1.0), 8);
    let blue = Frame::new(Array2::<f32>::from_elem((4, 4), 1.0), 8);
    let cf = ColorFrame { red, green, blue };
    let lum = luminance(&cf);
    assert!(
        (lum.data[[1, 1]] - 1.0).abs() < 1e-3,
        "white lum = {}",
        lum.data[[1, 1]]
    );
}

// ---------------------------------------------------------------------------
// SER integration tests
// ---------------------------------------------------------------------------

/// Build a synthetic Bayer RGGB SER file with 8-bit pixels.
fn build_bayer_ser(width: u32, height: u32, num_frames: usize) -> Vec<u8> {
    let mut buf = common::build_ser_header_full(width, height, 8, num_frames, 8);

    let w = width as usize;
    let h = height as usize;

    for frame_idx in 0..num_frames {
        let mut frame_data = vec![0u8; w * h];
        // Create a simple pattern: bright square that shifts slightly
        let center_y = h / 2;
        let center_x = w / 2;
        let jitter = frame_idx % 2;
        let square_size = 8;

        for row in 0..h {
            for col in 0..w {
                // Background: RGGB pattern with distinct values
                let bg = match (row % 2, col % 2) {
                    (0, 0) => 180u8, // R
                    (0, 1) | (1, 0) => 120u8, // G
                    _ => 60u8,                 // B
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

#[test]
fn test_ser_read_frame_color() {
    let ser_file = write_bayer_ser(32, 32, 3);
    let reader = jupiter_core::io::ser::SerReader::open(ser_file.path()).unwrap();

    assert!(reader.is_bayer(), "should detect Bayer pattern");
    assert!(reader.is_color(), "should be color");

    let cf = reader
        .read_frame_color(0, &DebayerMethod::Bilinear)
        .expect("should read color frame");

    assert_eq!(cf.red.data.dim(), (32, 32));
    assert_eq!(cf.green.data.dim(), (32, 32));
    assert_eq!(cf.blue.data.dim(), (32, 32));

    // Values should be in [0, 1]
    for row in 0..32 {
        for col in 0..32 {
            assert!(cf.red.data[[row, col]] >= 0.0 && cf.red.data[[row, col]] <= 1.0);
            assert!(cf.green.data[[row, col]] >= 0.0 && cf.green.data[[row, col]] <= 1.0);
            assert!(cf.blue.data[[row, col]] >= 0.0 && cf.blue.data[[row, col]] <= 1.0);
        }
    }
}

#[test]
fn test_color_pipeline_end_to_end() {
    let ser_file = write_bayer_ser(32, 32, 8);
    let out_dir = TempDir::new().unwrap();
    let output_path = out_dir.path().join("color_result.tiff");

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
        stacking: StackingConfig::default(),
        sharpening: None,
        filters: vec![],
    };

    let backend = Arc::new(CpuBackend);
    let result = run_pipeline(&config, backend, |_, _| {});
    assert!(result.is_ok(), "Color pipeline failed: {:?}", result.err());

    match result.unwrap() {
        PipelineOutput::Color(cf) => {
            assert_eq!(cf.red.data.dim(), (32, 32));
            assert_eq!(cf.green.data.dim(), (32, 32));
            assert_eq!(cf.blue.data.dim(), (32, 32));
        }
        PipelineOutput::Mono(_) => panic!("Expected color output for Bayer input with debayer enabled"),
    }

    assert!(output_path.exists(), "Color output file should exist");
}

#[test]
fn test_force_mono_ignores_bayer() {
    let ser_file = write_bayer_ser(32, 32, 4);
    let out_dir = TempDir::new().unwrap();
    let output_path = out_dir.path().join("mono_result.tiff");

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
        stacking: StackingConfig::default(),
        sharpening: None,
        filters: vec![],
    };

    let backend = Arc::new(CpuBackend);
    let result = run_pipeline(&config, backend, |_, _| {});
    assert!(result.is_ok(), "Mono pipeline failed: {:?}", result.err());

    match result.unwrap() {
        PipelineOutput::Mono(frame) => {
            assert_eq!(frame.width(), 32);
            assert_eq!(frame.height(), 32);
        }
        PipelineOutput::Color(_) => panic!("Expected mono output with force_mono=true"),
    }
}
