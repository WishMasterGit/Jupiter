#[allow(dead_code)]
mod common;

use std::sync::Arc;
use tempfile::TempDir;

use jupiter_core::align::phase_correlation::{compute_offset, shift_frame};
use jupiter_core::compute::cpu::CpuBackend;
use jupiter_core::io::image_io::{load_image, save_tiff};
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::{
    FrameSelectionConfig, PipelineConfig, SharpeningConfig, StackingConfig,
};
use jupiter_core::pipeline::run_pipeline;
use jupiter_core::quality::laplacian::rank_frames;
use jupiter_core::sharpen::wavelet::{self, WaveletParams};
use jupiter_core::stack::mean::mean_stack;

/// Build a synthetic SER file with a bright square that shifts across frames.
fn build_test_ser(
    width: u32,
    height: u32,
    num_frames: usize,
) -> Vec<u8> {
    let mut buf = common::build_ser_header(width, height, num_frames);

    // Generate frames: a bright square that shifts slightly between frames
    // simulating atmospheric jitter. Also vary the "sharpness" (edge contrast)
    // to make quality scoring meaningful.
    let w = width as usize;
    let h = height as usize;
    let square_size = 12;
    let center_y = h / 2;
    let center_x = w / 2;

    for frame_idx in 0..num_frames {
        let mut frame_data = vec![20u8; w * h]; // dark background with some noise

        // Small jitter: shift the square by a few pixels per frame
        let jitter_y = (frame_idx % 3) as isize - 1; // -1, 0, 1
        let jitter_x = ((frame_idx + 1) % 3) as isize - 1;

        // Vary brightness to simulate atmospheric seeing:
        // some frames are "sharper" (higher contrast edge)
        let sharpness = if frame_idx % 4 == 0 {
            255u8 // sharp frame
        } else if frame_idx % 4 == 2 {
            200u8 // medium
        } else {
            120u8 // blurry (low contrast)
        };

        let sy = (center_y as isize + jitter_y - square_size as isize / 2) as usize;
        let sx = (center_x as isize + jitter_x - square_size as isize / 2) as usize;

        for r in sy..sy + square_size {
            for c in sx..sx + square_size {
                if r < h && c < w {
                    frame_data[r * w + c] = sharpness;
                }
            }
        }

        buf.extend_from_slice(&frame_data);
    }

    buf
}

fn write_test_ser(num_frames: usize) -> tempfile::NamedTempFile {
    let ser_data = build_test_ser(64, 64, num_frames);
    common::write_test_ser(&ser_data)
}

#[test]
fn test_full_pipeline_end_to_end() {
    let ser_file = write_test_ser(20);
    let out_dir = TempDir::new().unwrap();
    let output_path = out_dir.path().join("result.tiff");

    let config = PipelineConfig {
        input: ser_file.path().to_path_buf(),
        output: output_path.clone(),
        device: Default::default(),
        memory: Default::default(),
        debayer: None,
        force_mono: false,
        frame_selection: FrameSelectionConfig {
            select_percentage: 0.5,
            ..Default::default()
        },
        alignment: Default::default(),
        stacking: StackingConfig::default(),
        sharpening: Some(SharpeningConfig::default()),
        filters: vec![],
    };

    let backend = Arc::new(CpuBackend);
    let result = run_pipeline(&config, backend, |_, _| {});

    assert!(result.is_ok(), "Pipeline failed: {:?}", result.err());

    let frame = result.unwrap().to_mono();
    assert_eq!(frame.width(), 64);
    assert_eq!(frame.height(), 64);

    // Output file should exist
    assert!(output_path.exists(), "Output file should be created");

    // Should be able to load it back
    let loaded = load_image(&output_path).unwrap();
    assert_eq!(loaded.width(), 64);
    assert_eq!(loaded.height(), 64);
}

#[test]
fn test_pipeline_without_sharpening() {
    let ser_file = write_test_ser(8);
    let out_dir = TempDir::new().unwrap();
    let output_path = out_dir.path().join("no_sharpen.tiff");

    let config = PipelineConfig {
        input: ser_file.path().to_path_buf(),
        output: output_path.clone(),
        device: Default::default(),
        memory: Default::default(),
        debayer: None,
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
    assert!(result.is_ok());
    assert!(output_path.exists());
}

#[test]
fn test_quality_scoring_ranks_sharp_frames_higher() {
    let ser_file = write_test_ser(12);
    let reader = SerReader::open(ser_file.path()).unwrap();
    let frames: Vec<_> = reader.frames().collect::<Result<_, _>>().unwrap();

    let ranked = rank_frames(&frames);

    // The "sharp" frames (idx 0, 4, 8 with brightness 255) should rank higher
    // than the "blurry" frames (idx 1, 3, 5, 7, 9, 11 with brightness 120)
    let top_3: Vec<usize> = ranked.iter().take(3).map(|(i, _)| *i).collect();

    // At least one of the sharp frames should be in top 3
    let sharp_indices = [0, 4, 8];
    let has_sharp = top_3.iter().any(|i| sharp_indices.contains(i));
    assert!(
        has_sharp,
        "Expected at least one sharp frame in top 3, got {:?}",
        top_3
    );
}

#[test]
fn test_align_stack_sharpen_manual_pipeline() {
    // Manual step-by-step pipeline (without the pipeline runner)
    let ser_file = write_test_ser(10);
    let reader = SerReader::open(ser_file.path()).unwrap();
    let frames: Vec<_> = reader.frames().collect::<Result<_, _>>().unwrap();
    assert_eq!(frames.len(), 10);

    // Quality assessment
    let ranked = rank_frames(&frames);
    assert_eq!(ranked.len(), 10);

    // Select top 50%
    let keep = 5;
    let selected: Vec<_> = ranked.iter().take(keep).map(|(i, _)| frames[*i].clone()).collect();
    assert_eq!(selected.len(), 5);

    // Align to best frame
    let reference = &selected[0];
    let mut aligned = vec![reference.clone()];
    for frame in selected.iter().skip(1) {
        let offset = compute_offset(reference, frame).unwrap();
        aligned.push(shift_frame(frame, &offset));
    }

    // Stack
    let stacked = mean_stack(&aligned).unwrap();
    assert_eq!(stacked.width(), 64);
    assert_eq!(stacked.height(), 64);

    // Stacked image should have non-zero content
    let center_val = stacked.data[[32, 32]];
    assert!(center_val > 0.1, "Center should be bright, got {}", center_val);

    // Sharpen
    let params = WaveletParams {
        num_layers: 4,
        coefficients: vec![1.5, 1.3, 1.1, 1.0],
        denoise: vec![],
    };
    let sharpened = wavelet::sharpen(&stacked, &params);
    assert_eq!(sharpened.width(), 64);

    // Save and reload
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("manual_result.tiff");
    save_tiff(&sharpened, &path).unwrap();
    let loaded = load_image(&path).unwrap();
    assert_eq!(loaded.width(), 64);
}

#[test]
fn test_ser_reader_source_info() {
    let ser_file = write_test_ser(5);
    let reader = SerReader::open(ser_file.path()).unwrap();
    let info = reader.source_info(ser_file.path());

    assert_eq!(info.total_frames, 5);
    assert_eq!(info.width, 64);
    assert_eq!(info.height, 64);
    assert_eq!(info.bit_depth, 8);
}

#[test]
fn test_wavelet_roundtrip_preserves_energy() {
    // The sum of all layers + residual should equal the original
    let ser_file = write_test_ser(1);
    let reader = SerReader::open(ser_file.path()).unwrap();
    let frame = reader.read_frame(0).unwrap();

    let (layers, residual) = wavelet::decompose(&frame.data, 6);

    // Reconstruct with all coefficients = 1.0
    let reconstructed = wavelet::reconstruct(&layers, &residual, &[1.0; 6], &[]);

    let (h, w) = frame.data.dim();
    let mut max_diff = 0.0f32;
    for r in 0..h {
        for c in 0..w {
            let diff = (frame.data[[r, c]] - reconstructed[[r, c]]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    assert!(
        max_diff < 1e-4,
        "Wavelet roundtrip max error: {} (should be < 1e-4)",
        max_diff
    );
}
