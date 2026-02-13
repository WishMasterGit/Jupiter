use std::io::Write;
use std::sync::Arc;
use tempfile::{NamedTempFile, TempDir};

use jupiter_core::align::phase_correlation::{compute_offset, compute_offsets_streaming};
use jupiter_core::compute::cpu::CpuBackend;
use jupiter_core::consts::LOW_MEMORY_THRESHOLD_BYTES;
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::{
    FrameSelectionConfig, MemoryStrategy, PipelineConfig, StackingConfig,
};
use jupiter_core::pipeline::run_pipeline;
use jupiter_core::color::debayer::{luminance, DebayerMethod};
use jupiter_core::quality::gradient::{rank_frames_gradient, rank_frames_gradient_color_streaming, rank_frames_gradient_streaming};
use jupiter_core::quality::laplacian::{rank_frames, rank_frames_color_streaming, rank_frames_streaming};
use jupiter_core::stack::mean::{mean_stack, StreamingMeanStacker};

const SER_HEADER_SIZE: usize = 178;

fn build_test_ser(width: u32, height: u32, num_frames: usize) -> Vec<u8> {
    let mut buf = Vec::new();

    // Header
    buf.extend_from_slice(b"LUCAM-RECORDER");
    buf.extend_from_slice(&0i32.to_le_bytes()); // LuID
    buf.extend_from_slice(&0i32.to_le_bytes()); // ColorID = MONO
    buf.extend_from_slice(&0i32.to_le_bytes()); // LittleEndian
    buf.extend_from_slice(&(width as i32).to_le_bytes());
    buf.extend_from_slice(&(height as i32).to_le_bytes());
    buf.extend_from_slice(&8i32.to_le_bytes()); // 8-bit
    buf.extend_from_slice(&(num_frames as i32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 40]); // Observer
    buf.extend_from_slice(&[0u8; 40]); // Instrument
    buf.extend_from_slice(&[0u8; 40]); // Telescope
    buf.extend_from_slice(&0u64.to_le_bytes()); // DateTime
    buf.extend_from_slice(&0u64.to_le_bytes()); // DateTimeUTC
    assert_eq!(buf.len(), SER_HEADER_SIZE);

    let w = width as usize;
    let h = height as usize;
    let square_size = 12;
    let center_y = h / 2;
    let center_x = w / 2;

    for frame_idx in 0..num_frames {
        let mut frame_data = vec![20u8; w * h];
        let jitter_y = (frame_idx % 3) as isize - 1;
        let jitter_x = ((frame_idx + 1) % 3) as isize - 1;
        let sharpness = if frame_idx % 4 == 0 {
            255u8
        } else if frame_idx % 4 == 2 {
            200u8
        } else {
            120u8
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

fn write_test_ser_file(num_frames: usize) -> NamedTempFile {
    let ser_data = build_test_ser(64, 64, num_frames);
    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(&ser_data).unwrap();
    tmpfile
}

/// Build a SER file with BayerRGGB pattern (color_id = 8).
fn build_test_bayer_ser(width: u32, height: u32, num_frames: usize) -> Vec<u8> {
    let mut buf = Vec::new();

    // Header
    buf.extend_from_slice(b"LUCAM-RECORDER");
    buf.extend_from_slice(&0i32.to_le_bytes()); // LuID
    buf.extend_from_slice(&8i32.to_le_bytes()); // ColorID = BayerRGGB
    buf.extend_from_slice(&0i32.to_le_bytes()); // LittleEndian
    buf.extend_from_slice(&(width as i32).to_le_bytes());
    buf.extend_from_slice(&(height as i32).to_le_bytes());
    buf.extend_from_slice(&8i32.to_le_bytes()); // 8-bit
    buf.extend_from_slice(&(num_frames as i32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 40]); // Observer
    buf.extend_from_slice(&[0u8; 40]); // Instrument
    buf.extend_from_slice(&[0u8; 40]); // Telescope
    buf.extend_from_slice(&0u64.to_le_bytes()); // DateTime
    buf.extend_from_slice(&0u64.to_le_bytes()); // DateTimeUTC
    assert_eq!(buf.len(), SER_HEADER_SIZE);

    let w = width as usize;
    let h = height as usize;
    let square_size = 12;
    let center_y = h / 2;
    let center_x = w / 2;

    for frame_idx in 0..num_frames {
        let mut frame_data = vec![40u8; w * h];
        let sharpness = if frame_idx % 4 == 0 {
            220u8
        } else if frame_idx % 4 == 2 {
            160u8
        } else {
            100u8
        };

        let sy = center_y - square_size / 2;
        let sx = center_x - square_size / 2;
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

fn write_test_bayer_ser_file(num_frames: usize) -> NamedTempFile {
    let ser_data = build_test_bayer_ser(64, 64, num_frames);
    let mut tmpfile = NamedTempFile::new().unwrap();
    tmpfile.write_all(&ser_data).unwrap();
    tmpfile
}

// --- Quality scoring tests ---

#[test]
fn test_streaming_laplacian_matches_eager() {
    let ser_file = write_test_ser_file(20);
    let reader = SerReader::open(ser_file.path()).unwrap();

    // Eager: load all frames, score
    let frames: Vec<_> = reader
        .frames()
        .collect::<jupiter_core::error::Result<_>>()
        .unwrap();
    let eager_ranked = rank_frames(&frames);

    // Streaming: score from reader
    let streaming_ranked = rank_frames_streaming(&reader).unwrap();

    // Same number of results
    assert_eq!(eager_ranked.len(), streaming_ranked.len());

    // Same ranking order (indices should match)
    for (eager, streaming) in eager_ranked.iter().zip(streaming_ranked.iter()) {
        assert_eq!(eager.0, streaming.0, "Frame indices differ");
        assert!(
            (eager.1.composite - streaming.1.composite).abs() < 1e-10,
            "Scores differ for frame {}: eager={}, streaming={}",
            eager.0,
            eager.1.composite,
            streaming.1.composite
        );
    }
}

#[test]
fn test_streaming_gradient_matches_eager() {
    let ser_file = write_test_ser_file(20);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let frames: Vec<_> = reader
        .frames()
        .collect::<jupiter_core::error::Result<_>>()
        .unwrap();
    let eager_ranked = rank_frames_gradient(&frames);
    let streaming_ranked = rank_frames_gradient_streaming(&reader).unwrap();

    assert_eq!(eager_ranked.len(), streaming_ranked.len());
    for (eager, streaming) in eager_ranked.iter().zip(streaming_ranked.iter()) {
        assert_eq!(eager.0, streaming.0);
        assert!(
            (eager.1.composite - streaming.1.composite).abs() < 1e-10,
            "Gradient scores differ for frame {}",
            eager.0
        );
    }
}

// --- StreamingMeanStacker tests ---

#[test]
fn test_streaming_mean_stacker_matches_batch() {
    let ser_file = write_test_ser_file(12);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let frames: Vec<_> = reader
        .frames()
        .collect::<jupiter_core::error::Result<_>>()
        .unwrap();

    // Batch mean
    let batch_result = mean_stack(&frames).unwrap();

    // Streaming mean
    let (h, w) = frames[0].data.dim();
    let mut stacker = StreamingMeanStacker::new(h, w, frames[0].original_bit_depth);
    for frame in &frames {
        stacker.add(frame);
    }
    let streaming_result = stacker.finalize().unwrap();

    // Should be pixel-identical
    assert_eq!(batch_result.data.dim(), streaming_result.data.dim());
    for row in 0..h {
        for col in 0..w {
            let diff = (batch_result.data[[row, col]] - streaming_result.data[[row, col]]).abs();
            assert!(
                diff < 1e-6,
                "Pixel [{},{}] differs: batch={}, streaming={}, diff={}",
                row,
                col,
                batch_result.data[[row, col]],
                streaming_result.data[[row, col]],
                diff
            );
        }
    }
}

#[test]
fn test_streaming_mean_stacker_empty_returns_error() {
    let stacker = StreamingMeanStacker::new(10, 10, 8);
    assert!(stacker.finalize().is_err());
}

// --- Streaming offset computation test ---

#[test]
fn test_compute_offsets_streaming_matches_batch() {
    let ser_file = write_test_ser_file(8);
    let reader = SerReader::open(ser_file.path()).unwrap();

    let frames: Vec<_> = reader
        .frames()
        .collect::<jupiter_core::error::Result<_>>()
        .unwrap();

    let backend = Arc::new(CpuBackend);
    let frame_indices: Vec<usize> = (0..frames.len()).collect();

    // Streaming offsets
    let streaming_offsets = compute_offsets_streaming(
        &reader,
        &frame_indices,
        0,
        backend.clone(),
        |_| {},
    )
    .unwrap();

    // Batch offsets
    let reference = &frames[0];
    let batch_offsets: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(i, frame)| {
            if i == 0 {
                jupiter_core::frame::AlignmentOffset::default()
            } else {
                compute_offset(reference, frame).unwrap_or_default()
            }
        })
        .collect();

    assert_eq!(streaming_offsets.len(), batch_offsets.len());
    for (i, (streaming, batch)) in streaming_offsets.iter().zip(batch_offsets.iter()).enumerate() {
        assert!(
            (streaming.dx - batch.dx).abs() < 1e-6,
            "Frame {} dx differs: streaming={}, batch={}",
            i,
            streaming.dx,
            batch.dx
        );
        assert!(
            (streaming.dy - batch.dy).abs() < 1e-6,
            "Frame {} dy differs: streaming={}, batch={}",
            i,
            streaming.dy,
            batch.dy
        );
    }
}

// --- Pipeline integration tests ---

#[test]
fn test_streaming_pipeline_mean_matches_eager() {
    let ser_file = write_test_ser_file(12);
    let out_eager = TempDir::new().unwrap();
    let out_streaming = TempDir::new().unwrap();

    let backend = Arc::new(CpuBackend);

    // Eager pipeline
    let config_eager = PipelineConfig {
        input: ser_file.path().to_path_buf(),
        output: out_eager.path().join("eager.tiff"),
        device: Default::default(),
        memory: MemoryStrategy::Eager,
        debayer: None,
        force_mono: false,
        frame_selection: FrameSelectionConfig {
            select_percentage: 0.5,
            ..Default::default()
        },
        stacking: StackingConfig::default(), // Mean
        sharpening: None,
        filters: vec![],
    };

    let eager_result = run_pipeline(&config_eager, backend.clone(), |_, _| {}).unwrap();

    // Streaming pipeline (force low-memory)
    let config_streaming = PipelineConfig {
        input: ser_file.path().to_path_buf(),
        output: out_streaming.path().join("streaming.tiff"),
        device: Default::default(),
        memory: MemoryStrategy::LowMemory,
        debayer: None,
        force_mono: false,
        frame_selection: FrameSelectionConfig {
            select_percentage: 0.5,
            ..Default::default()
        },
        stacking: StackingConfig::default(),
        sharpening: None,
        filters: vec![],
    };

    let streaming_result = run_pipeline(&config_streaming, backend, |_, _| {}).unwrap();

    let eager_frame = eager_result.to_mono();
    let streaming_frame = streaming_result.to_mono();

    // Dimensions should match
    assert_eq!(eager_frame.data.dim(), streaming_frame.data.dim());

    // Pixel values should be very close (both mean of same selected frames)
    let (h, w) = eager_frame.data.dim();
    let mut max_diff: f32 = 0.0;
    for row in 0..h {
        for col in 0..w {
            let diff = (eager_frame.data[[row, col]] - streaming_frame.data[[row, col]]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    assert!(
        max_diff < 1e-5,
        "Max pixel difference between eager and streaming: {}",
        max_diff
    );
}

#[test]
fn test_memory_strategy_display() {
    assert_eq!(format!("{}", MemoryStrategy::Auto), "Auto");
    assert_eq!(format!("{}", MemoryStrategy::Eager), "Eager");
    assert_eq!(format!("{}", MemoryStrategy::LowMemory), "Low Memory");
}

#[test]
fn test_auto_detection_threshold() {
    // A small file (12 frames of 64x64 f32 = 12 * 64 * 64 * 4 = 196,608 bytes)
    // should be well under the 1 GiB threshold
    let small_decoded = 12 * 64 * 64 * std::mem::size_of::<f32>();
    assert!(small_decoded < LOW_MEMORY_THRESHOLD_BYTES);

    // A large file (100 frames of 4096x4096 f32 = 100 * 4096 * 4096 * 4 = 6.7 GB)
    // should be well over the threshold
    let large_decoded = 100 * 4096 * 4096 * std::mem::size_of::<f32>();
    assert!(large_decoded > LOW_MEMORY_THRESHOLD_BYTES);
}

// --- Color streaming scoring tests ---

#[test]
fn test_color_streaming_laplacian_matches_eager() {
    let ser_file = write_test_bayer_ser_file(20);
    let reader = SerReader::open(ser_file.path()).unwrap();
    let color_mode = reader.header.color_mode();
    let method = DebayerMethod::default();

    // Eager: read all, debayer, luminance, score
    let color_frames: Vec<_> = (0..reader.frame_count())
        .map(|i| reader.read_frame_color(i, &method).unwrap())
        .collect();
    let lum_frames: Vec<_> = color_frames.iter().map(luminance).collect();
    let eager_ranked = rank_frames(&lum_frames);

    // Streaming
    let streaming_ranked = rank_frames_color_streaming(&reader, &color_mode, &method).unwrap();

    assert_eq!(eager_ranked.len(), streaming_ranked.len());
    for (eager, streaming) in eager_ranked.iter().zip(streaming_ranked.iter()) {
        assert_eq!(eager.0, streaming.0, "Frame indices differ");
        assert!(
            (eager.1.composite - streaming.1.composite).abs() < 1e-10,
            "Scores differ for frame {}: eager={}, streaming={}",
            eager.0,
            eager.1.composite,
            streaming.1.composite
        );
    }
}

#[test]
fn test_color_streaming_gradient_matches_eager() {
    let ser_file = write_test_bayer_ser_file(20);
    let reader = SerReader::open(ser_file.path()).unwrap();
    let color_mode = reader.header.color_mode();
    let method = DebayerMethod::default();

    // Eager: read all, debayer, luminance, score with gradient
    let color_frames: Vec<_> = (0..reader.frame_count())
        .map(|i| reader.read_frame_color(i, &method).unwrap())
        .collect();
    let lum_frames: Vec<_> = color_frames.iter().map(luminance).collect();
    let eager_ranked = rank_frames_gradient(&lum_frames);

    // Streaming
    let streaming_ranked =
        rank_frames_gradient_color_streaming(&reader, &color_mode, &method).unwrap();

    assert_eq!(eager_ranked.len(), streaming_ranked.len());
    for (eager, streaming) in eager_ranked.iter().zip(streaming_ranked.iter()) {
        assert_eq!(eager.0, streaming.0, "Frame indices differ");
        assert!(
            (eager.1.composite - streaming.1.composite).abs() < 1e-10,
            "Gradient scores differ for frame {}",
            eager.0
        );
    }
}
