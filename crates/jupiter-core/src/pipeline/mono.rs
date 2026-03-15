use std::sync::Arc;

use tracing::info;

use crate::align::{
    compute_centering_offsets, compute_common_overlap, compute_offsets_streaming_configured,
    detect_centroids, detect_centroids_streaming, pre_center_frames, shift_frame,
};
use crate::compute::ComputeBackend;
use crate::detection::config::DetectionConfig;
use crate::error::Result;
use crate::frame::{AlignmentOffset, Frame};
use crate::io::crop::CropRect;
use crate::io::disk_cache::DiskFrameStore;
use crate::io::image_io::save_image;
use crate::io::ser::SerReader;
use crate::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use crate::sharpen::wavelet;
use crate::stack::disk_backed::{median_stack_disk, sigma_clip_stack_disk, MmapFrameSlice};
use crate::stack::drizzle::{drizzle_output_dims, drizzle_single_frame};
use crate::stack::mean::StreamingMeanStacker;
use crate::stack::sigma_clip::SigmaClipParams;

use super::config::PipelineConfig;
use super::config::StackMethod;
use super::helpers::{
    apply_filter_step, compute_offsets_with_progress, drizzle_frames, rank_by_metric,
    rank_by_metric_streaming, select_frames, stack_frames_with_progress,
};
use super::types::{PipelineOutput, PipelineStage, ProgressReporter};

/// The existing mono pipeline path (unchanged logic).
pub(super) fn run_mono_pipeline(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    total: usize,
) -> Result<PipelineOutput> {
    let streaming = super::orchestrator::should_use_streaming(reader, config, false);
    if streaming {
        info!("Using low-memory streaming mode");
    }

    let stacked = if streaming {
        run_mono_standard_streaming(reader, config, backend, reporter, total)?
    } else {
        run_mono_standard(reader, config, backend, reporter, total)?
    };

    let output = apply_post_stack_mono(stacked, config, backend, reporter)?;
    Ok(output)
}

fn run_mono_standard(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    total: usize,
) -> Result<Frame> {
    // Read
    reporter.begin_stage(PipelineStage::Reading, Some(total));
    let frames: Vec<Frame> = reader.frames().collect::<Result<_>>()?;
    reporter.finish_stage();

    // Quality
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric(&frames, &config.frame_selection.metric);
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, _) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    let selected_frames: Vec<Frame> = selected_indices
        .iter()
        .map(|&i| frames[i].clone())
        .collect();
    info!(
        selected = selected_frames.len(),
        total, "Selected best frames"
    );
    reporter.finish_stage();

    // Pre-centering (optional)
    let selected_frames = if config.alignment.pre_center {
        let frame_count = selected_frames.len();
        reporter.begin_stage(PipelineStage::PreCentering, Some(frame_count));
        let detection_config = DetectionConfig::default();
        let centroids = detect_centroids(&selected_frames, &detection_config);
        let (h, w) = (selected_frames[0].height(), selected_frames[0].width());
        let result = if let Some(offsets) = compute_centering_offsets(&centroids, h, w) {
            let crop = compute_common_overlap(&offsets, h, w);
            info!(
                detected = centroids.iter().filter(|c| c.is_some()).count(),
                total = centroids.len(),
                crop = crop.is_some(),
                "Pre-centering frames"
            );
            pre_center_frames(&selected_frames, &offsets, crop.as_ref())
        } else {
            info!("Pre-centering skipped: too few planet detections");
            selected_frames
        };
        reporter.finish_stage();
        result
    } else {
        selected_frames
    };

    // Compute alignment offsets
    let offsets =
        compute_offsets_with_progress(&selected_frames, 0, &config.alignment, backend, reporter)?;

    // If drizzle enabled, project each frame onto high-res grid; otherwise shift
    let frames_to_stack = if let Some(ref drizzle_config) = config.stacking.drizzle {
        info!(
            scale = drizzle_config.scale,
            pixfrac = drizzle_config.pixfrac,
            "Drizzle-projecting frames"
        );
        drizzle_frames(&selected_frames, &offsets, drizzle_config)?
    } else {
        // Traditional integer-shift alignment
        let frame_count = selected_frames.len();
        reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
        // Offsets already computed; just re-use the shift-based approach
        // but we need to re-report alignment since compute_offsets_with_progress
        // already finished that stage. The alignment stage was consumed by offset
        // computation — shift is part of stacking prep.
        selected_frames
            .iter()
            .zip(offsets.iter())
            .enumerate()
            .map(|(i, (frame, offset))| {
                if i == 0 {
                    frame.clone()
                } else {
                    shift_frame(frame, offset)
                }
            })
            .collect()
    };

    // Stacking
    let frame_count = frames_to_stack.len();
    reporter.begin_stage(PipelineStage::Stacking, Some(frame_count));
    let r = reporter.clone();
    let result =
        stack_frames_with_progress(&frames_to_stack, &config.stacking.method, move |done| {
            r.advance(done);
        })?;
    info!(method = ?config.stacking.method, "Stacking complete");
    reporter.finish_stage();

    Ok(result)
}

/// Streaming mono pipeline: score -> select -> load-shift-stack one at a time.
///
/// For Mean: fully streaming -- each frame is loaded, shifted, accumulated, then dropped.
/// For Median/SigmaClip: semi-streaming -- offsets computed streaming, then M selected
/// frames loaded+shifted for the per-pixel stacking pass.
fn run_mono_standard_streaming(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    total: usize,
) -> Result<Frame> {
    // Quality (streaming: one batch at a time)
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric_streaming(reader, &config.frame_selection.metric)?;
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, _) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    info!(
        selected = selected_indices.len(),
        total, "Selected best frames (streaming)"
    );
    reporter.finish_stage();

    let frame_count = selected_indices.len();
    let h = reader.header.height as usize;
    let w = reader.header.width as usize;

    // Pre-centering (streaming): detect centroids and compute centering offsets
    let pre_center_data = if config.alignment.pre_center {
        reporter.begin_stage(PipelineStage::PreCentering, Some(frame_count));
        let detection_config = DetectionConfig::default();
        let centroids = detect_centroids_streaming(reader, &selected_indices, &detection_config)?;
        let result = if let Some(center_offsets) = compute_centering_offsets(&centroids, h, w) {
            let crop = compute_common_overlap(&center_offsets, h, w);
            info!(
                detected = centroids.iter().filter(|c| c.is_some()).count(),
                total = centroids.len(),
                crop = crop.is_some(),
                "Pre-centering frames (streaming)"
            );
            Some((center_offsets, crop))
        } else {
            info!("Pre-centering skipped: too few planet detections (streaming)");
            None
        };
        reporter.finish_stage();
        result
    } else {
        None
    };

    // Compute alignment offsets (streaming)
    reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
    let r = reporter.clone();
    let offsets = compute_offsets_streaming_configured(
        reader,
        &selected_indices,
        0,
        &config.alignment,
        backend.clone(),
        move |done| {
            r.advance(done);
        },
    )?;
    reporter.finish_stage();

    let drizzle_config = config.stacking.drizzle.as_ref();

    // Determine output dimensions (may be scaled up for drizzle)
    let (stack_h, stack_w) = {
        let (base_h, base_w) = if let Some((_, Some(ref crop))) = pre_center_data {
            (crop.height as usize, crop.width as usize)
        } else {
            (h, w)
        };
        if let Some(dc) = drizzle_config {
            drizzle_output_dims(base_h, base_w, dc.scale)
        } else {
            (base_h, base_w)
        }
    };

    match &config.stacking.method {
        StackMethod::Mean => {
            // Fully streaming: load, transform (shift or drizzle), accumulate one at a time
            reporter.begin_stage(PipelineStage::Stacking, Some(frame_count));
            let bit_depth = reader.header.pixel_depth as u8;
            let mut stacker = StreamingMeanStacker::new(stack_h, stack_w, bit_depth);
            for (i, (&frame_idx, offset)) in selected_indices.iter().zip(offsets.iter()).enumerate()
            {
                let frame = reader.read_frame(frame_idx)?;
                let transformed = if let Some(dc) = drizzle_config {
                    let combined = combine_offset(&pre_center_data, i, offset);
                    drizzle_single_frame(&frame, &combined, dc)?
                } else {
                    apply_combined_shift(&frame, i, offset, &pre_center_data)
                };
                stacker.add(&transformed);
                reporter.advance(i + 1);
            }
            let result = stacker.finalize()?;
            info!(method = "Mean", "Streaming stacking complete");
            reporter.finish_stage();
            Ok(result)
        }
        _ => {
            // Median/SigmaClip: disk-backed streaming — write transformed frames to
            // temp files, then mmap them for per-pixel stacking.
            let store = create_disk_store(config)?;
            reporter.begin_stage(PipelineStage::Reading, Some(frame_count));
            let mut disk_frames = Vec::with_capacity(frame_count);
            for (i, (&frame_idx, offset)) in selected_indices.iter().zip(offsets.iter()).enumerate()
            {
                let frame = reader.read_frame(frame_idx)?;
                let transformed = if let Some(dc) = drizzle_config {
                    let combined = combine_offset(&pre_center_data, i, offset);
                    drizzle_single_frame(&frame, &combined, dc)?
                } else {
                    apply_combined_shift(&frame, i, offset, &pre_center_data)
                };
                disk_frames.push(store.store_frame(&transformed)?);
                reporter.advance(i + 1);
            }
            reporter.finish_stage();

            // Mmap all stored files and stack from disk
            let slices: Vec<MmapFrameSlice> = disk_frames
                .iter()
                .map(MmapFrameSlice::from_disk_frame)
                .collect::<Result<_>>()?;

            let bit_depth = reader.header.pixel_depth as u8;
            reporter.begin_stage(PipelineStage::Stacking, Some(frame_count));
            let result = match &config.stacking.method {
                StackMethod::Median => median_stack_disk(&slices, bit_depth)?,
                StackMethod::SigmaClip(SigmaClipParams { sigma, iterations }) => {
                    sigma_clip_stack_disk(&slices, *sigma, *iterations, bit_depth)?
                }
                _ => unreachable!(),
            };
            info!(method = ?config.stacking.method, "Disk-backed streaming stacking complete");
            reporter.finish_stage();
            Ok(result)
        }
    }
}

/// Post-stacking processing for mono path: sharpen -> filter -> write -> return.
pub(super) fn apply_post_stack_mono(
    stacked: Frame,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<PipelineOutput> {
    // Sharpening
    let mut result = if let Some(ref sharpening_config) = config.sharpening {
        reporter.begin_stage(PipelineStage::Sharpening, None);
        let mut sharpened = stacked;
        if let Some(ref deconv_config) = sharpening_config.deconvolution {
            if backend.is_gpu() {
                sharpened = deconvolve_gpu(&sharpened, deconv_config, &**backend);
            } else {
                sharpened = deconvolve(&sharpened, deconv_config);
            }
            info!("Deconvolution complete");
        }
        sharpened = wavelet::sharpen(&sharpened, &sharpening_config.wavelet);
        info!("Wavelet sharpening complete");
        reporter.finish_stage();
        sharpened
    } else {
        stacked
    };

    // Filters
    if !config.filters.is_empty() {
        let total_filters = config.filters.len();
        reporter.begin_stage(PipelineStage::Filtering, Some(total_filters));
        for (i, step) in config.filters.iter().enumerate() {
            result = apply_filter_step(&result, step);
            reporter.advance(i + 1);
        }
        info!(count = total_filters, "Filters applied");
        reporter.finish_stage();
    }

    // Write
    reporter.begin_stage(PipelineStage::Writing, None);
    save_image(&result, &config.output)?;
    info!(output = %config.output.display(), "Output saved");
    reporter.finish_stage();

    Ok(PipelineOutput::Mono(result))
}

/// Create a [`DiskFrameStore`], optionally under a user-specified temp directory.
fn create_disk_store(config: &PipelineConfig) -> Result<DiskFrameStore> {
    if let Some(ref dir) = config.temp_dir {
        DiskFrameStore::with_parent(dir)
    } else {
        DiskFrameStore::new()
    }
}

/// Combine pre-centering offset with alignment offset into a single offset.
///
/// When pre-centering is active, the centering offset is added to the alignment
/// offset. When it's not active, returns just the alignment offset (or identity
/// for the reference frame).
fn combine_offset(
    pre_center_data: &Option<(Vec<AlignmentOffset>, Option<CropRect>)>,
    frame_index: usize,
    alignment_offset: &AlignmentOffset,
) -> AlignmentOffset {
    match pre_center_data {
        Some((center_offsets, _)) => AlignmentOffset {
            dx: center_offsets[frame_index].dx + alignment_offset.dx,
            dy: center_offsets[frame_index].dy + alignment_offset.dy,
        },
        None => {
            if frame_index == 0 {
                AlignmentOffset::default()
            } else {
                alignment_offset.clone()
            }
        }
    }
}

/// Apply combined centering + alignment shift to a single frame.
///
/// If pre-centering data is available, the centering offset is added to the
/// alignment offset, producing a single combined shift. The frame is then
/// optionally cropped to the common overlap region.
fn apply_combined_shift(
    frame: &Frame,
    frame_index: usize,
    alignment_offset: &AlignmentOffset,
    pre_center_data: &Option<(Vec<AlignmentOffset>, Option<CropRect>)>,
) -> Frame {
    use crate::align::pre_center::crop_frame;

    match pre_center_data {
        Some((center_offsets, crop)) => {
            let combined = AlignmentOffset {
                dx: center_offsets[frame_index].dx + alignment_offset.dx,
                dy: center_offsets[frame_index].dy + alignment_offset.dy,
            };
            let shifted = shift_frame(frame, &combined);
            match crop {
                Some(rect) => crop_frame(&shifted, rect),
                None => shifted,
            }
        }
        None => {
            if frame_index == 0 {
                frame.clone()
            } else {
                shift_frame(frame, alignment_offset)
            }
        }
    }
}
