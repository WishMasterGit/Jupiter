pub mod config;

use tracing::info;

use crate::align::phase_correlation::{compute_offset, shift_frame};
use crate::error::Result;
use crate::frame::Frame;
use crate::io::image_io::save_image;
use crate::io::ser::SerReader;
use crate::quality::laplacian::rank_frames;
use crate::sharpen::wavelet;
use crate::stack::mean::mean_stack;

use self::config::PipelineConfig;

/// Pipeline processing stage, used for progress reporting.
#[derive(Clone, Debug)]
pub enum PipelineStage {
    Reading,
    QualityAssessment,
    FrameSelection,
    Alignment,
    Stacking,
    Sharpening,
    Writing,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reading => write!(f, "Reading frames"),
            Self::QualityAssessment => write!(f, "Assessing quality"),
            Self::FrameSelection => write!(f, "Selecting best frames"),
            Self::Alignment => write!(f, "Aligning frames"),
            Self::Stacking => write!(f, "Stacking"),
            Self::Sharpening => write!(f, "Sharpening"),
            Self::Writing => write!(f, "Writing output"),
        }
    }
}

/// Run the full processing pipeline.
///
/// `on_progress` is called with (stage, fraction_complete) for UI updates.
pub fn run_pipeline<F>(config: &PipelineConfig, mut on_progress: F) -> Result<Frame>
where
    F: FnMut(PipelineStage, f32),
{
    // 1. Read frames
    on_progress(PipelineStage::Reading, 0.0);
    let reader = SerReader::open(&config.input)?;
    let total = reader.frame_count();
    info!(total_frames = total, "Reading SER file");

    let frames: Vec<Frame> = reader.frames().collect::<Result<_>>()?;
    on_progress(PipelineStage::Reading, 1.0);

    // 2. Quality assessment
    on_progress(PipelineStage::QualityAssessment, 0.0);
    let ranked = rank_frames(&frames);
    on_progress(PipelineStage::QualityAssessment, 1.0);

    // 3. Frame selection
    on_progress(PipelineStage::FrameSelection, 0.0);
    let keep = (total as f32 * config.frame_selection.select_percentage).ceil() as usize;
    let keep = keep.max(1).min(total);
    let selected_indices: Vec<usize> = ranked.iter().take(keep).map(|(i, _)| *i).collect();
    info!(
        selected = selected_indices.len(),
        total, "Selected best frames"
    );

    let selected_frames: Vec<Frame> = selected_indices
        .iter()
        .map(|&i| frames[i].clone())
        .collect();
    on_progress(PipelineStage::FrameSelection, 1.0);

    // 4. Alignment
    on_progress(PipelineStage::Alignment, 0.0);
    let aligned = if selected_frames.len() > 1 {
        // Use the first selected frame (best quality) as reference
        let reference = &selected_frames[0];
        let mut aligned = Vec::with_capacity(selected_frames.len());
        aligned.push(reference.clone());

        for (i, frame) in selected_frames.iter().enumerate().skip(1) {
            let offset = compute_offset(reference, frame)?;
            aligned.push(shift_frame(frame, &offset));
            on_progress(
                PipelineStage::Alignment,
                (i + 1) as f32 / selected_frames.len() as f32,
            );
        }
        aligned
    } else {
        selected_frames
    };
    on_progress(PipelineStage::Alignment, 1.0);

    // 5. Stacking
    on_progress(PipelineStage::Stacking, 0.0);
    let stacked = mean_stack(&aligned)?;
    info!("Stacking complete");
    on_progress(PipelineStage::Stacking, 1.0);

    // 6. Sharpening
    let result = if let Some(ref sharpening_config) = config.sharpening {
        on_progress(PipelineStage::Sharpening, 0.0);
        let sharpened = wavelet::sharpen(&stacked, &sharpening_config.wavelet);
        info!("Sharpening complete");
        on_progress(PipelineStage::Sharpening, 1.0);
        sharpened
    } else {
        stacked
    };

    // 7. Write output
    on_progress(PipelineStage::Writing, 0.0);
    save_image(&result, &config.output)?;
    info!(output = %config.output.display(), "Output saved");
    on_progress(PipelineStage::Writing, 1.0);

    Ok(result)
}
