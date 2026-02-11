pub mod config;

use tracing::info;

use crate::align::phase_correlation::{compute_offset, shift_frame};
use crate::error::Result;
use crate::filters::gaussian_blur::gaussian_blur;
use crate::filters::histogram::{auto_stretch, histogram_stretch};
use crate::filters::levels::{brightness_contrast, gamma_correct};
use crate::filters::unsharp_mask::unsharp_mask;
use crate::frame::Frame;
use crate::io::image_io::save_image;
use crate::io::ser::SerReader;
use crate::quality::gradient::rank_frames_gradient;
use crate::quality::laplacian::rank_frames;
use crate::sharpen::deconvolution::deconvolve;
use crate::sharpen::wavelet;
use crate::stack::mean::mean_stack;
use crate::stack::median::median_stack;
use crate::stack::multi_point::multi_point_stack;
use crate::stack::sigma_clip::sigma_clip_stack;

use self::config::{FilterStep, PipelineConfig, QualityMetric, StackMethod};

/// Pipeline processing stage, used for progress reporting.
#[derive(Clone, Debug)]
pub enum PipelineStage {
    Reading,
    QualityAssessment,
    FrameSelection,
    Alignment,
    Stacking,
    Sharpening,
    Filtering,
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
            Self::Filtering => write!(f, "Applying filters"),
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
    // For multi-point stacking, use a completely different flow
    let reader = SerReader::open(&config.input)?;
    let total = reader.frame_count();
    info!(total_frames = total, "Reading SER file");

    let stacked = if let StackMethod::MultiPoint(ref mp_config) = config.stacking.method {
        // Multi-point: global align → AP grid → per-AP score/select/align/stack → blend
        on_progress(PipelineStage::Stacking, 0.0);
        let result = multi_point_stack(&reader, mp_config, |progress| {
            on_progress(PipelineStage::Stacking, progress);
        })?;
        info!("Multi-point stacking complete");
        on_progress(PipelineStage::Stacking, 1.0);
        result
    } else {
        // Standard flow: read all → quality → select → align → stack
        on_progress(PipelineStage::Reading, 0.0);
        let frames: Vec<Frame> = reader.frames().collect::<Result<_>>()?;
        on_progress(PipelineStage::Reading, 1.0);

        // Quality assessment
        on_progress(PipelineStage::QualityAssessment, 0.0);
        let ranked = match config.frame_selection.metric {
            QualityMetric::Laplacian => rank_frames(&frames),
            QualityMetric::Gradient => rank_frames_gradient(&frames),
        };
        on_progress(PipelineStage::QualityAssessment, 1.0);

        // Frame selection
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

        // Alignment
        on_progress(PipelineStage::Alignment, 0.0);
        let aligned = if selected_frames.len() > 1 {
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

        // Stacking
        on_progress(PipelineStage::Stacking, 0.0);
        let result = match &config.stacking.method {
            StackMethod::Mean => mean_stack(&aligned)?,
            StackMethod::Median => median_stack(&aligned)?,
            StackMethod::SigmaClip(params) => sigma_clip_stack(&aligned, params)?,
            StackMethod::MultiPoint(_) => unreachable!(),
        };
        info!(method = ?config.stacking.method, "Stacking complete");
        on_progress(PipelineStage::Stacking, 1.0);
        result
    };

    // 6. Sharpening (deconvolution first, then wavelet)
    let mut result = if let Some(ref sharpening_config) = config.sharpening {
        on_progress(PipelineStage::Sharpening, 0.0);
        let mut sharpened = stacked;
        if let Some(ref deconv_config) = sharpening_config.deconvolution {
            sharpened = deconvolve(&sharpened, deconv_config);
            info!("Deconvolution complete");
        }
        sharpened = wavelet::sharpen(&sharpened, &sharpening_config.wavelet);
        info!("Wavelet sharpening complete");
        on_progress(PipelineStage::Sharpening, 1.0);
        sharpened
    } else {
        stacked
    };

    // 7. Post-processing filters
    if !config.filters.is_empty() {
        on_progress(PipelineStage::Filtering, 0.0);
        let total_filters = config.filters.len();
        for (i, step) in config.filters.iter().enumerate() {
            result = apply_filter_step(&result, step);
            on_progress(
                PipelineStage::Filtering,
                (i + 1) as f32 / total_filters as f32,
            );
        }
        info!(count = total_filters, "Filters applied");
    }

    // 8. Write output
    on_progress(PipelineStage::Writing, 0.0);
    save_image(&result, &config.output)?;
    info!(output = %config.output.display(), "Output saved");
    on_progress(PipelineStage::Writing, 1.0);

    Ok(result)
}

/// Apply a single filter step to a frame.
pub fn apply_filter_step(frame: &Frame, step: &FilterStep) -> Frame {
    match step {
        FilterStep::HistogramStretch {
            black_point,
            white_point,
        } => histogram_stretch(frame, *black_point, *white_point),
        FilterStep::AutoStretch {
            low_percentile,
            high_percentile,
        } => auto_stretch(frame, *low_percentile, *high_percentile),
        FilterStep::Gamma(gamma) => gamma_correct(frame, *gamma),
        FilterStep::BrightnessContrast {
            brightness,
            contrast,
        } => brightness_contrast(frame, *brightness, *contrast),
        FilterStep::UnsharpMask {
            radius,
            amount,
            threshold,
        } => unsharp_mask(frame, *radius, *amount, *threshold),
        FilterStep::GaussianBlur { sigma } => gaussian_blur(frame, *sigma),
    }
}
