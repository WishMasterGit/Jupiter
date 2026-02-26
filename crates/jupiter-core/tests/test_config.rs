use jupiter_core::pipeline::config::{
    AlignmentMethod, CentroidConfig, DeconvolutionMethod, EnhancedPhaseConfig, FilterStep,
    FrameSelectionConfig, MemoryStrategy, PsfModel, PyramidConfig, QualityMetric, StackMethod,
};
use jupiter_core::pipeline::PipelineStage;
use jupiter_core::stack::drizzle::DrizzleConfig;
use jupiter_core::stack::sigma_clip::SigmaClipParams;

// ---------------------------------------------------------------------------
// MemoryStrategy Display
// ---------------------------------------------------------------------------

#[test]
fn test_memory_strategy_display_auto() {
    assert_eq!(format!("{}", MemoryStrategy::Auto), "Auto");
}

#[test]
fn test_memory_strategy_display_eager() {
    assert_eq!(format!("{}", MemoryStrategy::Eager), "Eager");
}

#[test]
fn test_memory_strategy_display_low_memory() {
    assert_eq!(format!("{}", MemoryStrategy::LowMemory), "Low Memory");
}

#[test]
fn test_memory_strategy_default_is_auto() {
    let m = MemoryStrategy::default();
    assert_eq!(format!("{}", m), "Auto");
}

// ---------------------------------------------------------------------------
// QualityMetric Display
// ---------------------------------------------------------------------------

#[test]
fn test_quality_metric_display_laplacian() {
    assert_eq!(format!("{}", QualityMetric::Laplacian), "Laplacian");
}

#[test]
fn test_quality_metric_display_gradient() {
    assert_eq!(format!("{}", QualityMetric::Gradient), "Gradient");
}

#[test]
fn test_quality_metric_default() {
    let m = QualityMetric::default();
    assert_eq!(m, QualityMetric::Laplacian);
}

// ---------------------------------------------------------------------------
// AlignmentMethod Display
// ---------------------------------------------------------------------------

#[test]
fn test_alignment_method_display_phase_correlation() {
    let s = format!("{}", AlignmentMethod::PhaseCorrelation);
    assert_eq!(s, "Phase Correlation");
}

#[test]
fn test_alignment_method_display_enhanced_phase() {
    let m = AlignmentMethod::EnhancedPhaseCorrelation(EnhancedPhaseConfig {
        upsample_factor: 20,
    });
    let s = format!("{}", m);
    assert!(s.contains("Enhanced Phase"), "got: {s}");
    assert!(s.contains("20"), "got: {s}");
}

#[test]
fn test_alignment_method_display_centroid() {
    let m = AlignmentMethod::Centroid(CentroidConfig { threshold: 0.1 });
    let s = format!("{}", m);
    assert!(s.contains("Centroid"), "got: {s}");
    assert!(s.contains("0.1"), "got: {s}");
}

#[test]
fn test_alignment_method_display_gradient_correlation() {
    let s = format!("{}", AlignmentMethod::GradientCorrelation);
    assert_eq!(s, "Gradient Correlation");
}

#[test]
fn test_alignment_method_display_pyramid() {
    let m = AlignmentMethod::Pyramid(PyramidConfig { levels: 3 });
    let s = format!("{}", m);
    assert!(s.contains("Pyramid"), "got: {s}");
    assert!(s.contains("3"), "got: {s}");
}

// ---------------------------------------------------------------------------
// StackMethod Display
// ---------------------------------------------------------------------------

#[test]
fn test_stack_method_display_mean() {
    assert_eq!(format!("{}", StackMethod::Mean), "Mean");
}

#[test]
fn test_stack_method_display_median() {
    assert_eq!(format!("{}", StackMethod::Median), "Median");
}

#[test]
fn test_stack_method_display_sigma_clip() {
    let s = format!("{}", StackMethod::SigmaClip(SigmaClipParams::default()));
    assert_eq!(s, "Sigma Clip");
}

#[test]
fn test_stack_method_display_drizzle() {
    let cfg = DrizzleConfig {
        scale: 2.0,
        pixfrac: 0.7,
        quality_weighted: false,
        kernel: jupiter_core::stack::drizzle::DrizzleKernel::Square,
    };
    let s = format!("{}", StackMethod::Drizzle(cfg));
    assert!(s.contains("Drizzle"), "got: {s}");
    assert!(s.contains("2"), "got: {s}");
}

// ---------------------------------------------------------------------------
// DeconvolutionMethod Display
// ---------------------------------------------------------------------------

#[test]
fn test_deconvolution_method_display_rl() {
    let m = DeconvolutionMethod::RichardsonLucy { iterations: 10 };
    let s = format!("{}", m);
    assert!(s.contains("Richardson-Lucy"), "got: {s}");
    assert!(s.contains("10"), "got: {s}");
}

#[test]
fn test_deconvolution_method_display_wiener() {
    let m = DeconvolutionMethod::Wiener { noise_ratio: 0.01 };
    let s = format!("{}", m);
    assert!(s.contains("Wiener"), "got: {s}");
    assert!(s.contains("0.01"), "got: {s}");
}

// ---------------------------------------------------------------------------
// PsfModel Display
// ---------------------------------------------------------------------------

#[test]
fn test_psf_model_display_gaussian() {
    let m = PsfModel::Gaussian { sigma: 1.5 };
    let s = format!("{}", m);
    assert!(s.contains("Gaussian"), "got: {s}");
    assert!(s.contains("1.5"), "got: {s}");
}

#[test]
fn test_psf_model_display_kolmogorov() {
    let m = PsfModel::Kolmogorov { seeing: 2.0 };
    let s = format!("{}", m);
    assert!(s.contains("Kolmogorov"), "got: {s}");
    assert!(s.contains("2"), "got: {s}");
}

#[test]
fn test_psf_model_display_airy() {
    let m = PsfModel::Airy { radius: 3.0 };
    let s = format!("{}", m);
    assert!(s.contains("Airy"), "got: {s}");
    assert!(s.contains("3"), "got: {s}");
}

// ---------------------------------------------------------------------------
// FilterStep Display
// ---------------------------------------------------------------------------

#[test]
fn test_filter_step_display_histogram_stretch() {
    let step = FilterStep::HistogramStretch {
        black_point: 0.05,
        white_point: 0.95,
    };
    let s = format!("{}", step);
    assert!(s.contains("Histogram Stretch"), "got: {s}");
    assert!(s.contains("0.05"), "got: {s}");
    assert!(s.contains("0.95"), "got: {s}");
}

#[test]
fn test_filter_step_display_auto_stretch() {
    let step = FilterStep::AutoStretch {
        low_percentile: 0.001,
        high_percentile: 0.999,
    };
    let s = format!("{}", step);
    assert!(s.contains("Auto Stretch"), "got: {s}");
}

#[test]
fn test_filter_step_display_gamma() {
    let step = FilterStep::Gamma(2.2);
    let s = format!("{}", step);
    assert!(s.contains("Gamma"), "got: {s}");
    assert!(s.contains("2.2"), "got: {s}");
}

#[test]
fn test_filter_step_display_brightness_contrast() {
    let step = FilterStep::BrightnessContrast {
        brightness: 0.1,
        contrast: 1.5,
    };
    let s = format!("{}", step);
    assert!(s.contains("Brightness"), "got: {s}");
}

#[test]
fn test_filter_step_display_unsharp_mask() {
    let step = FilterStep::UnsharpMask {
        radius: 1.5,
        amount: 0.5,
        threshold: 0.02,
    };
    let s = format!("{}", step);
    assert!(s.contains("Unsharp"), "got: {s}");
}

#[test]
fn test_filter_step_display_gaussian_blur() {
    let step = FilterStep::GaussianBlur { sigma: 2.0 };
    let s = format!("{}", step);
    assert!(s.contains("Gaussian Blur"), "got: {s}");
    assert!(s.contains("2"), "got: {s}");
}

// ---------------------------------------------------------------------------
// FrameSelectionConfig default
// ---------------------------------------------------------------------------

#[test]
fn test_frame_selection_config_default() {
    let cfg = FrameSelectionConfig::default();
    assert!((cfg.select_percentage - 0.25).abs() < 1e-5);
    assert_eq!(cfg.metric, QualityMetric::Laplacian);
}

// ---------------------------------------------------------------------------
// PipelineStage Display
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_stage_display() {
    assert_eq!(format!("{}", PipelineStage::Reading), "Reading frames");
    assert_eq!(format!("{}", PipelineStage::Debayering), "Debayering");
    assert_eq!(
        format!("{}", PipelineStage::QualityAssessment),
        "Assessing quality"
    );
    assert_eq!(
        format!("{}", PipelineStage::FrameSelection),
        "Selecting best frames"
    );
    assert_eq!(format!("{}", PipelineStage::Alignment), "Aligning frames");
    assert_eq!(format!("{}", PipelineStage::Stacking), "Stacking");
    assert_eq!(format!("{}", PipelineStage::Sharpening), "Sharpening");
    assert_eq!(format!("{}", PipelineStage::Filtering), "Applying filters");
    assert_eq!(format!("{}", PipelineStage::Writing), "Writing output");
    assert_eq!(format!("{}", PipelineStage::Cropping), "Cropping");
}

// ---------------------------------------------------------------------------
// PipelineOutput
// ---------------------------------------------------------------------------

#[test]
fn test_pipeline_output_to_mono_from_mono() {
    use jupiter_core::frame::Frame;
    use jupiter_core::pipeline::PipelineOutput;
    use ndarray::Array2;

    let data = Array2::from_elem((8, 8), 0.4f32);
    let frame = Frame::new(data, 8);
    let output = PipelineOutput::Mono(frame.clone());
    let mono = output.to_mono();
    for (a, b) in frame.data.iter().zip(mono.data.iter()) {
        assert!((*a - *b).abs() < 1e-5);
    }
}

#[test]
fn test_pipeline_output_to_mono_from_color() {
    use jupiter_core::frame::{ColorFrame, Frame};
    use jupiter_core::pipeline::PipelineOutput;
    use ndarray::Array2;

    let h = 8;
    let w = 8;
    let make = |v: f32| Frame::new(Array2::from_elem((h, w), v), 8);
    let color = ColorFrame {
        red: make(0.6),
        green: make(0.6),
        blue: make(0.6),
    };
    let output = PipelineOutput::Color(color);
    let mono = output.to_mono();
    // luminance of (0.6, 0.6, 0.6) should be 0.6
    for v in mono.data.iter() {
        assert!(
            (*v - 0.6).abs() < 0.05,
            "luminance of gray should be ~0.6, got {v}"
        );
    }
}
