use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Args, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use jupiter_core::io::autocrop::{auto_detect_crop, AutoCropConfig, ThresholdMethod};
use jupiter_core::io::crop::crop_ser;
use jupiter_core::io::ser::SerReader;

#[derive(Clone, ValueEnum)]
pub enum ThresholdArg {
    /// Mean + sigma * stddev (default)
    Auto,
    /// Otsu's method (bimodal histogram)
    Otsu,
}

#[derive(Args)]
pub struct AutoCropArgs {
    /// Input SER file
    pub file: PathBuf,

    /// Output SER file (auto-generated if not provided)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Padding around the detected planet as fraction of diameter (0.0-1.0)
    #[arg(long, default_value = "0.15")]
    pub padding: f32,

    /// Number of frames to sample for detection
    #[arg(long, default_value = "30")]
    pub samples: usize,

    /// Threshold method
    #[arg(long, value_enum, default_value = "auto")]
    pub threshold: ThresholdArg,

    /// Sigma multiplier for auto threshold (mean + sigma * stddev)
    #[arg(long, default_value = "2.0")]
    pub sigma: f32,

    /// Fixed threshold value in [0.0, 1.0] (overrides --threshold)
    #[arg(long)]
    pub fixed_threshold: Option<f32>,

    /// Gaussian blur sigma for noise suppression before detection
    #[arg(long, default_value = "2.5")]
    pub blur_sigma: f32,

    /// Minimum connected component area (pixels) for planet detection
    #[arg(long, default_value = "100")]
    pub min_area: usize,
}

pub fn run(args: &AutoCropArgs) -> Result<()> {
    let reader = SerReader::open(&args.file)?;
    let info = reader.source_info(&args.file);

    println!(
        "Auto-crop: {} ({}x{}, {} frames)",
        info.filename.display(),
        info.width,
        info.height,
        info.total_frames
    );

    let threshold_method = if let Some(val) = args.fixed_threshold {
        ThresholdMethod::Fixed(val)
    } else {
        match args.threshold {
            ThresholdArg::Auto => ThresholdMethod::MeanPlusSigma,
            ThresholdArg::Otsu => ThresholdMethod::Otsu,
        }
    };

    let config = AutoCropConfig {
        sample_count: args.samples,
        padding_fraction: args.padding,
        threshold_method,
        sigma_multiplier: args.sigma,
        blur_sigma: args.blur_sigma,
        min_area: args.min_area,
        align_to_fft: true,
    };

    println!("Detecting planet...");
    let crop = auto_detect_crop(&reader, &config)?;
    println!(
        "Detected: {}x{} at ({}, {})",
        crop.width, crop.height, crop.x, crop.y
    );

    let output_path = args
        .output
        .clone()
        .unwrap_or_else(|| auto_crop_output_path(&args.file, crop.width, crop.height));

    let pb = ProgressBar::new(reader.frame_count() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("Cropping [{bar:40}] {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );

    crop_ser(&reader, &output_path, &crop, |done, _total| {
        pb.set_position(done as u64);
    })?;
    pb.finish();

    println!("Saved to {}", output_path.display());
    Ok(())
}

fn auto_crop_output_path(source: &Path, w: u32, h: u32) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = source
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("ser");
    let parent = source.parent().unwrap_or(Path::new("."));
    parent.join(format!("{stem}_crop{w}x{h}.{ext}"))
}
