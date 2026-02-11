use anyhow::Result;
use clap::{Args, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use jupiter_core::align::phase_correlation::{compute_offset, shift_frame};
use jupiter_core::io::image_io::save_image;
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::QualityMetric;
use jupiter_core::quality::laplacian::rank_frames;
use jupiter_core::stack::mean::mean_stack;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::multi_point::{multi_point_stack, MultiPointConfig};
use jupiter_core::stack::sigma_clip::{sigma_clip_stack, SigmaClipParams};
use std::path::PathBuf;

#[derive(Clone, ValueEnum)]
pub enum StackMethodArg {
    Mean,
    Median,
    SigmaClip,
    MultiPoint,
}

#[derive(Args)]
pub struct StackArgs {
    /// Input SER file
    pub file: PathBuf,

    /// Percentage of best frames to keep (1-100)
    #[arg(long, default_value = "25")]
    pub select: u32,

    /// Stacking method
    #[arg(long, value_enum, default_value = "mean")]
    pub method: StackMethodArg,

    /// Sigma threshold for sigma-clip stacking
    #[arg(long, default_value = "2.5")]
    pub sigma: f32,

    /// Alignment point size in pixels (multi-point mode)
    #[arg(long, default_value = "64")]
    pub ap_size: usize,

    /// Search radius around each AP for local alignment (multi-point mode)
    #[arg(long, default_value = "16")]
    pub search_radius: usize,

    /// Minimum mean brightness to place an alignment point (multi-point mode)
    #[arg(long, default_value = "0.05")]
    pub min_brightness: f32,

    /// Output file path
    #[arg(short, long, default_value = "stacked.tiff")]
    pub output: PathBuf,
}

pub fn run(args: &StackArgs) -> Result<()> {
    let reader = SerReader::open(&args.file)?;
    let percentage = (args.select as f32 / 100.0).clamp(0.01, 1.0);

    match args.method {
        StackMethodArg::MultiPoint => run_multi_point(&reader, args, percentage),
        _ => run_standard(&reader, args, percentage),
    }
}

fn run_multi_point(reader: &SerReader, args: &StackArgs, percentage: f32) -> Result<()> {
    let total = reader.frame_count();
    println!(
        "Multi-point stacking {} frames (ap_size={}, search_radius={})",
        total, args.ap_size, args.search_radius
    );

    let mp_config = MultiPointConfig {
        ap_size: args.ap_size,
        search_radius: args.search_radius,
        select_percentage: percentage,
        min_brightness: args.min_brightness,
        quality_metric: QualityMetric::Laplacian,
        ..Default::default()
    };

    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("Multi-point [{bar:40}] {pos}%")?
            .progress_chars("=> "),
    );

    let result = multi_point_stack(reader, &mp_config, |progress| {
        pb.set_position((progress * 100.0) as u64);
    })?;
    pb.finish();

    save_image(&result, &args.output)?;
    println!("Saved to {}", args.output.display());
    Ok(())
}

fn run_standard(reader: &SerReader, args: &StackArgs, percentage: f32) -> Result<()> {
    let total = reader.frame_count();

    println!("Reading {} frames...", total);
    let frames: Vec<_> = reader.frames().collect::<std::result::Result<_, _>>()?;

    println!("Scoring frames...");
    let ranked = rank_frames(&frames);

    let keep = (total as f32 * percentage).ceil() as usize;
    let keep = keep.max(1).min(total);
    println!("Selected {} best frames (top {}%)", keep, args.select);

    let selected: Vec<_> = ranked
        .iter()
        .take(keep)
        .map(|(i, _)| frames[*i].clone())
        .collect();

    let pb = ProgressBar::new(keep as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("Aligning [{bar:40}] {pos}/{len}")?
            .progress_chars("=> "),
    );

    let reference = &selected[0];
    let mut aligned = vec![reference.clone()];
    for (i, frame) in selected.iter().enumerate().skip(1) {
        let offset = compute_offset(reference, frame)?;
        aligned.push(shift_frame(frame, &offset));
        pb.set_position(i as u64 + 1);
    }
    pb.finish();

    let method_name = match args.method {
        StackMethodArg::Mean => "mean",
        StackMethodArg::Median => "median",
        StackMethodArg::SigmaClip => "sigma-clip",
        _ => unreachable!(),
    };
    println!("Stacking ({})...", method_name);

    let result = match args.method {
        StackMethodArg::Mean => mean_stack(&aligned)?,
        StackMethodArg::Median => median_stack(&aligned)?,
        StackMethodArg::SigmaClip => {
            let params = SigmaClipParams {
                sigma: args.sigma,
                ..Default::default()
            };
            sigma_clip_stack(&aligned, &params)?
        }
        _ => unreachable!(),
    };

    save_image(&result, &args.output)?;
    println!("Saved to {}", args.output.display());
    Ok(())
}
