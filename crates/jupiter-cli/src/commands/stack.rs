use anyhow::Result;
use clap::{Args, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use jupiter_core::align::phase_correlation::{align_frames_with_progress, compute_offset};
use jupiter_core::io::image_io::save_image;
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::config::QualityMetric;
use jupiter_core::quality::laplacian::rank_frames;
use jupiter_core::stack::drizzle::{drizzle_stack, DrizzleConfig};
use jupiter_core::stack::mean::mean_stack;
use jupiter_core::stack::median::median_stack;
use jupiter_core::stack::multi_point::{multi_point_stack, MultiPointConfig};
use jupiter_core::stack::sigma_clip::{sigma_clip_stack, SigmaClipParams};
use jupiter_core::stack::surface_warp::{surface_warp_stack, SurfaceWarpConfig};
use std::path::PathBuf;

#[derive(Clone, ValueEnum)]
pub enum StackMethodArg {
    Mean,
    Median,
    SigmaClip,
    MultiPoint,
    Drizzle,
    SurfaceWarp,
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

    /// Output scale factor for drizzle (e.g., 2.0 = 2x resolution)
    #[arg(long, default_value = "2.0")]
    pub drizzle_scale: f32,

    /// Pixfrac (drop size) for drizzle: 0.0-1.0
    #[arg(long, default_value = "0.7")]
    pub pixfrac: f32,

    /// Output file path
    #[arg(short, long, default_value = "stacked.tiff")]
    pub output: PathBuf,
}

pub fn run(args: &StackArgs) -> Result<()> {
    let reader = SerReader::open(&args.file)?;
    let percentage = (args.select as f32 / 100.0).clamp(0.01, 1.0);

    match args.method {
        StackMethodArg::MultiPoint => run_multi_point(&reader, args, percentage),
        StackMethodArg::Drizzle => run_drizzle(&reader, args, percentage),
        StackMethodArg::SurfaceWarp => run_surface_warp(&reader, args, percentage),
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

    let aligned = align_frames_with_progress(&selected, 0, |done| {
        pb.set_position(done as u64);
    })?;
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

fn run_drizzle(reader: &SerReader, args: &StackArgs, percentage: f32) -> Result<()> {
    let total = reader.frame_count();
    println!(
        "Drizzle stacking {} frames (scale={}, pixfrac={})",
        total, args.drizzle_scale, args.pixfrac
    );

    println!("Reading {} frames...", total);
    let frames: Vec<_> = reader.frames().collect::<std::result::Result<_, _>>()?;

    println!("Scoring frames...");
    let ranked = rank_frames(&frames);

    let keep = (total as f32 * percentage).ceil() as usize;
    let keep = keep.max(1).min(total);
    println!("Selected {} best frames (top {}%)", keep, args.select);

    let selected_indices: Vec<usize> = ranked.iter().take(keep).map(|(i, _)| *i).collect();
    let selected: Vec<_> = selected_indices
        .iter()
        .map(|&i| frames[i].clone())
        .collect();
    let quality_scores: Vec<f64> = selected_indices
        .iter()
        .map(|&i| {
            ranked
                .iter()
                .find(|(idx, _)| *idx == i)
                .unwrap()
                .1
                .composite
        })
        .collect();

    println!("Computing alignment offsets...");
    let pb = ProgressBar::new(keep as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("Aligning [{bar:40}] {pos}/{len}")?
            .progress_chars("=> "),
    );

    let reference = &selected[0];
    let offsets: Vec<_> = selected
        .iter()
        .enumerate()
        .map(|(i, frame)| {
            pb.set_position(i as u64 + 1);
            if i == 0 {
                jupiter_core::frame::AlignmentOffset::default()
            } else {
                compute_offset(reference, frame).unwrap_or_default()
            }
        })
        .collect();
    pb.finish();

    println!("Drizzle stacking...");
    let drizzle_config = DrizzleConfig {
        scale: args.drizzle_scale,
        pixfrac: args.pixfrac,
        quality_weighted: true,
        ..Default::default()
    };

    let result = drizzle_stack(&selected, &offsets, &drizzle_config, Some(&quality_scores))?;

    save_image(&result, &args.output)?;
    println!(
        "Saved {}x{} drizzle result to {}",
        result.width(),
        result.height(),
        args.output.display()
    );
    Ok(())
}

fn run_surface_warp(reader: &SerReader, args: &StackArgs, percentage: f32) -> Result<()> {
    let total = reader.frame_count();
    println!(
        "Surface warp stacking {} frames (ap_size={}, search_radius={})",
        total, args.ap_size, args.search_radius
    );

    let sw_config = SurfaceWarpConfig {
        ap_size: args.ap_size,
        search_radius: args.search_radius,
        select_percentage: percentage,
        min_brightness: args.min_brightness,
        ..Default::default()
    };

    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("Surface Warp [{bar:40}] {pos}%")?
            .progress_chars("=> "),
    );

    let result = surface_warp_stack(reader, &sw_config, |progress| {
        pb.set_position((progress * 100.0) as u64);
    })?;
    pb.finish();

    save_image(&result, &args.output)?;
    println!("Saved to {}", args.output.display());
    Ok(())
}
