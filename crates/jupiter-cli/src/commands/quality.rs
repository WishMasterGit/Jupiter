use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use jupiter_core::io::ser::SerReader;
use jupiter_core::quality::gradient::rank_frames_gradient;
use jupiter_core::quality::laplacian::rank_frames;

#[derive(Clone, ValueEnum)]
pub enum MetricArg {
    Laplacian,
    Gradient,
}

#[derive(Args)]
pub struct QualityArgs {
    /// Input SER file
    pub file: PathBuf,

    /// Show top N frames only
    #[arg(long, default_value = "20")]
    pub top: usize,

    /// Quality metric to use
    #[arg(long, value_enum, default_value = "laplacian")]
    pub metric: MetricArg,
}

pub fn run(args: &QualityArgs) -> Result<()> {
    let reader = SerReader::open(&args.file)?;
    let total = reader.frame_count();

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {pos}/{len}")?
            .progress_chars("=> "),
    );
    pb.set_message("Reading frames");

    let frames: Vec<_> = reader
        .frames()
        .enumerate()
        .map(|(i, f)| {
            pb.set_position(i as u64 + 1);
            f
        })
        .collect::<std::result::Result<_, _>>()?;

    let metric_name = match args.metric {
        MetricArg::Laplacian => "laplacian",
        MetricArg::Gradient => "gradient",
    };
    pb.finish_with_message(format!("Scoring frames ({})", metric_name));

    let ranked = match args.metric {
        MetricArg::Laplacian => rank_frames(&frames),
        MetricArg::Gradient => rank_frames_gradient(&frames),
    };

    println!(
        "\nTop {} frames by quality [{}] (of {}):",
        args.top.min(total),
        metric_name,
        total
    );
    println!("{:>5}  {:>12}  {:>8}", "Rank", "Frame #", "Score");
    println!("{}", "-".repeat(30));

    for (rank, (idx, score)) in ranked.iter().take(args.top).enumerate() {
        println!("{:>5}  {:>12}  {:>8.6}", rank + 1, idx, score.composite);
    }

    if total > 0 {
        let best = ranked.first().unwrap().1.composite;
        let worst = ranked.last().unwrap().1.composite;
        println!("\nBest score:  {:.6}", best);
        println!("Worst score: {:.6}", worst);
    }

    Ok(())
}
