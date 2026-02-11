use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use jupiter_core::align::phase_correlation::{compute_offset, shift_frame};
use jupiter_core::io::image_io::save_image;
use jupiter_core::io::ser::SerReader;
use jupiter_core::quality::laplacian::rank_frames;
use jupiter_core::stack::mean::mean_stack;

#[derive(Args)]
pub struct StackArgs {
    /// Input SER file
    pub file: PathBuf,

    /// Percentage of best frames to keep (1-100)
    #[arg(long, default_value = "25")]
    pub select: u32,

    /// Output file path
    #[arg(short, long, default_value = "stacked.tiff")]
    pub output: PathBuf,
}

pub fn run(args: &StackArgs) -> Result<()> {
    let reader = SerReader::open(&args.file)?;
    let total = reader.frame_count();
    let percentage = (args.select as f32 / 100.0).clamp(0.01, 1.0);

    println!("Reading {} frames...", total);
    let frames: Vec<_> = reader.frames().collect::<std::result::Result<_, _>>()?;

    println!("Scoring frames...");
    let ranked = rank_frames(&frames);

    let keep = (total as f32 * percentage).ceil() as usize;
    let keep = keep.max(1).min(total);
    println!("Selected {} best frames (top {}%)", keep, args.select);

    let selected: Vec<_> = ranked.iter().take(keep).map(|(i, _)| frames[*i].clone()).collect();

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

    println!("Stacking...");
    let result = mean_stack(&aligned)?;

    save_image(&result, &args.output)?;
    println!("Saved to {}", args.output.display());

    Ok(())
}
