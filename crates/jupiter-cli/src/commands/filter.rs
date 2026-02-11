use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use jupiter_core::filters::gaussian_blur::gaussian_blur;
use jupiter_core::filters::histogram::{auto_stretch, histogram_stretch};
use jupiter_core::filters::levels::{brightness_contrast, gamma_correct};
use jupiter_core::filters::unsharp_mask::unsharp_mask;
use jupiter_core::io::image_io::{load_image, save_image};

#[derive(Args)]
pub struct FilterArgs {
    /// Input image file (TIFF or PNG)
    pub file: PathBuf,

    /// Histogram stretch: "auto" or "black,white" (e.g. "0.01,0.99")
    #[arg(long)]
    pub stretch: Option<String>,

    /// Gamma correction value (e.g. 1.2)
    #[arg(long)]
    pub gamma: Option<f32>,

    /// Brightness adjustment (-1.0 to 1.0)
    #[arg(long)]
    pub brightness: Option<f32>,

    /// Contrast adjustment (1.0 = no change, >1.0 = more contrast)
    #[arg(long)]
    pub contrast: Option<f32>,

    /// Unsharp mask: "radius,amount,threshold" (e.g. "2.0,0.5,0.01")
    #[arg(long)]
    pub unsharp_mask: Option<String>,

    /// Gaussian blur sigma (e.g. 1.5)
    #[arg(long)]
    pub blur: Option<f32>,

    /// Output file path
    #[arg(short, long, default_value = "filtered.tiff")]
    pub output: PathBuf,
}

pub fn run(args: &FilterArgs) -> Result<()> {
    let mut frame = load_image(&args.file)
        .with_context(|| format!("Failed to load {}", args.file.display()))?;

    println!("Loaded {}x{} image", frame.width(), frame.height());

    if let Some(ref stretch_str) = args.stretch {
        if stretch_str == "auto" {
            println!("Applying auto histogram stretch");
            frame = auto_stretch(&frame, 0.001, 0.999);
        } else {
            let parts: Vec<f32> = stretch_str
                .split(',')
                .map(|s| s.trim().parse::<f32>())
                .collect::<std::result::Result<_, _>>()
                .context("Invalid stretch format (expected 'auto' or 'black,white')")?;
            if parts.len() != 2 {
                anyhow::bail!("Stretch requires exactly 2 values: black_point,white_point");
            }
            println!("Applying histogram stretch [{}, {}]", parts[0], parts[1]);
            frame = histogram_stretch(&frame, parts[0], parts[1]);
        }
    }

    if let Some(gamma) = args.gamma {
        println!("Applying gamma correction: {}", gamma);
        frame = gamma_correct(&frame, gamma);
    }

    if args.brightness.is_some() || args.contrast.is_some() {
        let b = args.brightness.unwrap_or(0.0);
        let c = args.contrast.unwrap_or(1.0);
        println!("Applying brightness={}, contrast={}", b, c);
        frame = brightness_contrast(&frame, b, c);
    }

    if let Some(ref usm_str) = args.unsharp_mask {
        let parts: Vec<f32> = usm_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<_, _>>()
            .context("Invalid unsharp mask format (expected 'radius,amount,threshold')")?;
        if parts.len() != 3 {
            anyhow::bail!("Unsharp mask requires exactly 3 values: radius,amount,threshold");
        }
        println!(
            "Applying unsharp mask: radius={}, amount={}, threshold={}",
            parts[0], parts[1], parts[2]
        );
        frame = unsharp_mask(&frame, parts[0], parts[1], parts[2]);
    }

    if let Some(sigma) = args.blur {
        println!("Applying Gaussian blur: sigma={}", sigma);
        frame = gaussian_blur(&frame, sigma);
    }

    save_image(&frame, &args.output)?;
    println!("Saved to {}", args.output.display());

    Ok(())
}
