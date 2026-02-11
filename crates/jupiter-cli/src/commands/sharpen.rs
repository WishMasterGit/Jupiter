use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use jupiter_core::io::image_io::{load_image, save_image};
use jupiter_core::sharpen::wavelet::{self, WaveletParams};

#[derive(Args)]
pub struct SharpenArgs {
    /// Input image file (TIFF or PNG)
    pub file: PathBuf,

    /// Number of wavelet layers
    #[arg(long, default_value = "6")]
    pub layers: usize,

    /// Comma-separated coefficients per layer (e.g. 1.5,1.3,1.2,1.1,1.0,1.0)
    #[arg(long)]
    pub coefficients: Option<String>,

    /// Comma-separated denoise thresholds per layer (e.g. 3.0,2.0,1.0,0,0,0)
    #[arg(long)]
    pub denoise: Option<String>,

    /// Output file path
    #[arg(short, long, default_value = "sharpened.tiff")]
    pub output: PathBuf,
}

pub fn run(args: &SharpenArgs) -> Result<()> {
    let frame = load_image(&args.file)
        .with_context(|| format!("Failed to load {}", args.file.display()))?;

    println!(
        "Loaded {}x{} image",
        frame.width(),
        frame.height()
    );

    let coefficients = if let Some(ref coeff_str) = args.coefficients {
        coeff_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Invalid coefficient format")?
    } else {
        WaveletParams::default().coefficients[..args.layers.min(6)].to_vec()
    };

    let denoise = if let Some(ref denoise_str) = args.denoise {
        denoise_str
            .split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Invalid denoise format")?
    } else {
        vec![]
    };

    let params = WaveletParams {
        num_layers: args.layers,
        coefficients,
        denoise: denoise.clone(),
    };

    println!(
        "Applying {}-layer wavelet sharpening with coefficients {:?}",
        params.num_layers, params.coefficients
    );
    if !denoise.is_empty() {
        println!("  Denoise thresholds: {:?}", denoise);
    }

    let sharpened = wavelet::sharpen(&frame, &params);

    save_image(&sharpened, &args.output)?;
    println!("Saved to {}", args.output.display());

    Ok(())
}
