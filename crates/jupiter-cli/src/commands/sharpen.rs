use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use jupiter_core::io::image_io::{load_image, save_image};
use jupiter_core::pipeline::config::{DeconvolutionConfig, DeconvolutionMethod, PsfModel};
use jupiter_core::sharpen::deconvolution::deconvolve;
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

    /// Deconvolution method (rl or wiener)
    #[arg(long)]
    pub deconv: Option<String>,

    /// PSF model (gaussian, kolmogorov, airy)
    #[arg(long, default_value = "gaussian")]
    pub psf: String,

    /// Gaussian PSF sigma in pixels
    #[arg(long, default_value = "2.0")]
    pub psf_sigma: f32,

    /// Kolmogorov seeing FWHM in pixels
    #[arg(long, default_value = "3.0")]
    pub seeing: f32,

    /// Airy first dark ring radius in pixels
    #[arg(long, default_value = "2.5")]
    pub airy_radius: f32,

    /// Richardson-Lucy iteration count
    #[arg(long, default_value = "20")]
    pub rl_iterations: usize,

    /// Wiener noise-to-signal ratio
    #[arg(long, default_value = "0.001")]
    pub noise_ratio: f32,

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

    let deconv_config = build_deconv_config(args);

    crate::summary::print_sharpen_summary(&params, deconv_config.as_ref());

    // Deconvolution (before wavelet)
    let frame = if let Some(ref deconv_config) = deconv_config {
        deconvolve(&frame, deconv_config)
    } else {
        frame
    };

    let sharpened = wavelet::sharpen(&frame, &params);

    save_image(&sharpened, &args.output)?;
    println!("Saved to {}", args.output.display());

    Ok(())
}

fn build_deconv_config(args: &SharpenArgs) -> Option<DeconvolutionConfig> {
    let method_str = args.deconv.as_deref()?;

    let method = match method_str {
        "rl" => DeconvolutionMethod::RichardsonLucy {
            iterations: args.rl_iterations,
        },
        "wiener" => DeconvolutionMethod::Wiener {
            noise_ratio: args.noise_ratio,
        },
        _ => {
            eprintln!("Unknown deconv method '{}', skipping", method_str);
            return None;
        }
    };

    let psf = match args.psf.as_str() {
        "gaussian" => PsfModel::Gaussian {
            sigma: args.psf_sigma,
        },
        "kolmogorov" => PsfModel::Kolmogorov {
            seeing: args.seeing,
        },
        "airy" => PsfModel::Airy {
            radius: args.airy_radius,
        },
        _ => {
            eprintln!("Unknown PSF model '{}', using Gaussian", args.psf);
            PsfModel::Gaussian {
                sigma: args.psf_sigma,
            }
        }
    };

    Some(DeconvolutionConfig { method, psf })
}
