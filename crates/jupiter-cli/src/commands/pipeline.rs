use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use jupiter_core::pipeline::config::{
    FilterStep, FrameSelectionConfig, PipelineConfig, SharpeningConfig, StackMethod, StackingConfig,
};
use jupiter_core::pipeline::run_pipeline;
use jupiter_core::sharpen::wavelet::WaveletParams;
use jupiter_core::stack::multi_point::MultiPointConfig;
use jupiter_core::stack::sigma_clip::SigmaClipParams;

use super::stack::StackMethodArg;

#[derive(Args)]
pub struct RunArgs {
    /// Input SER file
    pub file: PathBuf,

    /// Pipeline config file (TOML)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Percentage of best frames to keep (1-100)
    #[arg(long, default_value = "25")]
    pub select: u32,

    /// Stacking method
    #[arg(long, value_enum, default_value = "multi-point")]
    pub method: StackMethodArg,

    /// Sigma threshold for sigma-clip stacking
    #[arg(long, default_value = "2.5")]
    pub sigma: f32,

    /// Comma-separated wavelet sharpening coefficients
    #[arg(long)]
    pub sharpen: Option<String>,

    /// Comma-separated wavelet denoise thresholds
    #[arg(long)]
    pub denoise: Option<String>,

    /// Alignment point size in pixels (multi-point mode)
    #[arg(long, default_value = "64")]
    pub ap_size: usize,

    /// Search radius around each AP for local alignment (multi-point mode)
    #[arg(long, default_value = "16")]
    pub search_radius: usize,

    /// Minimum mean brightness to place an alignment point (multi-point mode)
    #[arg(long, default_value = "0.05")]
    pub min_brightness: f32,

    /// Disable sharpening
    #[arg(long)]
    pub no_sharpen: bool,

    /// Auto histogram stretch after processing
    #[arg(long)]
    pub auto_stretch: bool,

    /// Gamma correction after processing
    #[arg(long)]
    pub gamma: Option<f32>,

    /// Output file path
    #[arg(short, long, default_value = "result.tiff")]
    pub output: PathBuf,
}

pub fn run(args: &RunArgs) -> Result<()> {
    let config = if let Some(ref config_path) = args.config {
        let contents = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config {}", config_path.display()))?;
        toml::from_str(&contents).context("Invalid pipeline config")?
    } else {
        build_config_from_args(args)
    };

    println!("Jupiter Pipeline");
    println!("  Input:    {}", config.input.display());
    println!("  Output:   {}", config.output.display());
    println!(
        "  Select:   {:.0}%",
        config.frame_selection.select_percentage * 100.0
    );
    println!("  Stack:    {:?}", config.stacking.method);
    if let Some(ref sharp) = config.sharpening {
        println!("  Sharpen:  {:?}", sharp.wavelet.coefficients);
        if !sharp.wavelet.denoise.is_empty() {
            println!("  Denoise:  {:?}", sharp.wavelet.denoise);
        }
    } else {
        println!("  Sharpen:  disabled");
    }
    if !config.filters.is_empty() {
        println!("  Filters:  {} step(s)", config.filters.len());
    }
    println!();

    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg:20} [{bar:40}] {pos}%")?
            .progress_chars("=> "),
    );

    run_pipeline(&config, |stage, progress| {
        pb.set_message(stage.to_string());
        pb.set_position((progress * 100.0) as u64);
    })?;

    pb.finish_with_message("Done");
    println!("\nOutput saved to {}", config.output.display());

    Ok(())
}

fn build_config_from_args(args: &RunArgs) -> PipelineConfig {
    let sharpening = if args.no_sharpen {
        None
    } else {
        let wavelet = if let Some(ref coeff_str) = args.sharpen {
            let coefficients: Vec<f32> = coeff_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let denoise = args
                .denoise
                .as_ref()
                .map(|d| d.split(',').filter_map(|s| s.trim().parse().ok()).collect())
                .unwrap_or_default();
            WaveletParams {
                num_layers: coefficients.len(),
                coefficients,
                denoise,
            }
        } else {
            let mut params = WaveletParams::default();
            if let Some(ref d) = args.denoise {
                params.denoise = d.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            }
            params
        };
        Some(SharpeningConfig { wavelet })
    };

    let stacking_method = match args.method {
        StackMethodArg::Mean => StackMethod::Mean,
        StackMethodArg::Median => StackMethod::Median,
        StackMethodArg::SigmaClip => StackMethod::SigmaClip(SigmaClipParams {
            sigma: args.sigma,
            ..Default::default()
        }),
        StackMethodArg::MultiPoint => StackMethod::MultiPoint(MultiPointConfig {
            ap_size: args.ap_size,
            search_radius: args.search_radius,
            select_percentage: args.select as f32 / 100.0,
            min_brightness: args.min_brightness,
            ..Default::default()
        }),
    };

    let mut filters = Vec::new();
    if args.auto_stretch {
        filters.push(FilterStep::AutoStretch {
            low_percentile: 0.001,
            high_percentile: 0.999,
        });
    }
    if let Some(gamma) = args.gamma {
        filters.push(FilterStep::Gamma(gamma));
    }

    PipelineConfig {
        input: args.file.clone(),
        output: args.output.clone(),
        frame_selection: FrameSelectionConfig {
            select_percentage: args.select as f32 / 100.0,
            ..Default::default()
        },
        stacking: StackingConfig {
            method: stacking_method,
        },
        sharpening,
        filters,
    }
}
