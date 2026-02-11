use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use jupiter_core::pipeline::config::{
    FrameSelectionConfig, PipelineConfig, SharpeningConfig, StackingConfig,
};
use jupiter_core::pipeline::run_pipeline;
use jupiter_core::sharpen::wavelet::WaveletParams;

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

    /// Comma-separated wavelet sharpening coefficients
    #[arg(long)]
    pub sharpen: Option<String>,

    /// Disable sharpening
    #[arg(long)]
    pub no_sharpen: bool,

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
    println!("  Select:   {:.0}%", config.frame_selection.select_percentage * 100.0);
    if let Some(ref sharp) = config.sharpening {
        println!("  Sharpen:  {:?}", sharp.wavelet.coefficients);
    } else {
        println!("  Sharpen:  disabled");
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
            WaveletParams {
                num_layers: coefficients.len(),
                coefficients,
            }
        } else {
            WaveletParams::default()
        };
        Some(SharpeningConfig { wavelet })
    };

    PipelineConfig {
        input: args.file.clone(),
        output: args.output.clone(),
        frame_selection: FrameSelectionConfig {
            select_percentage: args.select as f32 / 100.0,
        },
        stacking: StackingConfig::default(),
        sharpening,
    }
}
