use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Args;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use jupiter_core::pipeline::config::{
    DeconvolutionConfig, DeconvolutionMethod, FilterStep, FrameSelectionConfig, PipelineConfig,
    PsfModel, SharpeningConfig, StackMethod, StackingConfig,
};
use jupiter_core::pipeline::{PipelineStage, ProgressReporter, run_pipeline_reported};
use jupiter_core::sharpen::wavelet::WaveletParams;
use jupiter_core::stack::multi_point::MultiPointConfig;
use jupiter_core::stack::sigma_clip::SigmaClipParams;

use super::stack::StackMethodArg;

#[derive(Args)]
pub struct RunArgs {
    /// Input SER file
    #[arg(required_unless_present = "config")]
    pub file: Option<PathBuf>,

    /// Pipeline config file (TOML)
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Percentage of best frames to keep (1-100)
    #[arg(long, default_value = "25")]
    pub select: u32,

    /// Stacking method
    #[arg(long, value_enum, default_value = "mean")]
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
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Save effective config as TOML and exit without processing
    #[arg(long)]
    pub save_config: Option<PathBuf>,
}

/// Progress reporter using indicatif MultiProgress with stage + detail bars.
struct MultiProgressReporter {
    stage_bar: ProgressBar,
    detail_bar: ProgressBar,
    stage_count: AtomicUsize,
    current_total: AtomicUsize,
}

impl MultiProgressReporter {
    fn new(multi: &MultiProgress) -> Result<Self> {
        let stage_bar = multi.add(ProgressBar::new(8));
        stage_bar.set_style(
            ProgressStyle::default_bar()
                .template("{msg:20} [{bar:40}] stage {pos}/{len}")?
                .progress_chars("=> "),
        );

        let detail_bar = multi.add(ProgressBar::new(0));
        detail_bar.set_style(
            ProgressStyle::default_bar()
                .template("  {msg:18} [{bar:40}] {pos}/{len}")?
                .progress_chars("=> "),
        );
        detail_bar.set_length(0);

        Ok(Self {
            stage_bar,
            detail_bar,
            stage_count: AtomicUsize::new(0),
            current_total: AtomicUsize::new(0),
        })
    }

    fn finish(&self) {
        self.detail_bar.finish_and_clear();
        self.stage_bar.finish_with_message("Done");
    }
}

impl ProgressReporter for MultiProgressReporter {
    fn begin_stage(&self, stage: PipelineStage, total_items: Option<usize>) {
        self.stage_bar.set_message(stage.to_string());

        if let Some(total) = total_items {
            self.current_total.store(total, Ordering::Relaxed);
            self.detail_bar.set_length(total as u64);
            self.detail_bar.set_position(0);
            self.detail_bar.set_message("items");
        } else {
            self.current_total.store(0, Ordering::Relaxed);
            self.detail_bar.set_length(0);
            self.detail_bar.set_position(0);
            self.detail_bar.set_message("");
        }
    }

    fn advance(&self, items_done: usize) {
        self.detail_bar.set_position(items_done as u64);
    }

    fn finish_stage(&self) {
        let count = self.stage_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.stage_bar.set_position(count as u64);

        // Clear detail bar between stages
        let total = self.current_total.load(Ordering::Relaxed);
        if total > 0 {
            self.detail_bar.set_position(total as u64);
        }
    }
}

pub fn run(args: &RunArgs) -> Result<()> {
    let mut config: PipelineConfig = if let Some(ref config_path) = args.config {
        let contents = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config {}", config_path.display()))?;
        toml::from_str(&contents).context("Invalid pipeline config")?
    } else {
        build_config_from_args(args)
    };

    // CLI overrides TOML
    if let Some(ref file) = args.file {
        config.input = file.clone();
    }
    if let Some(ref out) = args.output {
        config.output = out.clone();
    } else if config.output.as_os_str().is_empty() {
        config.output = PathBuf::from("result.tiff");
    }

    // Save config and exit if --save-config is set
    if let Some(ref save_path) = args.save_config {
        let toml_str =
            toml::to_string_pretty(&config).context("Failed to serialize pipeline config")?;
        std::fs::write(save_path, &toml_str)
            .with_context(|| format!("Failed to write config to {}", save_path.display()))?;
        println!("Config saved to {}", save_path.display());
        return Ok(());
    }

    crate::summary::print_pipeline_summary(&config);

    let multi = MultiProgress::new();
    let reporter = Arc::new(MultiProgressReporter::new(&multi)?);

    run_pipeline_reported(&config, reporter.clone())?;

    reporter.finish();
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
        let deconvolution = build_deconv_config(args);
        Some(SharpeningConfig {
            wavelet,
            deconvolution,
        })
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

    // file is always Some when --config is absent (required_unless_present)
    let input = args.file.clone().unwrap_or_default();
    let output = args.output.clone().unwrap_or_else(|| PathBuf::from("result.tiff"));

    PipelineConfig {
        input,
        output,
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

fn build_deconv_config(args: &RunArgs) -> Option<DeconvolutionConfig> {
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
