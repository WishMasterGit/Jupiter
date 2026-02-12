use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use jupiter_core::compute::DevicePreference;
use jupiter_core::pipeline::config::{
    FrameSelectionConfig, PipelineConfig, SharpeningConfig, StackingConfig,
};
use jupiter_core::sharpen::wavelet::WaveletParams;

#[derive(Args)]
pub struct ConfigArgs {
    /// Write config to a file instead of stdout
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Print or save a full default PipelineConfig as TOML.
pub fn run(args: &ConfigArgs) -> Result<()> {
    let config = PipelineConfig {
        input: PathBuf::from("input.ser"),
        output: PathBuf::from("result.tiff"),
        device: DevicePreference::Auto,
        frame_selection: FrameSelectionConfig::default(),
        stacking: StackingConfig::default(),
        sharpening: Some(SharpeningConfig {
            wavelet: WaveletParams::default(),
            deconvolution: None,
        }),
        filters: vec![],
    };
    let toml_str = toml::to_string_pretty(&config)?;

    if let Some(ref path) = args.output {
        std::fs::write(path, &toml_str)
            .with_context(|| format!("Failed to write config to {}", path.display()))?;
        println!("Default config saved to {}", path.display());
    } else {
        print!("{}", toml_str);
    }

    Ok(())
}
