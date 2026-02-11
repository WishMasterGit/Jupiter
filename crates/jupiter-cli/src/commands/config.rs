use std::path::PathBuf;

use anyhow::Result;
use jupiter_core::pipeline::config::{
    FrameSelectionConfig, PipelineConfig, SharpeningConfig, StackingConfig,
};
use jupiter_core::sharpen::wavelet::WaveletParams;

/// Print a full default PipelineConfig as TOML to stdout.
pub fn run() -> Result<()> {
    let config = PipelineConfig {
        input: PathBuf::from("input.ser"),
        output: PathBuf::from("result.tiff"),
        frame_selection: FrameSelectionConfig::default(),
        stacking: StackingConfig::default(),
        sharpening: Some(SharpeningConfig {
            wavelet: WaveletParams::default(),
            deconvolution: None,
        }),
        filters: vec![],
    };
    let toml_str = toml::to_string_pretty(&config)?;
    print!("{}", toml_str);
    Ok(())
}
