pub mod components;
pub(crate) mod config;
pub mod detection;
mod engine;
pub mod morphology;
pub mod temporal;
mod threshold;

pub use config::{AutoCropConfig, ThresholdMethod};
pub use engine::auto_detect_crop;
