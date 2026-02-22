pub(crate) mod config;
mod engine;
pub mod temporal;

pub use crate::detection::ThresholdMethod;
pub use config::AutoCropConfig;
pub use engine::auto_detect_crop;
