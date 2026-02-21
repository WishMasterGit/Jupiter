pub(crate) mod config;
mod engine;
pub mod temporal;

pub use config::AutoCropConfig;
pub use crate::detection::ThresholdMethod;
pub use engine::auto_detect_crop;
