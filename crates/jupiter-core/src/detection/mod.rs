pub mod components;
pub mod config;
pub mod morphology;
pub mod planet;
pub mod threshold;

pub use config::{DetectionConfig, ThresholdMethod};
pub use planet::{detect_planet_in_frame, FrameDetection};
