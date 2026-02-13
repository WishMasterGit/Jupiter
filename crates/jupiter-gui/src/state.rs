use std::path::PathBuf;

use jupiter_core::compute::DevicePreference;
use jupiter_core::frame::SourceInfo;
use jupiter_core::pipeline::config::{
    DeconvolutionConfig, DeconvolutionMethod, FilterStep, FrameSelectionConfig, PipelineConfig,
    PsfModel, QualityMetric, SharpeningConfig, StackMethod, StackingConfig,
};
use jupiter_core::pipeline::PipelineStage;
use jupiter_core::sharpen::wavelet::WaveletParams;
use jupiter_core::stack::drizzle::DrizzleConfig;
use jupiter_core::stack::multi_point::MultiPointConfig;
use jupiter_core::stack::sigma_clip::SigmaClipParams;

/// Overall UI state.
pub struct UIState {
    pub file_path: Option<PathBuf>,
    pub source_info: Option<SourceInfo>,
    pub preview_frame_index: usize,
    pub output_path: String,

    /// Which stage is currently running (None = idle).
    pub running_stage: Option<PipelineStage>,

    /// Cache status indicators.
    pub frames_scored: Option<usize>,
    pub ranked_preview: Vec<(usize, f64)>,
    pub stack_status: Option<String>,
    pub sharpen_status: bool,
    pub filter_status: Option<usize>,

    /// Log messages.
    pub log_messages: Vec<String>,

    /// Progress.
    pub progress_items_done: Option<usize>,
    pub progress_items_total: Option<usize>,

    /// Params changed since last run (stale indicators).
    pub score_params_dirty: bool,
    pub stack_params_dirty: bool,
    pub sharpen_params_dirty: bool,
    pub filter_params_dirty: bool,
}

impl Default for UIState {
    fn default() -> Self {
        Self {
            file_path: None,
            source_info: None,
            preview_frame_index: 0,
            output_path: String::new(),
            running_stage: None,
            frames_scored: None,
            ranked_preview: Vec::new(),
            stack_status: None,
            sharpen_status: false,
            filter_status: None,
            log_messages: Vec::new(),
            progress_items_done: None,
            progress_items_total: None,
            score_params_dirty: false,
            stack_params_dirty: false,
            sharpen_params_dirty: false,
            filter_params_dirty: false,
        }
    }
}

impl UIState {
    pub fn is_busy(&self) -> bool {
        self.running_stage.is_some()
    }

    pub fn add_log(&mut self, msg: String) {
        self.log_messages.push(msg);
    }
}

/// Viewport display state.
pub struct ViewportState {
    pub texture: Option<egui::TextureHandle>,
    pub image_size: Option<[usize; 2]>,
    pub zoom: f32,
    pub pan_offset: egui::Vec2,
    pub viewing_label: String,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            texture: None,
            image_size: None,
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            viewing_label: String::new(),
        }
    }
}

/// All pipeline configuration parameters as editable UI fields.
pub struct ConfigState {
    // Frame selection
    pub quality_metric: QualityMetric,
    pub select_percentage: f32,

    // Stacking
    pub stack_method_index: usize,
    // Sigma clip params
    pub sigma_clip_sigma: f32,
    pub sigma_clip_iterations: usize,
    // Multi-point params
    pub mp_ap_size: usize,
    pub mp_search_radius: usize,
    pub mp_min_brightness: f32,
    // Drizzle params
    pub drizzle_scale: f32,
    pub drizzle_pixfrac: f32,
    pub drizzle_quality_weighted: bool,

    // Sharpening
    pub sharpen_enabled: bool,
    pub wavelet_num_layers: usize,
    pub wavelet_coefficients: Vec<f32>,
    pub wavelet_denoise: Vec<f32>,
    // Deconvolution
    pub deconv_enabled: bool,
    pub deconv_method_index: usize,
    pub rl_iterations: usize,
    pub wiener_noise_ratio: f32,
    pub psf_model_index: usize,
    pub psf_gaussian_sigma: f32,
    pub psf_kolmogorov_seeing: f32,
    pub psf_airy_radius: f32,

    // Filters
    pub filters: Vec<FilterStep>,

    // Device
    pub device_index: usize,
}

pub const STACK_METHOD_NAMES: &[&str] = &["Mean", "Median", "Sigma Clip", "Multi-Point", "Drizzle"];
pub const DECONV_METHOD_NAMES: &[&str] = &["Richardson-Lucy", "Wiener"];
pub const PSF_MODEL_NAMES: &[&str] = &["Gaussian", "Kolmogorov", "Airy"];
pub const DEVICE_NAMES: &[&str] = &["Auto", "CPU", "GPU", "CUDA"];
pub const METRIC_NAMES: &[&str] = &["Laplacian", "Gradient"];
pub const FILTER_TYPE_NAMES: &[&str] = &[
    "Auto Stretch",
    "Histogram Stretch",
    "Gamma",
    "Brightness/Contrast",
    "Unsharp Mask",
    "Gaussian Blur",
];

impl Default for ConfigState {
    fn default() -> Self {
        Self {
            quality_metric: QualityMetric::default(),
            select_percentage: 0.25,

            stack_method_index: 0,
            sigma_clip_sigma: 2.5,
            sigma_clip_iterations: 2,
            mp_ap_size: 64,
            mp_search_radius: 16,
            mp_min_brightness: 0.05,
            drizzle_scale: 2.0,
            drizzle_pixfrac: 0.7,
            drizzle_quality_weighted: true,

            sharpen_enabled: true,
            wavelet_num_layers: 6,
            wavelet_coefficients: vec![1.5, 1.3, 1.2, 1.1, 1.0, 1.0],
            wavelet_denoise: vec![],
            deconv_enabled: false,
            deconv_method_index: 0,
            rl_iterations: 20,
            wiener_noise_ratio: 0.01,
            psf_model_index: 0,
            psf_gaussian_sigma: 1.5,
            psf_kolmogorov_seeing: 2.0,
            psf_airy_radius: 3.0,

            filters: Vec::new(),

            device_index: 0,
        }
    }
}

impl ConfigState {
    pub fn stack_method(&self) -> StackMethod {
        match self.stack_method_index {
            0 => StackMethod::Mean,
            1 => StackMethod::Median,
            2 => StackMethod::SigmaClip(SigmaClipParams {
                sigma: self.sigma_clip_sigma,
                iterations: self.sigma_clip_iterations,
            }),
            3 => StackMethod::MultiPoint(MultiPointConfig {
                ap_size: self.mp_ap_size,
                search_radius: self.mp_search_radius,
                select_percentage: self.select_percentage,
                min_brightness: self.mp_min_brightness,
                quality_metric: self.quality_metric.clone(),
                local_stack_method: Default::default(),
            }),
            4 => StackMethod::Drizzle(DrizzleConfig {
                scale: self.drizzle_scale,
                pixfrac: self.drizzle_pixfrac,
                quality_weighted: self.drizzle_quality_weighted,
                kernel: Default::default(),
            }),
            _ => StackMethod::Mean,
        }
    }

    pub fn device_preference(&self) -> DevicePreference {
        match self.device_index {
            0 => DevicePreference::Auto,
            1 => DevicePreference::Cpu,
            2 => DevicePreference::Gpu,
            3 => DevicePreference::Cuda,
            _ => DevicePreference::Auto,
        }
    }

    pub fn sharpening_config(&self) -> Option<SharpeningConfig> {
        if !self.sharpen_enabled {
            return None;
        }
        let deconvolution = if self.deconv_enabled {
            let method = match self.deconv_method_index {
                0 => DeconvolutionMethod::RichardsonLucy {
                    iterations: self.rl_iterations,
                },
                _ => DeconvolutionMethod::Wiener {
                    noise_ratio: self.wiener_noise_ratio,
                },
            };
            let psf = match self.psf_model_index {
                0 => PsfModel::Gaussian {
                    sigma: self.psf_gaussian_sigma,
                },
                1 => PsfModel::Kolmogorov {
                    seeing: self.psf_kolmogorov_seeing,
                },
                _ => PsfModel::Airy {
                    radius: self.psf_airy_radius,
                },
            };
            Some(DeconvolutionConfig { method, psf })
        } else {
            None
        };

        Some(SharpeningConfig {
            wavelet: WaveletParams {
                num_layers: self.wavelet_num_layers,
                coefficients: self.wavelet_coefficients.clone(),
                denoise: self.wavelet_denoise.clone(),
            },
            deconvolution,
        })
    }

    pub fn to_pipeline_config(&self, input: &std::path::Path, output: &std::path::Path) -> PipelineConfig {
        PipelineConfig {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            device: self.device_preference(),
            frame_selection: FrameSelectionConfig {
                select_percentage: self.select_percentage,
                metric: self.quality_metric.clone(),
            },
            stacking: StackingConfig {
                method: self.stack_method(),
            },
            sharpening: self.sharpening_config(),
            filters: self.filters.clone(),
        }
    }

    pub fn from_pipeline_config(config: &PipelineConfig) -> Self {
        let mut state = Self::default();
        state.quality_metric = config.frame_selection.metric.clone();
        state.select_percentage = config.frame_selection.select_percentage;

        match &config.stacking.method {
            StackMethod::Mean => state.stack_method_index = 0,
            StackMethod::Median => state.stack_method_index = 1,
            StackMethod::SigmaClip(p) => {
                state.stack_method_index = 2;
                state.sigma_clip_sigma = p.sigma;
                state.sigma_clip_iterations = p.iterations;
            }
            StackMethod::MultiPoint(p) => {
                state.stack_method_index = 3;
                state.mp_ap_size = p.ap_size;
                state.mp_search_radius = p.search_radius;
                state.mp_min_brightness = p.min_brightness;
            }
            StackMethod::Drizzle(p) => {
                state.stack_method_index = 4;
                state.drizzle_scale = p.scale;
                state.drizzle_pixfrac = p.pixfrac;
                state.drizzle_quality_weighted = p.quality_weighted;
            }
        }

        state.device_index = match config.device {
            DevicePreference::Auto => 0,
            DevicePreference::Cpu => 1,
            DevicePreference::Gpu => 2,
            DevicePreference::Cuda => 3,
        };

        if let Some(ref s) = config.sharpening {
            state.sharpen_enabled = true;
            state.wavelet_num_layers = s.wavelet.num_layers;
            state.wavelet_coefficients = s.wavelet.coefficients.clone();
            state.wavelet_denoise = s.wavelet.denoise.clone();
            if let Some(ref d) = s.deconvolution {
                state.deconv_enabled = true;
                match &d.method {
                    DeconvolutionMethod::RichardsonLucy { iterations } => {
                        state.deconv_method_index = 0;
                        state.rl_iterations = *iterations;
                    }
                    DeconvolutionMethod::Wiener { noise_ratio } => {
                        state.deconv_method_index = 1;
                        state.wiener_noise_ratio = *noise_ratio;
                    }
                }
                match &d.psf {
                    PsfModel::Gaussian { sigma } => {
                        state.psf_model_index = 0;
                        state.psf_gaussian_sigma = *sigma;
                    }
                    PsfModel::Kolmogorov { seeing } => {
                        state.psf_model_index = 1;
                        state.psf_kolmogorov_seeing = *seeing;
                    }
                    PsfModel::Airy { radius } => {
                        state.psf_model_index = 2;
                        state.psf_airy_radius = *radius;
                    }
                }
            }
        } else {
            state.sharpen_enabled = false;
        }

        state.filters = config.filters.clone();
        state
    }
}
