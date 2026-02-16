use std::path::PathBuf;

use jupiter_core::compute::DevicePreference;
use jupiter_core::frame::SourceInfo;
use jupiter_core::color::debayer::DebayerMethod;
use jupiter_core::pipeline::config::{
    AlignmentConfig, AlignmentMethod, CentroidConfig, DebayerConfig, DeconvolutionConfig,
    DeconvolutionMethod, EnhancedPhaseConfig, FilterStep, FrameSelectionConfig, PipelineConfig,
    PsfModel, PyramidConfig, QualityMetric, SharpeningConfig, StackMethod, StackingConfig,
};
use jupiter_core::pipeline::PipelineStage;
use jupiter_core::sharpen::wavelet::WaveletParams;
use jupiter_core::stack::drizzle::DrizzleConfig;
use jupiter_core::stack::multi_point::MultiPointConfig;
use jupiter_core::stack::sigma_clip::SigmaClipParams;

/// Crop rectangle in image pixel coordinates.
#[derive(Clone, Debug)]
pub struct CropRectPixels {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CropRectPixels {
    /// Convert to core `CropRect`, rounding to integers and snapping to even for Bayer.
    pub fn to_core_crop_rect(&self, is_bayer: bool) -> jupiter_core::io::crop::CropRect {
        let mut x = self.x.round() as u32;
        let mut y = self.y.round() as u32;
        let mut w = self.width.round() as u32;
        let mut h = self.height.round() as u32;

        if is_bayer {
            x = x & !1;
            y = y & !1;
            w = w & !1;
            h = h & !1;
        }

        jupiter_core::io::crop::CropRect {
            x,
            y,
            width: w,
            height: h,
        }
    }

    /// Snap the crop rect to the given aspect ratio (width/height), clamping to image bounds.
    pub fn snap_to_ratio(&mut self, ratio: f32, img_w: f32, img_h: f32) {
        let cx = self.x + self.width / 2.0;
        let cy = self.y + self.height / 2.0;

        // Try keeping width, adjust height
        let mut w = self.width;
        let mut h = w / ratio;

        if h > img_h {
            h = img_h;
            w = h * ratio;
        }
        if w > img_w {
            w = img_w;
            h = w / ratio;
        }

        self.x = (cx - w / 2.0).max(0.0).min(img_w - w);
        self.y = (cy - h / 2.0).max(0.0).min(img_h - h);
        self.width = w;
        self.height = h;
    }
}

pub const CROP_ASPECT_NAMES: &[&str] = &["Free", "1:1", "3:4", "4:3", "16:9"];

/// Return the width/height ratio for the given aspect ratio index, or `None` for free.
pub fn crop_aspect_value(index: usize) -> Option<f32> {
    match index {
        1 => Some(1.0),
        2 => Some(3.0 / 4.0),
        3 => Some(4.0 / 3.0),
        4 => Some(16.0 / 9.0),
        _ => None,
    }
}

/// State for crop mode.
#[derive(Default)]
pub struct CropState {
    /// Whether crop mode is active.
    pub active: bool,
    /// Current selection in image coords.
    pub rect: Option<CropRectPixels>,
    /// Screen coords of drag start (for creating new selection).
    pub drag_start: Option<egui::Pos2>,
    /// Worker processing flag.
    pub is_saving: bool,
    /// Selected aspect ratio index into `CROP_ASPECT_NAMES`.
    pub aspect_ratio_index: usize,
    /// True when the user is dragging to move an existing crop rect.
    pub moving: bool,
    /// Offset from pointer (image coords) to crop rect top-left when move started.
    pub move_offset: Option<egui::Vec2>,
}

/// Overall UI state.
#[derive(Default)]
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
    pub align_status: Option<String>,
    pub stack_status: Option<String>,
    pub sharpen_status: bool,
    pub filter_status: Option<usize>,

    /// Log messages.
    pub log_messages: Vec<String>,

    /// Progress.
    pub progress_items_done: Option<usize>,
    pub progress_items_total: Option<usize>,

    /// Crop state.
    pub crop_state: CropState,

    /// Params changed since last run (stale indicators).
    pub score_params_dirty: bool,
    pub align_params_dirty: bool,
    pub stack_params_dirty: bool,
    pub sharpen_params_dirty: bool,
    pub filter_params_dirty: bool,
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
    /// Original image size (before any display scaling).
    pub image_size: Option<[usize; 2]>,
    /// Scale factor applied for display (1.0 if no downscaling).
    pub display_scale: f32,
    pub zoom: f32,
    pub pan_offset: egui::Vec2,
    pub viewing_label: String,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            texture: None,
            image_size: None,
            display_scale: 1.0,
            zoom: 1.0,
            pan_offset: egui::Vec2::ZERO,
            viewing_label: String::new(),
        }
    }
}

pub const DEBAYER_METHOD_NAMES: &[&str] = &["Bilinear", "Malvar-He-Cutler"];

/// All pipeline configuration parameters as editable UI fields.
pub struct ConfigState {
    // Debayering
    pub debayer_enabled: bool,
    pub debayer_method_index: usize,

    // Frame selection
    pub quality_metric: QualityMetric,
    pub select_percentage: f32,

    // Alignment
    pub align_method_index: usize,
    pub enhanced_phase_upsample: usize,
    pub centroid_threshold: f32,
    pub pyramid_levels: usize,

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

pub const ALIGN_METHOD_NAMES: &[&str] = &[
    "Phase Correlation",
    "Enhanced Phase",
    "Centroid",
    "Gradient Correlation",
    "Pyramid",
];
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
            debayer_enabled: true,
            debayer_method_index: 0,

            quality_metric: QualityMetric::default(),
            select_percentage: 0.25,

            align_method_index: 0,
            enhanced_phase_upsample: 20,
            centroid_threshold: 0.1,
            pyramid_levels: 3,

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
    pub fn alignment_config(&self) -> AlignmentConfig {
        AlignmentConfig {
            method: match self.align_method_index {
                1 => AlignmentMethod::EnhancedPhaseCorrelation(EnhancedPhaseConfig {
                    upsample_factor: self.enhanced_phase_upsample,
                }),
                2 => AlignmentMethod::Centroid(CentroidConfig {
                    threshold: self.centroid_threshold,
                }),
                3 => AlignmentMethod::GradientCorrelation,
                4 => AlignmentMethod::Pyramid(PyramidConfig {
                    levels: self.pyramid_levels,
                }),
                _ => AlignmentMethod::PhaseCorrelation,
            },
        }
    }

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

    pub fn debayer_config(&self) -> Option<DebayerConfig> {
        if self.debayer_enabled {
            Some(DebayerConfig {
                method: match self.debayer_method_index {
                    0 => DebayerMethod::Bilinear,
                    _ => DebayerMethod::MalvarHeCutler,
                },
            })
        } else {
            None
        }
    }

    pub fn to_pipeline_config(&self, input: &std::path::Path, output: &std::path::Path) -> PipelineConfig {
        PipelineConfig {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            device: self.device_preference(),
            debayer: self.debayer_config(),
            force_mono: !self.debayer_enabled,
            frame_selection: FrameSelectionConfig {
                select_percentage: self.select_percentage,
                metric: self.quality_metric.clone(),
            },
            alignment: self.alignment_config(),
            stacking: StackingConfig {
                method: self.stack_method(),
            },
            sharpening: self.sharpening_config(),
            filters: self.filters.clone(),
            memory: Default::default(),
        }
    }

    pub fn from_pipeline_config(config: &PipelineConfig) -> Self {
        let mut state = Self::default();

        // Debayer
        if config.force_mono {
            state.debayer_enabled = false;
        } else if let Some(ref db) = config.debayer {
            state.debayer_enabled = true;
            state.debayer_method_index = match db.method {
                DebayerMethod::Bilinear => 0,
                DebayerMethod::MalvarHeCutler => 1,
            };
        }

        state.quality_metric = config.frame_selection.metric.clone();
        state.select_percentage = config.frame_selection.select_percentage;

        // Alignment
        match &config.alignment.method {
            AlignmentMethod::PhaseCorrelation => state.align_method_index = 0,
            AlignmentMethod::EnhancedPhaseCorrelation(p) => {
                state.align_method_index = 1;
                state.enhanced_phase_upsample = p.upsample_factor;
            }
            AlignmentMethod::Centroid(p) => {
                state.align_method_index = 2;
                state.centroid_threshold = p.threshold;
            }
            AlignmentMethod::GradientCorrelation => state.align_method_index = 3,
            AlignmentMethod::Pyramid(p) => {
                state.align_method_index = 4;
                state.pyramid_levels = p.levels;
            }
        }

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
