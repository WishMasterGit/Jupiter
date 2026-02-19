use jupiter_core::color::debayer::DebayerMethod;
use jupiter_core::compute::DevicePreference;
use jupiter_core::pipeline::config::{
    AlignmentConfig, AlignmentMethod, CentroidConfig, DebayerConfig, DeconvolutionConfig,
    DeconvolutionMethod, EnhancedPhaseConfig, FilterStep, FrameSelectionConfig, PipelineConfig,
    PsfModel, PyramidConfig, QualityMetric, SharpeningConfig, StackMethod, StackingConfig,
};
use jupiter_core::sharpen::wavelet::WaveletParams;
use jupiter_core::stack::drizzle::DrizzleConfig;
use jupiter_core::stack::multi_point::MultiPointConfig;
use jupiter_core::stack::sigma_clip::SigmaClipParams;

use super::choices::{
    AlignMethodChoice, DeconvMethodChoice, PsfModelChoice, StackMethodChoice,
};

/// All pipeline configuration parameters as editable UI fields.
pub struct ConfigState {
    // Debayering
    pub debayer_enabled: bool,
    pub debayer_method: DebayerMethod,

    // Frame selection
    pub quality_metric: QualityMetric,
    pub select_percentage: f32,

    // Alignment
    pub align_method: AlignMethodChoice,
    pub enhanced_phase_upsample: usize,
    pub centroid_threshold: f32,
    pub pyramid_levels: usize,

    // Stacking
    pub stack_method_choice: StackMethodChoice,
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
    pub deconv_method: DeconvMethodChoice,
    pub rl_iterations: usize,
    pub wiener_noise_ratio: f32,
    pub psf_model: PsfModelChoice,
    pub psf_gaussian_sigma: f32,
    pub psf_kolmogorov_seeing: f32,
    pub psf_airy_radius: f32,

    // Filters
    pub filters: Vec<FilterStep>,

    // Device
    pub device: DevicePreference,
}

impl Default for ConfigState {
    fn default() -> Self {
        Self {
            debayer_enabled: true,
            debayer_method: DebayerMethod::default(),

            quality_metric: QualityMetric::default(),
            select_percentage: 0.25,

            align_method: AlignMethodChoice::default(),
            enhanced_phase_upsample: 20,
            centroid_threshold: 0.1,
            pyramid_levels: 3,

            stack_method_choice: StackMethodChoice::default(),
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
            deconv_method: DeconvMethodChoice::default(),
            rl_iterations: 20,
            wiener_noise_ratio: 0.01,
            psf_model: PsfModelChoice::default(),
            psf_gaussian_sigma: 1.5,
            psf_kolmogorov_seeing: 2.0,
            psf_airy_radius: 3.0,

            filters: Vec::new(),

            device: DevicePreference::default(),
        }
    }
}

impl ConfigState {
    pub fn alignment_config(&self) -> AlignmentConfig {
        AlignmentConfig {
            method: match self.align_method {
                AlignMethodChoice::EnhancedPhase => {
                    AlignmentMethod::EnhancedPhaseCorrelation(EnhancedPhaseConfig {
                        upsample_factor: self.enhanced_phase_upsample,
                    })
                }
                AlignMethodChoice::Centroid => AlignmentMethod::Centroid(CentroidConfig {
                    threshold: self.centroid_threshold,
                }),
                AlignMethodChoice::GradientCorrelation => AlignmentMethod::GradientCorrelation,
                AlignMethodChoice::Pyramid => AlignmentMethod::Pyramid(PyramidConfig {
                    levels: self.pyramid_levels,
                }),
                AlignMethodChoice::PhaseCorrelation => AlignmentMethod::PhaseCorrelation,
            },
        }
    }

    pub fn stack_method(&self) -> StackMethod {
        match self.stack_method_choice {
            StackMethodChoice::Mean => StackMethod::Mean,
            StackMethodChoice::Median => StackMethod::Median,
            StackMethodChoice::SigmaClip => StackMethod::SigmaClip(SigmaClipParams {
                sigma: self.sigma_clip_sigma,
                iterations: self.sigma_clip_iterations,
            }),
            StackMethodChoice::MultiPoint => StackMethod::MultiPoint(MultiPointConfig {
                ap_size: self.mp_ap_size,
                search_radius: self.mp_search_radius,
                select_percentage: self.select_percentage,
                min_brightness: self.mp_min_brightness,
                quality_metric: self.quality_metric,
                local_stack_method: Default::default(),
            }),
            StackMethodChoice::Drizzle => StackMethod::Drizzle(DrizzleConfig {
                scale: self.drizzle_scale,
                pixfrac: self.drizzle_pixfrac,
                quality_weighted: self.drizzle_quality_weighted,
                kernel: Default::default(),
            }),
        }
    }

    pub fn device_preference(&self) -> DevicePreference {
        self.device
    }

    pub fn sharpening_config(&self) -> Option<SharpeningConfig> {
        if !self.sharpen_enabled {
            return None;
        }
        let deconvolution = if self.deconv_enabled {
            let method = match self.deconv_method {
                DeconvMethodChoice::RichardsonLucy => DeconvolutionMethod::RichardsonLucy {
                    iterations: self.rl_iterations,
                },
                DeconvMethodChoice::Wiener => DeconvolutionMethod::Wiener {
                    noise_ratio: self.wiener_noise_ratio,
                },
            };
            let psf = match self.psf_model {
                PsfModelChoice::Gaussian => PsfModel::Gaussian {
                    sigma: self.psf_gaussian_sigma,
                },
                PsfModelChoice::Kolmogorov => PsfModel::Kolmogorov {
                    seeing: self.psf_kolmogorov_seeing,
                },
                PsfModelChoice::Airy => PsfModel::Airy {
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
                method: self.debayer_method.clone(),
            })
        } else {
            None
        }
    }

    pub fn to_pipeline_config(
        &self,
        input: &std::path::Path,
        output: &std::path::Path,
    ) -> PipelineConfig {
        PipelineConfig {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            device: self.device_preference(),
            debayer: self.debayer_config(),
            force_mono: !self.debayer_enabled,
            frame_selection: FrameSelectionConfig {
                select_percentage: self.select_percentage,
                metric: self.quality_metric,
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
            state.debayer_method = db.method;
        }

        state.quality_metric = config.frame_selection.metric;
        state.select_percentage = config.frame_selection.select_percentage;

        // Alignment
        match &config.alignment.method {
            AlignmentMethod::PhaseCorrelation => {
                state.align_method = AlignMethodChoice::PhaseCorrelation;
            }
            AlignmentMethod::EnhancedPhaseCorrelation(p) => {
                state.align_method = AlignMethodChoice::EnhancedPhase;
                state.enhanced_phase_upsample = p.upsample_factor;
            }
            AlignmentMethod::Centroid(p) => {
                state.align_method = AlignMethodChoice::Centroid;
                state.centroid_threshold = p.threshold;
            }
            AlignmentMethod::GradientCorrelation => {
                state.align_method = AlignMethodChoice::GradientCorrelation;
            }
            AlignmentMethod::Pyramid(p) => {
                state.align_method = AlignMethodChoice::Pyramid;
                state.pyramid_levels = p.levels;
            }
        }

        // Stacking
        match &config.stacking.method {
            StackMethod::Mean => state.stack_method_choice = StackMethodChoice::Mean,
            StackMethod::Median => state.stack_method_choice = StackMethodChoice::Median,
            StackMethod::SigmaClip(p) => {
                state.stack_method_choice = StackMethodChoice::SigmaClip;
                state.sigma_clip_sigma = p.sigma;
                state.sigma_clip_iterations = p.iterations;
            }
            StackMethod::MultiPoint(p) => {
                state.stack_method_choice = StackMethodChoice::MultiPoint;
                state.mp_ap_size = p.ap_size;
                state.mp_search_radius = p.search_radius;
                state.mp_min_brightness = p.min_brightness;
            }
            StackMethod::Drizzle(p) => {
                state.stack_method_choice = StackMethodChoice::Drizzle;
                state.drizzle_scale = p.scale;
                state.drizzle_pixfrac = p.pixfrac;
                state.drizzle_quality_weighted = p.quality_weighted;
            }
        }

        // Device
        state.device = config.device;

        // Sharpening
        if let Some(ref s) = config.sharpening {
            state.sharpen_enabled = true;
            state.wavelet_num_layers = s.wavelet.num_layers;
            state.wavelet_coefficients = s.wavelet.coefficients.clone();
            state.wavelet_denoise = s.wavelet.denoise.clone();
            if let Some(ref d) = s.deconvolution {
                state.deconv_enabled = true;
                match &d.method {
                    DeconvolutionMethod::RichardsonLucy { iterations } => {
                        state.deconv_method = DeconvMethodChoice::RichardsonLucy;
                        state.rl_iterations = *iterations;
                    }
                    DeconvolutionMethod::Wiener { noise_ratio } => {
                        state.deconv_method = DeconvMethodChoice::Wiener;
                        state.wiener_noise_ratio = *noise_ratio;
                    }
                }
                match &d.psf {
                    PsfModel::Gaussian { sigma } => {
                        state.psf_model = PsfModelChoice::Gaussian;
                        state.psf_gaussian_sigma = *sigma;
                    }
                    PsfModel::Kolmogorov { seeing } => {
                        state.psf_model = PsfModelChoice::Kolmogorov;
                        state.psf_kolmogorov_seeing = *seeing;
                    }
                    PsfModel::Airy { radius } => {
                        state.psf_model = PsfModelChoice::Airy;
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
