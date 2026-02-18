use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::state::{DECONV_METHOD_NAMES, PSF_MODEL_NAMES};
use jupiter_core::pipeline::PipelineStage;

pub(super) fn sharpen_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = if app.config.sharpen_enabled && app.ui_state.sharpen_status { Some("Done") } else { None };
    crate::panels::section_header(ui, "Sharpening", status);
    ui.add_space(4.0);

    if ui.checkbox(&mut app.config.sharpen_enabled, "Enable sharpening").changed() {
        app.ui_state.mark_dirty_from_sharpen();
    }

    if app.config.sharpen_enabled {
        // Wavelet params
        ui.small("Wavelet Layers:");
        let mut layers = app.config.wavelet_num_layers as i32;
        if ui.add(egui::Slider::new(&mut layers, 1..=8).text("Layers")).changed() {
            app.config.wavelet_num_layers = layers as usize;
            // Resize coefficients to match
            app.config.wavelet_coefficients.resize(layers as usize, 1.0);
            app.ui_state.mark_dirty_from_sharpen();
        }

        for i in 0..app.config.wavelet_coefficients.len() {
            if ui
                .add(
                    egui::Slider::new(&mut app.config.wavelet_coefficients[i], 0.0..=3.0)
                        .text(format!("L{}", i + 1))
                        .fixed_decimals(2),
                )
                .changed()
            {
                app.ui_state.mark_dirty_from_sharpen();
            }
        }

        ui.add_space(4.0);

        // Deconvolution
        if ui.checkbox(&mut app.config.deconv_enabled, "Deconvolution").changed() {
            app.ui_state.mark_dirty_from_sharpen();
        }

        if app.config.deconv_enabled {
            if egui::ComboBox::from_label("Deconv")
                .selected_text(DECONV_METHOD_NAMES[app.config.deconv_method_index])
                .show_index(ui, &mut app.config.deconv_method_index, DECONV_METHOD_NAMES.len(), |i| {
                    DECONV_METHOD_NAMES[i].to_string()
                })
                .changed()
            {
                app.ui_state.mark_dirty_from_sharpen();
            }

            match app.config.deconv_method_index {
                0 => {
                    let mut iter = app.config.rl_iterations as i32;
                    if ui.add(egui::Slider::new(&mut iter, 1..=100).text("Iterations")).changed() {
                        app.config.rl_iterations = iter as usize;
                        app.ui_state.mark_dirty_from_sharpen();
                    }
                }
                _ => {
                    if ui.add(egui::Slider::new(&mut app.config.wiener_noise_ratio, 0.001..=0.1).text("Noise Ratio").logarithmic(true)).changed() {
                        app.ui_state.mark_dirty_from_sharpen();
                    }
                }
            }

            // PSF model
            if egui::ComboBox::from_label("PSF")
                .selected_text(PSF_MODEL_NAMES[app.config.psf_model_index])
                .show_index(ui, &mut app.config.psf_model_index, PSF_MODEL_NAMES.len(), |i| {
                    PSF_MODEL_NAMES[i].to_string()
                })
                .changed()
            {
                app.ui_state.mark_dirty_from_sharpen();
            }

            match app.config.psf_model_index {
                0 => {
                    if ui.add(egui::Slider::new(&mut app.config.psf_gaussian_sigma, 0.5..=5.0).text("Sigma")).changed() {
                        app.ui_state.mark_dirty_from_sharpen();
                    }
                }
                1 => {
                    if ui.add(egui::Slider::new(&mut app.config.psf_kolmogorov_seeing, 0.5..=10.0).text("Seeing")).changed() {
                        app.ui_state.mark_dirty_from_sharpen();
                    }
                }
                _ => {
                    if ui.add(egui::Slider::new(&mut app.config.psf_airy_radius, 0.5..=10.0).text("Radius")).changed() {
                        app.ui_state.mark_dirty_from_sharpen();
                    }
                }
            }
        }
    }

    // Sharpen button
    let can_sharpen = app.ui_state.stack_status.is_some()
        && !app.ui_state.is_busy()
        && app.config.sharpen_enabled;
    let sharpen_color = if app.ui_state.sharpen_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.sharpen_status {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Sharpen");
    let btn = if let Some(c) = sharpen_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_sharpen, btn).clicked() {
        if let Some(config) = app.config.sharpening_config() {
            app.ui_state.running_stage = Some(PipelineStage::Sharpening);
            app.send_command(WorkerCommand::Sharpen {
                config,
                device: app.config.device_preference(),
            });
        }
    }
}
