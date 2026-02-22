use crate::app::JupiterApp;
use crate::states::{DeconvMethodChoice, PsfModelChoice};
use jupiter_core::pipeline::PipelineStage;

pub(super) fn sharpen_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = if matches!(app.ui_state.running_stage, Some(PipelineStage::Sharpening)) {
        Some("Processing...")
    } else if app.config.sharpen_enabled {
        app.ui_state.stages.sharpen.label().map(|s| s as &str)
    } else {
        None
    };
    let color = if app.config.sharpen_enabled {
        app.ui_state.stages.sharpen.button_color()
    } else {
        None
    };
    crate::panels::section_header(ui, "4. Sharpening", status, color);
    ui.add_space(4.0);

    let enabled = app.ui_state.stages.stack.is_complete();
    ui.add_enabled_ui(enabled, |ui| {
        if ui
            .checkbox(&mut app.config.sharpen_enabled, "Enable sharpening")
            .changed()
        {
            app.ui_state.mark_dirty_from_sharpen();
            app.ui_state.request_sharpen();
        }

        if app.config.sharpen_enabled {
            // Wavelet params
            ui.small("Wavelet Layers:");
            let mut layers = app.config.wavelet_num_layers as i32;
            let resp = ui.add(egui::Slider::new(&mut layers, 1..=8).text("Layers"));
            if resp.changed() {
                app.config.wavelet_num_layers = layers as usize;
                app.config.wavelet_coefficients.resize(layers as usize, 1.0);
                app.ui_state.mark_dirty_from_sharpen();
            }
            if resp.drag_stopped() || resp.lost_focus() {
                app.ui_state.request_sharpen();
            }

            for i in 0..app.config.wavelet_coefficients.len() {
                let resp = ui.add(
                    egui::Slider::new(&mut app.config.wavelet_coefficients[i], 0.0..=3.0)
                        .text(format!("L{}", i + 1))
                        .fixed_decimals(2),
                );
                if resp.changed() {
                    app.ui_state.mark_dirty_from_sharpen();
                }
                if resp.drag_stopped() || resp.lost_focus() {
                    app.ui_state.request_sharpen();
                }
            }

            ui.add_space(4.0);

            // Deconvolution
            if ui
                .checkbox(&mut app.config.deconv_enabled, "Deconvolution")
                .changed()
            {
                app.ui_state.mark_dirty_from_sharpen();
                app.ui_state.request_sharpen();
            }

            if app.config.deconv_enabled {
                if crate::panels::enum_combo(
                    ui,
                    "Deconv",
                    &mut app.config.deconv_method,
                    DeconvMethodChoice::ALL,
                ) {
                    app.ui_state.mark_dirty_from_sharpen();
                    app.ui_state.request_sharpen();
                }

                match app.config.deconv_method {
                    DeconvMethodChoice::RichardsonLucy => {
                        let mut iter = app.config.rl_iterations as i32;
                        let resp = ui.add(egui::Slider::new(&mut iter, 1..=100).text("Iterations"));
                        if resp.changed() {
                            app.config.rl_iterations = iter as usize;
                            app.ui_state.mark_dirty_from_sharpen();
                        }
                        if resp.drag_stopped() || resp.lost_focus() {
                            app.ui_state.request_sharpen();
                        }
                    }
                    DeconvMethodChoice::Wiener => {
                        let resp = ui.add(
                            egui::Slider::new(&mut app.config.wiener_noise_ratio, 0.001..=0.1)
                                .text("Noise Ratio")
                                .logarithmic(true),
                        );
                        if resp.changed() {
                            app.ui_state.mark_dirty_from_sharpen();
                        }
                        if resp.drag_stopped() || resp.lost_focus() {
                            app.ui_state.request_sharpen();
                        }
                    }
                }

                // PSF model
                if crate::panels::enum_combo(
                    ui,
                    "PSF",
                    &mut app.config.psf_model,
                    PsfModelChoice::ALL,
                ) {
                    app.ui_state.mark_dirty_from_sharpen();
                    app.ui_state.request_sharpen();
                }

                match app.config.psf_model {
                    PsfModelChoice::Gaussian => {
                        let resp = ui.add(
                            egui::Slider::new(&mut app.config.psf_gaussian_sigma, 0.5..=5.0)
                                .text("Sigma"),
                        );
                        if resp.changed() {
                            app.ui_state.mark_dirty_from_sharpen();
                        }
                        if resp.drag_stopped() || resp.lost_focus() {
                            app.ui_state.request_sharpen();
                        }
                    }
                    PsfModelChoice::Kolmogorov => {
                        let resp = ui.add(
                            egui::Slider::new(&mut app.config.psf_kolmogorov_seeing, 0.5..=10.0)
                                .text("Seeing"),
                        );
                        if resp.changed() {
                            app.ui_state.mark_dirty_from_sharpen();
                        }
                        if resp.drag_stopped() || resp.lost_focus() {
                            app.ui_state.request_sharpen();
                        }
                    }
                    PsfModelChoice::Airy => {
                        let resp = ui.add(
                            egui::Slider::new(&mut app.config.psf_airy_radius, 0.5..=10.0)
                                .text("Radius"),
                        );
                        if resp.changed() {
                            app.ui_state.mark_dirty_from_sharpen();
                        }
                        if resp.drag_stopped() || resp.lost_focus() {
                            app.ui_state.request_sharpen();
                        }
                    }
                }
            }
        }
    });
}
