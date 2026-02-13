use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::state::{
    DECONV_METHOD_NAMES, DEVICE_NAMES, FILTER_TYPE_NAMES, METRIC_NAMES, PSF_MODEL_NAMES,
    STACK_METHOD_NAMES,
};
use jupiter_core::pipeline::config::{FilterStep, QualityMetric};
use jupiter_core::pipeline::PipelineStage;

const LEFT_PANEL_WIDTH: f32 = 280.0;

pub fn show(ctx: &egui::Context, app: &mut JupiterApp) {
    egui::SidePanel::left("controls")
        .default_width(LEFT_PANEL_WIDTH)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.set_min_width(LEFT_PANEL_WIDTH - 20.0);

                file_section(ui, app);
                ui.separator();
                score_section(ui, app);
                ui.separator();
                stack_section(ui, app);
                ui.separator();
                sharpen_section(ui, app);
                ui.separator();
                filter_section(ui, app);
                ui.separator();
                device_section(ui, app);
                ui.separator();
                actions_section(ui, app);
            });
        });
}

fn section_header(ui: &mut egui::Ui, label: &str, status: Option<&str>) {
    ui.horizontal(|ui| {
        ui.strong(label);
        if let Some(s) = status {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.small(s);
            });
        }
    });
}

fn file_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    section_header(ui, "File", None);
    ui.add_space(4.0);

    if ui.button("Open...").clicked() {
        let cmd_tx = app.cmd_tx.clone();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("SER files", &["ser"])
                .add_filter("All files", &["*"])
                .pick_file()
            {
                let _ = cmd_tx.send(WorkerCommand::LoadFileInfo { path });
            }
        });
    }

    if let Some(ref path) = app.ui_state.file_path {
        ui.label(
            path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
        );
    }

    if let Some(ref info) = app.ui_state.source_info {
        ui.small(format!("{}x{}, {} frames", info.width, info.height, info.total_frames));
        ui.small(format!("{}-bit, {:?}", info.bit_depth, info.color_mode));
        if let Some(ref obs) = info.observer {
            ui.small(format!("Observer: {obs}"));
        }
        if let Some(ref tel) = info.telescope {
            ui.small(format!("Telescope: {tel}"));
        }

        // Frame preview slider
        ui.add_space(4.0);
        let max_frame = info.total_frames.saturating_sub(1);
        let mut idx = app.ui_state.preview_frame_index;
        let response = ui.add(
            egui::Slider::new(&mut idx, 0..=max_frame)
                .text("Frame")
                .clamping(egui::SliderClamping::Always),
        );
        if response.changed() {
            app.ui_state.preview_frame_index = idx;
            if let Some(ref path) = app.ui_state.file_path {
                app.send_command(WorkerCommand::PreviewFrame {
                    path: path.clone(),
                    frame_index: idx,
                });
            }
        }
    }
}

fn score_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = app
        .ui_state
        .frames_scored
        .map(|n| format!("{n} scored"));
    section_header(
        ui,
        "Frame Selection",
        status.as_deref(),
    );
    ui.add_space(4.0);

    // Metric combo
    let mut metric_idx = match app.config.quality_metric {
        QualityMetric::Laplacian => 0,
        QualityMetric::Gradient => 1,
    };
    let changed_metric = egui::ComboBox::from_label("Metric")
        .selected_text(METRIC_NAMES[metric_idx])
        .show_index(ui, &mut metric_idx, METRIC_NAMES.len(), |i| METRIC_NAMES[i].to_string())
        .changed();
    if changed_metric {
        app.config.quality_metric = match metric_idx {
            0 => QualityMetric::Laplacian,
            _ => QualityMetric::Gradient,
        };
        app.ui_state.score_params_dirty = true;
    }

    // Keep percentage
    if ui
        .add(
            egui::Slider::new(&mut app.config.select_percentage, 0.01..=1.0)
                .text("Keep %")
                .fixed_decimals(2),
        )
        .changed()
    {
        app.ui_state.stack_params_dirty = true;
    }

    // Score button
    let can_score = app.ui_state.file_path.is_some() && !app.ui_state.is_busy();
    let score_color = if app.ui_state.score_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.frames_scored.is_some() {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Score Frames");
    let btn = if let Some(c) = score_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_score, btn).clicked() {
        if let Some(ref path) = app.ui_state.file_path {
            app.ui_state.running_stage = Some(PipelineStage::QualityAssessment);
            app.send_command(WorkerCommand::LoadAndScore {
                path: path.clone(),
                metric: app.config.quality_metric.clone(),
            });
        }
    }
}

fn stack_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    section_header(
        ui,
        "Stacking",
        app.ui_state.stack_status.as_deref(),
    );
    ui.add_space(4.0);

    // Method combo
    if egui::ComboBox::from_label("Method")
        .selected_text(STACK_METHOD_NAMES[app.config.stack_method_index])
        .show_index(ui, &mut app.config.stack_method_index, STACK_METHOD_NAMES.len(), |i| {
            STACK_METHOD_NAMES[i].to_string()
        })
        .changed()
    {
        app.ui_state.stack_params_dirty = true;
    }

    // Method-specific params
    match app.config.stack_method_index {
        2 => {
            // Sigma clip
            if ui.add(egui::Slider::new(&mut app.config.sigma_clip_sigma, 0.5..=5.0).text("Sigma")).changed() {
                app.ui_state.stack_params_dirty = true;
            }
            let mut iter = app.config.sigma_clip_iterations as i32;
            if ui.add(egui::Slider::new(&mut iter, 1..=10).text("Iterations")).changed() {
                app.config.sigma_clip_iterations = iter as usize;
                app.ui_state.stack_params_dirty = true;
            }
        }
        3 => {
            // Multi-point
            let mut ap = app.config.mp_ap_size as i32;
            if ui.add(egui::Slider::new(&mut ap, 16..=256).text("AP Size")).changed() {
                app.config.mp_ap_size = ap as usize;
                app.ui_state.stack_params_dirty = true;
            }
            let mut sr = app.config.mp_search_radius as i32;
            if ui.add(egui::Slider::new(&mut sr, 4..=64).text("Search Radius")).changed() {
                app.config.mp_search_radius = sr as usize;
                app.ui_state.stack_params_dirty = true;
            }
            if ui.add(egui::Slider::new(&mut app.config.mp_min_brightness, 0.0..=0.5).text("Min Bright")).changed() {
                app.ui_state.stack_params_dirty = true;
            }
        }
        4 => {
            // Drizzle
            if ui.add(egui::Slider::new(&mut app.config.drizzle_scale, 1.0..=4.0).text("Scale")).changed() {
                app.ui_state.stack_params_dirty = true;
            }
            if ui.add(egui::Slider::new(&mut app.config.drizzle_pixfrac, 0.1..=1.0).text("Pixfrac")).changed() {
                app.ui_state.stack_params_dirty = true;
            }
            if ui.checkbox(&mut app.config.drizzle_quality_weighted, "Quality weighted").changed() {
                app.ui_state.stack_params_dirty = true;
            }
        }
        _ => {}
    }

    // Stack button
    let can_stack = app.ui_state.frames_scored.is_some() && !app.ui_state.is_busy();
    let stack_color = if app.ui_state.stack_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.stack_status.is_some() {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Stack");
    let btn = if let Some(c) = stack_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_stack, btn).clicked() {
        app.ui_state.running_stage = Some(PipelineStage::Stacking);
        app.send_command(WorkerCommand::Stack {
            select_percentage: app.config.select_percentage,
            method: app.config.stack_method(),
            device: app.config.device_preference(),
        });
    }
}

fn sharpen_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = if app.ui_state.sharpen_status { Some("Done") } else { None };
    section_header(ui, "Sharpening", status);
    ui.add_space(4.0);

    if ui.checkbox(&mut app.config.sharpen_enabled, "Enable sharpening").changed() {
        app.ui_state.sharpen_params_dirty = true;
    }

    if app.config.sharpen_enabled {
        // Wavelet params
        ui.small("Wavelet Layers:");
        let mut layers = app.config.wavelet_num_layers as i32;
        if ui.add(egui::Slider::new(&mut layers, 1..=8).text("Layers")).changed() {
            app.config.wavelet_num_layers = layers as usize;
            // Resize coefficients to match
            app.config.wavelet_coefficients.resize(layers as usize, 1.0);
            app.ui_state.sharpen_params_dirty = true;
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
                app.ui_state.sharpen_params_dirty = true;
            }
        }

        ui.add_space(4.0);

        // Deconvolution
        if ui.checkbox(&mut app.config.deconv_enabled, "Deconvolution").changed() {
            app.ui_state.sharpen_params_dirty = true;
        }

        if app.config.deconv_enabled {
            if egui::ComboBox::from_label("Deconv")
                .selected_text(DECONV_METHOD_NAMES[app.config.deconv_method_index])
                .show_index(ui, &mut app.config.deconv_method_index, DECONV_METHOD_NAMES.len(), |i| {
                    DECONV_METHOD_NAMES[i].to_string()
                })
                .changed()
            {
                app.ui_state.sharpen_params_dirty = true;
            }

            match app.config.deconv_method_index {
                0 => {
                    let mut iter = app.config.rl_iterations as i32;
                    if ui.add(egui::Slider::new(&mut iter, 1..=100).text("Iterations")).changed() {
                        app.config.rl_iterations = iter as usize;
                        app.ui_state.sharpen_params_dirty = true;
                    }
                }
                _ => {
                    if ui.add(egui::Slider::new(&mut app.config.wiener_noise_ratio, 0.001..=0.1).text("Noise Ratio").logarithmic(true)).changed() {
                        app.ui_state.sharpen_params_dirty = true;
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
                app.ui_state.sharpen_params_dirty = true;
            }

            match app.config.psf_model_index {
                0 => {
                    if ui.add(egui::Slider::new(&mut app.config.psf_gaussian_sigma, 0.5..=5.0).text("Sigma")).changed() {
                        app.ui_state.sharpen_params_dirty = true;
                    }
                }
                1 => {
                    if ui.add(egui::Slider::new(&mut app.config.psf_kolmogorov_seeing, 0.5..=10.0).text("Seeing")).changed() {
                        app.ui_state.sharpen_params_dirty = true;
                    }
                }
                _ => {
                    if ui.add(egui::Slider::new(&mut app.config.psf_airy_radius, 0.5..=10.0).text("Radius")).changed() {
                        app.ui_state.sharpen_params_dirty = true;
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

fn filter_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = app
        .ui_state
        .filter_status
        .map(|n| format!("{n} applied"));
    section_header(ui, "Filters", status.as_deref());
    ui.add_space(4.0);

    let mut to_remove = None;
    for (i, filter) in app.config.filters.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("{}.", i + 1));
            match filter {
                FilterStep::AutoStretch {
                    low_percentile,
                    high_percentile,
                } => {
                    ui.label("Auto Stretch");
                    ui.add(egui::DragValue::new(low_percentile).speed(0.001).prefix("lo: "));
                    ui.add(egui::DragValue::new(high_percentile).speed(0.001).prefix("hi: "));
                }
                FilterStep::HistogramStretch {
                    black_point,
                    white_point,
                } => {
                    ui.label("Hist Stretch");
                    ui.add(egui::DragValue::new(black_point).speed(0.01).prefix("B: "));
                    ui.add(egui::DragValue::new(white_point).speed(0.01).prefix("W: "));
                }
                FilterStep::Gamma(g) => {
                    ui.label("Gamma");
                    ui.add(egui::DragValue::new(g).speed(0.05).range(0.1..=5.0));
                }
                FilterStep::BrightnessContrast {
                    brightness,
                    contrast,
                } => {
                    ui.label("B/C");
                    ui.add(egui::DragValue::new(brightness).speed(0.01).prefix("B: "));
                    ui.add(egui::DragValue::new(contrast).speed(0.05).prefix("C: "));
                }
                FilterStep::UnsharpMask {
                    radius,
                    amount,
                    threshold,
                } => {
                    ui.label("USM");
                    ui.add(egui::DragValue::new(radius).speed(0.1).prefix("R: "));
                    ui.add(egui::DragValue::new(amount).speed(0.05).prefix("A: "));
                    ui.add(egui::DragValue::new(threshold).speed(0.01).prefix("T: "));
                }
                FilterStep::GaussianBlur { sigma } => {
                    ui.label("Blur");
                    ui.add(egui::DragValue::new(sigma).speed(0.1).prefix("S: "));
                }
            }
            if ui.small_button("x").clicked() {
                to_remove = Some(i);
            }
        });
    }

    if let Some(i) = to_remove {
        app.config.filters.remove(i);
        app.ui_state.filter_params_dirty = true;
    }

    // Add filter menu
    ui.menu_button("+ Add Filter", |ui| {
        for (i, name) in FILTER_TYPE_NAMES.iter().enumerate() {
            if ui.button(*name).clicked() {
                let filter = match i {
                    0 => FilterStep::AutoStretch {
                        low_percentile: 0.001,
                        high_percentile: 0.999,
                    },
                    1 => FilterStep::HistogramStretch {
                        black_point: 0.0,
                        white_point: 1.0,
                    },
                    2 => FilterStep::Gamma(1.0),
                    3 => FilterStep::BrightnessContrast {
                        brightness: 0.0,
                        contrast: 1.0,
                    },
                    4 => FilterStep::UnsharpMask {
                        radius: 1.5,
                        amount: 0.5,
                        threshold: 0.0,
                    },
                    _ => FilterStep::GaussianBlur { sigma: 1.0 },
                };
                app.config.filters.push(filter);
                app.ui_state.filter_params_dirty = true;
                ui.close();
            }
        }
    });

    // Apply button
    let has_base = app.ui_state.stack_status.is_some() || app.ui_state.sharpen_status;
    let can_apply = has_base && !app.ui_state.is_busy() && !app.config.filters.is_empty();
    let filter_color = if app.ui_state.filter_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.filter_status.is_some() {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Apply Filters");
    let btn = if let Some(c) = filter_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_apply, btn).clicked() {
        app.ui_state.running_stage = Some(PipelineStage::Filtering);
        app.send_command(WorkerCommand::ApplyFilters {
            filters: app.config.filters.clone(),
        });
    }
}

fn device_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    section_header(ui, "Compute Device", None);
    ui.add_space(4.0);

    egui::ComboBox::from_label("Device")
        .selected_text(DEVICE_NAMES[app.config.device_index])
        .show_index(ui, &mut app.config.device_index, DEVICE_NAMES.len(), |i| {
            DEVICE_NAMES[i].to_string()
        });
}

fn actions_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    section_header(ui, "Actions", None);
    ui.add_space(4.0);

    // Output path
    ui.horizontal(|ui| {
        ui.label("Output:");
        ui.text_edit_singleline(&mut app.ui_state.output_path);
    });

    ui.add_space(4.0);

    // Run All button
    let can_run = app.ui_state.file_path.is_some() && !app.ui_state.is_busy();
    if ui
        .add_enabled(can_run, egui::Button::new("Run All").min_size(egui::vec2(ui.available_width(), 28.0)))
        .clicked()
    {
        if let Some(ref path) = app.ui_state.file_path {
            let output = if app.ui_state.output_path.is_empty() {
                path.with_extension("tiff")
            } else {
                std::path::PathBuf::from(&app.ui_state.output_path)
            };
            let config = app.config.to_pipeline_config(path, &output);
            app.ui_state.running_stage = Some(PipelineStage::Reading);
            app.send_command(WorkerCommand::RunAll { config });
        }
    }

    // Save button
    if ui
        .add_enabled(
            !app.ui_state.is_busy(),
            egui::Button::new("Save").min_size(egui::vec2(ui.available_width(), 28.0)),
        )
        .clicked()
    {
        let cmd_tx = app.cmd_tx.clone();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("TIFF", &["tiff", "tif"])
                .add_filter("PNG", &["png"])
                .set_file_name("output.tiff")
                .save_file()
            {
                let _ = cmd_tx.send(WorkerCommand::SaveImage { path });
            }
        });
    }
}
