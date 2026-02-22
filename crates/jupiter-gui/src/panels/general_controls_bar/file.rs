use crate::app::JupiterApp;
use crate::messages::WorkerCommand;

pub(super) fn file_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(ui, "File", None, None);
    ui.add_space(4.0);

    if ui.button("Open...").clicked() {
        let cmd_tx = app.cmd_tx.clone();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Video files", &["ser"])
                .add_filter("Image files", &["tiff", "tif", "png", "jpg", "jpeg"])
                .add_filter("All files", &["*"])
                .pick_file()
            {
                let cmd = match path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_ascii_lowercase())
                    .as_deref()
                {
                    Some("ser") => WorkerCommand::LoadFileInfo { path },
                    _ => WorkerCommand::LoadImageFile { path },
                };
                let _ = cmd_tx.send(cmd);
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

    // Extract info we need before any mutable borrows.
    let info_summary = app.ui_state.source_info.as_ref().map(|info| {
        (
            info.width,
            info.height,
            info.total_frames,
            info.bit_depth,
            info.color_mode.clone(),
            info.observer.clone(),
            info.telescope.clone(),
        )
    });

    if let Some((width, height, total_frames, bit_depth, color_mode, observer, telescope)) =
        info_summary
    {
        if app.ui_state.is_video {
            ui.small(format!("{}x{}, {} frames", width, height, total_frames));
            ui.small(format!("{}-bit, {:?}", bit_depth, color_mode));
        } else {
            ui.small(format!("{}x{}", width, height));
            ui.small(format!("{:?}", color_mode));
        }
        if let Some(ref obs) = observer {
            ui.small(format!("Observer: {obs}"));
        }
        if let Some(ref tel) = telescope {
            ui.small(format!("Telescope: {tel}"));
        }

        // View mode toggle + frame slider (video only)
        if app.ui_state.is_video {
            ui.add_space(4.0);

            // Raw / Processed toggle
            let has_processed = app.viewport.has_processed();
            let viewing_raw = app.ui_state.viewing_raw;
            let mut switch_to_raw = false;
            let mut switch_to_processed = false;

            ui.horizontal(|ui| {
                ui.label("View:");
                if ui
                    .add(egui::Button::new("Raw").selected(viewing_raw))
                    .clicked()
                    && !viewing_raw
                {
                    switch_to_raw = true;
                }
                let processed_response = ui.add_enabled(
                    has_processed,
                    egui::Button::new("Processed").selected(!viewing_raw),
                );
                if processed_response.clicked() && viewing_raw && has_processed {
                    switch_to_processed = true;
                }
            });

            if switch_to_raw {
                app.switch_to_raw();
            } else if switch_to_processed {
                app.switch_to_processed();
            }

            // Frame slider (only when viewing raw)
            if app.ui_state.viewing_raw {
                let max_frame = total_frames.saturating_sub(1);
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
    }
}
