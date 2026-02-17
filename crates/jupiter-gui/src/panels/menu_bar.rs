use crate::app::JupiterApp;
use crate::messages::{WorkerCommand, WorkerResult};
use crate::state::ConfigState;

pub fn show(ctx: &egui::Context, app: &mut JupiterApp) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                let open_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O);
                if ui.add(egui::Button::new("Open...").shortcut_text(ctx.format_shortcut(&open_shortcut))).clicked() {
                    ui.close();
                    open_file(app);
                }

                let save_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S);
                if ui.add(egui::Button::new("Save As...").shortcut_text(ctx.format_shortcut(&save_shortcut))).clicked() {
                    ui.close();
                    save_file(app);
                }

                ui.separator();

                if ui.button("Import Config...").clicked() {
                    ui.close();
                    import_config(app);
                }

                if ui.button("Export Config...").clicked() {
                    ui.close();
                    export_config(app);
                }

                ui.separator();

                let quit_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Q);
                if ui.add(egui::Button::new("Quit").shortcut_text(ctx.format_shortcut(&quit_shortcut))).clicked() {
                    ui.close();
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

            ui.menu_button("Edit", |ui| {
                if ui.button("Reset Defaults").clicked() {
                    ui.close();
                    app.config = ConfigState::default();
                    app.ui_state.mark_dirty_from_score();
                    app.ui_state.add_log("Config reset to defaults".into());
                }
            });

            ui.menu_button("Help", |ui| {
                if ui.button("About").clicked() {
                    ui.close();
                    app.show_about = true;
                }
            });
        });

        // Keyboard shortcuts (consumed outside menus)
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::O))) {
            open_file(app);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::S))) {
            save_file(app);
        }
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Q))) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    });
}

fn open_file(app: &mut JupiterApp) {
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

fn save_file(app: &mut JupiterApp) {
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

fn import_config(app: &mut JupiterApp) {
    let result_tx = app.result_tx.clone();
    std::thread::spawn(move || {
        let config = rfd::FileDialog::new()
            .add_filter("TOML", &["toml"])
            .pick_file()
            .and_then(|path| {
                let content = std::fs::read_to_string(&path).ok()?;
                toml::from_str(&content).ok()
            });
        if let Some(config) = config {
            let _ = result_tx.send(WorkerResult::ConfigImported { config });
        }
    });
}

fn export_config(app: &mut JupiterApp) {
    let input_path = app.ui_state.file_path.clone().unwrap_or_default();
    let output_path = std::path::PathBuf::from(&app.ui_state.output_path);
    let config = app.config.to_pipeline_config(&input_path, &output_path);

    std::thread::spawn(move || {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("TOML", &["toml"])
            .set_file_name("jupiter_config.toml")
            .save_file()
        {
            if let Ok(content) = toml::to_string_pretty(&config) {
                let _ = std::fs::write(path, content);
            }
        }
    });
}
