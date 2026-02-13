use crate::app::JupiterApp;
use crate::messages::WorkerCommand;

pub(super) fn file_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    super::section_header(ui, "File", None);
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
