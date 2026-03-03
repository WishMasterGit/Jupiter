mod app;
mod convert;
mod messages;
mod panels;
mod progress;
mod states;
mod workers;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let icon_data = load_icon();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Jupiter")
            .with_icon(std::sync::Arc::new(icon_data)),
        ..Default::default()
    };

    fn load_icon() -> egui::IconData {
        let (icon_rgba, icon_width, icon_height) = {
            let icon_bytes = include_bytes!("../Jupiter256.png");
            let image = image::load_from_memory(icon_bytes)
                .expect("Failed to load icon")
                .into_rgba8();
            let (width, height) = image.dimensions();
            (image.into_raw(), width, height)
        };

        egui::IconData {
            rgba: icon_rgba,
            width: icon_width,
            height: icon_height,
        }
    }

    eframe::run_native(
        "JupiterStack",
        options,
        Box::new(|cc| Ok(Box::new(app::JupiterApp::new(&cc.egui_ctx)))),
    )
}
