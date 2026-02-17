mod app;
mod convert;
mod messages;
mod panels;
mod progress;
mod state;
mod worker;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Jupiter"),
        ..Default::default()
    };

    eframe::run_native(
        "JupiterStack",
        options,
        Box::new(|cc| Ok(Box::new(app::JupiterApp::new(&cc.egui_ctx)))),
    )
}
