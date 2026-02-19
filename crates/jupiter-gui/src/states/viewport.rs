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
