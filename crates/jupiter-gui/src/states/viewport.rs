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

    /// Stored processed result for quick Raw/Processed switching.
    pub processed_texture: Option<egui::TextureHandle>,
    pub processed_image_size: Option<[usize; 2]>,
    pub processed_display_scale: f32,
    pub processed_label: String,
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
            processed_texture: None,
            processed_image_size: None,
            processed_display_scale: 1.0,
            processed_label: String::new(),
        }
    }
}

impl ViewportState {
    /// Whether a processed result is available for viewing.
    pub fn has_processed(&self) -> bool {
        self.processed_texture.is_some()
    }

    /// Clear the stored processed result (e.g., on file reload).
    pub fn clear_processed(&mut self) {
        self.processed_texture = None;
        self.processed_image_size = None;
        self.processed_display_scale = 1.0;
        self.processed_label.clear();
    }
}
