use std::fmt;

/// Crop rectangle in image pixel coordinates.
#[derive(Clone, Debug)]
pub struct CropRectPixels {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CropRectPixels {
    /// Convert to core `CropRect`, rounding to integers and snapping to even for Bayer.
    pub fn to_core_crop_rect(&self, is_bayer: bool) -> jupiter_core::io::crop::CropRect {
        let mut x = self.x.round() as u32;
        let mut y = self.y.round() as u32;
        let mut w = self.width.round() as u32;
        let mut h = self.height.round() as u32;

        if is_bayer {
            x &= !1;
            y &= !1;
            w &= !1;
            h &= !1;
        }

        jupiter_core::io::crop::CropRect {
            x,
            y,
            width: w,
            height: h,
        }
    }

    /// Snap the crop rect to the given aspect ratio (width/height), clamping to image bounds.
    pub fn snap_to_ratio(&mut self, ratio: f32, img_w: f32, img_h: f32) {
        let cx = self.x + self.width / 2.0;
        let cy = self.y + self.height / 2.0;

        // Try keeping width, adjust height
        let mut w = self.width;
        let mut h = w / ratio;

        if h > img_h {
            h = img_h;
            w = h * ratio;
        }
        if w > img_w {
            w = img_w;
            h = w / ratio;
        }

        self.x = (cx - w / 2.0).max(0.0).min(img_w - w);
        self.y = (cy - h / 2.0).max(0.0).min(img_h - h);
        self.width = w;
        self.height = h;
    }
}

/// Crop aspect ratio presets.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum CropAspect {
    #[default]
    Free,
    Square,
    ThreeByFour,
    FourByThree,
    SixteenByNine,
}

impl CropAspect {
    pub const ALL: &[Self] = &[
        Self::Free,
        Self::Square,
        Self::ThreeByFour,
        Self::FourByThree,
        Self::SixteenByNine,
    ];

    /// Return the width/height ratio, or `None` for free.
    pub fn ratio(&self) -> Option<f32> {
        match self {
            Self::Free => None,
            Self::Square => Some(1.0),
            Self::ThreeByFour => Some(3.0 / 4.0),
            Self::FourByThree => Some(4.0 / 3.0),
            Self::SixteenByNine => Some(16.0 / 9.0),
        }
    }
}

impl fmt::Display for CropAspect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Free => write!(f, "Free"),
            Self::Square => write!(f, "1:1"),
            Self::ThreeByFour => write!(f, "3:4"),
            Self::FourByThree => write!(f, "4:3"),
            Self::SixteenByNine => write!(f, "16:9"),
        }
    }
}

/// State for crop mode.
#[derive(Default)]
pub struct CropState {
    /// Whether crop mode is active.
    pub active: bool,
    /// Current selection in image coords.
    pub rect: Option<CropRectPixels>,
    /// Screen coords of drag start (for creating new selection).
    pub drag_start: Option<egui::Pos2>,
    /// Worker processing flag.
    pub is_saving: bool,
    /// Selected aspect ratio.
    pub aspect_ratio: CropAspect,
    /// True when the user is dragging to move an existing crop rect.
    pub moving: bool,
    /// Offset from pointer (image coords) to crop rect top-left when move started.
    pub move_offset: Option<egui::Vec2>,
}
