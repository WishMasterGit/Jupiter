# Changelog

### Pre-Centering Alignment System (New Feature)

Adds an optional pre-centering step that detects the planet in each frame and shifts it to image center before global alignment. This compensates for large planetary drift across frames (common with alt-azimuth mounts) where raw phase correlation can fail due to FFT wrap-around.

**Core algorithm** (`crates/jupiter-core/src/align/pre_center.rs` — new file):

- `detect_centroids()` / `detect_centroids_streaming()` — find planet centroids in frames (in-memory or streaming from SER)
- `compute_centering_offsets()` — compute per-frame shifts to move planet to image center; falls back to median centroid for frames where detection fails; returns `None` if fewer than 50% of frames have successful detections
- `compute_common_overlap()` — compute the intersection crop rectangle after centering shifts; returns `None` if overlap is below 25% of original area
- `pre_center_frames()` / `pre_center_color_frames()` — apply centering shifts and optional crop
- `crop_frame()` — crop a frame to a `CropRect` subregion

**New constants** (`consts.rs`):

- `PRE_CENTER_MIN_DETECTION_FRACTION` (0.5)
- `PRE_CENTER_MIN_OVERLAP_FRACTION` (0.25)

**Pipeline integration** (`pipeline/config.rs`, `pipeline/mono.rs`, `pipeline/color.rs`, `pipeline/types.rs`):

- Added `pre_center: bool` field to `AlignmentConfig` (serde default = false)
- Added `PreCentering` variant to `PipelineStage` enum
- Standard mono pipeline: pre-centering runs after frame reading, before alignment; combined offsets used for stacking; crop rect applied when present
- Streaming mono pipeline: same logic adapted for streaming reads with correct dimension handling
- Color pipeline: detects on luminance channel, applies centering shifts to all color channels
- Helper `apply_combined_shift()` chains centering + alignment offsets

**Multi-Point & Surface Warp stacking** (`stack/ap_grid.rs`, `stack/multi_point.rs`, `stack/surface_warp.rs`):

- Added `pre_center: bool` to both `MultiPointConfig` and `SurfaceWarpConfig`
- When enabled, Step 0 detects planet centroids and computes centering offsets; reference and target frames are shifted before phase correlation; final offsets combine centering + residual alignment
- No crop applied (multi-point/surface warp handle edge effects via their AP grid)
- `to_mp_config()` propagates `pre_center` from `SurfaceWarpConfig`

**Pipeline orchestrator** (`pipeline/orchestrator.rs`):

- Before calling `multi_point_stack*` / `surface_warp_stack*`, clones the stacking config and sets `pre_center = pipeline_config.alignment.pre_center`

### CLI Support

- Added `--pre-center` flag to `RunArgs` in `jupiter-cli/src/commands/pipeline.rs`
- Updated pattern match in `summary.rs` to handle new config fields with `..` catch-all

### GUI Enhancements

**Pre-centering UI** (`panels/controls/alignment.rs`, `states/config.rs`, `workers/align.rs`):

- Checkbox toggle for pre-centering with help text
- `ConfigState` passes `pre_center` when constructing `MultiPointConfig` and `SurfaceWarpConfig`
- Worker thread performs pre-centering and sends `PreCentering` stage updates; logs detected planet counts and crop status

**Quality score chart improvements** (`panels/controls/score.rs`):

- Bars sorted by score (highest to lowest rank order) instead of frame index
- Click-to-preview: clicking a bar previews that frame

**System monitoring** (`states/ui.rs`, `panels/status.rs`, `app.rs`, `Cargo.toml`):

- New `SystemStats` struct tracks CPU and memory usage (1-second refresh via `sysinfo` crate)
- Status bar shows resolved device name (e.g. "Auto (Apple M1 Pro)" instead of just "Auto")
- Status bar shows live CPU% and memory usage
- `refresh_device_name()` called when device preference changes

### Documentation

- `Claude.md`: added architecture section (workspace structure, pipeline flow, key types), common commands, troubleshooting

### Tests

- `tests/test_pre_center.rs` (new) — full coverage for pre-centering core functions
- `tests/test_pre_center_stacking.rs` (new) — pre_center=true/false for multi-point and surface warp; graceful fallback when detection fails on uniform dark frames
- `tests/test_alignment_methods.rs` — updated config constructors to use `..Default::default()`

### Dependencies

- Added `sysinfo = "0.38.4"` to `jupiter-gui`
