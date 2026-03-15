# Changelog

## Refactor Drizzle: From Stacking Algorithm to Stacking Option

Drizzle is no longer a standalone `StackMethod::Drizzle(DrizzleConfig)` variant. It is now an **option** on `StackingConfig` that can be enabled for any standard stacking method (Mean+Drizzle, Median+Drizzle, SigmaClip+Drizzle). This better reflects drizzle's nature as a geometric super-resolution projection, not a pixel-combination strategy.

When drizzle is enabled, each frame is individually projected onto a high-resolution grid using its alignment offset before the chosen stacking method combines them.

### Core library (`jupiter-core`)

1. **`stack/drizzle.rs`** — Rewrote to expose `drizzle_single_frame()` and `drizzle_output_dims()` as the public API. Removed `quality_weighted` from `DrizzleConfig`, removed `drizzle_stack()`, `drizzle_stack_with_progress()`, `drizzle_stack_parallel()`, `drizzle_stack_sequential()`, `drizzle_stack_streaming()`, and `default_true()`. Added `Display` impl for `DrizzleConfig`.

2. **`pipeline/config.rs`** — Removed `Drizzle(DrizzleConfig)` from `StackMethod` enum. Added `drizzle: Option<DrizzleConfig>` to `StackingConfig`. Added `Display` impl for `StackingConfig` that appends " + Drizzle" when enabled.

3. **`pipeline/helpers.rs`** — Removed `drizzle_flow()` and `drizzle_color_channels_parallel()`. Added `drizzle_frames()` and `drizzle_color_frames()` helpers that drizzle individual frames. Removed `StackMethod::Drizzle` from unreachable arm.

4. **`pipeline/mono.rs`** — Removed `run_mono_drizzle()` and `run_mono_drizzle_streaming()`. Simplified `run_mono_pipeline()` to just streaming vs non-streaming branching. Integrated drizzle as a pre-stack step: if `config.stacking.drizzle` is `Some`, frames are drizzled instead of shifted before stacking. Added `combine_offset()` helper for streaming drizzle with pre-centering.

5. **`pipeline/color.rs`** — Removed `color_drizzle_flow()` and all `StackMethod::Drizzle` branches. Integrated drizzle into `color_standard_flow()` and `color_disk_backed_flow()` as a pre-stack projection step.

6. **`tests/test_drizzle.rs`** — Rewrote all tests to use `drizzle_single_frame()` API. Added `test_drizzle_then_mean_stack`, `test_drizzle_then_median_stack`, `test_drizzle_then_sigma_clip_stack`, `test_drizzle_output_dims_helper`, and `test_drizzle_subpixel_offset`.

### CLI (`jupiter-cli`)

7. **`commands/stack.rs`** — Removed `Drizzle` from `StackMethodArg`. Added `--drizzle` bool flag. Removed `run_drizzle()`. In `run_standard()`, if `--drizzle`, frames are drizzled before stacking.

8. **`commands/pipeline.rs`** — Added `--drizzle` flag to `RunArgs`. Build `drizzle: Option<DrizzleConfig>` from the flag and set on `StackingConfig`.

9. **`summary.rs`** — Changed `print_stack_sub_params()` to accept `&StackingConfig`. Removed `StackMethod::Drizzle` arm. Added drizzle info printing when `stacking.drizzle.is_some()`.

### GUI (`jupiter-gui`)

10. **`states/choices.rs`** — Removed `Drizzle` from `StackMethodChoice`.

11. **`states/config.rs`** — Replaced `drizzle_quality_weighted` with `drizzle_enabled: bool`. Added `drizzle_config()` method. Updated `to_pipeline_config()` and `from_pipeline_config()`.

12. **`panels/controls/stack.rs`** — Removed `StackMethodChoice::Drizzle` match arm. Added drizzle section below method params (checkbox + sliders), visible only for Mean/Median/SigmaClip.

13. **`messages.rs`** — Changed `Stack { method }` to `Stack { method, drizzle }`.

14. **`workers/stacking/mod.rs`** — Removed `mod drizzle` and `StackMethod::Drizzle` arm. Passes `drizzle` to `handle_standard()`.

15. **`workers/stacking/standard.rs`** — Accepts `drizzle: Option<&DrizzleConfig>`. If drizzle is `Some`, calls `drizzle_single_frame` for each frame instead of `shift_frame`.

16. **`workers/stacking/drizzle.rs`** — Deleted.
