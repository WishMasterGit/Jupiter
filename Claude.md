# Planet imagery processing tool

## What is it about

this is planet image/video processing tool to achieve crisp results from the earth based telescopes

## Technical decisions

- Main language rust
- Try to use constants from std for inline numbers
  - if constants aren't available in the std try checking libraries for corresponding constants
  - if there are no libraries, create constants so it is easier to understand the meaning (when it applicable)
- based on research lets use egui as our UI rendering engine to keep it crossplatform
- app should be crossplatform Windows/Mac/Linux but not mobile phones
- write tests for all jupiter-core logic

## Code Style

- For the cases when we need to use "match" to branch the logic, lets avoid numeric indexes in matching and instead use named enums

- Prefer splitting logic into separate files under the folder, for example when you need to implement multiple different worker handlers, create workers folder and make ##\_hanlder.rs for different handlers instead of putting everything into one file

- mod files should only be used to define module structure and re-exporting

- When there is if/else branching due to parallel/non-parallel cases present, code for parallel and non-parallel should be in the corresponding methods, aka compute_parallel.., compute_sequential..

## Claude workstream

- make sure to build with features gpu and normally while testing
- place tests in separate files in "tests" folder
- save all researches in the docs folder
- run cargo llvm-cov after changes in core library and make sure that core library methods have full test coverage
- always run cargo clippy and cargo fmt once you done with coding

## Project management

- Claude should read tasks from todo.md, from top to bottom, skip ones that marked [Done], after completing the todo Claud should mark it [Done]

## Architecture

### Workspace structure
```
jupiter/
├── crates/
│   ├── jupiter-core/       # Library — all algorithms and pipeline logic
│   │   ├── src/
│   │   │   ├── align/       # Alignment methods (phase correlation, triangle, feature-based, surface warp)
│   │   │   ├── color/       # Debayering and color processing
│   │   │   ├── compute/     # GPU compute shaders and CPU fallbacks
│   │   │   ├── detection/   # Planet/feature detection
│   │   │   ├── filters/     # Post-processing filters
│   │   │   ├── io/          # SER/AVI/TIFF/PNG I/O, disk cache
│   │   │   ├── pipeline/    # Pipeline orchestration (mono, color, config, types)
│   │   │   ├── quality/     # Frame quality scoring
│   │   │   ├── sharpen/     # Wavelet sharpening
│   │   │   ├── stack/       # Stacking algorithms (mean, median, sigma-clip, disk-backed)
│   │   │   ├── consts.rs    # Project-wide constants and thresholds
│   │   │   ├── error.rs     # JupiterError type
│   │   │   ├── frame.rs     # Frame types
│   │   │   └── lib.rs       # Public API re-exports
│   │   └── tests/           # Integration tests (one file per module)
│   ├── jupiter-cli/         # CLI binary
│   └── jupiter-gui/         # egui desktop GUI
```

### Processing pipeline flow
```
SER/AVI → Read Frames → Score Quality → Select Best N%
  → Align (phase correlation / triangle / feature / surface warp)
  → Stack (mean / median / sigma-clip)
  → Sharpen (wavelet layers)
  → Filter (post-processing)
  → Output (TIFF / PNG)
```

### Key types
- `ndarray::Array2<f32>` — pixel data in [0.0, 1.0]
- `JupiterError` — unified error type (`crates/jupiter-core/src/error.rs`)
- `PipelineConfig` — full pipeline configuration (`crates/jupiter-core/src/pipeline/config.rs`)
- `AlignmentMethod` — enum of alignment algorithms
- `StackMethod` — enum of stacking algorithms

## Common Commands

```bash
# Full quality check
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo clippy --workspace --features gpu -- -D warnings
cargo test --workspace
cargo test --workspace --features gpu

# Coverage (jupiter-core only)
cargo llvm-cov --package jupiter-core

# Run GUI
cargo run --package jupiter-gui

# Run CLI
cargo run --package jupiter-cli -- --help
```

## Troubleshooting

- **Linux build deps**: `apt install libgtk-3-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev`
- **GPU features fail**: The `gpu` feature requires wgpu support. Falls back to CPU automatically at runtime if GPU is unavailable. Build without `--features gpu` to skip GPU compilation entirely.
- **Coverage tool not found**: Install with `cargo install cargo-llvm-cov`
