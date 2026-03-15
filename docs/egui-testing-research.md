# E2E & Visual Testing for Rust + egui

## Motivation

Jupiter GUI (`crates/jupiter-gui/`) currently has zero GUI-level tests. All testing is concentrated in `jupiter-core` (unit and integration tests for algorithms). As the GUI grows in complexity — panels, controls, pipeline interactions — we need a strategy for verifying UI behavior and catching visual regressions without relying solely on manual QA.

This article surveys the available e2e and visual testing solutions for Rust applications built with egui, evaluates their trade-offs, and recommends a testing strategy for Jupiter.

## Available Solutions

### 1. egui_kittest (Official)

- **Crate**: [`egui_kittest`](https://crates.io/crates/egui_kittest) (part of the egui ecosystem, by emilk)
- **Based on**: `kittest` by Rerun (AccessKit-based GUI testing library)
- **Introduced**: egui 0.30.0
- **Docs**: [docs.rs/egui_kittest](https://docs.rs/egui_kittest)

This is the official testing framework shipped with egui. It provides two main capabilities:

**Automation testing** — simulates clicks, keypresses, and other events via AccessKit widget tree queries. The API is similar to [Testing Library](https://testing-library.com/) in the JavaScript ecosystem: you find widgets by label, role, or other accessibility properties, then interact with them programmatically.

**Snapshot/visual regression testing** — renders UI to images via wgpu and compares against stored baselines. Requires enabling the `snapshot` and `wgpu` cargo features.

Additional features:
- **Widget queries** — find widgets by label, role, or other accessibility properties
- **Masking** — `Harness::mask()` to exclude volatile regions (e.g., timestamps) from snapshot comparisons
- **Tolerance control** — `SnapshotOptions::threshold` and pixel count tolerance for handling minor cross-platform/GPU rendering differences

### 2. egui-screenshot-testing (Community)

- **Crate**: [`egui-screenshot-testing`](https://lib.rs/crates/egui-screenshot-testing)

A community crate focused purely on visual comparison. It renders app state via a `TestBackend` and compares screenshots. Simpler API than egui_kittest but significantly less mature, with no automation/interaction capabilities.

### 3. kittest (Framework-agnostic)

- **Crate**: [`kittest`](https://github.com/rerun-io/kittest) (by Rerun)

An AccessKit-based GUI testing library that works with any Rust GUI framework, not just egui. `egui_kittest` is essentially a thin wrapper around `kittest` with egui-specific integrations. Useful to know about if you ever need to test non-egui Rust GUIs.

### 4. insta (Complementary — text/data snapshots)

- **Crate**: [`insta`](https://github.com/mitsuhiko/insta) (by mitsuhiko)

Snapshot testing for serializable data — not visual, but useful for testing UI state or output as structured text. The egui_kittest docs themselves recommend using insta for non-visual assertions where possible, since text snapshots are deterministic across platforms and much faster to run.

## How egui_kittest Works

### The Harness

The core abstraction is `Harness`, which creates an egui context without a real window. You provide a closure that builds your UI, and the harness manages the egui frame lifecycle.

```rust
use egui_kittest::Harness;

let mut harness = Harness::new_ui(|ui| {
    ui.label("Hello World");
    ui.button("Click me");
});
```

### Widget Queries via AccessKit

Under the hood, egui exposes its widget tree through [AccessKit](https://github.com/AccessKit/accesskit) (an accessibility toolkit). `kittest` queries this tree to find widgets, similar to how Testing Library queries the DOM in web testing:

```rust
// Find by label text
let button = harness.get_by_label("Click me");

// Check state
assert!(!button.is_selected());

// Interact
button.click();
harness.run(); // advance one frame to process the click
```

### Automation Test Example

```rust
use egui_kittest::Harness;

#[test]
fn test_checkbox_toggle() {
    let mut checked = false;
    let mut harness = Harness::new_ui(|ui| {
        ui.checkbox(&mut checked, "My Checkbox");
    });

    // Verify initial state
    let checkbox = harness.get_by_label("My Checkbox");
    assert!(!checkbox.is_selected());

    // Click and verify toggle
    checkbox.click();
    harness.run();

    assert!(harness.get_by_label("My Checkbox").is_selected());
}
```

### Snapshot Test Example

```rust
use egui_kittest::Harness;

// Requires: egui_kittest = { features = ["snapshot", "wgpu"] }

#[test]
fn test_layout_snapshot() {
    let harness = Harness::new_ui(|ui| {
        ui.heading("Pipeline Controls");
        ui.separator();
        ui.label("Frame selection: 50%");
        ui.button("Run Pipeline");
    });

    // Renders to image and compares against tests/snapshots/pipeline_controls.png
    // On first run, creates the baseline. On subsequent runs, diffs against it.
    harness.snapshot("pipeline_controls");
}
```

## Cross-Platform Caveats

Visual snapshot tests are the most fragile part of this stack:

- **GPU/driver variance** — different GPUs (and even different driver versions on the same GPU) produce slightly different renderings. Anti-aliasing, font rasterization, and subpixel positioning all vary.
- **OS differences** — font fallback chains, system DPI, and default font rendering differ between macOS, Linux, and Windows.
- **Brittleness** — unrelated style changes (e.g., tweaking a color constant) can break snapshots across the entire test suite.

Mitigations:
- Use `SnapshotOptions::threshold` to allow small per-pixel differences
- Use `Harness::mask()` to exclude regions that vary (timestamps, dynamic content)
- Run snapshot tests only on CI with a pinned OS/GPU configuration
- Prefer automation tests (state assertions) over snapshot tests where possible

## Comparison

| Feature                   | egui_kittest              | egui-screenshot-testing | kittest           |
|---------------------------|---------------------------|-------------------------|-------------------|
| Official egui support     | Yes                       | No                      | Underlying lib    |
| Automation (clicks, etc.) | Yes                       | No                      | Yes               |
| Visual snapshots          | Yes (via wgpu)            | Yes                     | No                |
| AccessKit integration     | Yes                       | No                      | Yes               |
| Maturity                  | Good (shipped with egui)  | Early                   | Good              |
| Cross-platform stability  | Fragile (GPU-dependent)   | Fragile                 | N/A (no visuals)  |

## Recommended Testing Strategy for Jupiter

A layered approach, ordered by reliability and speed:

1. **Regular unit tests** (already in place) — test core algorithms in `jupiter-core`. These are fast, deterministic, and the foundation of the test suite.

2. **egui_kittest automation tests** — test UI behavior: button clicks trigger correct state changes, panel interactions update the pipeline config, error states display correctly. These query the AccessKit widget tree and assert on state, so they are deterministic across platforms.

3. **egui_kittest snapshot tests** (sparingly) — for critical visual layouts only, such as the main control panel layout or the status bar. Use tolerance settings and masking. Run on a single pinned CI configuration to avoid cross-platform flakiness.

4. **insta text snapshots** (optional complement) — for testing serialized UI state or configuration output where visual appearance doesn't matter but data correctness does.

## Sources

- [egui_kittest docs](https://docs.rs/egui_kittest)
- [egui_kittest on crates.io](https://crates.io/crates/egui_kittest)
- [kittest repo](https://github.com/rerun-io/kittest)
- [egui issue #3926 — Improved test support](https://github.com/emilk/egui/issues/3926)
- [egui 0.30.0 release (introduced kittest)](https://github.com/emilk/egui/releases/tag/0.30.0)
- [egui-screenshot-testing](https://lib.rs/crates/egui-screenshot-testing)
- [insta snapshot testing](https://github.com/mitsuhiko/insta)
