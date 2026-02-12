# Rust Desktop UI Frameworks: A 2026 Research Guide

The Rust GUI ecosystem has grown significantly — a [2025 survey by boringcactus](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html) catalogued **43 libraries**. However, most aren't production-ready. Below are the frameworks that actually matter for desktop (macOS / Windows / Linux).

---

## Tier 1: Production-Viable

### Slint
**Architecture:** Declarative DSL (`.slint` files) compiled to native code
**Rendering:** OpenGL, software fallback
**Version:** 1.x stable (last updated Jan 2026)
**GitHub:** [slint-ui/slint](https://github.com/slint-ui/slint)

Slint is the most **mature and stable** option. It has a dedicated DSL for UI layout that compiles to native code, VS Code tooling with live preview, and is the only Rust GUI framework at a stable 1.x release with a commitment to no breaking changes.

- **Licensing:** Free (royalty-free) for desktop apps, even proprietary ones. GPL v3 also available. Commercial license only needed for embedded.
- **Accessibility:** Best-in-class — perfect screen reader (Narrator) support on Windows per the boringcactus survey.
- **Tradeoff:** You must learn the `.slint` DSL. Business logic stays in Rust, but the UI layer is a separate language. Also supports C++, JS, Python bindings if you ever need them.
- **Runtime footprint:** <300 KiB RAM.

**Best for:** Apps that need polished UI, accessibility, and long-term API stability.

---

### Dioxus
**Architecture:** React/JSX-like (RSX macros), virtual DOM
**Rendering:** System WebView (WebView2 on Windows, WebKitGTK on Linux)
**Version:** 0.7.3 (Jan 2026)
**GitHub:** [DioxusLabs/dioxus](https://github.com/DioxusLabs/dioxus)

Dioxus brings a React-like developer experience to Rust. Version 0.7 introduced **Rust hot-patching** (edit Rust code, see changes without losing state), Tailwind CSS integration, and Radix UI components.

- **Accessibility:** Strong — good screen reader and IME support.
- **Developer experience:** Excellent. If you know React, you're productive immediately. CLI handles building, bundling, and serving.
- **Tradeoff:** Uses system WebView under the hood ("Diet Electron"). Your UI runs in a web rendering engine, not a native widget set. This means web-class layout flexibility but also web-class overhead.
- **Team:** Small fulltime team backed by FutureWei, Satellite.im, and GitHub Accelerator.

**Best for:** Developers comfortable with web/React paradigms who want to use Rust for everything.

---

### egui
**Architecture:** Immediate mode (no retained widget tree)
**Rendering:** OpenGL (glow) or wgpu
**Version:** Active development, requires Rust 1.88+
**GitHub:** [emilk/egui](https://github.com/emilk/egui)

egui is the **easiest to get started with** — no DSL, no macros, no build steps. You describe your UI every frame in plain Rust. Widely used in game dev tooling, debug panels, and data visualization.

- **Accessibility:** Reasonable screen reader support, but IME (international text input) is broken.
- **Performance:** 1-2ms per frame typical. Scales well for tool-type UIs but not ideal for text-heavy or document-editing apps.
- **Tradeoff:** Immediate mode means the entire UI redraws every frame. Looks consistent across platforms but not "native". No complex text editing out of the box.
- **Ecosystem:** Used as the UI for [Rerun](https://rerun.io/), a well-funded data visualization startup, which validates its production use.

**Best for:** Developer tools, data visualization, debug/inspection UIs, and apps where fast iteration matters more than native look.

---

## Tier 2: Strong Contenders with Caveats

### Iced
**Architecture:** Elm-inspired (Model → Message → Update → View)
**Rendering:** wgpu (GPU-accelerated)
**Version:** 0.14 (Dec 2025), experimental
**GitHub:** [iced-rs/iced](https://github.com/iced-rs/iced)

Iced is the only Rust toolkit that **solves application architecture upfront** with its Elm-style message passing. This makes complex state management clean and predictable.

- **Rendering:** Pure wgpu — custom rendering, not system WebView or native widgets. Looks the same on all platforms.
- **Tradeoff:** **No accessibility support** (no screen reader, no IME). Still labeled "experimental" by maintainers. Breaking changes between versions.
- **Community:** Sponsored by the Cryptowatch team at Kraken.

**Best for:** Apps where you value architectural correctness and don't need accessibility (internal tools, personal projects).

---

### Tauri
**Architecture:** Web frontend + Rust backend, IPC bridge
**Rendering:** System WebView
**Version:** 2.0 (stable, Jan 2025)
**GitHub:** [tauri-apps/tauri](https://github.com/tauri-apps/tauri) — 83k+ stars

Tauri is the **most popular** option by GitHub stars. It's essentially "Electron but with Rust backend and system WebView instead of bundled Chromium." Much smaller binaries, lower memory usage.

- **Frontend:** Any web framework (React, Svelte, Vue, etc.) — your UI is HTML/CSS/JS.
- **Tradeoff:** Your app is split-brain: JS frontend communicating with Rust backend over IPC. You lose Rust's compile-time type safety at the boundary. The boringcactus survey author specifically [called this out](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html) as a significant downside.
- **When it makes sense:** If you already have a web UI or a team that knows web tech, and you just want Rust for the backend/system access layer.

**Best for:** Teams with web frontend expertise who want smaller binaries than Electron.

---

### Freya
**Architecture:** Dioxus component model + custom Skia rendering
**Rendering:** Skia (no WebView)
**Version:** 0.3.x, heavy rewrite in progress
**GitHub:** [marc2332/freya](https://github.com/marc2332/freya)

Freya takes Dioxus's component model but throws away the WebView, replacing it with a custom rendering pipeline built on Skia. This gives you Dioxus's developer experience with native-quality rendering.

- **Features:** Built-in scroll views, animations, rich text editing, terminal emulator integration.
- **Tradeoff:** **Not production-ready.** Major rewrite means the main branch diverges significantly from stable releases. Partial accessibility.

**Best for:** Watching and experimenting with — could become the best of both worlds (Dioxus DX + native rendering) once it stabilizes.

---

## Tier 3: Worth Watching

### Xilem (Linebender)
**Architecture:** SwiftUI-inspired declarative, diffing views
**Rendering:** Vello (GPU vector renderer)
**Version:** Alpha, no stable crate release
**GitHub:** [linebender/xilem](https://github.com/linebender/xilem)

Xilem is backed by serious infrastructure: Vello for GPU rendering, Parley for text layout, Masonry for the widget system. The architecture looks excellent on paper, but it's still in alpha with no stable release.

**Best for:** Following if you care about the long-term future of pure-Rust native GUI.

### GTK 4 (via gtk-rs)
The mature option with decades of widget development — but accessibility is **completely broken on Windows** (screen readers can't even see window chrome), and it looks out of place on macOS/Windows. Only really viable if you're targeting Linux exclusively.

---

## Comparison Matrix

| Framework | Rendering | Architecture | Stable? | Accessibility | Native Look | Learning Curve |
|-----------|-----------|-------------|---------|---------------|-------------|----------------|
| **Slint** | OpenGL/SW | Declarative DSL | 1.x | Excellent | Custom (polished) | Medium (DSL) |
| **Dioxus** | WebView | React-like RSX | 0.7 | Good | Web-style | Low (if React exp.) |
| **egui** | OpenGL/wgpu | Immediate mode | ~stable | Partial | Custom | Low |
| **Iced** | wgpu | Elm MVU | 0.14 | None | Custom | Medium |
| **Tauri** | WebView | Web + Rust IPC | 2.0 | Web-dependent | Web-style | Low (web devs) |
| **Freya** | Skia | Dioxus components | 0.3 | Partial | Custom | Low-Medium |
| **Xilem** | Vello GPU | SwiftUI-like | Alpha | WIP | Custom | Medium |

---

## Recommendations by Use Case

**Image processing app (like Jupiter):**
**Slint** or **egui** are the strongest fits. Slint if you want a polished user-facing app with controls, panels, and accessibility. egui if you want fast iteration, easy integration of custom rendering (image previews, histograms), and don't mind the immediate-mode style.

**Data-heavy / visualization tool:**
**egui** — immediate mode shines for real-time data display, and it integrates naturally with wgpu for custom rendering.

**App with complex forms / text:**
**Slint** or **Dioxus** — both handle text input and complex layouts well.

**Team with web experience:**
**Dioxus** (all-Rust) or **Tauri** (keep your existing web stack).

---

## The Honest Summary

No Rust GUI framework is as mature as Qt, WPF, or SwiftUI. The ecosystem improved dramatically from 2023-2026, but there's still no "obviously correct choice" ([boringcactus's conclusion](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html)). Slint comes closest to a production-grade toolkit with its 1.x stability guarantee and accessibility. egui is the pragmatic choice for developer tools. Everything else involves meaningful trade-offs or stability risk.

---

**Sources:**
- [A 2025 Survey of Rust GUI Libraries — boringcactus](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html)
- [Are we GUI yet?](https://areweguiyet.com/)
- [Slint — Declarative GUI for Rust](https://slint.dev/)
- [Slint Pricing / Licensing](https://slint.dev/pricing)
- [Dioxus — Fullstack crossplatform app framework](https://dioxuslabs.com/)
- [Dioxus 0.7 release](https://medium.com/@trivajay259/dioxus-0-7-the-rust-ui-release-that-finally-feels-full-stack-everywhere-89f482ee97e3)
- [egui — GitHub](https://github.com/emilk/egui)
- [Iced — GitHub](https://github.com/iced-rs/iced)
- [Tauri 2.0](https://v2.tauri.app/)
- [Freya — GitHub](https://github.com/marc2332/freya)
- [Xilem / Linebender](https://github.com/linebender/xilem)
- [The State of Rust GUI — Rust Bytes](https://weeklyrust.substack.com/p/the-state-of-rust-gui-the-good-and)
- [Making Slint Desktop-Ready — Slint Blog](https://slint.dev/blog/making-slint-desktop-ready)
