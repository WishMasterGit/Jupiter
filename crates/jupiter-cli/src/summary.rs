use console::Style;
use jupiter_core::pipeline::config::{DeconvolutionConfig, PipelineConfig, StackMethod};
use jupiter_core::sharpen::wavelet::WaveletParams;
use jupiter_core::stack::multi_point::MultiPointConfig;
use jupiter_core::stack::sigma_clip::SigmaClipParams;

struct Styles {
    title: Style,
    header: Style,
    label: Style,
    value: Style,
    method: Style,
    disabled: Style,
    path: Style,
}

impl Styles {
    fn new() -> Self {
        Self {
            title: Style::new().cyan().bold(),
            header: Style::new().cyan().bold(),
            label: Style::new().dim(),
            value: Style::new().bold().white(),
            method: Style::new().green(),
            disabled: Style::new().dim().yellow(),
            path: Style::new().underlined(),
        }
    }
}

pub fn print_pipeline_summary(config: &PipelineConfig) {
    let s = Styles::new();

    println!();
    println!("  {}", s.title.apply_to("Jupiter Pipeline"));
    println!("  {}", s.title.apply_to("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"));
    println!();

    // Input / Output
    println!(
        "  {:<14}{}",
        s.label.apply_to("Input"),
        s.path.apply_to(config.input.display())
    );
    println!(
        "  {:<14}{}",
        s.label.apply_to("Output"),
        s.path.apply_to(config.output.display())
    );
    println!();

    // Frame Selection
    println!("  {}", s.header.apply_to("Frame Selection"));
    println!(
        "    {:<12}{}",
        s.label.apply_to("Metric"),
        s.value.apply_to(&config.frame_selection.metric)
    );
    println!(
        "    {:<12}{}",
        s.label.apply_to("Keep"),
        s.value.apply_to(format!(
            "{:.0}%",
            config.frame_selection.select_percentage * 100.0
        ))
    );
    println!();

    // Stacking
    println!("  {}", s.header.apply_to("Stacking"));
    println!(
        "    {:<12}{}",
        s.label.apply_to("Method"),
        s.method.apply_to(&config.stacking.method)
    );
    print_stack_sub_params(&s, &config.stacking.method);
    println!();

    // Sharpening
    if let Some(ref sharp) = config.sharpening {
        print_sharpening_section(&s, &sharp.wavelet, sharp.deconvolution.as_ref());
    } else {
        println!(
            "  {:<14}{}",
            s.header.apply_to("Sharpening"),
            s.disabled.apply_to("disabled")
        );
        println!();
    }

    // Filters
    if config.filters.is_empty() {
        println!(
            "  {:<14}{}",
            s.header.apply_to("Filters"),
            s.disabled.apply_to("none")
        );
    } else {
        println!("  {}", s.header.apply_to("Filters"));
        for (i, filter) in config.filters.iter().enumerate() {
            println!(
                "    {}. {}",
                s.label.apply_to(i + 1),
                s.value.apply_to(filter)
            );
        }
    }
    println!();
}

pub fn print_sharpen_summary(
    params: &WaveletParams,
    deconv: Option<&DeconvolutionConfig>,
) {
    let s = Styles::new();

    println!();
    println!("  {}", s.title.apply_to("Sharpening"));
    println!("  {}", s.title.apply_to("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}"));
    println!();

    print_sharpening_section(&s, params, deconv);
}

fn print_sharpening_section(
    s: &Styles,
    params: &WaveletParams,
    deconv: Option<&DeconvolutionConfig>,
) {
    println!("  {}", s.header.apply_to("Sharpening"));
    println!(
        "    {:<14}{}",
        s.label.apply_to("Wavelet"),
        s.value.apply_to(format!("{} layers", params.num_layers))
    );
    println!(
        "    {:<14}{:?}",
        s.label.apply_to("Coefficients"),
        params.coefficients
    );
    if params.denoise.is_empty() {
        println!(
            "    {:<14}{}",
            s.label.apply_to("Denoise"),
            s.disabled.apply_to("disabled")
        );
    } else {
        println!(
            "    {:<14}{:?}",
            s.label.apply_to("Denoise"),
            params.denoise
        );
    }
    println!();

    // Deconvolution
    if let Some(deconv) = deconv {
        println!("  {}", s.header.apply_to("Deconvolution"));
        println!(
            "    {:<14}{}",
            s.label.apply_to("Method"),
            s.method.apply_to(&deconv.method)
        );
        println!(
            "    {:<14}{}",
            s.label.apply_to("PSF"),
            s.value.apply_to(&deconv.psf)
        );
        println!();
    } else {
        println!(
            "  {:<14}{}",
            s.header.apply_to("Deconvolution"),
            s.disabled.apply_to("disabled")
        );
        println!();
    }
}

fn print_stack_sub_params(s: &Styles, method: &StackMethod) {
    match method {
        StackMethod::SigmaClip(SigmaClipParams { sigma, iterations }) => {
            println!(
                "    {:<12}{}",
                s.label.apply_to("Sigma"),
                s.value.apply_to(sigma)
            );
            println!(
                "    {:<12}{}",
                s.label.apply_to("Iterations"),
                s.value.apply_to(iterations)
            );
        }
        StackMethod::MultiPoint(MultiPointConfig {
            ap_size,
            search_radius,
            select_percentage,
            min_brightness,
            quality_metric,
            local_stack_method,
        }) => {
            println!(
                "    {:<12}{}",
                s.label.apply_to("AP Size"),
                s.value.apply_to(format!("{ap_size} px"))
            );
            println!(
                "    {:<12}{}",
                s.label.apply_to("Search"),
                s.value.apply_to(format!("{search_radius} px"))
            );
            println!(
                "    {:<12}{}",
                s.label.apply_to("Keep"),
                s.value.apply_to(format!("{:.0}%", select_percentage * 100.0))
            );
            println!(
                "    {:<12}{}",
                s.label.apply_to("Min Bright"),
                s.value.apply_to(min_brightness)
            );
            println!(
                "    {:<12}{}",
                s.label.apply_to("Metric"),
                s.value.apply_to(quality_metric)
            );
            println!(
                "    {:<12}{}",
                s.label.apply_to("Local Stack"),
                s.method.apply_to(local_stack_method)
            );
        }
        _ => {}
    }
}
