mod commands;
mod summary;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "jupiter", about = "Planetary image processing tool")]
#[command(version)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show SER/image file metadata
    Info(commands::info::InfoArgs),
    /// Score and rank frames by quality
    Quality(commands::quality::QualityArgs),
    /// Align and stack the best frames
    Stack(commands::stack::StackArgs),
    /// Apply wavelet sharpening to an image
    Sharpen(commands::sharpen::SharpenArgs),
    /// Apply post-processing filters to an image
    Filter(commands::filter::FilterArgs),
    /// Run the full processing pipeline
    Run(commands::pipeline::RunArgs),
    /// Print a default pipeline config as TOML
    Config,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("warn")
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match &cli.command {
        Commands::Info(args) => commands::info::run(args),
        Commands::Quality(args) => commands::quality::run(args),
        Commands::Stack(args) => commands::stack::run(args),
        Commands::Sharpen(args) => commands::sharpen::run(args),
        Commands::Filter(args) => commands::filter::run(args),
        Commands::Run(args) => commands::pipeline::run(args),
        Commands::Config => commands::config::run(),
    }
}
