mod commands;

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
    /// Run the full processing pipeline
    Run(commands::pipeline::RunArgs),
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
        Commands::Run(args) => commands::pipeline::run(args),
    }
}
