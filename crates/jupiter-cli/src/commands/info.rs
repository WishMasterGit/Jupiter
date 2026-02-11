use std::path::PathBuf;

use anyhow::Result;
use clap::Args;
use jupiter_core::io::ser::SerReader;

#[derive(Args)]
pub struct InfoArgs {
    /// Input SER file
    pub file: PathBuf,
}

pub fn run(args: &InfoArgs) -> Result<()> {
    let reader = SerReader::open(&args.file)?;
    let info = reader.source_info(&args.file);

    println!("File:        {}", info.filename.display());
    println!("Frames:      {}", info.total_frames);
    println!("Dimensions:  {}x{}", info.width, info.height);
    println!("Bit depth:   {}", info.bit_depth);
    println!("Color mode:  {:?}", info.color_mode);

    if let Some(ref obs) = info.observer {
        println!("Observer:    {}", obs);
    }
    if let Some(ref tel) = info.telescope {
        println!("Telescope:   {}", tel);
    }
    if let Some(ref inst) = info.instrument {
        println!("Instrument:  {}", inst);
    }

    let frame_bytes = reader.header.frame_byte_size();
    let total_mb = (frame_bytes * info.total_frames) as f64 / (1024.0 * 1024.0);
    println!("Data size:   {:.1} MB", total_mb);

    Ok(())
}
