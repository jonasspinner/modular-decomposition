use std::error::Error;
use std::path::{Path, PathBuf};
use clap::{Parser, ValueEnum};
use common::io::{read_metis, read_pace2023, write_metis, GraphFileType};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long)]
    input_type: GraphFileType,
    #[arg(long)]
    output_type: GraphFileType,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>>{
    println!("convert");
    let cli = Cli::parse();

    let graph = match cli.input_type {
        GraphFileType::Pace2023 => { read_pace2023(cli.input)? }
        GraphFileType::Metis => { read_metis(cli.input)? }
    };

    match cli.output_type {
        GraphFileType::Pace2023 => { unimplemented!() }
        GraphFileType::Metis => { write_metis(cli.output, &graph)? },
    }
    Ok(())
}