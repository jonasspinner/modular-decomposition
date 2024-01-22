use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use clap::{Parser, ValueEnum};
use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::FmtSubscriber;
use tracing_subscriber::util::SubscriberInitExt;
use common::io::{GraphFileType, read_edgelist, read_metis, read_pace2023, write_md_tree_adj};


#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum Algo {
    Miz23Rust,
    Miz23Cpp,
    MS00,
    Kar19Rust,
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long)]
    input_type: GraphFileType,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    stats: Option<PathBuf>,
    #[arg(long, value_enum)]
    algo: Algo,
    #[arg(long, value_enum)]
    log_level: Option<Level>,
}


fn write_stats(stats: &Option<PathBuf>, input: &Path, algo: Algo, time: Option<std::time::Duration>, status: &str) -> Result<(), Box<dyn Error>> {
    let Some(stats) = stats else { return Ok(()); };
    let mut out = BufWriter::new(File::create(stats)?);
    let input = input.file_name().unwrap().to_string_lossy();
    let algo = algo.to_possible_value().unwrap().get_name().to_string();
    let time = time.map(|t| t.as_secs_f64()).unwrap_or(f64::NAN);
    writeln!(out, "{{\"input\": \"{input}\", \"algo\": \"{algo}\", \"time\": {time}, \"status\": \"{status}\"}}")?;
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    write_stats(&cli.stats, &cli.input, cli.algo, None, "unfinished")?;
    let graph = match cli.input_type {
        GraphFileType::Pace2023 => read_pace2023(&cli.input)?,
        GraphFileType::Metis => read_metis(&cli.input)?,
        GraphFileType::EdgeList => read_edgelist(&cli.input)?,
    };

    if let Some(level) = cli.log_level {
        let subscriber = FmtSubscriber::builder()
            .with_max_level(level)
            .with_span_events(FmtSpan::CLOSE)
            //.json()
            .finish();

        subscriber.init();
    };

    let (t, md) = match cli.algo {
        Algo::Miz23Rust => {
            let problem = miz23_md_rs::prepare(&graph);
            let start = Instant::now();
            let result = problem.compute();
            (start.elapsed(), result.finalize())
        }
        Algo::Miz23Cpp => {
            let problem = miz23_md_cpp::prepare(&graph);
            let start = Instant::now();
            let result = problem.compute();
            (start.elapsed(), result.finalize())
        }
        Algo::MS00 => {
            let problem = ms00::prepare(&graph);
            let start = Instant::now();
            let result = problem.compute();
            (start.elapsed(), result.finalize())
        }
        Algo::Kar19Rust => {
            let start = Instant::now();
            let md = kar19_rs::modular_decomposition(&graph);
            (start.elapsed(), md)
        }
    };

    if let Some(output) = cli.output {
        let mut out = BufWriter::new(File::create(output)?);
        write_md_tree_adj(&mut out, &md)?;
    }
    write_stats(&cli.stats, &cli.input, cli.algo, Some(t), "finished")?;
    Ok(())
}