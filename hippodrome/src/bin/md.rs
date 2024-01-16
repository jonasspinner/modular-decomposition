use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use clap::{Parser, ValueEnum};
use petgraph::visit::IntoNodeReferences;
use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::FmtSubscriber;
use tracing_subscriber::util::SubscriberInitExt;
use common::io::{GraphFileType, read_edgelist, read_metis, read_pace2023, write_md_tree_adj};
use common::modular_decomposition::MDNodeKind;


#[derive(Debug, Clone, Eq, PartialEq, ValueEnum)]
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


fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let graph = match cli.input_type {
        GraphFileType::Pace2023 => read_pace2023(&cli.input)?,
        GraphFileType::Metis => read_metis(&cli.input)?,
        GraphFileType::EdgeList => read_edgelist(&cli.input)?,
    };

    if let Some(level) = cli.log_level {
        let subscriber = FmtSubscriber::builder()
            .with_max_level(level)
            .with_span_events(FmtSpan::CLOSE)
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

    let (prime, series, parallel, vertex) = md.node_references()
        .fold((0, 0, 0, 0),
              |(prime, series, parallel, vertex), (_, k)| {
                  match k {
                      MDNodeKind::Prime => (prime + 1, series, parallel, vertex),
                      MDNodeKind::Series => (prime, series + 1, parallel, vertex),
                      MDNodeKind::Parallel => (prime, series, parallel + 1, vertex),
                      MDNodeKind::Vertex(_) => (prime, series, parallel, vertex + 1)
                  }
              });

    if let Some(output) = cli.output {
        let mut out = BufWriter::new(File::create(output)?);
        write_md_tree_adj(&mut out, &md)?;
    }
    if let Some(stats) = cli.stats {
        let mut out = BufWriter::new(File::create(stats)?);
        writeln!(out, "input {}", cli.input.file_name().unwrap().to_string_lossy())?;
        writeln!(out, "algo {:#?}", cli.algo.to_possible_value().unwrap().get_name())?;
        writeln!(out, "time {}", t.as_micros())?;
        writeln!(out, "num_prime {}", prime)?;
        writeln!(out, "num_series {}", series)?;
        writeln!(out, "num_parallel {}", parallel)?;
        writeln!(out, "num_vertex {}", vertex)?;
    }
    Ok(())
}