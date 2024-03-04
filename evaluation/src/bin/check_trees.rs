use std::error::Error;
use std::path::PathBuf;
use clap::Parser;
use common::io::read_md_tree_adj;
use canonicalize::canonicalize;

#[path = "../canonicalize.rs"]
mod canonicalize;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(num_args(0..))]
    tree_paths: Vec<PathBuf>,
}


fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let trees : Result<Vec<_>, _> = cli.tree_paths.iter().map(read_md_tree_adj).collect();
    let trees = trees?;

    let trees : Vec<_> = trees.iter().map(canonicalize).collect();
    for i in 1..trees.len() {
        assert_eq!(trees[0], trees[i]);
    }
    Ok(())
}