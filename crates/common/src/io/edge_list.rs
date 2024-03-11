use petgraph::graph::UnGraph;
use std::fs::File;
use std::io::BufRead;
use std::num::ParseIntError;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReadEdgeListError {
    #[error("invalid line (got {0})")]
    InvalidLine(String),
    #[error("parse int error")]
    ParseInt(#[from] ParseIntError),
    #[error("io error")]
    IoError(#[from] std::io::Error),
}

pub fn read_edge_list<P>(path: P) -> Result<UnGraph<(), ()>, ReadEdgeListError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;

    let mut n = 0;
    let mut edges = vec![];
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        let mut tokens = line.split_ascii_whitespace();
        let Some(a) = tokens.next() else {
            return Err(ReadEdgeListError::InvalidLine(line));
        };
        let Some(b) = tokens.next() else {
            return Err(ReadEdgeListError::InvalidLine(line));
        };
        if tokens.next().is_some() {
            return Err(ReadEdgeListError::InvalidLine(line));
        }

        let u: u32 = a.parse()?;
        let v: u32 = b.parse()?;
        edges.push((u, v));
        n = n.max(u + 1).max(v + 1);
    }

    let mut graph = UnGraph::with_capacity(n as usize, edges.len());
    graph.extend_with_edges(edges);
    Ok(graph)
}
