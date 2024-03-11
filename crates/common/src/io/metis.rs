use petgraph::graph::{NodeIndex, UnGraph};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::num::ParseIntError;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReadMetisError {
    #[error("invalid format")]
    NotMetisFormat,
    #[error("missing header")]
    MissingHeader,
    #[error("invalid header (expected '(n) (m)', got {0})")]
    WrongHeader(String),
    #[error("wrong n given in header (expected {expected}, got {actual})")]
    WrongN { expected: usize, actual: usize },
    #[error("wrong m given in header (expected {expected}, got {actual})")]
    WrongM { expected: usize, actual: usize },
    #[error("found zero index (indices must be at least 1)")]
    ZeroIndex,
    #[error("number of reverse edges not matching (expected {expected}, got {actual})")]
    NumReverseEdgesNotMatching { expected: usize, actual: usize },
    #[error("found self loop")]
    SelfLoop(usize),
    #[error("parse error")]
    ParseInt(#[from] ParseIntError),
    #[error("io error")]
    IoError(#[from] std::io::Error),
}

/// Implements a subset of the metis graph format. See [metis].
///
/// The format supports undirected, unweighted, loop-less graphs and specifies
/// the number of nodes and the number edges in its header. The vertices indices
/// start with 1.
///
/// The restrictions are
/// + only allows comments before header line
/// + does only allow '(n) (m)' as header
/// + does not support vertex and edge weights
///
/// [metis]: https://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html
pub fn read_metis<P>(path: P) -> Result<UnGraph<(), ()>, ReadMetisError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let mut lines = std::io::BufReader::new(file).lines().enumerate();

    let first_line = loop {
        let line = lines.next().ok_or(ReadMetisError::MissingHeader)?.1?;
        if !line.starts_with('%') {
            break line;
        }
    };

    let (n, m) = {
        let err = || ReadMetisError::WrongHeader(first_line.clone());
        let mut iter = first_line.split_ascii_whitespace();
        let n: usize = iter.next().ok_or_else(err)?.parse().map_err(|_| err())?;
        let m: usize = iter.next().ok_or_else(err)?.parse().map_err(|_| err())?;
        if iter.next().is_some() {
            unimplemented!("cannot handle more than 'n m' first line");
        }
        assert!(n < u32::MAX as _, "too large");
        assert!(m < u32::MAX as _, "too large");
        (n, m)
    };

    let mut graph = UnGraph::with_capacity(n, m);
    let (n_expected, m_expected) = (n, m);

    for _ in 0..n {
        graph.add_node(());
    }

    let mut m_reverse = 0;
    let mut u = 1;
    for (_line_idx, line) in lines {
        let line = line?;
        if line.starts_with('%') {
            unimplemented!("cannot handle later comment lines")
        }
        for v in line.split_ascii_whitespace() {
            let v: usize = v.parse()?;
            if v == 0 {
                return Err(ReadMetisError::ZeroIndex);
            }
            let (u, v) = (NodeIndex::new(u - 1), NodeIndex::new(v - 1));
            match u.cmp(&v) {
                Ordering::Less => {
                    graph.add_edge(u, v, ());
                }
                Ordering::Equal => return Err(ReadMetisError::SelfLoop(u.index() + 1)),
                Ordering::Greater => {
                    m_reverse += 1;
                }
            }
        }
        u += 1;
    }

    let n_actual = u - 1;
    if n_actual != n_expected {
        return Err(ReadMetisError::WrongN { actual: n_actual, expected: n_expected });
    }
    if graph.edge_count() != m_expected {
        return Err(ReadMetisError::WrongM { actual: graph.edge_count(), expected: m_expected });
    }
    if m_reverse != m_expected {
        return Err(ReadMetisError::NumReverseEdgesNotMatching { actual: m_reverse, expected: m_expected });
    }

    Ok(graph)
}

#[derive(Error, Debug)]
pub enum WriteMetisError {
    #[error("io error")]
    IoError(#[from] std::io::Error),
}

pub fn write_metis<P>(path: P, graph: &UnGraph<(), ()>) -> Result<(), WriteMetisError>
where
    P: AsRef<Path>,
{
    let file = File::create(path)?;
    let mut file = BufWriter::new(file);

    writeln!(file, "{} {}", graph.node_count(), graph.edge_count())?;
    for u in graph.node_indices() {
        let mut first = true;
        let mut neighbors: Vec<_> = graph.neighbors(u).collect();
        neighbors.sort();
        for v in neighbors {
            let v = v.index() + 1;
            if first {
                first = false;
            } else {
                write!(file, " ")?;
            }
            write!(file, "{v}")?;
        }
        if u.index() + 1 != graph.node_count() {
            writeln!(file)?;
        }
    }
    file.flush()?;
    Ok(())
}
