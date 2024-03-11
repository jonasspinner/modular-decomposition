use petgraph::graph::UnGraph;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::num::ParseIntError;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Pace2023Error {
    #[error("not pace2023 format")]
    InvalidFormat,
    #[error("missing header")]
    MissingHeader,
    #[error("invalid header (expected 'p tww (n) (m)')")]
    InvalidHeader,
    #[error("could not read file")]
    IoError(#[from] io::Error),
}

impl From<ParseIntError> for Pace2023Error {
    fn from(_value: ParseIntError) -> Self {
        Pace2023Error::InvalidFormat
    }
}

pub fn read_pace2023<P>(path: P) -> Result<UnGraph<(), ()>, Pace2023Error>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();
    let first_line = lines.next().ok_or(Pace2023Error::MissingHeader)??;
    let mut iter = first_line.split_ascii_whitespace();

    let p = iter.next().ok_or(Pace2023Error::InvalidHeader)?;
    let tww = iter.next().ok_or(Pace2023Error::InvalidHeader)?;
    if p != "p" || tww != "tww" {
        return Err(Pace2023Error::InvalidHeader);
    }
    let n: usize = iter.next().ok_or(Pace2023Error::InvalidHeader)?.parse()?;
    let m: usize = iter.next().ok_or(Pace2023Error::InvalidHeader)?.parse()?;

    let mut graph = UnGraph::with_capacity(n, m);
    for _ in 0..n {
        graph.add_node(());
    }

    for line in lines {
        let line = line?;
        let mut iter = line.split_ascii_whitespace();
        let mut next_number =
            || -> Result<u32, Pace2023Error> { Ok(iter.next().ok_or(Pace2023Error::InvalidFormat)?.parse()?) };
        let u = next_number()? - 1;
        let v = next_number()? - 1;
        assert!(u < n as u32);
        assert!(v < n as u32);
        graph.add_edge(u.into(), v.into(), ());
    }
    Ok(graph)
}
