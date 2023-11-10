use std::error::Error;
use std::ffi::OsStr;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io;
use std::io::{BufRead, BufWriter, Write};
use std::num::ParseIntError;
use std::path::Path;
use clap::ValueEnum;
use petgraph::adj::DefaultIx;
use petgraph::{Graph, Undirected};
use petgraph::graph::{NodeIndex, UnGraph};


#[derive(Debug, Clone, Eq, PartialEq, ValueEnum)]
pub enum GraphFileType {
    Pace2023,
    Metis,
}


pub enum Pace2023Error {
    NotPace2023Format,
    MissingHeader,
    WrongHeaderFormat,
    IoError(io::Error),
}

impl From<io::Error> for Pace2023Error {
    fn from(value: io::Error) -> Self { Pace2023Error::IoError(value) }
}

impl From<ParseIntError> for Pace2023Error {
    fn from(_value: ParseIntError) -> Self { Pace2023Error::NotPace2023Format }
}

impl Display for Pace2023Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Pace2023Error::NotPace2023Format => { write!(f, "not pace2023 format") }
            Pace2023Error::MissingHeader => { write!(f, "missing header") }
            Pace2023Error::WrongHeaderFormat => { write!(f, "wrong header format. expected 'p tww (n) (m)'") }
            Pace2023Error::IoError(err) => { write!(f, "{}", err) }
        }
    }
}

impl Debug for Pace2023Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Error for Pace2023Error {}

pub fn read_pace2023<P>(path: P) -> Result<UnGraph<(), ()>, Pace2023Error>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();
    let first_line = lines.next().ok_or(Pace2023Error::MissingHeader)??;
    let mut iter = first_line.split_ascii_whitespace();

    let p = iter.next().ok_or(Pace2023Error::NotPace2023Format)?;
    let tww = iter.next().ok_or(Pace2023Error::NotPace2023Format)?;
    if p != "p" || tww != "tww" {
        return Err(Pace2023Error::WrongHeaderFormat);
    }
    let n: usize = iter.next().ok_or(Pace2023Error::WrongHeaderFormat)?.parse()?;
    let m: usize = iter.next().ok_or(Pace2023Error::WrongHeaderFormat)?.parse()?;

    let mut graph = UnGraph::with_capacity(n, m);
    for _ in 0..n { graph.add_node(()); }

    for line in lines {
        let line = line?;
        let mut iter = line.split_ascii_whitespace();
        let mut next_number = || -> Result<u32, Pace2023Error> { Ok(iter.next().ok_or(Pace2023Error::NotPace2023Format)?.parse()?) };
        let u = next_number()? - 1;
        let v = next_number()? - 1;
        assert!(u < n as u32);
        assert!(v < n as u32);
        graph.add_edge(u.into(), v.into(), ());
    }
    Ok(graph)
}


pub enum ReadMetisError {
    NotMetisFormat,
    MissingHeader,
    WrongHeader(String),
    WrongN(usize, usize),
    WrongM(usize, usize),
    ZeroIndex,
    MissingReverseEdge(usize, usize),
    SelfLoop(usize),
    ParseInt(ParseIntError),
    IoError(io::Error),
}

impl From<io::Error> for ReadMetisError {
    fn from(value: io::Error) -> Self { ReadMetisError::IoError(value) }
}

impl From<ParseIntError> for ReadMetisError {
    fn from(value: ParseIntError) -> Self { ReadMetisError::ParseInt(value) }
}

impl Display for ReadMetisError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadMetisError::NotMetisFormat => { write!(f, "not metis format") }
            ReadMetisError::MissingHeader => { write!(f, "missing header") }
            ReadMetisError::WrongHeader(line) => { write!(f, "wrong header format. expected '(n) (m)', got '{line}'") }
            ReadMetisError::WrongN(actual, expected) => { write!(f, "wrong n given in header. expected {expected}, got {actual}") }
            ReadMetisError::WrongM(actual, expected) => { write!(f, "wrong m given in header. expected {expected}, got {actual}") }
            ReadMetisError::ZeroIndex => { write!(f, "zero index. metis indices start from 1") }
            ReadMetisError::MissingReverseEdge(u, v) => { write!(f, "missing reverse edge ({u} {v})") }
            ReadMetisError::SelfLoop(u) => { write!(f, "self loop not allowed ({u}, {u})") }
            ReadMetisError::ParseInt(err) => { write!(f, "parse error {err}") }
            ReadMetisError::IoError(err) => { write!(f, "io error {err}") }
        }
    }
}

impl Debug for ReadMetisError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Error for ReadMetisError {}

/// Implements a subset of the metis graph format. See [metis].
///
/// The format supports undirected, unweighted, loop-less graphs and specifies the number of nodes
/// and the number edges in its header. The vertices indices start with 1.
///
/// The restrictions are
/// + only allows comments before header line
/// + does only allow '(n) (m)' as header
/// + does not support vertex and edge weights
///
/// [metis]: https://people.sc.fsu.edu/~jburkardt/data/metis_graph/metis_graph.html
pub fn read_metis<P>(path: P) -> Result<UnGraph<(), ()>, ReadMetisError>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines().enumerate();

    let first_line = loop {
        let line = lines.next().ok_or(ReadMetisError::MissingHeader)?.1?;
        if !line.starts_with('%') { break line; }
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

    for _ in 0..n { graph.add_node(()); }

    let mut u = 1;
    for (_line_idx, line) in lines {
        let line = line?;
        if line.starts_with('%') { unimplemented!("cannot handle later comment lines") }
        for v in line.split_ascii_whitespace() {
            let v: usize = v.parse()?;
            if v == 0 {
                return Err(ReadMetisError::ZeroIndex);
            }
            let (u, v) = (NodeIndex::new(u - 1), NodeIndex::new(v - 1));
            if u < v {
                graph.add_edge(u, v, ());
            } else if u == v {
                return Err(ReadMetisError::SelfLoop(u.index() + 1));
            } else if graph.find_edge(u, v).is_none() {
                return Err(ReadMetisError::MissingReverseEdge(u.index() + 1, v.index() + 1));
            }
        }
        u += 1;
    }

    let n_actual = u - 1;
    if n_actual != n_expected {
        return Err(ReadMetisError::WrongN(n_actual, n_expected));
    }
    if graph.edge_count() != m_expected {
        return Err(ReadMetisError::WrongM(graph.edge_count(), m_expected));
    }

    Ok(graph)
}

#[derive(Debug)]
pub enum WriteMetisError {
    IoError(io::Error),
}

impl From<io::Error> for WriteMetisError { fn from(value: io::Error) -> Self { WriteMetisError::IoError(value) } }

impl Display for WriteMetisError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteMetisError::IoError(err) => { write!(f, "{}", err) }
        }
    }
}

impl Error for WriteMetisError {}

pub fn write_metis<P>(path: P, graph: &UnGraph<(), ()>) -> Result<(), WriteMetisError> where P: AsRef<Path> {
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
    Ok(())
}

#[allow(dead_code)]
fn read_edge_list<P>(path: P) -> Result<UnGraph<(), ()>, Box<dyn Error>>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let lines = io::BufReader::new(file).lines();

    let mut edges = vec![];
    for line in lines {
        let line = line?;
        let mut iter = line.split_ascii_whitespace();
        let mut next_number = || -> Result<u32, Box<dyn Error>> { Ok(iter.next().ok_or("cannot read number")?.parse()?) };
        let u = next_number()?;
        let v = next_number()?;
        edges.push((u, v));
    }
    let mut graph = UnGraph::with_capacity(0, 0);
    graph.extend_with_edges(edges);
    Ok(graph)
}

#[allow(dead_code)]
fn read_dat<P>(path: P) -> Result<UnGraph<(), (), DefaultIx>, Box<dyn Error>>
    where P: AsRef<Path> {
    if path.as_ref().extension().and_then(OsStr::to_str) != Some("dat") {
        return Err("extension is not .dat".into());
    }

    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();

    let first_line = lines.next().ok_or("cannot read first line")??;
    let mut iter = first_line.split_ascii_whitespace();
    let n: usize = iter.next().ok_or("no n in first line")?.parse()?;
    let m: usize = iter.next().ok_or("no m in first line")?.parse()?;
    if iter.next().is_some() { return Err("unrecognized value in first line".into()); }

    let mut graph = Graph::<(), (), Undirected, DefaultIx>::with_capacity(n, m);

    for _ in 0..n {
        graph.add_node(());
    }

    for (u, line) in lines.enumerate() {
        assert!(u < n);

        let line = line.map_err(|_| "cannot read line")?;
        let mut iter = line.split_ascii_whitespace();

        let d: usize = iter.next().ok_or("cannot read degree")?.parse()?;

        for _ in 0..d {
            let v: usize = iter.next().ok_or("missing neighbor")?.parse()?;
            assert!(v < n);
            if u < v {
                graph.add_edge(NodeIndex::new(u), NodeIndex::new(v), ());
            } else {
                println!("{} {}", u, v);
                assert!(graph.find_edge(NodeIndex::new(u), NodeIndex::new(v)).is_some());
            }
        }
        if iter.next().is_some() { return Err("to many neighbors in line".into()); }
    }

    assert_eq!(graph.node_count(), n);
    assert_eq!(graph.edge_count(), m);

    Ok(graph)
}