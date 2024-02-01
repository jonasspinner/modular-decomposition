use std::cmp::Ordering;
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
use petgraph::{Direction, Graph, Undirected};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use crate::modular_decomposition::MDNodeKind;


#[derive(Debug, Clone, Eq, PartialEq, ValueEnum)]
pub enum GraphFileType {
    Pace2023,
    Metis,
    EdgeList,
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
    NumReverseEdgesNotMatching(usize, usize),
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
            ReadMetisError::NumReverseEdgesNotMatching(actual, expected) => { write!(f, "number of reverse edges not matching. expected {expected}, got {actual}") }
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

    let mut m_reverse = 0;
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
            match u.cmp(&v) {
                Ordering::Less => { graph.add_edge(u, v, ()); }
                Ordering::Equal => return Err(ReadMetisError::SelfLoop(u.index() + 1)),
                Ordering::Greater => { m_reverse += 1; }
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
    if m_reverse != m_expected {
        return Err(ReadMetisError::NumReverseEdgesNotMatching(m_reverse, m_expected));
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
    file.flush()?;
    Ok(())
}



pub enum ReadMDTreeError {
    NotMDTreeFormat,
    MissingHeader,
    WrongM(usize, usize),
    WrongHeader(String),
    ParseInt(ParseIntError),
    IoError(io::Error),
}

impl From<io::Error> for ReadMDTreeError {
    fn from(value: io::Error) -> Self { ReadMDTreeError::IoError(value) }
}

impl From<ParseIntError> for ReadMDTreeError {
    fn from(value: ParseIntError) -> Self { ReadMDTreeError::ParseInt(value) }
}

impl Display for ReadMDTreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadMDTreeError::NotMDTreeFormat => { write!(f, "not metis format") }
            ReadMDTreeError::MissingHeader => { write!(f, "missing header") }
            ReadMDTreeError::WrongHeader(line) => { write!(f, "wrong header format. expected '(n) (m) 10', got '{line}'") }
            ReadMDTreeError::WrongM(actual, expected) => { write!(f, "wrong m given in header. expected {expected}, got {actual}") }
            ReadMDTreeError::ParseInt(err) => { write!(f, "parse error {err}") }
            ReadMDTreeError::IoError(err) => { write!(f, "io error {err}") }
        }
    }
}

impl Debug for ReadMDTreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Error for ReadMDTreeError {}

pub fn read_md_tree_adj<P>(path: P) -> Result<DiGraph<MDNodeKind, ()>, ReadMDTreeError>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();
    let first_line = loop {
        let line = lines.next().ok_or(ReadMDTreeError::MissingHeader)??;
        if !line.starts_with('%') { break line; }
    };

    let (n, m) = {
        let err = || ReadMDTreeError::WrongHeader(first_line.clone());
        let mut iter = first_line.split_ascii_whitespace();
        let n: usize = iter.next().ok_or_else(err)?.parse().map_err(|_| err())?;
        let m: usize = iter.next().ok_or_else(err)?.parse().map_err(|_| err())?;
        let fmt: usize = iter.next().ok_or_else(err)?.parse().map_err(|_| err())?;
        if fmt != 10 || iter.next().is_some() {
            return Err(err());
        }
        (n, m)
    };

    if n > 0 && m != n - 1 {
        return Err(ReadMDTreeError::WrongM(n - 1, m));
    }

    let mut tree = DiGraph::with_capacity(n, m);

    for _ in 0..n { tree.add_node(MDNodeKind::Vertex(usize::MAX)); }

    for (u, line) in lines.enumerate() {
        let line = line?;
        assert!(u < n, "n={n} m={m} u={u} line=\"{line}\"");
        let u = NodeIndex::new(u);
        if line.starts_with('%') { unimplemented!("cannot handle later comment lines") }
        let mut values = line.split_ascii_whitespace();
        let kind : usize = values.next().ok_or(ReadMDTreeError::NotMDTreeFormat)?.parse()?;
        let kind = match kind {
            0 => MDNodeKind::Prime,
            1 => MDNodeKind::Series,
            2 => MDNodeKind::Parallel,
            v => MDNodeKind::Vertex(v-3) };
        tree[u] = kind;
        for v in values {
            let v: usize = v.parse()?;
            let v = NodeIndex::new(v);
            assert_ne!(u, v);
            tree.add_edge(u, v, ());
        }
    }

    if tree.edge_count() != m {
        return Err(ReadMDTreeError::WrongM(tree.edge_count(), m));
    }

    Ok(tree)
}


#[derive(Debug)]
pub enum WriteMDTreeError {
    IoError(io::Error),
}

impl From<io::Error> for WriteMDTreeError { fn from(value: io::Error) -> Self { WriteMDTreeError::IoError(value) } }

impl Display for WriteMDTreeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self { WriteMDTreeError::IoError(err) => { write!(f, "{}", err) } }
    }
}

impl Error for WriteMDTreeError {}

pub fn write_md_tree_adj<W: Write>(out: &mut W, md: &DiGraph<MDNodeKind, ()>) -> Result<(), WriteMDTreeError> {
    writeln!(out, "%% modular decomposition tree")?;
    writeln!(out, "%% 1st line:   n m fmt")?;
    writeln!(out, "%% ff lines:   weight children...")?;
    writeln!(out, "%%   weight:   0 => Prime, 1 => Series, 2 => Parallel, 3 + v => v")?;
    if let Some(root) = md.externals(Direction::Incoming).next() {
        writeln!(out, "% root {}", root.index())?;
    }
    writeln!(out, "{} {} 10", md.node_count(), md.edge_count())?;
    for u in md.node_indices() {
        // write!(out, "{} ", u.index())?;
        let w = match md[u] {
            MDNodeKind::Prime => { 0 }
            MDNodeKind::Series => { 1 }
            MDNodeKind::Parallel => { 2 }
            MDNodeKind::Vertex(v) => { 3 + v }
        };
        write!(out, "{w}")?;
        for e in md.edges_directed(u, Direction::Outgoing) {
            write!(out, " {}", e.target().index())?;
        }
        writeln!(out)?;
    }
    out.flush()?;
    Ok(())
}


pub enum ReadEdgeListError {
    WrongLineFormat,
    ParseInt(ParseIntError),
    IoError(io::Error),
}

impl From<io::Error> for ReadEdgeListError {
    fn from(value: io::Error) -> Self { ReadEdgeListError::IoError(value) }
}

impl From<ParseIntError> for ReadEdgeListError {
    fn from(value: ParseIntError) -> Self { ReadEdgeListError::ParseInt(value) }
}

impl Display for ReadEdgeListError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadEdgeListError::WrongLineFormat => { write!(f, "wrong line format") }
            ReadEdgeListError::ParseInt(err) => { write!(f, "parse error {err}") }
            ReadEdgeListError::IoError(err) => { write!(f, "io error {err}") }
        }
    }
}

impl Debug for ReadEdgeListError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{self}") }
}

impl Error for ReadEdgeListError {}

pub fn read_edgelist<P>(path: P) -> Result<UnGraph<(), ()>, ReadEdgeListError>
    where P: AsRef<Path> {
    let file = File::open(path)?;

    let mut n = 0;
    let mut edges = vec![];
    for line in io::BufReader::new(file).lines() {
        let line = line?;
        let mut tokens = line.split_ascii_whitespace();
        let Some(a) = tokens.next() else { return Err(ReadEdgeListError::WrongLineFormat); };
        let Some(b) = tokens.next() else { return Err(ReadEdgeListError::WrongLineFormat); };
        if tokens.next().is_some() { return Err(ReadEdgeListError::WrongLineFormat); }

        let u: u32 = a.parse()?;
        let v: u32 = b.parse()?;
        edges.push((u, v));
        n = n.max(u + 1).max(v + 1);
    }

    let mut graph = UnGraph::with_capacity(n as usize, edges.len());
    graph.extend_with_edges(edges);
    Ok(graph)
}

#[cfg(test)]
mod test {
    use petgraph::graph::DiGraph;
    use crate::io::write_md_tree_adj;
    use crate::modular_decomposition::MDNodeKind;

    #[test]
    fn small_tree() {
        let mut md = DiGraph::<MDNodeKind, ()>::new();
        let nodes = [
            MDNodeKind::Prime, MDNodeKind::Parallel, MDNodeKind::Series,
            MDNodeKind::Vertex(0), MDNodeKind::Vertex(1), MDNodeKind::Vertex(2),
            MDNodeKind::Vertex(3), MDNodeKind::Vertex(4), MDNodeKind::Vertex(5),
            MDNodeKind::Vertex(6)
        ].map(|w| md.add_node(w));
        for (i, j) in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (0, 7), (0, 8), (0, 9)] {
            md.add_edge(nodes[i], nodes[j], ());
        }

        let mut out = Vec::new();
        write_md_tree_adj(&mut out, &md).unwrap();
        let out = String::from_utf8(out).unwrap();
        let expected = r"%% modular decomposition tree
%% 1st line:   n m fmt
%% ff lines:   weight children...
%%   weight:   0 => Prime, 1 => Series, 2 => Parallel, 3 + v => v
% root 0
10 9 10
0 9 8 7 2 1
2 4 3
1 6 5
3
4
5
6
7
8
9
";
        assert_eq!(out, expected);
    }
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