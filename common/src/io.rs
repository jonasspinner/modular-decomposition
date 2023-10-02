use std::error::Error;
use std::ffi::OsStr;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::num::ParseIntError;
use std::path::Path;
use petgraph::adj::DefaultIx;
use petgraph::{Graph, Undirected};
use petgraph::graph::{NodeIndex, UnGraph};


#[derive(Debug)]
pub enum Pace2023Error {
    NotPace2023Format,
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
            Pace2023Error::NotPace2023Format => { write!(f, "not pace format") }
            Pace2023Error::IoError(err) => { write!(f, "{}", err) }
        }
    }
}

impl Error for Pace2023Error {}

pub fn read_pace2023<P>(path: P) -> Result<UnGraph<(), ()>, Pace2023Error>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();
    let first_line = lines.next().ok_or(Pace2023Error::NotPace2023Format)??;
    let mut iter = first_line.split_ascii_whitespace();

    let p = iter.next().ok_or(Pace2023Error::NotPace2023Format)?;
    let tww = iter.next().ok_or(Pace2023Error::NotPace2023Format)?;
    if p != "p" || tww != "tww" {
        return Err(Pace2023Error::NotPace2023Format);
    }
    let n: usize = iter.next().ok_or(Pace2023Error::NotPace2023Format)?.parse()?;
    let m: usize = iter.next().ok_or(Pace2023Error::NotPace2023Format)?.parse()?;

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

pub fn read_edgelist<P>(path: P) -> Result<UnGraph<(), ()>, Box<dyn Error>>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let lines = io::BufReader::new(file).lines();

    let mut edges = vec![];
    for line in lines {
        let line = line?;
        let mut iter = line.split_ascii_whitespace();
        let mut next_number = || -> Result<u32, Box<dyn Error>> { Ok(iter.next().ok_or_else(|| "cannot read number")?.parse()?) };
        let u = next_number()?;
        let v = next_number()?;
        edges.push((u, v));
    }
    let mut graph = UnGraph::with_capacity(0, 0);
    graph.extend_with_edges(edges);
    Ok(graph)
}

pub fn read_dat<P>(path: P) -> Result<UnGraph<(), (), DefaultIx>, Box<dyn Error>>
    where P: AsRef<Path> {
    if path.as_ref().extension().and_then(OsStr::to_str) != Some("dat") {
        return Err("extension is not .dat".into());
    }

    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();

    let first_line = lines.next().ok_or_else(|| "cannot read first line")??;
    let mut iter = first_line.split_ascii_whitespace();
    let n: usize = iter.next().ok_or_else(|| "no n in first line")?.parse()?;
    let m: usize = iter.next().ok_or_else(|| "no m in first line")?.parse()?;
    if iter.next().is_some() { return Err("unrecognized value in first line".into()); }

    let mut graph = Graph::<(), (), Undirected, DefaultIx>::with_capacity(n, m);

    for _ in 0..n {
        graph.add_node(());
    }

    for (u, line) in lines.enumerate() {
        assert!(u < n);

        let line = line.map_err(|_| "cannot read line")?;
        let mut iter = line.split_ascii_whitespace();

        let d: usize = iter.next().ok_or_else(|| "cannot read degree")?.parse()?;

        for _ in 0..d {
            let v: usize = iter.next().ok_or_else(|| "missing neighbor")?.parse()?;
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