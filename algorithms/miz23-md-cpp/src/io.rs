use std::error::Error;
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::path::Path;
use petgraph::graph::UnGraph;

#[allow(unused)]
pub(crate) fn read_pace2023<P>(path: P) -> Result<UnGraph<(), ()>, Box<dyn Error>>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();
    let first_line = lines.next().ok_or_else(|| "cannot read first line")??;
    let mut iter = first_line.split_ascii_whitespace();

    let p = iter.next().ok_or_else(|| "p")?;
    let tww = iter.next().ok_or_else(|| "")?;
    let n: usize = iter.next().ok_or_else(|| "")?.parse()?;
    let m: usize = iter.next().ok_or_else(|| "")?.parse()?;

    assert_eq!(p, "p");
    assert_eq!(tww, "tww");

    let mut graph = UnGraph::with_capacity(n, m);
    for _ in 0..n { graph.add_node(()); }

    for line in lines {
        let line = line.map_err(|_| "cannot read line")?;
        let mut iter = line.split_ascii_whitespace();
        let mut next_number = || -> Result<u32, Box<dyn Error>> { Ok(iter.next().ok_or_else(|| "cannot read number")?.parse()?) };
        let u = next_number()? - 1;
        let v = next_number()? - 1;
        assert!(u < n as u32);
        assert!(v < n as u32);
        graph.add_edge(u.into(), v.into(), ());
    }
    Ok(graph)
}

#[allow(unused)]
pub(crate) fn read_edgelist<P>(path: P) -> Result<UnGraph<(), ()>, Box<dyn Error>>
    where P: AsRef<Path> {
    let file = File::open(path)?;
    let mut lines = io::BufReader::new(file).lines();

    let mut edges = vec![];
    for line in lines {
        let line = line.map_err(|_| "cannot read line")?;
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