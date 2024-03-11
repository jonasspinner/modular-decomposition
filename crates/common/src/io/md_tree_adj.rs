use crate::modular_decomposition::MDNodeKind;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::fs::File;
use std::io::{BufRead, Write};
use std::num::ParseIntError;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ReadMDTreeError {
    #[error("invalid format")]
    InvalidFormat,
    #[error("missing error")]
    MissingHeader,
    #[error("wrong m given in header (expected {expected}, got {actual})")]
    WrongM { expected: usize, actual: usize },
    #[error("invalid header (got {0})")]
    InvalidHeader(String),
    #[error("parse int error")]
    ParseInt(#[from] ParseIntError),
    #[error("io error")]
    IoError(#[from] std::io::Error),
}

pub fn read_md_tree_adj<P>(path: P) -> Result<DiGraph<MDNodeKind, ()>, ReadMDTreeError>
where
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let mut lines = std::io::BufReader::new(file).lines();
    let first_line = loop {
        let line = lines.next().ok_or(ReadMDTreeError::MissingHeader)??;
        if !line.starts_with('%') {
            break line;
        }
    };

    let (n, m) = {
        let err = || ReadMDTreeError::InvalidHeader(first_line.clone());
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
        return Err(ReadMDTreeError::WrongM { expected: n - 1, actual: m });
    }

    let mut tree = DiGraph::with_capacity(n, m);

    for _ in 0..n {
        tree.add_node(MDNodeKind::Vertex(usize::MAX));
    }

    for (u, line) in lines.enumerate() {
        let line = line?;
        assert!(u < n, "n={n} m={m} u={u} line=\"{line}\"");
        let u = NodeIndex::new(u);
        if line.starts_with('%') {
            unimplemented!("cannot handle later comment lines")
        }
        let mut values = line.split_ascii_whitespace();
        let kind: usize = values.next().ok_or(ReadMDTreeError::InvalidFormat)?.parse()?;
        let kind = match kind {
            0 => MDNodeKind::Prime,
            1 => MDNodeKind::Series,
            2 => MDNodeKind::Parallel,
            v => MDNodeKind::Vertex(v - 3),
        };
        tree[u] = kind;
        for v in values {
            let v: usize = v.parse()?;
            let v = NodeIndex::new(v);
            assert_ne!(u, v);
            tree.add_edge(u, v, ());
        }
    }

    if tree.edge_count() != m {
        return Err(ReadMDTreeError::WrongM { actual: tree.edge_count(), expected: m });
    }

    Ok(tree)
}

#[derive(Error, Debug)]
pub enum WriteMDTreeError {
    #[error("io error")]
    IoError(#[from] std::io::Error),
}

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
            MDNodeKind::Prime => 0,
            MDNodeKind::Series => 1,
            MDNodeKind::Parallel => 2,
            MDNodeKind::Vertex(v) => 3 + v,
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

#[cfg(test)]
mod test {
    use crate::io::write_md_tree_adj;
    use crate::modular_decomposition::MDNodeKind;
    use petgraph::graph::DiGraph;

    #[test]
    fn small_tree() {
        let mut md = DiGraph::<MDNodeKind, ()>::new();
        let nodes = [
            MDNodeKind::Prime,
            MDNodeKind::Parallel,
            MDNodeKind::Series,
            MDNodeKind::Vertex(0),
            MDNodeKind::Vertex(1),
            MDNodeKind::Vertex(2),
            MDNodeKind::Vertex(3),
            MDNodeKind::Vertex(4),
            MDNodeKind::Vertex(5),
            MDNodeKind::Vertex(6),
        ]
        .map(|w| md.add_node(w));
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
