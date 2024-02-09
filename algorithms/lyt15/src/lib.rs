use std::env::VarError;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::process::Command;
use std::string::FromUtf8Error;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use common::modular_decomposition::MDNodeKind;


struct Edge(usize, usize);

impl Debug for Edge {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{},{}}}", self.0, self.1)
    }
}

pub fn modular_decomposition_from_env(graph: &UnGraph<(), ()>) -> Result<DiGraph<MDNodeKind, ()>, Lyt15Error> {
    let lyt15_bin_path = std::env::var("LYT15_BIN")?;
    modular_decomposition_from_bin_path(&lyt15_bin_path, graph)
}

pub fn modular_decomposition_from_bin_path(lyt15_bin_path: &str, graph: &UnGraph<(), ()>) -> Result<DiGraph<MDNodeKind, ()>, Lyt15Error> {
    let g = format!("({:?}, {:?})", graph.node_indices().map(|u| u.index()).collect::<Vec<_>>(), graph.edge_references().map(|e| Edge(e.source().index(), e.target().index())).collect::<Vec<_>>());
    let output = Command::new(lyt15_bin_path)
        .arg(g)
        .output()?;
    let output = String::from_utf8(output.stdout)?;

    let mut md = DiGraph::new();
    for (line_number, line) in output.lines().enumerate() {
        let err_line = || { FormatError { line_number, line: line.to_string() } };
        if line.contains('[') {
            let mut parts = line.split('[');
            let _u: usize = parts
                .next().ok_or_else(err_line)?
                .parse().map_err(|_| err_line())?;
            let label = parts
                .next().ok_or_else(err_line)?
                .strip_prefix("label=").ok_or_else(err_line)?
                .strip_suffix("];").ok_or_else(err_line)?;
            let kind = match label {
                "Parallel" => { MDNodeKind::Parallel }
                "Series" => { MDNodeKind::Series }
                "Prime" => { MDNodeKind::Prime }
                vertex => {
                    let vertex = vertex.strip_prefix('v').ok_or_else(err_line)?;
                    let vertex = vertex.parse().map_err(|_| err_line())?;
                    MDNodeKind::Vertex(vertex)
                }
            };
            md.add_node(kind);
        } else if line.contains("->") {
            let parts: Vec<_> = line
                .strip_suffix(" ;").ok_or_else(err_line)?
                .split("->").collect();
            if parts.len() != 2 { return Err(err_line().into()); }
            let u: usize = parts[0].parse().map_err(|_| err_line())?;
            let v: usize = parts[1].parse().map_err(|_| err_line())?;
            md.add_edge(NodeIndex::new(u), NodeIndex::new(v), ());
        } else {
            return Err(err_line().into());
        }
    }
    Ok(md)
}

#[derive(Debug)]
pub struct FormatError {
    line_number: usize,
    line: String,
}

#[derive(Debug)]
pub enum Lyt15Error {
    MissingEnvVar(VarError),
    IoError(std::io::Error),
    FormatError(FormatError),
    FromUtf8Error(FromUtf8Error),
}

impl From<FormatError> for Lyt15Error {
    fn from(value: FormatError) -> Self { Lyt15Error::FormatError(value) }
}

impl From<std::io::Error> for Lyt15Error {
    fn from(value: std::io::Error) -> Self { Self::IoError(value) }
}

impl From<VarError> for Lyt15Error {
    fn from(value: VarError) -> Self { Self::MissingEnvVar(value) }
}

impl From<FromUtf8Error> for Lyt15Error {
    fn from(value: FromUtf8Error) -> Self { Self::FromUtf8Error(value) }
}

impl Display for Lyt15Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Lyt15Error::MissingEnvVar(e) => write!(f, "missing binary path. {e}"),
            Lyt15Error::IoError(e) => write!(f, "{e}"),
            Lyt15Error::FormatError(e) => write!(f, "line({}): {}", e.line_number, e.line),
            Lyt15Error::FromUtf8Error(e) => write!(f, "{e}"),
        }
    }
}

impl Error for Lyt15Error {}
