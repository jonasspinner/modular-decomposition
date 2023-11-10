use std::fmt::{Debug, Formatter};
use std::fs;
use std::process::Command;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use common::modular_decomposition::MDNodeKind;


struct Edge(usize, usize);

impl Debug for Edge {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{},{}}}", self.0, self.1)
    }
}


pub fn modular_decomposition(graph: &UnGraph<(), ()>) -> DiGraph<MDNodeKind, ()> {
    let g = format!("({:?}, {:?})", graph.node_indices().map(|u| u.index()).collect::<Vec<_>>(), graph.edge_references().map(|e| Edge(e.source().index(), e.target().index())).collect::<Vec<_>>());
    let candidate_paths : Vec<_> = [
        "extern/lyt15/build/mod_dec",
        "../extern/lyt15/build/mod_dec",
        "../../extern/lyt15/build/mod_dec"
    ].iter().filter_map(|p| fs::metadata(p).is_ok().then(|| p.to_string())).collect();

    let lyt15_bin_path = std::env::var("LYT15_BIN")
        .ok().or(candidate_paths.first().cloned())
        .unwrap_or("mod_dec".to_string());
    let output = Command::new(lyt15_bin_path)
        .arg(g)
        .output().unwrap();
    let output = String::from_utf8(output.stdout).unwrap();

    let mut md = DiGraph::new();
    for line in output.lines() {
        if line.contains('[') {
            let mut parts = line.split('[');
            let _u: usize = parts.next().unwrap().parse().unwrap();
            let label = parts.next().unwrap().strip_prefix("label=").unwrap().strip_suffix("];").unwrap();
            let kind = match label {
                "Parallel" => { MDNodeKind::Parallel }
                "Series" => { MDNodeKind::Series }
                "Prime" => { MDNodeKind::Prime }
                vertex => {
                    MDNodeKind::Vertex(vertex.strip_prefix('v').unwrap().parse().unwrap())
                }
            };
            md.add_node(kind);
        } else if line.contains("->") {
            let parts: Vec<_> = line.strip_suffix(" ;").unwrap().split("->").collect();
            let u: usize = parts[0].parse().unwrap();
            let v: usize = parts[1].parse().unwrap();
            md.add_edge(NodeIndex::new(u), NodeIndex::new(v), ());
        }
    }
    md
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ted08_test0() {
        let graph = common::instances::ted08_test0();

        let md = modular_decomposition(&graph);

        println!("{:?}", md);
    }
}
