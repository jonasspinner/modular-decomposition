use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use common::modular_decomposition::MDNodeKind;
use crate::seq::algos::{Kind, TreeNode};

mod linked_list;
mod seq;

#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        //println!($($x),*)
    }
}


pub struct Prepared {
    graph: seq::graph::Graph,
    partition: seq::partition::Partition,
}

pub fn prepare<N, E>(graph: &UnGraph<N, E>) -> Prepared
{
    let edges : Vec<_> = graph.edge_references().map(|e| {
        let u = seq::graph::NodeIndex::new(e.source().index());
        let v = seq::graph::NodeIndex::new(e.target().index());
        (u, v)
    }).collect();
    let mut graph = seq::graph::Graph::from_edges(
        graph.node_count(), edges.iter().copied());
    let mut partition = seq::partition::Partition::new(graph.node_count());
    Prepared { graph, partition }
}

pub struct Computed {
    tree: Vec<TreeNode>,
}

impl Prepared {
    pub fn compute(mut self) -> Computed
    {
        let p = seq::partition::Part::new(&self.partition);
        let mut tree = Vec::with_capacity(2 * self.graph.node_count());

        let root = seq::algos::TreeNodeIndex::new(0);
        tree.push(seq::algos::TreeNode::default());
        seq::algos::modular_decomposition(&mut self.graph, p, &mut self.partition, &mut tree, root);
        Computed { tree }
    }
}

impl Computed {
    pub fn finalize(&self) -> DiGraph<MDNodeKind, ()> {
        let mut md = DiGraph::new();
        for i in 0..self.tree.len() {
            let kind = match self.tree[i].kind {
                Kind::Prime => { MDNodeKind::Prime }
                Kind::Series => { MDNodeKind::Series }
                Kind::Parallel => { MDNodeKind::Parallel }
                Kind::Vertex(u) => { MDNodeKind::Vertex(u.index())}
                Kind::UnderConstruction => { panic!() }
            };
            md.add_node(kind);
        }
        for i in 0..self.tree.len() {
            for j in &self.tree[i].children {
                md.add_edge(NodeIndex::new(i), NodeIndex::new(j.index()), ());
            }
        }
        md
    }
}

pub fn modular_decomposition<N, E>(graph: &UnGraph<N, E>) -> DiGraph<MDNodeKind, ()> {
    prepare(graph).compute().finalize()
}
