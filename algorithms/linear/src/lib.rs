mod forest;
mod compute;
mod graph;
mod set;

use std::collections::VecDeque;
use petgraph::adj::DefaultIx;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::prelude::EdgeRef;

use common::modular_decomposition::MDNodeKind;
use crate::compute::{MDComputeNode, NodeType, Operation};
use crate::forest::{Forest, NodeIdx};
use crate::graph::Graph;


#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        //println!($($x),*)
    }
}

pub struct Prepared {
    graph: Graph,
}

pub fn prepare<N, E>(graph: &UnGraph<N, E>) -> Prepared
{
    let mut g = Graph::new(graph.node_count());
    for edge in graph.edge_references() {
        g.add_edge(edge.source().index(), edge.target().index());
    }
    Prepared { graph: g }
}

pub struct Computed {
    forest: Forest<MDComputeNode>,
    root: Option<NodeIdx>,
}

impl Prepared {
    pub fn compute(&self) -> Computed
    {
        let (forest, root) = compute::compute(&self.graph);
        Computed { forest, root }
    }
}

impl Computed {
    pub fn finalize(&self) -> DiGraph<MDNodeKind, ()> {
        let Some(root) = self.root else {
            assert_eq!(self.forest.size(), 0);
            return DiGraph::new();
        };

        let mut tree = DiGraph::<MDNodeKind, ()>::with_capacity(self.forest.size(), self.forest.size());

        let add_node = |forest: &Forest<MDComputeNode>, tree: &mut DiGraph<MDNodeKind, ()>, u| -> NodeIndex<DefaultIx> {
            let kind: MDNodeKind = {
                let data = &forest[u].data;
                match data.node_type {
                    NodeType::Vertex => {
                        MDNodeKind::Vertex(data.vertex.idx())
                    }
                    NodeType::Operation => {
                        match data.op_type {
                            Operation::Prime => { MDNodeKind::Prime }
                            Operation::Series => { MDNodeKind::Series }
                            Operation::Parallel => { MDNodeKind::Parallel }
                        }
                    }
                    NodeType::Problem => { panic!("problem nodes should be non existent") }
                }
            };
            tree.add_node(kind)
        };

        let mut queue = VecDeque::new();
        queue.push_back((root, add_node(&self.forest, &mut tree, root)));
        while let Some((x, a)) = queue.pop_front() {
            for y in self.forest.children(x) {
                let b = add_node(&self.forest, &mut tree, y);
                tree.add_edge(a, b, ());
                queue.push_back((y, b));
            }
        }

        tree
    }
}


pub fn modular_decomposition<N, E>(graph: &UnGraph<N, E>) -> DiGraph<MDNodeKind, ()>
{
    prepare(graph).compute().finalize()
}

#[cfg(test)]
mod test {
    use petgraph::dot::{Config, Dot};
    use petgraph::graph::{DiGraph, UnGraph};
    use petgraph::visit::IntoNodeReferences;
    use common::instances::ted08_test0;
    use common::modular_decomposition::MDNodeKind;
    use crate::modular_decomposition;

    #[test]
    fn modular_decomposition_test() {
        let graph = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2), (2, 3)]);
        let md = modular_decomposition(&graph);

        println!("{:?}", Dot::with_config(&md, &[Config::EdgeNoLabel]));
    }

    #[test]
    fn test0() {
        let graph = ted08_test0();

        let md = modular_decomposition(&graph);

        let count_node_kinds = |tree: &DiGraph<MDNodeKind, ()>| -> (usize, usize, usize, usize) {
            tree.node_references()
                .fold((0, 0, 0, 0),
                      |(prime, series, parallel, vertex), (_, k)| {
                          match k {
                              MDNodeKind::Prime => (prime + 1, series, parallel, vertex),
                              MDNodeKind::Series => (prime, series + 1, parallel, vertex),
                              MDNodeKind::Parallel => (prime, series, parallel + 1, vertex),
                              MDNodeKind::Vertex(_) => (prime, series, parallel, vertex + 1)
                          }
                      })
        };

        assert_eq!(md.node_count(), 27);
        assert_eq!(count_node_kinds(&md), (2, 4, 3, 18));
    }

    #[test]
    fn empty() {
        let graph = UnGraph::<(), ()>::new_undirected();
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 0);
        assert_eq!(md.edge_count(), 0);
    }
}
