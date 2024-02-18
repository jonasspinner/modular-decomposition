#![allow(dead_code)]

use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use tracing::instrument;
use common::modular_decomposition::MDNodeKind;
use crate::algos::tree::{Kind, TreeNode, TreeNodeIndex};

pub(crate) mod graph;
pub(crate) mod partition;
mod ordered_vertex_partition;
pub(crate) mod algos;
mod testing;

#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        //println!($($x),*)
    }
}


pub struct Prepared {
    graph: graph::Graph,
}

#[instrument(skip_all)]
pub fn prepare<N, E>(graph: &UnGraph<N, E>) -> Prepared
{
    let edges: Vec<_> = graph.edge_references().map(|e| {
        let u = graph::NodeIndex::new(e.source().index());
        let v = graph::NodeIndex::new(e.target().index());
        (u, v)
    }).collect();
    let graph = graph::Graph::from_edges(graph.node_count(), edges.iter().copied());
    Prepared { graph }
}

pub struct Computed {
    tree: Vec<TreeNode>,
}

impl Prepared {
    #[instrument(skip_all)]
    pub fn compute(mut self) -> Computed
    {
        let tree = algos::modular_decomposition(&mut self.graph);
        Computed { tree }
    }
}

impl Computed {
    #[instrument(skip_all)]
    pub fn finalize(&self) -> DiGraph<MDNodeKind, ()> {
        let add_node = |md: &mut DiGraph<MDNodeKind, ()>, i: TreeNodeIndex| -> NodeIndex {
            md.add_node(self.tree[i.index()].kind.into())
        };

        let mut md = DiGraph::new();

        if self.tree.is_empty() { return md; }

        let root = TreeNodeIndex::new(0);
        let mut stack = vec![root];
        let mut map = vec![NodeIndex::end(); self.tree.len()];
        map[root.index()] = add_node(&mut md, root);
        while let Some(i) = stack.pop() {
            for &j in &self.tree[i.index()].children {
                stack.push(j);
                if self.tree[j.index()].kind == self.tree[i.index()].kind && self.tree[j.index()].kind != Kind::Prime {
                    map[j.index()] = map[i.index()];
                } else {
                    map[j.index()] = add_node(&mut md, j);
                    md.add_edge(map[i.index()], map[j.index()], ());
                }
            }
        }
        md
    }
}

pub fn modular_decomposition<N, E>(graph: &UnGraph<N, E>) -> DiGraph<MDNodeKind, ()> {
    prepare(graph).compute().finalize()
}


#[cfg(test)]
mod test {
    use std::path::Path;
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