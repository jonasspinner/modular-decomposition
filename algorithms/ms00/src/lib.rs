#![feature(step_trait)]
#![allow(dead_code)]

use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use common::modular_decomposition::MDNodeKind;
use crate::seq::algos::{Kind, TreeNode, TreeNodeIndex};

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
    let edges: Vec<_> = graph.edge_references().map(|e| {
        let u = seq::graph::NodeIndex::new(e.source().index());
        let v = seq::graph::NodeIndex::new(e.target().index());
        (u, v)
    }).collect();
    let graph = seq::graph::Graph::from_edges(graph.node_count(), edges.iter().copied());
    let partition = seq::partition::Partition::new(graph.node_count());
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

        if self.graph.node_count() != 0 {
            let root = seq::algos::TreeNodeIndex::new(0);
            tree.push(seq::algos::TreeNode::default());
            seq::algos::modular_decomposition(&mut self.graph, p, &mut self.partition, &mut tree, root);
        }
        Computed { tree }
    }
}

impl Computed {
    pub fn finalize(&self) -> DiGraph<MDNodeKind, ()> {
        let add_node = |md: &mut DiGraph<MDNodeKind, ()>, i: TreeNodeIndex| -> NodeIndex {
            let kind = match self.tree[i.index()].kind {
                Kind::Prime => { MDNodeKind::Prime }
                Kind::Series => { MDNodeKind::Series }
                Kind::Parallel => { MDNodeKind::Parallel }
                Kind::Vertex(u) => { MDNodeKind::Vertex(u.index()) }
                Kind::UnderConstruction => { panic!() }
            };
            md.add_node(kind)
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

    #[test]
    fn exact_053() {
        let path = Path::new("../../hippodrome/instances/pace2023/exact_053.gr");
        let graph = common::io::read_pace2023(path).unwrap();
        let md = modular_decomposition(&graph);

        println!("{:?}", Dot::with_config(&md, &[Config::EdgeNoLabel]));
    }
}