mod base;
mod improved;

use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use common::modular_decomposition::MDNodeKind;


#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        tracing::trace!($($x),*)
    }
}


pub fn modular_decomposition<N, E>(graph: &UnGraph<N, E>) -> DiGraph<MDNodeKind, ()> {
    prepare(graph).compute().finalize()
}

pub struct Prepared {
    graph: Vec<Vec<NodeIndex>>,
}

pub fn prepare<N, E>(graph: &UnGraph<N, E>) -> Prepared
{
    let mut graph: Vec<Vec<NodeIndex>> = graph.node_indices().map(|u| graph.neighbors(u).collect()).collect();
    graph.iter_mut().for_each(|neighbors| neighbors.sort());
    Prepared { graph }
}

pub struct Computed {
    tree: DiGraph<MDNodeKind, ()>,
}

impl Prepared {
    pub fn compute(mut self) -> Computed
    {
        let tree = improved::modular_decomposition(&mut self.graph);
        Computed { tree }
    }
}

impl Computed {
    pub fn finalize(self) -> DiGraph<MDNodeKind, ()> {
        self.tree
    }
}


#[cfg(test)]
#[allow(non_snake_case)]
mod test {
    use std::collections::HashSet;
    use common::instances;
    use super::*;

    #[test]
    fn E_0() {
        let graph = instances::empty_graph(0);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 0);
    }

    #[test]
    fn E_1() {
        let graph = instances::empty_graph(1);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 1);
    }

    #[test]
    fn E_2() {
        let graph = instances::empty_graph(2);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 3);
        assert_eq!(md.node_weights().copied().collect::<HashSet<_>>(),
                   HashSet::from_iter([MDNodeKind::Vertex(0), MDNodeKind::Vertex(1), MDNodeKind::Parallel]));
    }

    #[test]
    fn K_2() {
        let graph = instances::complete_graph(2);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 3);
        assert_eq!(md.node_weights().copied().collect::<HashSet<_>>(),
                   HashSet::from_iter([MDNodeKind::Vertex(0), MDNodeKind::Vertex(1), MDNodeKind::Series]));
    }

    #[test]
    fn K_32() {
        let graph = instances::complete_graph(32);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 33);
    }

    #[test]
    fn P_4() {
        let graph = instances::path_graph(4);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 5);
        assert_eq!(md.node_weights().copied().collect::<HashSet<_>>(),
                   HashSet::from_iter([MDNodeKind::Vertex(0), MDNodeKind::Vertex(1), MDNodeKind::Vertex(2), MDNodeKind::Vertex(3), MDNodeKind::Prime]));
    }

    #[test]
    fn P_32() {
        let graph = instances::path_graph(32);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 33);
    }
}