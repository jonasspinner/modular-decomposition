mod base;

use common::modular_decomposition::MDNodeKind;
use modular_decomposition::fracture::modular_decomposition as fracture_modular_decomposition;
use modular_decomposition::ModuleKind;
use petgraph::graph::{DiGraph, UnGraph};
use tracing::info;

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
    graph: UnGraph<(), ()>,
}

pub fn prepare<N, E>(graph: &UnGraph<N, E>) -> Prepared {
    info!(n = graph.node_count(), m = graph.edge_count());
    let graph = graph.map(|_, _| (), |_, _| ());
    Prepared { graph }
}

pub struct Computed {
    tree: DiGraph<ModuleKind, ()>,
}

impl Prepared {
    pub fn compute(self) -> Computed {
        let tree = fracture_modular_decomposition(&self.graph).map(|tree| tree.into_digraph()).unwrap_or_default();
        Computed { tree }
    }
}

impl Computed {
    pub fn finalize(self) -> DiGraph<MDNodeKind, ()> {
        self.tree.map(
            |_, kind| match kind {
                ModuleKind::Prime => MDNodeKind::Prime,
                ModuleKind::Series => MDNodeKind::Series,
                ModuleKind::Parallel => MDNodeKind::Parallel,
                ModuleKind::Node(v) => MDNodeKind::Vertex(v.index()),
            },
            |_, _| (),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use common::instances;
    use std::collections::HashSet;

    #[test]
    fn empty_0() {
        let graph = instances::empty_graph(0);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 0);
    }

    #[test]
    fn empty_1() {
        let graph = instances::empty_graph(1);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 1);
    }

    #[test]
    fn empty_2() {
        let graph = instances::empty_graph(2);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 3);
        assert_eq!(
            md.node_weights().copied().collect::<HashSet<_>>(),
            HashSet::from_iter([MDNodeKind::Vertex(0), MDNodeKind::Vertex(1), MDNodeKind::Parallel])
        );
    }

    #[test]
    fn complete_2() {
        let graph = instances::complete_graph(2);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 3);
        assert_eq!(
            md.node_weights().copied().collect::<HashSet<_>>(),
            HashSet::from_iter([MDNodeKind::Vertex(0), MDNodeKind::Vertex(1), MDNodeKind::Series])
        );
    }

    #[test]
    fn complete_32() {
        let graph = instances::complete_graph(32);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 33);
    }

    #[test]
    fn path_4() {
        let graph = instances::path_graph(4);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 5);
        assert_eq!(
            md.node_weights().copied().collect::<HashSet<_>>(),
            HashSet::from_iter([
                MDNodeKind::Vertex(0),
                MDNodeKind::Vertex(1),
                MDNodeKind::Vertex(2),
                MDNodeKind::Vertex(3),
                MDNodeKind::Prime
            ])
        );
    }

    #[test]
    fn path_32() {
        let graph = instances::path_graph(32);
        let md = modular_decomposition(&graph);
        assert_eq!(md.node_count(), 33);
    }
}
