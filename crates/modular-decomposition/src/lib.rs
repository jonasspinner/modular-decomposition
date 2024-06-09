//! This is a library to compute the [modular decomposition](https://en.wikipedia.org/wiki/Modular_decomposition) of a simple, undirected graph.
//!
//! A node set *M* is a *module* if every node has the same neighborhood outside
//! *M*. The set of all nodes *V* and the sets with a single node *{u}* are
//! trivial modules.
//!
//! # Examples
//!
//! The smallest prime graph is the path graph on 4 nodes.
//! ```rust
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use petgraph::graph::UnGraph;
//! use modular_decomposition::{ModuleKind, modular_decomposition};
//!
//! // a path graph with 4 nodes
//! let graph = UnGraph::<(), ()>::from_edges([(0, 1), (1, 2), (2, 3)]);
//! let md = modular_decomposition(&graph)?;
//!
//! assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Prime));
//! # Ok(())
//! # }
//! ```
//!
//! Determining whether a graph is a [cograph](https://en.wikipedia.org/wiki/Cograph).
//! ```rust
//! # use std::error::Error;
//! #
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use petgraph::graph::UnGraph;
//! use modular_decomposition::{ModuleKind, modular_decomposition};
//!
//! // a complete graph with 3 nodes
//! let graph = UnGraph::<(), ()>::from_edges([(0, 1), (0, 2), (1, 2)]);
//! let md = modular_decomposition(&graph)?;
//!
//! // a graph is a cograph exactly if none of its modules is prime
//! let is_cograph = md.module_kinds().all(|kind| *kind != ModuleKind::Prime);
//! assert!(is_cograph);
//! # Ok(())
//! # }
//! ```
//!
//! # Generics
//!
//! The algorithm is implemented for structs that implement the `petgraph`
//! traits `NodeCompactIndexable`, `IntoNeighbors`, and `GraphProp<EdgeType =
//! Undirected>`.
//!
//! # References
//! + \[HPV99\]: Michel Habib, Christophe Paul, and Laurent Viennot. “Partition Refinement Techniques: An Interesting Algorithmic Tool Kit”. <https://doi.org/10.1142/S0129054199000125>.
//! + \[CHM02\]: Christian Capelle, Michel Habib, and Fabien Montgolfier. “Graph
//!   Decompositions and Factorizing Permutations”. <https://doi.org/10.46298/dmtcs.298>.

#![forbid(unsafe_code)]
#![doc(test(attr(deny(warnings, rust_2018_idioms), allow(dead_code))))]
#![warn(missing_docs, missing_debug_implementations, rust_2018_idioms, unreachable_pub)]

/// A modular decomposition algorithm.
pub mod fracture;
mod index;
mod md_tree;

mod deque;
mod segmented_stack;
#[cfg(test)]
mod tests;

pub use fracture::modular_decomposition;
pub use md_tree::MDTree;
pub use md_tree::ModuleKind;
pub use md_tree::ModuleIndex;

#[cfg(test)]
mod test {
    use super::*;
    use crate::md_tree::{MDTree, ModuleKind, NodeIndex};

    #[derive(Default, Debug)]
    struct ModuleKindCounts {
        prime: usize,
        series: usize,
        parallel: usize,
        vertex: usize,
    }

    impl PartialEq<[usize; 4]> for ModuleKindCounts {
        fn eq(&self, &[prime, series, parallel, vertex]: &[usize; 4]) -> bool {
            self.prime == prime && self.series == series && self.parallel == parallel && self.vertex == vertex
        }
    }

    fn count_module_kinds(md: &MDTree) -> ModuleKindCounts {
        let mut counts = ModuleKindCounts::default();
        for kind in md.module_kinds() {
            match kind {
                ModuleKind::Prime => counts.prime += 1,
                ModuleKind::Series => counts.series += 1,
                ModuleKind::Parallel => counts.parallel += 1,
                ModuleKind::Node(_) => counts.vertex += 1,
            }
        }
        counts
    }

    #[test]
    fn empty_0() {
        let graph = tests::empty_graph(0);
        let md = modular_decomposition(&graph);
        assert!(md.is_err())
    }

    #[test]
    fn empty_1() {
        let graph = tests::empty_graph(1);
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(md.node_count(), 1);
        assert_eq!(count_module_kinds(&md), [0, 0, 0, 1]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Node(NodeIndex::new(0))));
    }

    #[test]
    fn empty_2() {
        let graph = tests::empty_graph(2);
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(md.node_count(), 3);
        assert_eq!(count_module_kinds(&md), [0, 0, 1, 2]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Parallel));
        assert_eq!(md.children(md.root()).count(), 2);
    }

    #[test]
    fn complete_2() {
        let graph = tests::complete_graph(2);
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(md.node_count(), 3);
        assert_eq!(count_module_kinds(&md), [0, 1, 0, 2]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Series));
        assert_eq!(md.children(md.root()).count(), 2);
    }

    #[test]
    fn complete_4() {
        let graph = tests::complete_graph(4);
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(count_module_kinds(&md), [0, 1, 0, 4]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Series));
        assert_eq!(md.children(md.root()).count(), 4);
    }

    #[test]
    fn complete_32() {
        let graph = tests::complete_graph(32);
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(count_module_kinds(&md), [0, 1, 0, 32]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Series));
        assert_eq!(md.children(md.root()).count(), 32);
    }

    #[test]
    fn path_4() {
        let graph = tests::path_graph(4);
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(md.node_count(), 5);
        assert_eq!(count_module_kinds(&md), [1, 0, 0, 4]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Prime));
        assert_eq!(md.children(md.root()).count(), 4);
    }

    #[test]
    fn path_32() {
        let graph = tests::path_graph(32);
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(count_module_kinds(&md), [1, 0, 0, 32]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Prime));
        assert_eq!(md.children(md.root()).count(), 32);
    }

    #[test]
    fn pace2023_exact_024() {
        let graph = tests::pace2023_exact_024();
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(count_module_kinds(&md), [1, 2, 2, 40]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Parallel));
        assert_eq!(md.children(md.root()).count(), 3);
    }

    #[test]
    fn pace2023_exact_054() {
        let graph = tests::pace2023_exact_054();
        let md = modular_decomposition(&graph).unwrap();
        assert_eq!(count_module_kinds(&md), [3, 9, 5, 73]);
        assert_eq!(md.module_kind(md.root()), Some(&ModuleKind::Parallel));
        assert_eq!(md.children(md.root()).count(), 11);
    }
}
