use std::fmt::{Debug, Display, Formatter};
use std::iter::from_fn;

use petgraph::graph::DiGraph;
use petgraph::{Incoming, Outgoing};

use crate::index::make_index;

make_index!(pub(crate) NodeIndex);

/// Module kinds of nodes in a [MDTree].
///
/// Each module corresponds to a set of nodes in the original graph, the leaves of the subtree
/// rooted at that node.
///
/// The module kinds are determined by the quotient graph of a module that is obtained by taking a
/// single node from each child module.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ModuleKind<NodeId: Copy + PartialEq> {
    /// A prime module. Its quotient graph has only trivial modules.
    Prime,
    /// A series module. Its quotient graph is a complete graph.
    Series,
    /// A parallel module. Its quotient graph is an empty graph.
    Parallel,
    /// A trivial module with a single vertex. This is leaf node in the [MDTree].
    Node(NodeId),
}

impl<NodeId: Debug + Copy + PartialEq> Debug for ModuleKind<NodeId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleKind::Prime => {
                write!(f, "Prime")
            }
            ModuleKind::Series => {
                write!(f, "Series")
            }
            ModuleKind::Parallel => {
                write!(f, "Parallel")
            }
            ModuleKind::Node(v) => {
                write!(f, "{v:?}")
            }
        }
    }
}

/// A modular decomposition tree. The tree contains at least one node.
#[derive(Clone, Debug)]
pub struct MDTree<NodeId: Copy + PartialEq> {
    tree: DiGraph<ModuleKind<NodeId>, ()>,
    root: ModuleIndex,
}

/// Module identifier.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ModuleIndex(pub(crate) petgraph::graph::NodeIndex);

impl Debug for ModuleIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ModuleIndex").field(&self.0.index()).finish()
    }
}

impl ModuleIndex {
    /// Create new index from `usize`.
    pub fn new(x: usize) -> Self {
        Self(petgraph::graph::NodeIndex::new(x))
    }

    /// Returns the index as `usize`.
    pub fn index(&self) -> usize {
        self.0.index()
    }
}

impl<NodeId: Copy + PartialEq> MDTree<NodeId> {
    /// Create a new modular decomposition tree.
    ///
    /// Assumes that the input `DiGraph` is rooted tree with node weights
    /// `Prime`, `Series`, and `Parallel` for inner nodes and `Vertex(_)`
    /// for leaf nodes. This is not checked explicitly.
    ///
    /// Return `NullGraph` if the input graph does not have any nodes.
    ///
    /// Panics if all nodes have a non-zero in-degree.
    pub(crate) fn from_digraph(tree: DiGraph<ModuleKind<NodeId>, ()>) -> Result<Self, NullGraphError> {
        if tree.node_count() == 0 {
            return Err(NullGraphError);
        }
        let root = tree.externals(Incoming).next().expect("non-null trees must have a root");
        let root = ModuleIndex(root);
        Ok(Self { tree, root })
    }

    /// Return the number of nodes in the modular decomposition tree.
    #[inline(always)]
    pub fn node_count(&self) -> usize {
        self.tree.node_count()
    }

    /// Return the root node index.
    #[inline(always)]
    pub fn root(&self) -> ModuleIndex {
        self.root
    }

    /// Access the [ModuleKind] of a module.
    ///
    /// If the module does not exist, return None.
    pub fn module_kind(&self, module: ModuleIndex) -> Option<&ModuleKind<NodeId>> {
        self.tree.node_weight(module.0)
    }

    /// Return an iterator yielding references to [ModuleKind]s for all nodes.
    pub fn module_kinds(&self) -> impl Iterator<Item = &ModuleKind<NodeId>> {
        self.tree.node_weights()
    }

    /// Return an iterator for the children of a node.
    pub fn children(&self, module: ModuleIndex) -> impl Iterator<Item = ModuleIndex> + '_ {
        self.tree.neighbors_directed(module.0, Outgoing).map(ModuleIndex)
    }

    /// Return the nodes of a module.
    ///
    /// The nodes represent nodes of the original graph and not modules of the [MDTree].
    /// These are the leaves of the subtree rooted at `module` in the [MDTree].
    pub fn nodes(&self, module: ModuleIndex) -> impl Iterator<Item = NodeId> + '_ {
        let mut stack = vec![module.0];
        from_fn(move || {
            while let Some(next) = stack.pop() {
                if let Some(ModuleKind::Node(node)) = self.tree.node_weight(next) {
                    return Some(*node);
                } else {
                    let children = self.tree.neighbors_directed(next, Outgoing);
                    stack.extend(children);
                }
            }
            None
        })
    }

    /// Convert to [DiGraph].
    ///
    /// This allows the use of [petgraph] algorithms. Use [ModuleIndex::index] and
    /// [petgraph::graph::NodeIndex::new] to convert the root index.
    ///
    /// ```rust
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use petgraph::graph::{NodeIndex, UnGraph};
    /// use petgraph::visit::Dfs;
    /// use modular_decomposition::{modular_decomposition, ModuleKind};
    ///
    /// let graph = UnGraph::<(), ()>::from_edges([(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)]);
    /// let md = modular_decomposition(&graph)?;
    ///
    /// let root = NodeIndex::new(md.root().index());
    /// let digraph = md.into_digraph();
    ///
    /// let mut dfs = Dfs::new(&digraph, root);
    /// let mut node_order = vec![];
    /// while let Some(node) = dfs.next(&digraph) { node_order.push(*digraph.node_weight(node).unwrap()); }
    ///
    /// let expected_node_order = [
    ///     ModuleKind::Prime,
    ///     ModuleKind::Node(NodeIndex::new(2)),
    ///     ModuleKind::Parallel,
    ///     ModuleKind::Node(NodeIndex::new(0)),
    ///     ModuleKind::Node(NodeIndex::new(1)),
    ///     ModuleKind::Node(NodeIndex::new(3)),
    ///     ModuleKind::Parallel,
    ///     ModuleKind::Node(NodeIndex::new(4)),
    ///     ModuleKind::Node(NodeIndex::new(5)),
    /// ];
    /// assert_eq!(node_order, expected_node_order);
    /// # Ok(())
    /// # }
    /// ```
    pub fn into_digraph(self) -> DiGraph<ModuleKind<NodeId>, ()> {
        self.tree
    }
}

/// A graph does not contain any nodes or edges.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct NullGraphError;

impl Display for NullGraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("graph does not contain any nodes or edges")
    }
}

impl std::error::Error for NullGraphError {}

#[cfg(test)]
mod test {
    use petgraph::graph::{DiGraph, NodeIndex};
    use petgraph::visit::IntoNodeIdentifiers;
    use petgraph::Outgoing;

    use crate::md_tree::NullGraphError;
    use crate::tests::{complete_graph, pace2023_exact_024};
    use crate::{modular_decomposition, MDTree, ModuleIndex, ModuleKind};

    #[test]
    fn nodes() {
        let graph = pace2023_exact_024();
        let md = modular_decomposition(&graph).unwrap();
        let mut module_nodes: Vec<(ModuleKind<_>, Vec<_>)> = (0..md.node_count())
            .map(ModuleIndex::new)
            .map(|module| (*md.module_kind(module).unwrap(), md.nodes(module).map(|node| node.index()).collect()))
            .collect();
        module_nodes.iter_mut().for_each(|(_, nodes)| nodes.sort());
        module_nodes.sort();

        for (kind, nodes) in &module_nodes {
            if nodes.len() == 1 {
                assert_eq!(*kind, ModuleKind::Node(NodeIndex::new(nodes[0])));
            }
            if nodes.len() == graph.node_count() {
                assert_eq!(*nodes, graph.node_identifiers().map(|node| node.index()).collect::<Vec<_>>());
            }
        }

        module_nodes.retain(|(_, nodes)| nodes.len() > 1);

        let expected = [
            (
                ModuleKind::Prime,
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 26, 27, 28, 29, 30, 31,
                    32, 33, 34, 35, 36, 37, 38, 39,
                ],
            ),
            (ModuleKind::Series, vec![17, 18, 19]),
            (ModuleKind::Series, vec![24, 25]),
            (
                ModuleKind::Parallel,
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                ],
            ),
            (ModuleKind::Parallel, vec![20, 30]),
        ];
        assert_eq!(module_nodes, expected);
    }

    #[test]
    fn mdtree_and_digraph_are_equivalent() {
        let graph = complete_graph(5);
        let md = modular_decomposition(&graph).unwrap();
        let root = md.root();

        assert_eq!(md.module_kind(root), Some(&ModuleKind::Series));

        let children: Vec<_> = md.children(root).collect();
        assert_eq!(md.module_kind(children[0]), Some(&ModuleKind::Node(NodeIndex::new(0))));

        let md = md.into_digraph();
        let root = NodeIndex::new(root.index());

        let children: Vec<_> = md.neighbors_directed(root, Outgoing).collect();
        assert_eq!(md.node_weight(root), Some(&ModuleKind::Series));
        assert_eq!(md.node_weight(children[0]), Some(&ModuleKind::Node(NodeIndex::new(0))));
    }

    #[test]
    fn null_graph_error() {
        let digraph: DiGraph<ModuleKind<NodeIndex>, ()> = Default::default();
        let err = MDTree::from_digraph(digraph).unwrap_err();
        assert_eq!(err, NullGraphError);
        assert_eq!(format!("{}", err), "graph does not contain any nodes or edges".to_string());
    }

    #[test]
    fn module_index_fmt() {
        let idx = ModuleIndex::new(42);
        assert_eq!(format!("{:?}", idx), "ModuleIndex(42)".to_string())
    }
}
