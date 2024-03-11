use crate::index::make_index;
use petgraph::graph::DiGraph;
use petgraph::{Incoming, Outgoing};
use std::fmt::{Debug, Formatter};

make_index!(pub NodeIndex);

/// Module kinds of nodes in a [MDTree].
#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ModuleKind {
    /// A prime module.
    Prime,
    /// A series module.
    Series,
    /// A parallel module.
    Parallel,
    /// A trivial module with a single vertex.
    Vertex(NodeIndex),
}

impl Debug for ModuleKind {
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
            ModuleKind::Vertex(v) => {
                write!(f, "{v}")
            }
        }
    }
}

/// A modular decomposition tree. The tree contains at least one node.
#[derive(Clone, Debug)]
pub struct MDTree {
    tree: DiGraph<ModuleKind, ()>,
    root: petgraph::graph::NodeIndex,
}

/// Module identifier.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ModuleIndex(petgraph::graph::NodeIndex);

impl MDTree {
    /// Create a new modular decomposition tree.
    ///
    /// Assumes that the input `DiGraph` is rooted tree with node weights
    /// `Prime`, `Series`, and `Parallel` for inner nodes and `Vertex(_)`
    /// for leaf nodes. This is not checked explicitly.
    ///
    /// Return `NullGraph` if the input graph does not have any nodes.
    ///
    /// Panics if all nodes have a non-zero in-degree.
    pub(crate) fn from_digraph(tree: DiGraph<ModuleKind, ()>) -> Result<Self, NullGraph> {
        if tree.node_count() == 0 {
            return Err(NullGraph);
        }
        let root = tree.externals(Incoming).next().expect("non-null trees must have a root");
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
        ModuleIndex(self.root)
    }

    /// Access the [ModuleKind] of a module.
    ///
    /// If the module does not exist, return None.
    pub fn module_kind(&self, module: ModuleIndex) -> Option<&ModuleKind> {
        self.tree.node_weight(module.0)
    }

    /// Return an iterator yielding references to [ModuleKind]s for all nodes.
    pub fn module_kinds(&self) -> impl Iterator<Item = &ModuleKind> {
        self.tree.node_weights()
    }

    /// Return an iterator for the children of a node.
    pub fn children(&self, module: ModuleIndex) -> impl Iterator<Item = ModuleIndex> + '_ {
        self.tree.neighbors_directed(module.0, Outgoing).map(ModuleIndex)
    }

    /// Convert to [DiGraph].
    pub fn into_digraph(self) -> DiGraph<ModuleKind, ()> {
        self.tree
    }
}

/// A graph does not contain any nodes or edges.
#[derive(Copy, Clone, Debug)]
pub struct NullGraph;
