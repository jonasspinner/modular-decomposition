use crate::{MDTree, ModuleIndex, ModuleKind};

impl<NodeId: Copy + PartialEq> MDTree<NodeId> {
    /// Returns whether the original graph is a [cograph](https://en.wikipedia.org/wiki/Cograph).
    pub fn is_cograph(&self) -> bool {
        self.module_kinds().all(|node| *node != ModuleKind::Prime)
    }
}

struct TwinsBase<'tree, NodeId: Copy + PartialEq, P> {
    md_tree: &'tree MDTree<NodeId>,
    stack: Vec<ModuleIndex>,
    pred: P,
}

impl<'tree, NodeId: Copy + PartialEq, P: FnMut(&ModuleKind<NodeId>) -> bool> TwinsBase<'tree, NodeId, P> {
    fn new(md_tree: &'tree MDTree<NodeId>, pred: P) -> Self {
        let stack = match md_tree.module_kind(md_tree.root()) {
            Some(ModuleKind::Node(_)) => vec![],
            _ => vec![md_tree.root()],
        };
        Self { md_tree, stack, pred }
    }
}

impl<'tree, NodeId: Copy + PartialEq, P: FnMut(&ModuleKind<NodeId>) -> bool> Iterator for TwinsBase<'tree, NodeId, P> {
    type Item = Vec<NodeId>;

    fn next(&mut self) -> Option<Self::Item> {
        // Walk the tree with a dfs on the internal nodes.
        // If a node has a matching type `$internal_node_type` then collect its leaf children
        // and return them if there are at least two.
        while let Some(node) = self.stack.pop() {
            let kind = self.md_tree.module_kind(node).expect("node is valid");
            if (self.pred)(kind) {
                let mut leaf_children = vec![];
                for child in self.md_tree.children(node) {
                    if let Some(ModuleKind::Node(u)) = self.md_tree.module_kind(child) {
                        leaf_children.push(*u);
                    } else {
                        self.stack.push(child);
                    }
                }
                if leaf_children.len() >= 2 {
                    return Some(leaf_children);
                }
            } else {
                let children = self
                    .md_tree
                    .children(node)
                    .filter(|child| !matches!(self.md_tree.module_kind(*child), Some(ModuleKind::Node(_))));
                self.stack.extend(children);
            }
        }
        None
    }
}

impl<NodeId: Copy + PartialEq> MDTree<NodeId> {
    /// Returns an iterator for the set of nodes of the original graph that are twins.
    ///
    /// Two nodes are twins when they have the same neighborhood excluding themselves, i.e. N(u) \ {v} = N(v) \ {u}.
    ///
    /// Twins are either true or false twins. See [MDTree::true_twins] and [MDTree::false_twins].
    ///
    /// ```rust
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use petgraph::graph::{NodeIndex, UnGraph};
    /// use modular_decomposition::modular_decomposition;
    ///
    /// // a K_2 + 2 K_1
    /// let graph = UnGraph::<(), ()>::from_edges([(2, 3)]);
    /// let md = modular_decomposition(&graph)?;
    ///
    /// let mut twins: Vec<_> = md.twins().collect();
    /// twins.iter_mut().for_each(|nodes| nodes.sort());
    /// twins.sort();
    /// assert_eq!(twins, [[NodeIndex::new(0), NodeIndex::new(1)], [NodeIndex::new(2), NodeIndex::new(3)]]);
    /// # Ok(())
    /// # }
    pub fn twins(&self) -> impl Iterator<Item = Vec<NodeId>> + '_ {
        TwinsBase::new(self, |kind| matches!(kind, ModuleKind::Series | ModuleKind::Parallel))
    }

    /// Returns an iterator for the set of nodes of the original graph that are true twins.
    ///
    /// Two nodes are true twins when they have the same closed neighborhood, i.e. N(u) = N(v).
    ///
    /// ```rust
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use petgraph::graph::{NodeIndex, UnGraph};
    /// use modular_decomposition::modular_decomposition;
    ///
    /// // a K_2 + 2 K_1
    /// let graph = UnGraph::<(), ()>::from_edges([(2, 3)]);
    /// let md = modular_decomposition(&graph)?;
    ///
    /// let mut twins: Vec<_> = md.true_twins().collect();
    /// twins.iter_mut().for_each(|nodes| nodes.sort());
    /// twins.sort();
    /// assert_eq!(twins, [[NodeIndex::new(2), NodeIndex::new(3)]]);
    /// # Ok(())
    /// # }
    pub fn true_twins(&self) -> impl Iterator<Item = Vec<NodeId>> + '_ {
        TwinsBase::new(self, |kind| kind == &ModuleKind::Series)
    }

    /// Returns an iterator for the set of nodes of the original graph that false twins.
    ///
    /// Two nodes are false twins when they have the same open neighborhood, i.e. N(u) u {u} = N(v) u {v}.
    ///
    /// ```rust
    /// # use std::error::Error;
    /// #
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// use petgraph::graph::{NodeIndex, UnGraph};
    /// use modular_decomposition::modular_decomposition;
    ///
    /// // a K_2 + 2 K_1
    /// let graph = UnGraph::<(), ()>::from_edges([(2, 3)]);
    /// let md = modular_decomposition(&graph)?;
    ///
    /// let mut twins: Vec<_> = md.false_twins().collect();
    /// twins.iter_mut().for_each(|nodes| nodes.sort());
    /// twins.sort();
    /// assert_eq!(twins, [[NodeIndex::new(0), NodeIndex::new(1)]]);
    /// # Ok(())
    /// # }
    pub fn false_twins(&self) -> impl Iterator<Item = Vec<NodeId>> + '_ {
        TwinsBase::new(self, |kind| kind == &ModuleKind::Parallel)
    }
}

#[cfg(test)]
mod test {
    use petgraph::graph::{NodeIndex, UnGraph};

    use crate::modular_decomposition;
    use crate::tests;

    fn to_usize(node_vecs: &[Vec<NodeIndex>]) -> Vec<Vec<usize>> {
        let mut node_vecs: Vec<Vec<usize>> =
            node_vecs.iter().map(|nodes| nodes.iter().map(|node| node.index()).collect()).collect();
        node_vecs.iter_mut().for_each(|nodes| nodes.sort());
        node_vecs.sort();
        node_vecs
    }

    #[test]
    fn cograph_k5() {
        let graph = tests::complete_graph(5);
        let md = modular_decomposition(&graph).unwrap();
        assert!(md.is_cograph());
    }

    #[test]
    fn cograph_e5() {
        let graph = tests::empty_graph(5);
        let md = modular_decomposition(&graph).unwrap();
        assert!(md.is_cograph());
    }

    #[test]
    fn cograph_p5() {
        let graph = tests::path_graph(5);
        let md = modular_decomposition(&graph).unwrap();
        assert!(!md.is_cograph());
    }

    #[test]
    fn twins_k1() {
        let graph = tests::complete_graph(1);
        let md = modular_decomposition(&graph).unwrap();
        let twins: Vec<_> = md.twins().collect();
        assert!(twins.is_empty());
        let true_twins: Vec<_> = md.true_twins().collect();
        assert!(true_twins.is_empty());
        let false_twins: Vec<_> = md.false_twins().collect();
        assert!(false_twins.is_empty());
    }

    #[test]
    fn twins_k3() {
        let graph = tests::complete_graph(3);
        let md = modular_decomposition(&graph).unwrap();
        let twins: Vec<_> = md.twins().collect();
        assert_eq!(to_usize(&twins), [[0, 1, 2]]);
        let true_twins: Vec<_> = md.true_twins().collect();
        assert_eq!(to_usize(&true_twins), [[0, 1, 2]]);
        let false_twins: Vec<_> = md.false_twins().collect();
        assert_eq!(to_usize(&false_twins), Vec::<Vec<_>>::new());
    }

    #[test]
    fn twins_e3() {
        let graph = tests::empty_graph(3);
        let md = modular_decomposition(&graph).unwrap();
        let twins: Vec<_> = md.twins().collect();
        assert_eq!(to_usize(&twins), [[0, 1, 2]]);
        let true_twins: Vec<_> = md.true_twins().collect();
        assert_eq!(to_usize(&true_twins), Vec::<Vec<_>>::new());
        let false_twins: Vec<_> = md.false_twins().collect();
        assert_eq!(to_usize(&false_twins), [[0, 1, 2]]);
    }

    #[test]
    fn twins_k2_plus_2k1() {
        let graph = UnGraph::<(), ()>::from_edges([(2, 3)]);
        let md = modular_decomposition(&graph).unwrap();
        let twins: Vec<_> = md.twins().collect();
        assert_eq!(to_usize(&twins), [[0, 1], [2, 3]]);
        let true_twins: Vec<_> = md.true_twins().collect();
        assert_eq!(to_usize(&true_twins), [[2, 3]]);
        let false_twins: Vec<_> = md.false_twins().collect();
        assert_eq!(to_usize(&false_twins), [[0, 1]]);
    }

    #[test]
    fn twins_k2_plus_2k1_complement() {
        let graph = UnGraph::<(), ()>::from_edges([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        let md = modular_decomposition(&graph).unwrap();
        let twins: Vec<_> = md.twins().collect();
        assert_eq!(to_usize(&twins), [[0, 1], [2, 3]]);
        let true_twins: Vec<_> = md.true_twins().collect();
        assert_eq!(to_usize(&true_twins), [[0, 1]]);
        let false_twins: Vec<_> = md.false_twins().collect();
        assert_eq!(to_usize(&false_twins), [[2, 3]]);
    }

    #[test]
    fn twins_pace2023_exact_054() {
        let graph = tests::pace2023_exact_054();
        let md_tree = modular_decomposition(&graph).unwrap();

        let mut twins: Vec<_> = md_tree.twins().collect();
        twins.iter_mut().for_each(|nodes| nodes.sort());
        twins.sort();
        assert_eq!(
            to_usize(&twins),
            [
                vec![24, 25],
                vec![32, 35],
                vec![44, 63, 64],
                vec![45, 46],
                vec![48, 49, 50],
                vec![52, 70],
                vec![53, 54],
                vec![55, 56],
                vec![57, 62],
                vec![65, 67],
                vec![68, 69],
                vec![71, 72]
            ]
        );

        let mut true_twins: Vec<_> = md_tree.true_twins().collect();
        true_twins.iter_mut().for_each(|nodes| nodes.sort());
        true_twins.sort();
        assert_eq!(
            to_usize(&true_twins),
            [
                vec![24, 25],
                vec![45, 46],
                vec![48, 49, 50],
                vec![53, 54],
                vec![55, 56],
                vec![65, 67],
                vec![68, 69],
                vec![71, 72]
            ]
        );

        let mut false_twins: Vec<_> = md_tree.false_twins().collect();
        false_twins.iter_mut().for_each(|nodes| nodes.sort());
        false_twins.sort();
        assert_eq!(to_usize(&false_twins), [vec![32, 35], vec![44, 63, 64], vec![52, 70], vec![57, 62]]);
    }
}
