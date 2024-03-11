use crate::deque::{Deque, DequeIndex};
use crate::index::make_index;
use crate::md_tree::NodeIndex;
use petgraph::visit::{GraphProp, IntoNeighbors, NodeCompactIndexable};
use petgraph::Undirected;
use std::ops::{Index, Range};
use tracing::instrument;

/// This algorithm uses an ordered partition datastructure that maintains a
/// permutation and its inverse sequentially in memory.
///
/// example representation of
///         [[1 4] [2] [0 3]]
/// positions: [4 0 2 3 1]    node -> position
///     nodes: [1 4 2 3 0]    position -> node
///     parts: [0 0 1 2 2]    position -> part
///
/// This allows parts and the elements of sequential parts to be represented by
/// a consecutive range of nodes. Therefore we can use a pair of numbers instead
/// of a whole set as a key for first_pivot and the elements of pivots and
/// modules.
///
/// For a sequence of nodes, which does not divide any parts on its boundaries,
/// the set of nodes in it will never change in the rest of the algorithm. It
/// will only be divided up into more and more parts, but the the nodes in it
/// will not change.
#[instrument(skip_all)]
pub(crate) fn factorizing_permutation<G>(graph: G) -> Permutation
where
    G: NodeCompactIndexable + IntoNeighbors + GraphProp<EdgeType = Undirected>,
{
    let mut state = State::new(&graph);
    state.partition_refinement();
    state.into_permutation()
}

#[allow(non_snake_case)]
impl<'g, G> State<'g, G>
where
    G: NodeCompactIndexable + IntoNeighbors + GraphProp<EdgeType = Undirected>,
{
    fn partition_refinement(&mut self) {
        while self.init_partition() {
            while let Some(Y) = self.pop_pivot() {
                let Y = self.part(Y).seq;
                for y_pos in Y.positions() {
                    self.refine(y_pos, Y);
                }
            }
        }
    }

    fn init_partition(&mut self) -> bool {
        assert!(self.pivots.is_empty());

        let Some(non_singleton) = self.next_non_singleton() else {
            return false;
        };

        if let Some(X) = self.pop_front_module() {
            let X = self.part(X).seq;
            let x = self.node(X.first()).node.min(self.node(X.last()).node);

            // Directly call refine.
            // This avoids adding {x} as a new part and directly removing it again.
            let x_pos = self.position(x);
            self.refine(x_pos, X);
        } else {
            let X_idx = non_singleton;
            let X = self.part(X_idx).seq;

            let x = self.node(X.first()).node.min(self.node(X.last()).node);

            let (S, L) = self.refine_central(X_idx, x);

            if let Some(S) = S {
                self.push_pivot(S);
            }
            self.push_back_module(L);

            assert!(self.pivots.len() <= 1);
            assert_eq!(self.modules.len(), 1);
        }
        true
    }

    fn refine_central(&mut self, X_idx: PartIndex, x: NodeIndex) -> (Option<PartIndex>, PartIndex) {
        // Divide X into A, {x}, N, where A = X n N(x), N = X \ (N(x) u {x})
        // This takes O(|N(x)|) time.

        let mut X = self.part(X_idx).seq;
        self.current_subgraph = X;

        // Create A
        let A_idx = self.new_part(X.first(), 0, false);
        let mut A = self.part(A_idx).seq;
        for v in Self::neighbors(self.graph, x) {
            let v_pos = self.position(v);
            if X.contains(v_pos) {
                self.swap_nodes(X.first(), v_pos);
                self.node_mut(X.first()).part = A_idx;
                A.grow_right();
                X.shrink_left();
            }
        }

        self.part_mut(A_idx).seq = A;

        let A_idx = if A.is_empty() {
            self.remove_part(A_idx);
            None
        } else {
            Some(A_idx)
        };

        // Create {x}
        let x_part_idx = self.new_part(X.first(), 1, false);
        self.swap_nodes(X.first(), self.position(x));
        self.node_mut(X.first()).part = x_part_idx;
        let x_pos = X.first();
        X.shrink_left();
        self.part_mut(X_idx).seq = X;

        // N is already there.

        self.center_pos = x_pos;

        let N_idx = if X.is_empty() {
            self.remove_part(X_idx);
            None
        } else {
            Some(X_idx)
        };

        match (A_idx, N_idx) {
            (Some(A_idx), Some(N_idx)) => {
                let (S, L) =
                    if self.part(A_idx).seq.len <= self.part(N_idx).seq.len { (A_idx, N_idx) } else { (N_idx, A_idx) };
                (Some(S), L)
            }
            (Some(A_idx), None) => (None, A_idx),
            (None, Some(N_idx)) => (None, N_idx),
            _ => unreachable!(),
        }
    }

    fn refine(&mut self, y_pos: NodePos, Y: Seq) {
        let y = self.node(y_pos).node;

        // The pivot set S is N(y) \ E. S is implicitly computed by iterating over N(y)
        // and checking if a neighbor of y is in the set E. This can be done
        // efficiently as E is consecutive range.

        for u in Self::neighbors(self.graph, y) {
            let u_pos = self.position(u);
            if Y.contains(u_pos) || !self.current_subgraph.contains(u_pos) {
                continue;
            }

            let X = self.part(self.node(u_pos).part).seq;
            if X.contains(self.center_pos) || X.contains(y_pos) {
                continue;
            }

            if self.should_insert_right(X, y_pos) {
                self.insert_right(u_pos);
            } else {
                self.insert_left(u_pos);
            }
        }

        // All refinement steps have been done at this point. Exactly the parts which
        // properly overlap the pivot set have the current gen. Get the proper
        // X, X_a pair depending on the location and call add_pivot(X, X_a).
        for u in Self::neighbors(self.graph, y) {
            let u_pos = self.position(u);
            if Y.contains(u_pos) || !self.current_subgraph.contains(u_pos) {
                continue;
            }
            let X_a_idx = self.node(u_pos).part;

            if self.part(X_a_idx).is_marked() {
                self.part_mut(X_a_idx).set_marked(false);

                let X_a = self.part(X_a_idx).seq;
                let X_idx = if self.should_insert_right(X_a, y_pos) {
                    self.nodes[X_a.first().index() - 1].part
                } else {
                    self.nodes[X_a.last().index() + 1].part
                };

                self.add_pivot(X_idx, X_a_idx);
            }
        }
    }

    fn add_pivot(&mut self, X_b: PartIndex, X_a: PartIndex) {
        if self.part(X_b).is_in_pivots() {
            self.push_pivot(X_a);
        } else {
            let X_b_len = self.part(X_b).seq.len;
            let X_a_len = self.part(X_a).seq.len;
            let (S, L) = if X_b_len <= X_a_len { (X_b, X_a) } else { (X_a, X_b) };
            self.push_pivot(S);

            if self.part(X_b).is_in_modules() {
                self.replace_module(X_b, L);
            } else {
                self.push_back_module(L);
            }
        }
    }

    #[inline]
    fn push_pivot(&mut self, pivot: PartIndex) {
        debug_assert!(!self.part(pivot).is_in_pivots());
        self.pivots.push(pivot);
        self.part_mut(pivot).set_in_pivots(true);
    }

    #[inline]
    fn pop_pivot(&mut self) -> Option<PartIndex> {
        let pivot = self.pivots.pop()?;
        debug_assert!(self.part(pivot).is_in_pivots());
        self.part_mut(pivot).set_in_pivots(false);
        Some(pivot)
    }

    #[inline]
    fn push_back_module(&mut self, module: PartIndex) {
        debug_assert!(!self.part(module).is_in_modules());
        let idx = self.modules.push_back(module);
        self.part_mut(module).modules_idx = idx;
    }

    #[inline]
    fn pop_front_module(&mut self) -> Option<PartIndex> {
        let module = self.modules.pop_front()?;
        debug_assert!(self.part(module).is_in_modules());
        self.part_mut(module).modules_idx = DequeIndex::end();
        Some(module)
    }

    fn replace_module(&mut self, old: PartIndex, new: PartIndex) {
        if old != new {
            let idx = std::mem::replace(&mut self.part_mut(old).modules_idx, DequeIndex::end());
            self.modules[idx] = new;
            self.part_mut(new).modules_idx = idx;
        }
    }

    #[inline]
    fn should_insert_right(&self, X: Seq, y_pos: NodePos) -> bool {
        let (a, b) = if y_pos < self.center_pos { (y_pos, self.center_pos) } else { (self.center_pos, y_pos) };
        let x = X.first;
        a < x && x < b
    }

    /// Inserts the node at `u_pos` into a part to the right of its original
    /// part. If the part to the right of it has been created in the current
    /// generation we can add to it. Otherwise we create a new part exactly
    /// to the right.
    fn insert_right(&mut self, u_pos: NodePos) {
        let part = self.node(u_pos).part;
        debug_assert!(!self.part(part).is_marked());
        let last = self.part(part).seq.last();

        let next = (last.index() + 1 != self.nodes.len())
            .then(|| self.nodes[last.index() + 1].part)
            .filter(|&next| self.part(next).is_marked())
            .unwrap_or_else(|| self.new_part(NodePos::new(last.index() + 1), 0, true));

        debug_assert!(!self.part(next).is_in_pivots());
        debug_assert!(!self.part(next).is_in_modules());
        debug_assert!(self.part(next).is_marked());

        self.swap_nodes(last, u_pos);
        self.part_mut(next).seq.grow_left();
        self.part_mut(part).seq.shrink_right();
        self.node_mut(last).part = next;

        if self.part(part).seq.is_empty() {
            // We moved all elements from X to X n S.
            // Undo all that and unmark the part to prevent interfering with neighboring
            // parts.
            self.part_mut(next).set_marked(false);
            self.part_mut(part).seq = self.part(next).seq;
            self.part_mut(next).seq.len = 0;
            self.remove_part(next);
            let range = self.part(part).seq.range();
            self.nodes[range].iter_mut().for_each(|n| {
                n.part = part;
            });
        }

        debug_assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
        debug_assert!(self.part(self.node(last).part).seq.contains(last));
    }

    fn insert_left(&mut self, u_pos: NodePos) {
        let part = self.node(u_pos).part;
        debug_assert!(!self.part(part).is_marked());
        let first = self.part(part).seq.first();

        let prev = (first.index() != 0)
            .then(|| self.nodes[first.index() - 1].part)
            .filter(|&prev| self.part(prev).is_marked())
            .unwrap_or_else(|| self.new_part(first, 0, true));

        debug_assert!(!self.part(prev).is_in_pivots());
        debug_assert!(!self.part(prev).is_in_modules());
        debug_assert!(self.part(prev).is_marked());

        self.swap_nodes(first, u_pos);
        self.part_mut(prev).seq.grow_right();
        self.part_mut(part).seq.shrink_left();
        self.node_mut(first).part = prev;

        if self.part(part).seq.is_empty() {
            // We moved all elements from X to X n S.
            // Undo all that and unmark the part to prevent interfering with neighboring
            // parts.
            self.part_mut(prev).set_marked(false);
            self.part_mut(part).seq = self.part(prev).seq;
            self.part_mut(prev).seq.len = 0;
            self.remove_part(prev);
            let range = self.part(part).seq.range();
            self.nodes[range].iter_mut().for_each(|n| {
                n.part = part;
            });
        }

        debug_assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
        debug_assert!(self.part(self.node(first).part).seq.contains(first));
    }

    #[inline]
    fn swap_nodes(&mut self, a: NodePos, b: NodePos) {
        let Node { part: a_part, node: u } = *self.node(a);
        let Node { part: b_part, node: v } = *self.node(b);
        debug_assert_eq!(a_part, b_part);
        self.positions.swap(u.index(), v.index());
        self.nodes.swap(a.index(), b.index());

        debug_assert_eq!(self.node(self.position(u)).node, u);
        debug_assert_eq!(self.node(self.position(v)).node, v);
    }

    fn neighbors(graph: &G, u: NodeIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        graph.neighbors(graph.from_index(u.index())).map(|v| NodeIndex::new(graph.to_index(v)))
    }

    #[inline(always)]
    fn part(&self, idx: PartIndex) -> &Part {
        &self.parts[idx.index()]
    }

    #[inline(always)]
    fn part_mut(&mut self, idx: PartIndex) -> &mut Part {
        &mut self.parts[idx.index()]
    }

    #[inline(always)]
    fn node(&self, pos: NodePos) -> &Node {
        &self.nodes[pos.index()]
    }

    #[inline(always)]
    fn node_mut(&mut self, pos: NodePos) -> &mut Node {
        &mut self.nodes[pos.index()]
    }

    #[inline(always)]
    fn position(&self, node: NodeIndex) -> NodePos {
        self.positions[node.index()]
    }

    #[inline(always)]
    fn remove_part(&mut self, idx: PartIndex) {
        debug_assert!(self.part(idx).seq.is_empty());
        debug_assert!(!self.part(idx).is_in_pivots());
        debug_assert!(!self.part(idx).is_in_modules());
        self.removed.push(idx);
    }

    fn new_part(&mut self, first: NodePos, len: u32, marked: bool) -> PartIndex {
        let idx = self.removed.pop().unwrap_or_else(|| {
            let idx = PartIndex(self.parts.len() as _);
            self.parts.push(Part::new(Seq::new(NodePos(0), 0)));
            idx
        });
        *self.part_mut(idx) = Part::new(Seq { first, len });
        self.part_mut(idx).set_marked(marked);
        idx
    }

    /// Returns the index of a part which has at least two elements.
    ///
    /// Parts never get bigger during the partition refinement. The function
    /// caches a index that moves left to right.
    ///
    /// The total complexity is O(n + k) where k is the number of times the
    /// function is called.
    fn next_non_singleton(&mut self) -> Option<PartIndex> {
        let mut j = self.non_singleton_idx.index();
        while j + 1 < self.nodes.len() {
            if self.nodes[j].part == self.nodes[j + 1].part {
                self.non_singleton_idx = NodePos::new(j);
                return Some(self.nodes[j].part);
            }
            j += 1;
        }
        None
    }
}

struct State<'g, G>
where
    G: NodeCompactIndexable + IntoNeighbors + GraphProp<EdgeType = Undirected>,
{
    graph: &'g G,

    positions: Vec<NodePos>,
    nodes: Vec<Node>,
    parts: Vec<Part>,
    removed: Vec<PartIndex>,

    pivots: Vec<PartIndex>,
    modules: Deque<PartIndex>,
    center_pos: NodePos,
    non_singleton_idx: NodePos,

    current_subgraph: Seq,
}

impl<'g, G> State<'g, G>
where
    G: NodeCompactIndexable + IntoNeighbors + GraphProp<EdgeType = Undirected>,
{
    fn new(graph: &'g G) -> Self {
        let n = graph.node_bound();

        let positions = (0..n).map(NodePos::new).collect();
        let nodes = (0..n).map(|u| Node { node: NodeIndex::new(u), part: PartIndex::new(0) }).collect();
        let mut parts = Vec::with_capacity(n + 1);
        parts.push(Part::new(Seq::new(NodePos::new(0), n as u32)));
        let removed = vec![];
        let pivots = vec![];
        let modules = Deque::with_capacity(0);
        let center_pos = NodePos(u32::MAX);
        let non_singleton_idx = NodePos::new(0);
        let current_subgraph = Seq::new(NodePos::new(0), n as u32);

        State {
            graph,
            positions,
            nodes,
            parts,
            removed,
            pivots,
            modules,
            center_pos,
            non_singleton_idx,
            current_subgraph,
        }
    }

    fn into_permutation(self) -> Permutation {
        Permutation { positions: self.positions, nodes: self.nodes }
    }
}

make_index!(NodePos);

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
struct Seq {
    first: NodePos,
    len: u32,
}

impl Seq {
    fn new(first: NodePos, len: u32) -> Self {
        Self { first, len }
    }
    fn is_empty(&self) -> bool {
        self.len == 0
    }
    fn first(&self) -> NodePos {
        debug_assert!(!self.is_empty());
        self.first
    }

    fn last(&self) -> NodePos {
        debug_assert!(!self.is_empty());
        NodePos::new(self.first.index() + (self.len as usize) - 1)
    }

    fn contains(&self, pos: NodePos) -> bool {
        let s = self.first.0;
        let e = s + self.len;
        (s <= pos.0) & (pos.0 < e)
    }

    fn range(&self) -> Range<usize> {
        let start = self.first.index();
        start..(start + self.len as usize)
    }

    fn positions(&self) -> impl Iterator<Item = NodePos> {
        self.range().map(NodePos::new)
    }

    fn grow_right(&mut self) {
        self.len += 1;
    }
    fn shrink_right(&mut self) {
        debug_assert!(!self.is_empty());
        self.len -= 1;
    }

    fn grow_left(&mut self) {
        debug_assert_ne!(self.first, NodePos(0));
        self.first.0 -= 1;
        self.len += 1;
    }
    fn shrink_left(&mut self) {
        debug_assert!(!self.is_empty());
        self.first.0 += 1;
        self.len -= 1;
    }
}

make_index!(PartIndex);

#[derive(Debug)]
struct Node {
    node: NodeIndex,
    part: PartIndex,
}

struct Part {
    seq: Seq,
    modules_idx: DequeIndex,
    flags: u8,
}

impl Part {
    fn new(seq: Seq) -> Self {
        Self { seq, modules_idx: DequeIndex::end(), flags: 0 }
    }
    fn is_marked(&self) -> bool {
        self.flags & 1 != 0
    }
    fn set_marked(&mut self, b: bool) {
        self.flags = if b { self.flags | 1 } else { self.flags & !1 };
    }
    fn is_in_pivots(&self) -> bool {
        self.flags & 2 != 0
    }
    fn set_in_pivots(&mut self, b: bool) {
        self.flags = if b { self.flags | 2 } else { self.flags & !2 };
    }
    fn is_in_modules(&self) -> bool {
        self.modules_idx != DequeIndex::end()
    }
}

pub(crate) struct Permutation {
    positions: Vec<NodePos>,
    nodes: Vec<Node>,
}

impl Index<usize> for Permutation {
    type Output = NodeIndex;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index].node
    }
}

impl Permutation {
    #[allow(unused)]
    pub(crate) fn new_identity(size: usize) -> Self {
        let positions = (0..size).map(NodePos::new).collect();
        let nodes = (0..size).map(NodeIndex::new).map(|node| Node { node, part: Default::default() }).collect();
        Self { positions, nodes }
    }
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
    pub(crate) fn position(&self, u: NodeIndex) -> usize {
        self.positions[u.index()].index()
    }
}

impl<'a> Permutation {
    pub(crate) fn iter(&'a self) -> impl Iterator<Item = NodeIndex> + 'a {
        self.nodes.iter().map(|n| n.node)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn empty_graph() {
        let graph = UnGraph::<(), ()>::new_undirected();
        let p = factorizing_permutation(&graph);
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn empty_one_vertex_graph() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        graph.add_node(());
        let p = factorizing_permutation(&graph);
        assert_eq!(p.len(), 1);
    }
}
