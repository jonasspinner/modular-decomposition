use std::iter::zip;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use common::modular_decomposition::MDNodeKind;
use tracing::{info, instrument};
use crate::improved::factorizing_permutation::Permutation;


#[allow(dead_code)]
pub(crate) fn modular_decomposition(graph: &UnGraph<(), ()>) -> DiGraph<MDNodeKind, ()> {
    let mut graph: Vec<Vec<NodeIndex>> = graph.node_indices().map(|u| graph.neighbors(u).collect()).collect();
    graph.iter_mut().for_each(|neighbors| neighbors.sort());

    let p = factorizing_permutation(&graph);

    let n = p.len();
    let mut op = vec![0; n];
    op[0] = 1;
    let mut cl = vec![0; n];
    cl[n - 1] = 1;
    let mut lc: Vec<_> = (0..n - 1).map(|i| i as u32).collect();
    let mut uc: Vec<_> = (1..n).map(|i| i as u32).collect();

    build_parenthesizing(&mut graph, &mut op, &mut cl, &mut lc, &mut uc, &p);

    info!(n);

    remove_non_module_dummy_nodes(&mut op, &mut cl, &lc, &uc);

    create_consecutive_twin_nodes(&mut op, &mut cl, &lc, &uc);

    remove_singleton_dummy_nodes(&mut op, &mut cl);

    let tree = build_tree(&graph, &op, &cl, &p);

    info!(number_of_nodes = tree.node_count(), number_of_inner_nodes = tree.node_count() - graph.len());

    tree
}

#[instrument(skip_all)]
pub(crate) fn factorizing_permutation(graph: &[Vec<NodeIndex>]) -> Permutation {
    factorizing_permutation::factorizing_permutation(graph)
}

#[instrument(skip_all)]
pub(crate) fn build_parenthesizing(graph: &mut [Vec<NodeIndex>], op: &mut [u32], cl: &mut [u32], lc: &mut [u32], uc: &mut [u32], p: &Permutation) {
    graph.iter_mut().for_each(|neighbors| neighbors.sort_by_key(|u| { p.position(*u) }));

    fn merge_iter_next<'a>(mut a: impl Iterator<Item=&'a NodeIndex>, mut b: impl Iterator<Item=&'a NodeIndex>, f: impl Fn(NodeIndex) -> usize, g: impl Fn(usize, usize) -> usize) -> Option<usize> {
        loop {
            match (a.next(), b.next()) {
                (Some(i), Some(j)) => { if i != j { break Some(g(f(*i), f(*j))); } }
                (Some(i), None) => break Some(f(*i)),
                (None, Some(j)) => break Some(f(*j)),
                (None, None) => break None,
            }
        }
    }

    let first = |a: &[NodeIndex], b: &[NodeIndex]| -> Option<usize> {
        merge_iter_next(a.iter(), b.iter(), |u| p.position(u), usize::min)
    };
    let last = |a: &[NodeIndex], b: &[NodeIndex]| -> Option<usize> {
        merge_iter_next(a.iter().rev(), b.iter().rev(), |u| p.position(u), usize::max)
    };

    let n = p.len();
    for j in 0..n - 1 {
        let pos_j0 = &graph[p[j].index()];
        let pos_j1 = &graph[p[j + 1].index()];

        let mut lc_idx = first(pos_j0, pos_j1);
        let mut uc_idx = if lc_idx.is_some() { last(pos_j0, pos_j1) } else { None };
        lc_idx = lc_idx.filter(|i| *i < j);
        uc_idx = uc_idx.filter(|i| *i > j + 1);

        if let Some(i) = lc_idx {
            assert!(i < j);
            debug_assert_ne!(graph[p[i].index()].contains(&p[j]), graph[p[i].index()].contains(&p[j + 1]));
            op[i] += 1;
            cl[j] += 1;
            lc[j] = i as u32;
        }
        if let Some(i) = uc_idx {
            assert!(i > j + 1 && i < n);
            debug_assert_ne!(graph[p[i].index()].contains(&p[j]), graph[p[i].index()].contains(&p[j + 1]));
            op[j + 1] += 1;
            cl[i] += 1;
            uc[j] = i as u32;
        }
    }
}


pub(crate) struct BuildStack<T> {
    values: Vec<T>,
    starts: Vec<(u32, u32)>,
    len: usize,
}

impl<T> BuildStack<T> {
    pub(crate) fn new() -> Self {
        Self { values: vec![], starts: vec![(0, 1)], len: 0 }
    }
    pub(crate) fn extend(&mut self, n: usize) {
        let start = self.values.len() as u32;
        if n > 0 { self.starts.push((start, n as u32)); }
    }
    pub(crate) fn add(&mut self, value: T) {
        self.values.truncate(self.len);
        self.values.push(value);
        self.len += 1;
    }
    pub(crate) fn pop(&mut self) -> &[T] {
        let (i, last_count) = self.starts.last_mut().unwrap();
        let i = *i as usize;
        *last_count -= 1;
        if *last_count == 0 { self.starts.pop().unwrap(); }
        self.len = i;
        &self.values[i..]
    }
    pub(crate) fn is_empty(&self) -> bool { self.starts.is_empty() }
}

#[instrument(skip_all)]
pub(crate) fn remove_non_module_dummy_nodes(op: &mut [u32], cl: &mut [u32], lc: &[u32], uc: &[u32]) {
    #[derive(Clone)]
    struct Info {
        first_vertex: u32,
        last_vertex: u32,
        first_cutter: u32,
        last_cutter: u32,
    }
    impl Info {
        fn new(first_vertex: u32, last_vertex: u32, first_cutter: u32, last_cutter: u32) -> Self {
            Self { first_vertex, last_vertex, first_cutter, last_cutter }
        }
    }
    fn create_vertex_node(position: usize) -> Info {
        let pos = position as u32;
        Info::new(pos, pos, pos, pos)
    }

    let create_inner_node = |op: &mut [u32], cl: &mut [u32], nodes: &[Info]| -> Info {
        let k = nodes.len();

        assert!((0..k - 1).all(|i| nodes[i].last_vertex + 1 == nodes[i + 1].first_vertex));

        let first_vertex = nodes[0].first_vertex;
        let last_vertex = nodes[k - 1].last_vertex;
        let first_cutter = u32::min(
            first_vertex,
            nodes[0..k - 1].iter().map(|n| lc[n.last_vertex as usize])
                .chain(nodes.iter().map(|n| n.first_cutter)).min().unwrap_or(first_vertex));
        let last_cutter = u32::max(
            last_vertex,
            nodes[0..k - 1].iter().map(|n| uc[n.last_vertex as usize])
                .chain(nodes.iter().map(|n| n.last_cutter)).max().unwrap_or(last_vertex));

        let info = Info::new(first_vertex, last_vertex, first_cutter, last_cutter);
        if first_vertex < last_vertex && first_vertex <= first_cutter && last_cutter <= last_vertex {
            return info;
        }
        assert!(op[info.first_vertex as usize] >= 1);
        assert!(cl[info.last_vertex as usize] >= 1);
        op[info.first_vertex as usize] -= 1;
        cl[info.last_vertex as usize] -= 1;
        info
    };


    let mut s = BuildStack::new();
    for j in 0..op.len() {
        s.extend(op[j] as _);
        s.add(create_vertex_node(j));
        for _ in 0..cl[j] {
            let info = create_inner_node(op, cl, s.pop());
            s.add(info);
        }
    }
    create_inner_node(op, cl, s.pop());
    debug_assert_eq!(op.iter().sum::<u32>(), cl.iter().sum());
    assert!(s.is_empty());
}

#[instrument(skip_all)]
pub(crate) fn create_consecutive_twin_nodes(op: &mut [u32], cl: &mut [u32], lc: &[u32], uc: &[u32]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);
    let mut l = 0;
    for k in 0..n {
        s.push((k, l));
        l = k;
        s.extend(std::iter::repeat((k, k)).take(op[k] as _));
        for c in (0..cl[k] + 1).rev() {
            let (j, i) = s.pop().unwrap();

            l = i; // continue twin chain by default
            if i >= j { continue; }
            if i <= (lc[j - 1] as usize) && (lc[j - 1] as usize) < (uc[j - 1] as usize) && (uc[j - 1] as usize) <= k {
                // this node and prev are twins
                if c > 0 {
                    // not last parens ∴ last twin
                    op[i] += 1;
                    cl[k] += 1;
                    l = k + 1;
                }
            } else {
                // this node and prev aren't twins
                if i < j - 1 {
                    op[i] += 1;
                    cl[j - 1] += 1;
                }
                l = j; // this node starts new chain
            }
        }
    }
}

#[instrument(skip_all)]
pub(crate) fn remove_singleton_dummy_nodes(op: &mut [u32], cl: &mut [u32]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);

    for j in 0..n {
        s.extend(std::iter::repeat(j).take(op[j] as _));
        let mut i_ = usize::MAX;
        for _ in 0..cl[j] {
            let i = s.pop().unwrap();
            if i == i_ {
                op[i] -= 1;
                cl[j] -= 1;
            }
            i_ = i;
        }
    }
    op[0] -= 1;
    cl[n - 1] -= 1;
}

#[instrument(skip_all)]
#[allow(unreachable_code)]
pub(crate) fn build_tree(graph: &[Vec<NodeIndex>], op: &[u32], cl: &[u32], p: &Permutation) -> DiGraph<MDNodeKind, ()> {
    let n = graph.len();

    // Calculate the degrees between children of a module.
    // Every module keeps a graph node as a representative. Mark the representatives of the children.
    // For each representative, iterate over its neighbors. If a neighbor is also a representative,
    // increment the degree.

    let mut marked = vec![false; n];
    let mut degrees: Vec<usize> = vec![];

    let mut add_node = |t: &mut DiGraph<MDNodeKind, ()>, nodes: &[(NodeIndex, NodeIndex)]| -> (NodeIndex, NodeIndex) {
        for (x, _) in nodes { marked[x.index()] = true; }

        // Prepare a vec with nodes.len() zeros at the front.
        let clear_len = degrees.len().min(nodes.len());
        degrees[0..clear_len].iter_mut().for_each(|d| *d = 0);
        degrees.resize(nodes.len(), 0);

        for (d, (x, _)) in zip(degrees.iter_mut(), nodes.iter()) {
            *d = graph[x.index()].iter().copied().map(|v| marked[v.index()] as usize).sum();
        }
        for (x, _) in nodes { marked[x.index()] = false; }

        let n = nodes.len();
        let degree_sum = degrees.iter().sum::<usize>();
        assert!(degree_sum <= n * (n - 1));
        let kind = if degree_sum == 0 { MDNodeKind::Parallel } else if degree_sum == n * (n - 1) { MDNodeKind::Series } else { MDNodeKind::Prime };

        let idx = if let Some((_, u)) =
            nodes.iter()
                .find(|(_, u)| t[*u] == kind && kind != MDNodeKind::Prime) {
            *u
        } else {
            t.add_node(kind)
        };
        for (_, u) in nodes {
            if *u != idx {
                t.add_edge(idx, *u, ());
            }
        }
        // Use the representative of the first child as representative
        // Might not be optimal if it has a larger degree than other nodes in the module.
        // Has been compared with i) child representative with the smallest degree, and
        // ii) child representative with the smallest degree from the first 32 children only.
        // Neither option was better for the instances investigated.
        let x = nodes[0].0;
        (x, idx)
    };

    let mut t = DiGraph::new();

    let mut s = BuildStack::new();
    for (j, x) in p.iter().enumerate() {
        s.extend(op[j] as _);
        let (x, idx) = (x, t.add_node(MDNodeKind::Vertex(x.index())));
        s.add((x, idx));
        for _ in 0..cl[j] {
            let (x, idx) = add_node(&mut t, s.pop());
            s.add((x, idx));
        }
    }
    add_node(&mut t, s.pop());
    assert!(s.is_empty());
    t
}

pub(crate) mod factorizing_permutation {
    use std::ops::{Index, IndexMut, Range};
    use petgraph::graph::NodeIndex;
    use common::make_index;

    /// This algorithm uses an ordered partition datastructure that maintains a permutation and its
    /// inverse sequentially in memory.
    ///
    /// example representation of
    ///         [[1 4] [2] [0 3]]
    /// positions: [4 0 2 3 1]    node -> position
    ///     nodes: [1 4 2 3 0]    position -> node
    ///     parts: [0 0 1 2 2]    position -> part
    ///
    /// This allows parts and the elements of sequential parts to be represented by a consecutive
    /// range of nodes. Therefore we can use a pair of numbers instead of a whole set as a key for
    /// first_pivot and the elements of pivots and modules.
    ///
    /// For a sequence of nodes, which does not divide any parts on its boundaries, the set of nodes
    /// in it will never change in the rest of the algorithm. It will only be divided up into
    /// more and more parts, but the the nodes in it will not change.
    pub(crate) fn factorizing_permutation(graph: &[Vec<NodeIndex>]) -> Permutation {
        let mut state = State::new(graph);
        state.partition_refinement();
        state.into_permutation()
    }

    #[allow(non_snake_case)]
    impl<'a> State<'a> {
        fn partition_refinement(&mut self) {
            while self.init_partition() {
                while let Some(E) = self.pop_pivot() {
                    let E = self.part(E).seq;
                    // NOTE: E may or may not be a single part
                    for x_pos in E.positions() {
                        let x = self.node(x_pos).node;
                        self.refine(x, E);
                    }
                }
            }
        }

        fn init_partition(&mut self) -> bool {
            assert!(self.pivots.is_empty());

            let Some(non_singleton) = self.next_non_singleton() else { return false; };

            if let Some(X) = self.pop_front_module() {
                let X = self.part(X).seq;
                let x = self.node(X.first()).node.min(self.node(X.last()).node);

                // Directly call refine.
                // This avoids adding {x} as a new part and directly removing it again.
                self.refine(x, X);
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

            // Create A
            let A_idx = self.new_part(X.first(), 0, false);
            let mut A = self.part(A_idx).seq;
            for &v in &self.graph[x.index()] {
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
                self.remove_if_empty(A_idx);
                None
            } else { Some(A_idx) };

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
                self.remove_if_empty(X_idx);
                None
            } else { Some(X_idx) };

            if let Some(A_idx) = A_idx {
                if let Some(N_idx) = N_idx {
                    if self.part(A_idx).seq.len <= self.part(N_idx).seq.len {
                        (Some(A_idx), N_idx)
                    } else { (Some(N_idx), A_idx) }
                } else { (None, A_idx) }
            } else { (None, N_idx.unwrap()) }
        }

        fn refine(&mut self, y: NodeIndex, E: Seq) {
            let y_pos = self.position(y);

            // The pivot set S is N(y) \ E. S is implicitly computed by iterating over N(y) and
            // checking if a neighbor of y is in the set E. This can be done efficiently as E is
            // consecutive range.

            for &u in &self.graph[y.index()] {
                let u_pos = self.position(u);
                if E.contains(u_pos) { continue; }

                let X = self.part(self.node(u_pos).part).seq;
                if X.contains(self.center_pos) || X.contains(y_pos) { continue; }

                if self.should_insert_right(X, y_pos) {
                    self.insert_right(u_pos);
                } else {
                    self.insert_left(u_pos);
                }
            }

            // All refinement steps have been done at this point. Exactly the parts which properly
            // overlap the pivot set have the current gen. Get the proper X, X_a pair depending on
            // the location and call add_pivot(X, X_a).
            for &u in &self.graph[y.index()] {
                let u_pos = self.position(u);

                let X_a_idx = self.node(u_pos).part;
                let X_a = self.part(X_a_idx).seq;

                if self.part(X_a_idx).is_marked() {
                    self.part_mut(X_a_idx).set_marked(false);

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
                    if X_b != L {
                        let idx = std::mem::replace(&mut self.part_mut(X_b).modules_idx, u32::MAX);
                        self.modules[idx] = L;
                        self.part_mut(L).modules_idx = idx;
                    }
                } else {
                    self.push_back_module(L);
                }
            }
        }


        fn push_pivot(&mut self, pivot: PartIndex) {
            assert!(!self.part(pivot).is_in_pivots());
            self.pivots.push(pivot);
            self.part_mut(pivot).set_in_pivots(true);
        }
        fn pop_pivot(&mut self) -> Option<PartIndex> {
            let pivot = self.pivots.pop()?;
            assert!(self.part(pivot).is_in_pivots());
            self.part_mut(pivot).set_in_pivots(false);
            Some(pivot)
        }
        fn push_back_module(&mut self, module: PartIndex) {
            assert!(!self.part(module).is_in_modules());
            let idx = self.modules.push_back(module);
            self.part_mut(module).modules_idx = idx;
        }
        fn pop_front_module(&mut self) -> Option<PartIndex> {
            let module = self.modules.pop_front()?;
            assert!(self.part(module).is_in_modules());
            self.part_mut(module).modules_idx = u32::MAX;
            Some(module)
        }

        fn should_insert_right(&self, X: Seq, y_pos: NodePos) -> bool {
            let (a, b) = if y_pos < self.center_pos { (y_pos, self.center_pos) } else { (self.center_pos, y_pos) };
            let x = X.first;
            a < x && x < b
        }


        /// Inserts the node at `u_pos` into a part to the right of its original part.
        /// If the part to the right of it has been created in the current generation we can add to
        /// it. Otherwise we create a new part exactly to the right.
        fn insert_right(&mut self, u_pos: NodePos) {
            let part = self.node(u_pos).part;
            assert!(!self.part(part).is_marked());
            let last = self.part(part).seq.last();

            let next = (last.index() + 1 != self.nodes.len()).then(|| {
                self.nodes[last.index() + 1].part
            }).filter(|&next| self.part(next).is_marked())
                .unwrap_or_else(|| {
                    self.new_part(NodePos::new(last.index() + 1), 0, true)
                });

            assert!(!self.part(next).is_in_pivots());
            assert!(!self.part(next).is_in_modules());
            assert!(self.part(next).is_marked());

            self.swap_nodes(last, u_pos);
            self.part_mut(next).seq.grow_left();
            self.part_mut(part).seq.shrink_right();
            self.node_mut(last).part = next;

            if self.part(part).seq.is_empty() {
                // We moved all elements from X to X n S.
                // Undo all that and unmark the part to prevent interfering with neighboring parts.
                self.remove_from_current_gen(next);
                self.part_mut(part).seq = self.part(next).seq;
                self.part_mut(next).seq.len = 0;
                self.remove_if_empty(next);
                let range = self.part(part).seq.range();
                self.nodes[range].iter_mut().for_each(|n| { n.part = part; });
            }

            debug_assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
            debug_assert!(self.part(self.node(last).part).seq.contains(last));
        }

        fn insert_left(&mut self, u_pos: NodePos) {
            let part = self.node(u_pos).part;
            assert!(!self.is_current_gen(part));
            let first = self.part(part).seq.first();

            let prev = (first.index() != 0).then(|| {
                self.nodes[first.index() - 1].part
            }).filter(|&prev| self.is_current_gen(prev))
                .unwrap_or_else(|| self.new_part(first, 0, true));

            assert!(!self.part(prev).is_in_pivots());
            assert!(!self.part(prev).is_in_modules());
            assert!(self.is_current_gen(prev));

            self.swap_nodes(first, u_pos);
            self.part_mut(prev).seq.grow_right();
            self.part_mut(part).seq.shrink_left();
            self.node_mut(first).part = prev;

            if self.part(part).seq.is_empty() {
                // We moved all elements from X to X n S.
                // Undo all that and unmark the part to prevent interfering with neighboring parts.
                self.remove_from_current_gen(prev);
                self.part_mut(part).seq = self.part(prev).seq;
                self.part_mut(prev).seq.len = 0;
                self.remove_if_empty(prev);
                let range = self.part(part).seq.range();
                self.nodes[range].iter_mut().for_each(|n| { n.part = part; });
            }

            debug_assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
            debug_assert!(self.part(self.node(first).part).seq.contains(first));
        }

        fn swap_nodes(&mut self, a: NodePos, b: NodePos) {
            let Node { part: a_part, node: u } = *self.node(a);
            let Node { part: b_part, node: v } = *self.node(b);
            assert_eq!(a_part, b_part);
            self.positions.swap(u.index(), v.index());
            self.nodes.swap(a.index(), b.index());

            debug_assert_eq!(self.node(self.position(u)).node, u);
            debug_assert_eq!(self.node(self.position(v)).node, v);
        }

        fn part(&self, idx: PartIndex) -> &Part { &self.parts[idx.index()] }
        fn part_mut(&mut self, idx: PartIndex) -> &mut Part { &mut self.parts[idx.index()] }
        fn node(&self, pos: NodePos) -> &Node { &self.nodes[pos.index()] }
        fn node_mut(&mut self, pos: NodePos) -> &mut Node { &mut self.nodes[pos.index()] }
        fn position(&self, node: NodeIndex) -> NodePos { self.positions[node.index()] }

        fn remove_if_empty(&mut self, idx: PartIndex) {
            assert!(!self.part(idx).is_in_pivots());
            assert!(!self.part(idx).is_in_modules());
            if self.part(idx).seq.is_empty() {
                self.removed.push(idx);
            }
        }

        fn new_part(&mut self, first: NodePos, len: u32, is_current_gen: bool) -> PartIndex {
            let idx = if let Some(idx) = self.removed.pop() {
                *self.part_mut(idx) = Part::new(Seq { first, len });
                idx
            } else {
                let idx = PartIndex::new(self.parts.len());
                self.parts.push(Part::new(Seq { first, len }));
                idx
            };
            self.part_mut(idx).set_marked(is_current_gen);
            idx
        }

        fn is_current_gen(&self, idx: PartIndex) -> bool { self.part(idx).is_marked() }
        fn remove_from_current_gen(&mut self, idx: PartIndex) { self.part_mut(idx).set_marked(false); }

        /// Returns the index of a part which has at least two elements.
        ///
        /// Parts never get bigger during the partition refinement. The function caches a index that
        /// moves left to right.
        ///
        /// The total complexity is O(n + k) where k is the number of times the function is called.
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

    struct State<'graph> {
        graph: &'graph [Vec<NodeIndex>],

        positions: Vec<NodePos>,
        nodes: Vec<Node>,
        parts: Vec<Part>,
        removed: Vec<PartIndex>,

        pivots: Vec<PartIndex>,
        modules: Deque<PartIndex>,
        center_pos: NodePos,
        non_singleton_idx: NodePos,
    }

    impl<'a> State<'a> {
        fn new(graph: &'a [Vec<NodeIndex>]) -> Self {
            let n = graph.len();

            let positions = (0..n).map(NodePos::new).collect();
            let nodes = (0..n).map(|u| Node { node: NodeIndex::new(u), part: PartIndex::new(0) }).collect();
            let mut parts = Vec::with_capacity(n);
            parts.push(Part::new(Seq::new(NodePos::new(0), n as u32)));
            let removed = vec![];
            let pivots = vec![];
            let modules = Deque::new();
            let center_pos = NodePos(u32::MAX);
            let non_singleton_idx = NodePos::new(0);

            State { graph, positions, nodes, parts, removed, pivots, modules, center_pos, non_singleton_idx }
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
        fn new(first: NodePos, len: u32) -> Self { Self { first, len } }
        fn is_empty(&self) -> bool { self.len == 0 }
        fn first(&self) -> NodePos {
            assert!(!self.is_empty());
            self.first
        }

        fn last(&self) -> NodePos {
            assert!(!self.is_empty());
            NodePos::new(self.first.index() + (self.len as usize) - 1)
        }

        fn contains(&self, pos: NodePos) -> bool { self.range().contains(&pos.index()) }

        fn range(&self) -> Range<usize> {
            let start = self.first.index();
            start..(start + self.len as usize)
        }

        fn positions(&self) -> impl Iterator<Item=NodePos> {
            self.range().map(NodePos::new)
        }

        fn grow_right(&mut self) { self.len += 1; }
        fn shrink_right(&mut self) {
            assert!(!self.is_empty());
            self.len -= 1;
        }

        fn grow_left(&mut self) {
            assert_ne!(self.first, NodePos(0));
            self.first.0 -= 1;
            self.len += 1;
        }
        fn shrink_left(&mut self) {
            assert!(!self.is_empty());
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

    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    struct Gen(u32);

    struct Part {
        seq: Seq,
        modules_idx: u32,
        flags: u8,
    }

    impl Part {
        fn new(seq: Seq) -> Self {
            Self { seq, modules_idx: u32::MAX, flags: 0 }
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
            self.modules_idx != u32::MAX
        }
    }

    pub(crate) struct Permutation {
        positions: Vec<NodePos>,
        nodes: Vec<Node>,
    }

    impl Index<usize> for Permutation {
        type Output = NodeIndex;

        fn index(&self, index: usize) -> &Self::Output { &self.nodes[index].node }
    }

    impl Permutation {
        pub(crate) fn len(&self) -> usize { self.nodes.len() }
        pub(crate) fn position(&self, u: NodeIndex) -> usize { self.positions[u.index()].index() }
    }

    impl<'a> Permutation {
        pub(crate) fn iter(&'a self) -> impl Iterator<Item=NodeIndex> + 'a { self.nodes.iter().map(|n| n.node) }
    }


    struct Deque<T> {
        elements: Vec<T>,
        head: u32,
        tail: u32,
    }

    impl<T> Deque<T> {
        fn new() -> Self {
            Self { elements: vec![], head: 0, tail: 0 }
        }
        fn len(&self) -> usize {
            (self.head - self.tail) as usize
        }
        fn from_index_to_pos(idx: u32, n: usize) -> usize {
            idx as usize % n
        }
    }

    impl<T: Copy + Default> Deque<T> {
        fn push_back(&mut self, element: T) -> u32 {
            if self.head == self.elements.len() as u32 + self.tail {
                let old_len = self.elements.len();
                let new_len = 1024.max(old_len * 2);
                self.elements.resize(new_len, T::default());
                if self.head != self.tail {
                    let h = Self::from_index_to_pos(self.head, old_len);
                    let t = Self::from_index_to_pos(self.tail, old_len);
                    self.elements.copy_within(0..h, t);
                }
            }
            let idx = self.head;
            let pos = Self::from_index_to_pos(idx, self.elements.len());
            self.elements[pos] = element;
            self.head += 1;
            idx
        }
        fn pop_front(&mut self) -> Option<T> {
            if self.head != self.tail {
                let pos = Self::from_index_to_pos(self.tail, self.elements.len());
                let element = self.elements[pos];
                self.tail += 1;
                if self.head == self.tail {
                    self.head = 0;
                    self.tail = 0;
                }
                Some(element)
            } else {
                None
            }
        }
    }

    impl<T> Index<u32> for Deque<T> {
        type Output = T;

        fn index(&self, index: u32) -> &Self::Output {
            assert!(self.tail <= index);
            assert!(index < self.head);
            let pos = Self::from_index_to_pos(index, self.elements.len());
            &self.elements[pos]
        }
    }

    impl<T> IndexMut<u32> for Deque<T> {
        fn index_mut(&mut self, index: u32) -> &mut Self::Output {
            assert!(self.tail <= index);
            assert!(index < self.head);
            let pos = Self::from_index_to_pos(index, self.elements.len());
            &mut self.elements[pos]
        }
    }

    #[cfg(test)]
    mod test {
        use super::factorizing_permutation;

        #[test]
        fn ted08_test0_graph() {
            let graph = common::instances::ted08_test0();
            let graph: Vec<_> = graph.node_indices().map(|u| graph.neighbors(u).collect()).collect();

            let p = factorizing_permutation(&graph);
            println!("{:?}", p.iter().map(|n| n.index()).collect::<Vec<_>>());
        }
    }
}

