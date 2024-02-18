use petgraph::data::{Element, FromElements};
use petgraph::graph::{DiGraph, NodeIndex};
use common::modular_decomposition::MDNodeKind;
use tracing::{info, info_span, instrument};
use crate::improved::factorizing_permutation::Permutation;


#[instrument(skip_all)]
pub(crate) fn modular_decomposition(graph: &mut [Vec<NodeIndex>]) -> DiGraph<MDNodeKind, ()> {
    let n = graph.len();
    if n == 0 { return DiGraph::new(); }
    if n == 1 {
        return DiGraph::from_elements([Element::Node { weight: MDNodeKind::Vertex(0) }]);
    }

    let p = factorizing_permutation(graph);

    let (mut op, mut cl, mut lc, mut uc) = init_parenthesizing(n);

    build_parenthesizing(graph, &mut op, &mut cl, &mut lc, &mut uc, &p);

    remove_non_module_dummy_nodes(&mut op, &mut cl, &lc, &uc);

    create_consecutive_twin_nodes(&mut op, &mut cl, &lc, &uc);

    let tree = convert_to_md_tree(graph, &op, &cl, &p);

    info!(number_of_nodes = tree.node_count(), number_of_inner_nodes = tree.node_count() - graph.len());

    tree
}

#[instrument(skip_all)]
pub(crate) fn factorizing_permutation(graph: &[Vec<NodeIndex>]) -> Permutation {
    factorizing_permutation::factorizing_permutation(graph)
}


pub(crate) fn init_parenthesizing(n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut op = vec![0; n];
    op[0] = 1;
    let mut cl = vec![0; n];
    cl[n - 1] = 1;
    let lc = (0..n - 1).map(|i| i as u32).collect();
    let uc = (1..n).map(|i| i as u32).collect();
    (op, cl, lc, uc)
}

#[instrument(skip_all)]
pub(crate) fn build_parenthesizing(graph: &mut [Vec<NodeIndex>], op: &mut [u32], cl: &mut [u32], lc: &mut [u32], uc: &mut [u32], p: &Permutation) {
    {
        let _span = info_span!("sort_neighbors_by_pos").entered();
        graph.iter_mut().for_each(|neighbors| neighbors.sort_unstable_by_key(|u| { p.position(*u) as u32 }));
    }

    fn next_unequal<'a>(mut a: impl Iterator<Item=&'a NodeIndex>, mut b: impl Iterator<Item=&'a NodeIndex>, f: impl Fn(NodeIndex) -> usize, g: impl Fn(usize, usize) -> usize) -> Option<usize> {
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
        next_unequal(a.iter(), b.iter(), |u| p.position(u), usize::min)
    };
    let last = |a: &[NodeIndex], b: &[NodeIndex]| -> Option<usize> {
        next_unequal(a.iter().rev(), b.iter().rev(), |u| p.position(u), usize::max)
    };

    // To find the first and last vertices (in respect to the permutation) which are not in both
    // neighborhoods of consecutive vertices, the sorted neighborhoods are traversed from the
    // beginning and end, stopping when they first disagree.
    // Every neighborhood is traversed at most twice, once for each adjacent vertex in the
    // permutation. This results in a total O(n + m) running time.

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


pub(crate) struct SegmentedStack<T> {
    values: Vec<T>,
    starts: Vec<(u32, u32)>,
    len: usize,
}

impl<T> SegmentedStack<T> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let starts = Vec::with_capacity(128);
        Self { values: Vec::with_capacity(capacity), starts, len: 0 }
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
    fn handle_vertex_node(position: usize) -> Info {
        let pos = position as u32;
        Info::new(pos, pos, pos, pos)
    }

    let handle_inner_node = |op: &mut [u32], cl: &mut [u32], nodes: &[Info]| -> Info {
        let k = nodes.len();

        debug_assert!((0..k - 1).all(|i| nodes[i].last_vertex + 1 == nodes[i + 1].first_vertex));

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
        // Check if we are at a genuine node, i.e. the node of the fracture tree represents a module.
        if first_vertex < last_vertex && first_vertex <= first_cutter && last_cutter <= last_vertex {
            return info;
        }
        // As this is a post-order traversal, we already walked the parenthesis up to the last
        // vertex and we can change the parenthesis expression without affecting the remaining
        // traversal.
        // Removing the pair parenthesis deletes the node from the fracture tree.
        op[info.first_vertex as usize] -= 1;
        cl[info.last_vertex as usize] -= 1;
        info
    };


    let mut s = SegmentedStack::with_capacity(op.len());
    // We do a post-order traversal of the fracture tree induced by the parenthesized permutation.
    for j in 0..op.len() {
        s.extend(op[j] as _);
        s.add(handle_vertex_node(j));
        for _ in 0..cl[j] {
            let info = handle_inner_node(op, cl, s.pop());
            s.add(info);
        }
    }
    assert!(s.is_empty());
    debug_assert_eq!(op.iter().sum::<u32>(), cl.iter().sum());
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

            l = i;
            if i >= j { continue; }
            let (a, b) = (lc[j - 1] as usize, uc[j - 1] as usize);
            if i <= a && a < b && b <= k {
                if c > 0 {
                    op[i] += 1;
                    cl[k] += 1;
                    l = k + 1;
                }
            } else {
                if i < j - 1 {
                    op[i] += 1;
                    cl[j - 1] += 1;
                }
                l = j;
            }
        }
    }
}

#[instrument(skip_all)]
pub(crate) fn convert_to_md_tree(graph: &[Vec<NodeIndex>], op: &[u32], cl: &[u32], p: &Permutation) -> DiGraph<MDNodeKind, ()> {
    // Do a post-order traversal of the fracture tree (represented by op, cl and p), skipping nodes
    // with a single child, determining the module type of the others and adding it to the modular
    // decomposition tree.
    let n = graph.len();

    let handle_vertex_node = |t: &mut DiGraph<MDNodeKind, ()>, x: NodeIndex| -> (NodeIndex, NodeIndex) {
        (x, t.add_node(MDNodeKind::Vertex(x.index())))
    };

    let mut marked = vec![0; n];
    let mut gen = 0;

    let mut determine_node_kind = |t: &mut DiGraph<MDNodeKind, ()>, nodes: &[(NodeIndex, NodeIndex)]| -> (NodeIndex, MDNodeKind) {
        // Calculate the degrees between children of a module.
        // Every module keeps a graph node as a representative. Mark the representatives of the children.
        // For each representative, iterate over its neighbors. If a neighbor is also a representative,
        // increment the degree.

        gen += 1;
        for (x, _) in nodes { marked[x.index()] = gen; }

        let quotient_degree = |x: NodeIndex| -> usize {
            graph[x.index()].iter().filter(|w| marked[w.index()] == gen).count()
        };
        // Choosing the node with the minimum degree ensures a linear total running time.
        // Let n be the number of leaves corresponding to the vertices of the graph. In the worst
        // case, every inner node has exactly two children and we have at most n - 1 inner nodes.
        // For each inner node, we can choose one of the leaves of its subtree, such that every
        // inner node gets assigned a unique vertex as a representative. Leaving one vertex
        // unassigned in the tree. Every inner node has work in order of the number of its children
        // and the degree of its representative in the original graph. The work for the children is
        // O(n) in total and the work for the representatives is O(m) in total. By choosing the
        // representatives to be the vertex with the lowest degree in the subtree, the work is
        // bounded by the previous assignment of representatives. This proves a total running time
        // of O(n + m) with this strategy.
        let &(y, _) = nodes.iter().min_by_key(|(x, _)| graph[x.index()].len()).unwrap();

        // It is enough to look at a single vertex and its degree in the quotient graph. Let y be one
        // of the vertices and k be the number of vertices.
        // If the quotient has only two vertices, it is either series or parallel and we can
        // distinguish these case by calculating the degree of one of the vertices.
        // Now assume there are at least three vertices. If y does not have degree 0
        // or k - 1, the graph cannot be series or parallel. If y does have degree 0, then y and
        // the other vertices would be the children of a parallel node. It y does have degree k - 1,
        // then they would be the children of a series node.
        let d0 = quotient_degree(y);
        let kind = if d0 == 0 {
            MDNodeKind::Parallel
        } else if d0 == (nodes.len() - 1) {
            MDNodeKind::Series
        } else { MDNodeKind::Prime };

        if kind != MDNodeKind::Prime {
            debug_assert!(nodes.iter().map(|(y, _)| quotient_degree(*y)).all(|d| d == d0));
            debug_assert!(nodes.iter().all(|(_, u)| t[*u] != kind), "{:?}", kind);
        }

        (y, kind)
    };

    let mut handle_inner_node = |t: &mut DiGraph<MDNodeKind, ()>, nodes: &[(NodeIndex, NodeIndex)]| -> (NodeIndex, NodeIndex) {
        if nodes.len() == 1 { return nodes[0]; }
        let (y, kind) = determine_node_kind(t, nodes);
        let idx = t.add_node(kind);
        for (_, u) in nodes { t.add_edge(idx, *u, ()); }
        (y, idx)
    };

    let mut t = DiGraph::with_capacity(n + 256, n + 256);

    // We keep track of (y, u) where u is a node in the tree and y is a representative of the
    // vertices in the subtree of u, i.e. one of the its leaves.
    let mut s = SegmentedStack::with_capacity(n);
    // We do a post-order traversal of the fracture tree induced by the parenthesized permutation.
    for (j, x) in p.iter().enumerate() {
        s.extend(op[j] as _);
        s.add(handle_vertex_node(&mut t, x));
        for _ in 0..cl[j] {
            let (x, idx) = handle_inner_node(&mut t, s.pop());
            s.add((x, idx));
        }
    }
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

            let Some(non_singleton) = self.next_non_singleton() else { return false; };

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
                self.remove_part(A_idx);
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
                self.remove_part(X_idx);
                None
            } else { Some(X_idx) };

            match (A_idx, N_idx) {
                (Some(A_idx), Some(N_idx)) => {
                    let (S, L) = if self.part(A_idx).seq.len <= self.part(N_idx).seq.len {
                        (A_idx, N_idx)
                    } else { (N_idx, A_idx) };
                    (Some(S), L)
                }
                (Some(A_idx), None) => (None, A_idx),
                (None, Some(N_idx)) => (None, N_idx),
                _ => unreachable!(),
            }
        }

        fn refine(&mut self, y_pos: NodePos, Y: Seq) {
            let y = self.node(y_pos).node;

            // The pivot set S is N(y) \ E. S is implicitly computed by iterating over N(y) and
            // checking if a neighbor of y is in the set E. This can be done efficiently as E is
            // consecutive range.


            for &u in &self.graph[y.index()] {
                let u_pos = self.position(u);
                if Y.contains(u_pos) || !self.current_subgraph.contains(u_pos) { continue; }

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
                if Y.contains(u_pos) || !self.current_subgraph.contains(u_pos) { continue; }
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


        fn push_pivot(&mut self, pivot: PartIndex) {
            debug_assert!(!self.part(pivot).is_in_pivots());
            self.pivots.push(pivot);
            self.part_mut(pivot).set_in_pivots(true);
        }
        fn pop_pivot(&mut self) -> Option<PartIndex> {
            let pivot = self.pivots.pop()?;
            debug_assert!(self.part(pivot).is_in_pivots());
            self.part_mut(pivot).set_in_pivots(false);
            Some(pivot)
        }
        fn push_back_module(&mut self, module: PartIndex) {
            debug_assert!(!self.part(module).is_in_modules());
            let idx = self.modules.push_back(module);
            self.part_mut(module).modules_idx = idx;
        }
        fn pop_front_module(&mut self) -> Option<PartIndex> {
            let module = self.modules.pop_front()?;
            debug_assert!(self.part(module).is_in_modules());
            self.part_mut(module).modules_idx = DequeIndex::invalid();
            Some(module)
        }

        fn replace_module(&mut self, old: PartIndex, new: PartIndex) {
            if old != new {
                let idx = std::mem::replace(&mut self.part_mut(old).modules_idx, DequeIndex::invalid());
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


        /// Inserts the node at `u_pos` into a part to the right of its original part.
        /// If the part to the right of it has been created in the current generation we can add to
        /// it. Otherwise we create a new part exactly to the right.
        fn insert_right(&mut self, u_pos: NodePos) {
            let part = self.node(u_pos).part;
            debug_assert!(!self.part(part).is_marked());
            let last = self.part(part).seq.last();

            let next = (last.index() + 1 != self.nodes.len()).then(|| {
                self.nodes[last.index() + 1].part
            }).filter(|&next| self.part(next).is_marked())
                .unwrap_or_else(|| {
                    self.new_part(NodePos::new(last.index() + 1), 0, true)
                });

            debug_assert!(!self.part(next).is_in_pivots());
            debug_assert!(!self.part(next).is_in_modules());
            debug_assert!(self.part(next).is_marked());

            self.swap_nodes(last, u_pos);
            self.part_mut(next).seq.grow_left();
            self.part_mut(part).seq.shrink_right();
            self.node_mut(last).part = next;

            if self.part(part).seq.is_empty() {
                // We moved all elements from X to X n S.
                // Undo all that and unmark the part to prevent interfering with neighboring parts.
                self.part_mut(next).set_marked(false);
                self.part_mut(part).seq = self.part(next).seq;
                self.part_mut(next).seq.len = 0;
                self.remove_part(next);
                let range = self.part(part).seq.range();
                self.nodes[range].iter_mut().for_each(|n| { n.part = part; });
            }

            debug_assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
            debug_assert!(self.part(self.node(last).part).seq.contains(last));
        }

        fn insert_left(&mut self, u_pos: NodePos) {
            let part = self.node(u_pos).part;
            debug_assert!(!self.part(part).is_marked());
            let first = self.part(part).seq.first();

            let prev = (first.index() != 0).then(|| {
                self.nodes[first.index() - 1].part
            }).filter(|&prev| self.part(prev).is_marked())
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
                // Undo all that and unmark the part to prevent interfering with neighboring parts.
                self.part_mut(prev).set_marked(false);
                self.part_mut(part).seq = self.part(prev).seq;
                self.part_mut(prev).seq.len = 0;
                self.remove_part(prev);
                let range = self.part(part).seq.range();
                self.nodes[range].iter_mut().for_each(|n| { n.part = part; });
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

        #[inline]
        fn part(&self, idx: PartIndex) -> &Part { &self.parts[idx.index()] }
        #[inline]
        fn part_mut(&mut self, idx: PartIndex) -> &mut Part { &mut self.parts[idx.index()] }
        #[inline]
        fn node(&self, pos: NodePos) -> &Node { &self.nodes[pos.index()] }
        #[inline]
        fn node_mut(&mut self, pos: NodePos) -> &mut Node { &mut self.nodes[pos.index()] }
        #[inline]
        fn position(&self, node: NodeIndex) -> NodePos { self.positions[node.index()] }

        #[inline]
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

        current_subgraph: Seq,
    }

    impl<'a> State<'a> {
        fn new(graph: &'a [Vec<NodeIndex>]) -> Self {
            let n = graph.len();

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

            State { graph, positions, nodes, parts, removed, pivots, modules, center_pos, non_singleton_idx, current_subgraph }
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

        fn positions(&self) -> impl Iterator<Item=NodePos> {
            self.range().map(NodePos::new)
        }

        fn grow_right(&mut self) { self.len += 1; }
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
            Self { seq, modules_idx: DequeIndex::invalid(), flags: 0 }
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
            self.modules_idx.is_valid()
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
        #[allow(unused)]
        pub(crate) fn new_identity(size: usize) -> Self {
            let positions = (0..size).map(NodePos::new).collect();
            let nodes = (0..size).map(NodeIndex::new).map(|node| Node { node, part: Default::default() }).collect();
            Self { positions, nodes }
        }
        pub(crate) fn len(&self) -> usize { self.nodes.len() }
        pub(crate) fn position(&self, u: NodeIndex) -> usize { self.positions[u.index()].index() }
    }

    impl<'a> Permutation {
        pub(crate) fn iter(&'a self) -> impl Iterator<Item=NodeIndex> + 'a { self.nodes.iter().map(|n| n.node) }
    }

    make_index!(DequeIndex);

    struct Deque<T> {
        elements: Vec<T>,
        head: DequeIndex,
        tail: DequeIndex,
    }

    impl<T> Deque<T> {
        fn with_capacity(capacity: usize) -> Self {
            Self { elements: Vec::with_capacity(capacity), head: DequeIndex::new(0), tail: DequeIndex::new(0) }
        }
        fn len(&self) -> usize {
            (self.head.0 - self.tail.0) as usize
        }
        fn from_index_to_pos(idx: DequeIndex, n: usize) -> usize {
            idx.index() % n
        }
    }

    impl<T: Copy + Default> Deque<T> {
        fn push_back(&mut self, element: T) -> DequeIndex {
            if self.head.0 == self.elements.len() as u32 + self.tail.0 {
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
            self.head.0 += 1;
            idx
        }
        fn pop_front(&mut self) -> Option<T> {
            if self.head != self.tail {
                let pos = Self::from_index_to_pos(self.tail, self.elements.len());
                let element = self.elements[pos];
                self.tail.0 += 1;
                if self.head == self.tail {
                    self.head.0 = 0;
                    self.tail.0 = 0;
                }
                Some(element)
            } else {
                None
            }
        }
    }

    impl<T> Index<DequeIndex> for Deque<T> {
        type Output = T;

        fn index(&self, index: DequeIndex) -> &Self::Output {
            assert!(self.tail <= index);
            assert!(index < self.head);
            let pos = Self::from_index_to_pos(index, self.elements.len());
            &self.elements[pos]
        }
    }

    impl<T> IndexMut<DequeIndex> for Deque<T> {
        fn index_mut(&mut self, index: DequeIndex) -> &mut Self::Output {
            assert!(self.tail <= index);
            assert!(index < self.head);
            let pos = Self::from_index_to_pos(index, self.elements.len());
            &mut self.elements[pos]
        }
    }

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn empty_graph() {
            let p = factorizing_permutation(&[]);
            assert_eq!(p.len(), 0);
        }

        #[test]
        fn empty_one_vertex_graph() {
            let p = factorizing_permutation(&[vec![]]);
            assert_eq!(p.len(), 1);
        }

        #[test]
        fn ted08_test0_graph() {
            let graph = common::instances::ted08_test0();
            let graph: Vec<_> = graph.node_indices().map(|u| graph.neighbors(u).collect()).collect();

            let p = factorizing_permutation(&graph);
            assert_eq!(p.iter().map(|n| n.index()).collect::<Vec<_>>(), [15, 8, 17, 3, 4, 2, 7, 6, 5, 11, 10, 1, 9, 13, 14, 12, 0, 16]);
        }
    }
}

#[cfg(test)]
mod test {
    use petgraph::dot::{Config, Dot};
    use petgraph::graph::{NodeIndex, UnGraph};
    use crate::improved::{build_parenthesizing, create_consecutive_twin_nodes, init_parenthesizing, modular_decomposition, remove_non_module_dummy_nodes};
    use crate::improved::factorizing_permutation::Permutation;

    fn print_parenthesis(op: &[u32], cl: &[u32], permutation: &Permutation) {
        let n = op.len();
        for j in 0..n {
            for _ in 0..op[j] { print!("("); }
            print!("{}", permutation[j].index());
            for _ in 0..cl[j] { print!(")"); }
        }
        println!();
    }

    #[test]
    fn parenthesized_factorizing_permutation() {
        let n = 7;
        let permutation = Permutation::new_identity(n);
        let graph: UnGraph<(), ()> = UnGraph::from_edges([(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)]);
        let mut graph: Vec<Vec<NodeIndex>> = graph.node_indices().map(|u| graph.neighbors(u).collect()).collect();

        let (mut op, mut cl, mut lc, mut uc) = init_parenthesizing(n);

        build_parenthesizing(&mut graph, &mut op, &mut cl, &mut lc, &mut uc, &permutation);


        assert_eq!(op, [3, 0, 0, 0, 2, 1, 0]);
        assert_eq!(cl, [0, 1, 0, 0, 1, 2, 2]);

        print_parenthesis(&op, &cl, &permutation);

        remove_non_module_dummy_nodes(&mut op, &mut cl, &lc, &uc);

        assert_eq!(op, [3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(cl, [0, 1, 0, 0, 0, 0, 2]);

        print_parenthesis(&op, &cl, &permutation);

        create_consecutive_twin_nodes(&mut op, &mut cl, &lc, &uc);

        assert_eq!(op, [5, 0, 0, 0, 0, 0, 0]);
        assert_eq!(cl, [0, 2, 0, 1, 0, 0, 2]);

        print_parenthesis(&op, &cl, &permutation);


        let tree = modular_decomposition(&mut graph);
        println!("{:?}", Dot::with_config(&tree, &[Config::EdgeNoLabel]));
    }
}

