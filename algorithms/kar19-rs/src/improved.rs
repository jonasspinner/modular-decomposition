use std::iter::zip;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use common::modular_decomposition::MDNodeKind;
use tracing::{info, instrument};
use crate::improved::factorizing_permutation::Permutation;
use crate::shared;


#[allow(dead_code)]
pub(crate) fn modular_decomposition(graph: &UnGraph<(), ()>) -> DiGraph<MDNodeKind, ()> {
    let p = factorizing_permutation(graph);

    let n = p.len();
    let mut op = vec![0; n];
    op[0] = 1;
    let mut cl = vec![0; n];
    cl[n - 1] = 1;
    let mut lc: Vec<_> = (0..n - 1).collect();
    let mut uc: Vec<_> = (1..n).collect();

    build_parenthesizing(graph, &mut op, &mut cl, &mut lc, &mut uc, &p);

    info!(n, op = op.iter().sum::<usize>(), cl = cl.iter().sum::<usize>());

    remove_non_module_dummy_nodes(&mut op, &mut cl, &mut lc, &mut uc);

    shared::create_consecutive_twin_nodes(&mut op, &mut cl, &lc, &uc);

    shared::remove_singleton_dummy_nodes(&mut op, &mut cl);

    let tree = build_tree(graph, &op, &cl, &p);

    info!(number_of_nodes = tree.node_count(), number_of_inner_nodes = tree.node_count() - graph.node_count());

    tree
}

#[instrument(skip_all)]
fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Permutation {
    factorizing_permutation::factorizing_permutation(graph)
}

#[instrument(skip_all)]
fn build_parenthesizing(graph: &UnGraph<(), ()>, op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize], p: &Permutation) {
    let get_neighbor_positions = |idx: usize| -> Vec<usize> {
        let mut pos: Vec<_> = graph.neighbors(p[idx]).map(|v| p.position(v)).collect();
        pos.sort();
        pos
    };
    let mut pos_j0;
    let mut pos_j1 = get_neighbor_positions(0);

    fn merge_iter_next<'a>(mut a: impl Iterator<Item=&'a usize>, mut b: impl Iterator<Item=&'a usize>, f: impl Fn(usize, usize) -> usize) -> Option<usize> {
        loop {
            match (a.next(), b.next()) {
                (Some(i), Some(j)) => { if i != j { break Some(f(*i, *j)); } }
                (Some(i), None) => break Some(*i),
                (None, Some(j)) => break Some(*j),
                (None, None) => break None,
            }
        }
    }

    let first = |a: &[usize], b: &[usize]| -> Option<usize> {
        merge_iter_next(a.iter(), b.iter(), usize::min)
    };
    let last = |a: &[usize], b: &[usize]| -> Option<usize> {
        merge_iter_next(a.iter().rev(), b.iter().rev(), usize::max)
    };

    let n = p.len();
    for j in 0..n - 1 {
        pos_j0 = std::mem::replace(&mut pos_j1, get_neighbor_positions(j + 1));
        let mut lc_idx = first(&pos_j0, &pos_j1);
        let mut uc_idx = if lc_idx.is_some() { last(&pos_j0, &pos_j1) } else { None };
        lc_idx = lc_idx.filter(|i| *i < j);
        uc_idx = uc_idx.filter(|i| *i > j + 1);
        //println!("{:?} {:?}", pos_j0, pos_j1);
        //println!("# {j} {:?} {:?} {:?}", min_i, max_i, HashSet::<usize>::from_iter(pos_j0).symmetric_difference(&HashSet::<usize>::from_iter(pos_j1.iter().copied())).collect::<Vec<_>>());
        if let Some(i) = lc_idx {
            assert!(i < j);
            debug_assert_ne!(graph.find_edge(p[i], p[j]).is_some(), graph.find_edge(p[i], p[j + 1]).is_some());
            op[i] += 1;
            cl[j] += 1;
            lc[j] = i;
        }
        if let Some(i) = uc_idx {
            assert!(i > j + 1 && i < n);
            debug_assert_ne!(graph.find_edge(p[i], p[j]).is_some(), graph.find_edge(p[i], p[j + 1]).is_some());
            op[j + 1] += 1;
            cl[i] += 1;
            uc[j] = i;
        }
    }
}

#[instrument(skip_all)]
fn remove_non_module_dummy_nodes(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);
    for j in 0..n {
        s.push((j, op[j]));
        let mut k = cl[j];
        while k > 0 {
            let (i, count) = s.pop().unwrap();
            if i < j {
                let l = (i..j).map(|k| lc[k]).min().unwrap();
                let u = (i..j).map(|k| uc[k]).max().unwrap();
                if i <= l && u <= j {
                    if k >= count { k -= count; } else {
                        s.push((i, count - k));
                        break;
                    }
                    continue;
                }
            }
            op[i] -= count.min(k);
            cl[j] -= count.min(k);
            if k >= count { k -= count; } else {
                s.push((i, count - k));
                break;
            }
        }
    }
}

#[instrument(skip_all)]
#[allow(unreachable_code)]
fn build_tree(graph: &UnGraph<(), ()>, op: &[usize], cl: &[usize], p: &Permutation) -> DiGraph<MDNodeKind, ()> {
    let n = graph.node_count();

    // Calculate the degrees between children of a module.
    // Every module keeps a graph node as a representative. Mark the representatives of the children.
    // For each representative, iterate over its neighbors. If a neighbor is also a representative,
    // increment the degree.
    let mut gens = vec![0; n];
    let mut gen = 0;
    let mut degrees: Vec<usize> = vec![];

    let mut add_node = |t: &mut DiGraph<MDNodeKind, ()>, nodes: &[(NodeIndex, NodeIndex)]| -> (NodeIndex, NodeIndex) {
        gen += 1;
        for (x, _) in nodes { gens[x.index()] = gen; }

        // Prepare a vec with nodes.len() zeros at the front.
        let clear_len = degrees.len().min(nodes.len());
        degrees[0..clear_len].iter_mut().for_each(|d| *d = 0);
        degrees.resize(nodes.len(), 0);

        for (d, (x, _)) in zip(degrees.iter_mut(), nodes.iter()) {
            *d = graph.neighbors(*x).map(|v| (gens[v.index()] == gen) as usize).sum();
        }

        let n = nodes.len();
        let degree_sum = degrees.iter().sum::<usize>();
        assert!(degree_sum <= n * (n - 1));
        let kind = if degree_sum == 0 { MDNodeKind::Parallel } else if degree_sum == n * (n - 1) { MDNodeKind::Series } else { MDNodeKind::Prime };
        let idx = t.add_node(kind);

        for (_, u) in nodes {
            if t[*u] == kind && kind != MDNodeKind::Prime {
                // TODO: investigate
                // panic!("This case does not seem to occur. Weird.");
                let children: Vec<_> = t.neighbors(*u).collect();
                for v in children {
                    t.add_edge(idx, v, ());
                }
                t.remove_node(*u);
            } else {
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
    let mut s = vec![vec![]];

    for (j, x) in p.iter().enumerate() {
        s.extend(std::iter::repeat(vec![]).take(op[j]));
        let (x, idx) = (x, t.add_node(MDNodeKind::Vertex(x.index())));
        s.last_mut().unwrap().push((x, idx));
        for _ in 0..cl[j] {
            let nodes = s.pop().unwrap();
            let (x, idx) = add_node(&mut t, &nodes);
            s.last_mut().unwrap().push((x, idx));
        }
    }
    add_node(&mut t, &s.pop().unwrap());
    t
}

mod factorizing_permutation {
    use std::collections::{HashMap, VecDeque};
    use std::ops::Range;
    use petgraph::graph::{NodeIndex, UnGraph};
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
    pub(crate) fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Permutation {
        let mut state = State::new(graph);
        state.partition_refinement();
        state.into_permutation()
    }

    impl<'a> State<'a> {
        #[allow(non_snake_case)]
        fn partition_refinement(&mut self) {
            while self.init_partition() {
                while let Some(E) = self.pivots.pop() {
                    for x_pos in E.positions() {
                        let x = self.node(x_pos).node;
                        self.refine(x, E);
                    }
                }
            }
        }

        #[allow(non_snake_case)]
        fn init_partition(&mut self) -> bool {
            let Some(non_singleton) = self.next_non_singleton() else { return false; };

            if let Some(X) = self.modules.pop_front() {
                let x_pos = X.first().unwrap();
                let x = self.node(x_pos).node;
                self.pivots.push(Seq { first: x_pos, len: 1 });
                self.first_pivot.insert(X, x);
            } else {
                let X_idx = non_singleton;
                let X = self.part(X_idx).seq;

                let x = self.first_pivot.get(&X).copied().unwrap_or_else(|| {
                    self.node(X.first().unwrap()).node
                });

                // Divide X into A, {x}, N, where A = X n N(x), N = X \ (N(x) u {x})
                // This takes O(|N(x)|) time.

                // Create A
                self.increment_gen();
                for v in self.graph.neighbors(x) {
                    let v_pos = self.position(v);
                    if X.contains(v_pos) {
                        self.insert_left(v_pos);
                    }
                }

                // Create {x}
                self.increment_gen();
                self.insert_left(self.position(x));

                // N is already there.

                // Find A and N
                let x_pos = self.position(x);
                let [A, N] = [-1_isize, 1_isize]
                    .map(|diff| {
                        let pos = NodePos::new(x_pos.index().wrapping_add_signed(diff));
                        if X.contains(pos) {
                            self.part(self.node(pos).part).seq
                        } else {
                            Seq::new(NodePos(0), 0)
                        }
                    });

                let (S, L) = if A.len <= N.len { (A, N) } else { (N, A) };
                self.center_pos = x_pos;
                self.pivots.push(S);
                self.modules.push_back(L);
            }
            true
        }

        #[allow(non_snake_case)]
        fn refine(&mut self, y: NodeIndex, E: Seq) {
            let y_pos = self.position(y);

            self.increment_gen();

            // The pivot set S is N(y) \ E. S is implicitly computed by iterating over N(y) and
            // checking if a neighbor of y is in the set E. This can be done efficiently as E is
            // consecutive range.

            for u in self.graph.neighbors(y) {
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
            for u in self.graph.neighbors(y) {
                let u_pos = self.position(u);

                let X_a_idx = self.node(u_pos).part;
                let X_a = self.part(X_a_idx).seq;

                if self.is_current_gen(X_a_idx) {
                    self.remove_from_current_gen(X_a_idx);

                    let X_idx = if self.should_insert_right(X_a, y_pos) {
                        self.nodes[X_a.first().unwrap().index() - 1].part
                    } else {
                        self.nodes[X_a.last().unwrap().index() + 1].part
                    };

                    let X = self.part(X_idx).seq;

                    self.add_pivot(X, X_a);
                }
            }
        }


        #[allow(non_snake_case)]
        fn add_pivot(&mut self, X: Seq, X_a: Seq) {
            if self.pivots.contains(&X) {
                self.pivots.push(X_a);
            } else {
                let i = self.modules.iter().position(|Y| Y == &X);
                let (S, L) = if X.len <= X_a.len { (X, X_a) } else { (X_a, X) };
                self.pivots.push(S);
                if let Some(i) = i {
                    self.modules[i] = L;
                } else {
                    self.modules.push_back(L);
                }
            }
        }


        #[allow(non_snake_case)]
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
            assert!(!self.is_current_gen(part));
            let last = self.part(part).seq.last().unwrap();

            let next = (last.index() + 1 != self.nodes.len()).then(|| {
                self.nodes[last.index() + 1].part
            }).filter(|&next| self.is_current_gen(next))
                .unwrap_or_else(|| self.new_part(NodePos::new(last.index() + 1), 0));

            self.swap_nodes(last, u_pos);
            self.part_mut(next).seq.grow_left();
            self.part_mut(part).seq.shrink_right();
            self.node_mut(last).part = next;

            if self.part(part).seq.is_empty() {
                self.remove_from_current_gen(next);
                self.remove_if_empty(part);
            }

            assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
            assert!(self.part(self.node(last).part).seq.contains(last));
        }

        fn insert_left(&mut self, u_pos: NodePos) {
            let part = self.node(u_pos).part;
            assert!(!self.is_current_gen(part));
            let first = self.part(part).seq.first().unwrap();

            let prev = (first.index() != 0).then(|| {
                self.nodes[first.index() - 1].part
            }).filter(|&prev| self.is_current_gen(prev))
                .unwrap_or_else(|| self.new_part(first, 0));

            self.swap_nodes(first, u_pos);
            self.part_mut(prev).seq.grow_right();
            self.part_mut(part).seq.shrink_left();
            self.node_mut(first).part = prev;

            if self.part(part).seq.is_empty() {
                self.remove_from_current_gen(prev);
                self.remove_if_empty(part);
            }

            assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
            assert!(self.part(self.node(first).part).seq.contains(first));
        }

        fn swap_nodes(&mut self, a: NodePos, b: NodePos) {
            let Node { part: a_part, node: u } = *self.node(a);
            let Node { part: b_part, node: v } = *self.node(b);
            assert_eq!(a_part, b_part);
            self.positions.swap(u.index(), v.index());
            self.nodes.swap(a.index(), b.index());

            assert_eq!(self.node(self.position(u)).node, u);
            assert_eq!(self.node(self.position(v)).node, v);
        }

        fn part(&self, idx: PartIndex) -> &Part { &self.parts[idx.index()] }
        fn part_mut(&mut self, idx: PartIndex) -> &mut Part { &mut self.parts[idx.index()] }
        fn node(&self, pos: NodePos) -> &Node { &self.nodes[pos.index()] }
        fn node_mut(&mut self, pos: NodePos) -> &mut Node { &mut self.nodes[pos.index()] }
        fn position(&self, node: NodeIndex) -> NodePos { self.positions[node.index()] }

        fn remove_if_empty(&mut self, idx: PartIndex) {
            if self.part(idx).seq.is_empty() {
                self.removed.push(idx);
            }
        }

        fn new_part(&mut self, first: NodePos, len: u32) -> PartIndex {
            if let Some(idx) = self.removed.pop() {
                *self.part_mut(idx) = Part { seq: Seq { first, len }, gen: self.gen };
                idx
            } else {
                let idx = PartIndex::new(self.parts.len());
                self.parts.push(Part { seq: Seq { first, len }, gen: self.gen });
                idx
            }
        }

        fn increment_gen(&mut self) { self.gen.0 += 1; }
        fn is_current_gen(&self, idx: PartIndex) -> bool { self.part(idx).gen == self.gen }
        fn remove_from_current_gen(&mut self, idx: PartIndex) { self.part_mut(idx).gen.0 -= 1; }

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
        graph: &'graph UnGraph<(), ()>,

        positions: Vec<NodePos>,
        nodes: Vec<Node>,
        parts: Vec<Part>,
        removed: Vec<PartIndex>,

        gen: Gen,

        pivots: Vec<Seq>,
        modules: VecDeque<Seq>,
        first_pivot: HashMap<Seq, NodeIndex>,
        center_pos: NodePos,
        non_singleton_idx: NodePos,
    }

    impl<'a> State<'a> {
        fn new(graph: &'a UnGraph<(), ()>) -> Self {
            let n = graph.node_count();

            let positions = (0..n).map(NodePos::new).collect();
            let nodes = (0..n).map(|u| Node { node: NodeIndex::new(u), part: PartIndex::new(0) }).collect();
            let mut parts = Vec::with_capacity(n);
            parts.push(Part { seq: Seq::new(NodePos::new(0), n as u32), gen: Gen(0) });
            let removed = vec![];
            let gen = Gen(0);
            let pivots = vec![];
            let modules = VecDeque::new();
            let first_pivot = HashMap::new();
            let center_pos = NodePos(u32::MAX);
            let non_singleton_idx = NodePos::new(0);

            State { graph, positions, nodes, parts, removed, gen, pivots, modules, first_pivot, center_pos, non_singleton_idx }
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
        fn first(&self) -> Option<NodePos> { (!self.is_empty()).then_some(self.first) }

        fn last(&self) -> Option<NodePos> { (!self.is_empty()).then_some(NodePos::new(self.first.index() + (self.len as usize) - 1)) }

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
        gen: Gen,
    }

    pub(crate) struct Permutation {
        positions: Vec<NodePos>,
        nodes: Vec<Node>,
    }

    impl std::ops::Index<usize> for Permutation {
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
}

