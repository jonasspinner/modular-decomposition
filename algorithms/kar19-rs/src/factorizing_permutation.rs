pub(crate) mod kar19 {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::collections::hash_map::RandomState;
    use petgraph::graph::{NodeIndex, UnGraph};
    use crate::factorizing_permutation::kar19::util::{smaller_larger, splice};
    #[allow(unused_imports)]
    use crate::{d1, d2, trace};

    #[allow(non_snake_case, dead_code)]
    pub(crate) fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Vec<NodeIndex> {
        let V = graph.node_indices().collect();
        let mut P = vec![V];
        let mut center = NodeIndex::new(0);
        let mut pivots = vec![];
        let mut modules = VecDeque::new();
        let mut first_pivot = HashMap::<Vec<NodeIndex>, NodeIndex>::new();

        partition_refinement(&mut P, &mut center, &mut pivots, &mut modules, &mut first_pivot, graph);
        P.iter().map(|part| part[0]).collect()
    }

    #[allow(non_snake_case)]
    fn refine(P: &mut Vec<Vec<NodeIndex>>,
              S: HashSet<NodeIndex>,
              x: NodeIndex,
              center: &NodeIndex,
              pivots: &mut Vec<Vec<NodeIndex>>,
              modules: &mut VecDeque<Vec<NodeIndex>>) {
        trace!("refine: {} {:?}", x.index(), S.iter().map(|u| u.index()).collect::<Vec<_>>());
        let mut i = 0_usize.wrapping_sub(1);
        let mut between = false;
        while i.wrapping_add(1) < P.len() {
            i = i.wrapping_add(1);
            let X = &P[i];
            if X.contains(center) || X.contains(&x) {
                between = !between;
                continue;
            }
            let (X_a, X): (Vec<_>, Vec<_>) = X.iter().partition(|&y| S.contains(y));
            if X_a.is_empty() || X.is_empty() { continue; }
            trace!("refine:P:0: {:?}", d2(&*P));
            P[i] = X.clone();
            P.insert(i + between as usize, X_a.clone());
            trace!("refine:P:1: {:?}", d2(&*P));
            add_pivot(X, X_a, pivots, modules);
            i += 1;
        }
    }

    #[allow(non_snake_case)]
    fn add_pivot(
        X: Vec<NodeIndex>,
        X_a: Vec<NodeIndex>,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>) {
        trace!("add_pivot: {:?} {:?}", d1(&X), d1(&X_a));
        if pivots.contains(&X) {
            pivots.push(X_a);
        } else {
            let i = modules.iter().position(|Y| Y == &X);
            let (S, L) = smaller_larger(X, X_a);
            pivots.push(S);
            if let Some(i) = i {
                modules[i] = L;
            } else {
                modules.push_back(L);
            }
        }
    }

    #[allow(non_snake_case)]
    fn partition_refinement(
        P: &mut Vec<Vec<NodeIndex>>,
        center: &mut NodeIndex,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
        first_pivot: &mut HashMap<Vec<NodeIndex>, NodeIndex>,
        graph: &UnGraph<(), ()>) {
        while init_partition(P, center, pivots, modules, first_pivot, graph) {
            trace!("P: {:?}", d2(&*P));
            trace!("pivots: {:?}", d2(&*pivots));
            trace!("modules: {:?}", d2(&*modules));
            while let Some(E) = pivots.pop() {
                let E_h: HashSet<_, RandomState> = HashSet::from_iter(E.clone());
                for &x in &E {
                    let S = graph.neighbors(x).filter(|v| !E_h.contains(v)).collect();
                    refine(P, S, x, center, pivots, modules);
                }
            }
        }
    }

    #[allow(non_snake_case)]
    fn init_partition(
        P: &mut Vec<Vec<NodeIndex>>,
        center: &mut NodeIndex,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
        first_pivot: &mut HashMap<Vec<NodeIndex>, NodeIndex>,
        graph: &UnGraph<(), ()>) -> bool {
        if P.iter().all(|p| p.len() <= 1) { return false; }
        if let Some(X) = modules.pop_front() {
            let x = X[0];
            pivots.push(vec![x]);
            first_pivot.insert(X, x);
        } else {
            for (i, X) in P.iter().enumerate() {
                if X.len() <= 1 { continue; }
                let x = first_pivot.get(X).copied().unwrap_or(X[0]);
                let adj: HashSet<_> = graph.neighbors(x).collect();
                let (A, mut N): (Vec<_>, Vec<_>) = X.iter().partition(|&y| *y != x && adj.contains(y));
                N.retain(|y| *y != x);
                splice(P, i, A.clone(), x, N.clone());
                let (S, L) = smaller_larger(A, N);
                *center = x;
                pivots.push(S);
                modules.push_back(L);
                break;
            }
        }
        true
    }

    mod util {
        pub(crate) fn smaller_larger<T>(a: Vec<T>, b: Vec<T>) -> (Vec<T>, Vec<T>) {
            if a.len() <= b.len() { (a, b) } else { (b, a) }
        }

        pub(crate) fn splice<T>(vec: &mut Vec<Vec<T>>, i: usize, first: Vec<T>, second: T, third: Vec<T>) {
            match (first.is_empty(), third.is_empty()) {
                (true, true) => { vec[i] = vec![second] }
                (true, false) => {
                    vec[i] = vec![second];
                    vec.insert(i + 1, third)
                }
                (false, true) => {
                    vec[i] = first;
                    vec.insert(i + 1, vec![second]);
                }
                (false, false) => {
                    vec[i] = first;
                    vec.insert(i + 1, vec![second]);
                    vec.insert(i + 2, third)
                }
            }
        }
    }
}

pub(crate) mod seq {
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
    pub fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Permutation {
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
