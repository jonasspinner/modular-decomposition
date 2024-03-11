use petgraph::graph::NodeIndex;

#[macro_export]
macro_rules! traceln {
    ($($x:expr),*) => {
        //println!($($x),*)
    };
}

#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        //print!($($x),*)
    };
}

#[allow(dead_code)]
fn d2<'a, P>(p: P) -> Vec<Vec<usize>>
where
    P: IntoIterator<Item = &'a Vec<NodeIndex>>,
{
    p.into_iter().map(|q| d1(q)).collect()
}

fn d1(p: &[NodeIndex]) -> Vec<usize> {
    p.iter().map(|u| u.index()).collect()
}

pub mod kar19_checked_impl {
    #[allow(unused_imports)]
    use super::{d1, d2};
    use common::make_index;
    use petgraph::graph::{NodeIndex, UnGraph};
    use std::collections::hash_map::RandomState;
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::iter::zip;
    use std::ops::Range;

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
        fn first(&self) -> Option<NodePos> {
            (!self.is_empty()).then_some(self.first)
        }

        fn last(&self) -> Option<NodePos> {
            (!self.is_empty()).then_some(NodePos::new(self.first.index() + (self.len as usize) - 1))
        }

        fn contains(&self, pos: NodePos) -> bool {
            self.range().contains(&pos.index())
        }

        fn range(&self) -> Range<usize> {
            let start = self.first.index();
            start..(start + self.len as usize)
        }

        fn grow_right(&mut self) {
            self.len += 1;
        }
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

    #[derive(Copy, Clone, Eq, PartialEq)]
    struct Gen(u32);

    struct Part {
        seq: Seq,
        gen: Gen,
    }

    #[allow(dead_code, non_snake_case)]
    struct State {
        P: Vec<Vec<NodeIndex>>,
        center: NodeIndex,
        pivots: Vec<Vec<NodeIndex>>,
        modules: VecDeque<Vec<NodeIndex>>,
        first_pivot: HashMap<Vec<NodeIndex>, NodeIndex>,

        _positions: Vec<NodePos>,
        _nodes: Vec<Node>,
        _parts: Vec<Part>,
        _removed: Vec<PartIndex>,

        _gen: Gen,

        _pivots: Vec<Seq>,
        _modules: VecDeque<Seq>,
        _first_pivot: HashMap<Seq, NodeIndex>,
        _center_pos: NodePos,
        _non_singleton_idx: NodePos,
    }

    #[allow(non_snake_case, dead_code)]
    pub fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Vec<NodeIndex> {
        let P = vec![graph.node_indices().collect()];
        let center = NodeIndex::new(0);
        let pivots = vec![];
        let modules = VecDeque::new();
        let first_pivot = HashMap::<Vec<NodeIndex>, NodeIndex>::new();

        let n = graph.node_count();
        let _positions = (0..n).map(NodePos::new).collect();
        let _nodes = (0..n).map(|u| Node { node: NodeIndex::new(u), part: PartIndex::new(0) }).collect();
        let mut _parts = Vec::with_capacity(n);
        _parts.push(Part { seq: Seq::new(NodePos::new(0), n as u32), gen: Gen(0) });
        let _removed = vec![];
        let _gen = Gen(0);
        let _pivots = vec![];
        let _modules = VecDeque::new();
        let _first_pivot = HashMap::new();
        let _center_pos = NodePos(u32::MAX);
        let _non_singleton_idx = NodePos::new(0);

        let mut state = State {
            P,
            center,
            pivots,
            modules,
            first_pivot,
            _positions,
            _nodes,
            _parts,
            _removed,
            _gen,
            _pivots,
            _modules,
            _first_pivot,
            _center_pos,
            _non_singleton_idx,
        };
        state.check_consistency();
        state.partition_refinement(graph);

        state.P.iter().map(|part| part[0]).collect()
    }

    #[allow(non_snake_case)]
    fn smaller_larger(A: Vec<NodeIndex>, B: Vec<NodeIndex>) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
        if A.len() <= B.len() {
            (A, B)
        } else {
            (B, A)
        }
    }

    impl State {
        #[allow(non_snake_case)]
        fn _debug_print(&self) {
            println!(
                "#       P: {:?}",
                self.P.iter().map(|X| X.iter().map(|u| u.index()).collect::<Vec<_>>()).collect::<Vec<_>>()
            );
            println!("_     pos: {:?}", self._positions.iter().map(|pos| pos.index()).collect::<Vec<_>>());
            println!("_    node: {:?}", self._nodes.iter().map(|n| n.node.index()).collect::<Vec<_>>());
            println!("_    part: {:?}", self._nodes.iter().map(|n| n.part.index()).collect::<Vec<_>>());
            println!(
                "_   parts: {:?}",
                self._parts.iter().map(|s| (s.seq.first.index(), s.seq.len, s.gen.0)).collect::<Vec<_>>()
            );
            println!(
                "#  pivots: {:?}",
                self.pivots.iter().map(|X| X.iter().map(|u| u.index()).collect::<Vec<_>>()).collect::<Vec<_>>()
            );
            //println!("_  pivots: {:?}", self._pivots.iter().map(|s| (s.first.index(),
            // s.len)).collect::<Vec<_>>());
            println!(
                "_  pivots: {:?}",
                self._pivots
                    .iter()
                    .map(|X| X.range().map(|pos| self._nodes[pos].node.index()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );
            println!(
                "# modules: {:?}",
                self.modules.iter().map(|X| X.iter().map(|u| u.index()).collect::<Vec<_>>()).collect::<Vec<_>>()
            );
            println!(
                "_ modules: {:?}",
                self._modules
                    .iter()
                    .map(|X| X.range().map(|pos| self._nodes[pos].node.index()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );
            //println!("_ modules: {:?}", self._modules.iter().map(|s|
            // (s.first.index(), s.len)).collect::<Vec<_>>())
        }
        fn check_consistency(&self) {
            //self._debug_print();
            let mut i = 0;
            let mut j = 0;
            while i < self._nodes.len() {
                let part = self._nodes[i].part;
                let start = i;
                let end = start + self._parts[part.index()].seq.len as usize;
                assert_eq!(self._parts[part.index()].seq.first.index(), start);
                i = end;
                let a: HashSet<NodeIndex> = self._nodes[start..end].iter().map(|n| n.node).collect();
                let b: HashSet<NodeIndex> = self.P[j].iter().copied().collect();
                if a != b {
                    self._debug_print();
                }
                assert_eq!(a, b);
                j += 1;
            }
            for (a, b) in zip(self.pivots.iter(), self._pivots.iter()) {
                let a: HashSet<NodeIndex> = a.iter().copied().collect();
                let b: HashSet<NodeIndex> = self._nodes[b.range()].iter().map(|n| n.node).collect();
                if a != b {
                    self._debug_print();
                }
                assert_eq!(a, b);
            }
            for (a, b) in zip(self.modules.iter(), self._modules.iter()) {
                let a: HashSet<NodeIndex> = a.iter().copied().collect();
                let b: HashSet<NodeIndex> = self._nodes[b.range()].iter().map(|n| n.node).collect();
                if a != b {
                    self._debug_print();
                }
                assert_eq!(a, b);
            }
        }
        #[allow(non_snake_case)]
        fn refine(&mut self, S: HashSet<NodeIndex>, x: NodeIndex, graph: &UnGraph<(), ()>, _E: Seq) {
            traceln!("refine: {} {:?}", x.index(), S.iter().map(|u| u.index()).collect::<Vec<_>>());
            let mut i = 0_usize.wrapping_sub(1);
            let mut between = false;
            while i.wrapping_add(1) < self.P.len() {
                i = i.wrapping_add(1);
                let X = &self.P[i];
                //traceln!("refine:X: {} {:?}", i, d1(&X));
                if X.contains(&self.center) || X.contains(&x) {
                    between = !between;
                    continue;
                }
                let (X_a, X): (Vec<_>, Vec<_>) = X.iter().partition(|&y| S.contains(y));
                if X_a.is_empty() || X.is_empty() {
                    continue;
                }
                //traceln!("refine:P:0: {:?}", d2(&*self.P));
                self.P[i] = X.clone();
                self.P.insert(i + between as usize, X_a.clone());
                //traceln!("refine:P:1: {:?}", d2(&*self.P));
                self.add_pivot(X, X_a);
                i += 1;
            }

            if false {
                let x_pos = self._positions[x.index()].index();
                let mut i = 0;
                let mut between = false;
                while i < self._nodes.len() {
                    let X_idx = self._nodes[i].part;
                    let X = self._parts[X_idx.index()].seq;
                    i += X.len as usize;
                    if X.range().contains(&self._center_pos.index()) || X.range().contains(&x_pos) {
                        between = !between;
                        continue;
                    }

                    //let X_a_idx = PartIndex::new(self._parts.len());
                    //self._parts.push(Seq { first: X.first, len: 0 });
                    let X_a_idx = self._new_part(X.first, 0);

                    if between {
                        let mut j = X.range().end - 1;
                        for k in X.range().rev() {
                            if S.contains(&self._nodes[k].node) {
                                let (u, v) = (self._nodes[j].node, self._nodes[k].node);
                                self._positions.swap(u.index(), v.index());
                                self._nodes.swap(j, k);
                                self._nodes[j].part = X_a_idx;
                                j = j.wrapping_sub(1);
                            }
                        }
                        let X_len = (j.wrapping_add(1) - X.first.index()) as u32;
                        self._parts[X_idx.index()].seq.len = X_len;
                        self._parts[X_a_idx.index()].seq =
                            Seq { first: NodePos::new(X.first.index() + X_len as usize), len: X.len - X_len };
                    } else {
                        let mut j = X.range().start;
                        for k in X.range() {
                            if S.contains(&self._nodes[k].node) {
                                let (u, v) = (self._nodes[j].node, self._nodes[k].node);
                                self._positions.swap(u.index(), v.index());
                                self._nodes.swap(j, k);
                                self._nodes[j].part = X_a_idx;
                                j += 1;
                            }
                        }
                        self._parts[X_a_idx.index()].seq.len =
                            (j - self._parts[X_a_idx.index()].seq.first.index()) as u32;
                        self._parts[X_idx.index()].seq =
                            Seq { first: NodePos::new(j), len: X.len - self._parts[X_a_idx.index()].seq.len };
                    }

                    let X = self.part(X_idx).seq;
                    let X_a = self.part(X_a_idx).seq;

                    if !X.is_empty() && !X_a.is_empty() {
                        self._add_pivot(X, X_a);
                    } else {
                        self._remove_if_empty(X_idx);
                        self._remove_if_empty(X_a_idx);
                    }
                }
            } else {
                let y = x;
                let y_pos = self.position(y);

                self._gen.0 += 1;

                for u in graph.neighbors(y) {
                    let u_pos = self.position(u);
                    if _E.contains(u_pos) {
                        continue;
                    }

                    let X = self.part(self.node(u_pos).part).seq;
                    if X.contains(self._center_pos) || X.contains(y_pos) {
                        continue;
                    }

                    if self._should_insert_right(X, y_pos) {
                        self._insert_right(u_pos);
                    } else {
                        self._insert_left(u_pos);
                    }
                }

                let mut todo_pivots = vec![];

                for u in graph.neighbors(y) {
                    let u_pos = self.position(u);

                    let X_a_idx = self.node(u_pos).part;
                    let X_a = self.part(X_a_idx).seq;

                    if self.part(X_a_idx).gen == self._gen {
                        self.part_mut(X_a_idx).gen.0 -= 1;

                        let X_idx = if self._should_insert_right(X_a, y_pos) {
                            self._nodes[X_a.first().unwrap().index() - 1].part
                        } else {
                            self._nodes[X_a.last().unwrap().index() + 1].part
                        };

                        let X = self.part(X_idx).seq;

                        todo_pivots.push((u_pos, X, X_a));
                    }
                }

                todo_pivots.sort_by_key(|(pos, _, _)| pos.index());
                for (_, X, X_a) in todo_pivots {
                    self._add_pivot(X, X_a);
                }
            }
            self.check_consistency();
        }

        #[allow(non_snake_case)]
        fn _should_insert_right(&self, X: Seq, y_pos: NodePos) -> bool {
            let (a, b) = if y_pos < self._center_pos { (y_pos, self._center_pos) } else { (self._center_pos, y_pos) };
            let x = X.first;
            a < x && x < b
        }

        fn _insert_right(&mut self, u_pos: NodePos) {
            let part = self._nodes[u_pos.index()].part;
            let last = self.part(part).seq.last().unwrap();

            let next = (last.index() + 1 != self._nodes.len())
                .then(|| self._nodes[last.index() + 1].part)
                .filter(|&next| self.part(next).gen == self._gen)
                .unwrap_or_else(|| self._new_part(NodePos::new(last.index() + 1), 0));

            self.swap_nodes(last, u_pos);
            self.part_mut(next).seq.grow_left();
            self.part_mut(part).seq.shrink_right();
            self._nodes[last.index()].part = next;

            if self.part(part).seq.is_empty() {
                self.part_mut(next).gen.0 -= 1;
                self._remove_if_empty(part);
            }

            assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
            assert!(self.part(self.node(last).part).seq.contains(last));
        }

        fn _insert_left(&mut self, u_pos: NodePos) {
            let part = self._nodes[u_pos.index()].part;
            let first = self.part(part).seq.first().unwrap();

            let prev = (first.index() != 0)
                .then(|| self._nodes[first.index() - 1].part)
                .filter(|&prev| self.part(prev).gen == self._gen)
                .unwrap_or_else(|| self._new_part(first, 0));

            self.swap_nodes(first, u_pos);
            self.part_mut(prev).seq.grow_right();
            self.part_mut(part).seq.shrink_left();

            self._nodes[first.index()].part = prev;

            if self.part(part).seq.is_empty() {
                self.part_mut(prev).gen.0 -= 1;
                self._remove_if_empty(part);
            }

            assert!(self.part(self.node(u_pos).part).seq.contains(u_pos));
            assert!(self.part(self.node(first).part).seq.contains(first));
        }

        fn swap_nodes(&mut self, a: NodePos, b: NodePos) {
            let Node { part: a_part, node: u } = *self.node(a);
            let Node { part: b_part, node: v } = *self.node(b);
            assert_eq!(a_part, b_part);
            self._positions.swap(u.index(), v.index());
            self._nodes.swap(a.index(), b.index());

            assert_eq!(self.node(self.position(u)).node, u);
            assert_eq!(self.node(self.position(v)).node, v);
        }

        fn part(&self, idx: PartIndex) -> &Part {
            &self._parts[idx.index()]
        }
        fn part_mut(&mut self, idx: PartIndex) -> &mut Part {
            &mut self._parts[idx.index()]
        }
        fn node(&self, idx: NodePos) -> &Node {
            &self._nodes[idx.index()]
        }
        fn position(&self, node: NodeIndex) -> NodePos {
            self._positions[node.index()]
        }

        fn _remove_if_empty(&mut self, idx: PartIndex) {
            if self.part(idx).seq.is_empty() {
                self._removed.push(idx);
            }
        }

        fn _new_part(&mut self, first: NodePos, len: u32) -> PartIndex {
            if let Some(idx) = self._removed.pop() {
                self._parts[idx.index()] = Part { seq: Seq { first, len }, gen: self._gen };
                idx
            } else {
                let idx = PartIndex::new(self._parts.len());
                self._parts.push(Part { seq: Seq { first, len }, gen: self._gen });
                idx
            }
        }

        #[allow(non_snake_case)]
        fn add_pivot(&mut self, X: Vec<NodeIndex>, X_a: Vec<NodeIndex>) {
            traceln!("add_pivot: {:?} {:?}", d1(&X), d1(&X_a));
            if self.pivots.contains(&X) {
                self.pivots.push(X_a);
            } else {
                let i = self.modules.iter().position(|Y| Y == &X);
                let (S, L) = smaller_larger(X, X_a);
                self.pivots.push(S);
                if let Some(i) = i {
                    self.modules[i] = L;
                } else {
                    self.modules.push_back(L);
                }
            }
        }

        #[allow(non_snake_case)]
        fn _add_pivot(&mut self, X: Seq, X_a: Seq) {
            if self._pivots.contains(&X) {
                self._pivots.push(X_a);
            } else {
                let i = self._modules.iter().position(|Y| Y == &X);
                let (S, L) = if X.len <= X_a.len { (X, X_a) } else { (X_a, X) };
                self._pivots.push(S);
                if let Some(i) = i {
                    self._modules[i] = L;
                } else {
                    self._modules.push_back(L);
                }
            }
        }

        #[allow(non_snake_case)]
        fn partition_refinement(&mut self, graph: &UnGraph<(), ()>) {
            while self.init_partition(graph) {
                while let Some(E) = self.pivots.pop() {
                    let _E = self._pivots.pop().unwrap();
                    traceln!("P: {:?}", d2(&*self.P));
                    traceln!("pivots: {:?}", d2(&self.pivots));
                    traceln!("modules: {:?}", d2(&self.modules));
                    traceln!("pivot: {:?}", d1(&E));
                    let E_h: HashSet<_, RandomState> = HashSet::from_iter(E.clone());
                    for &x in &E {
                        let S = graph.neighbors(x).filter(|v| !E_h.contains(v)).collect();
                        let _S: HashSet<_> = graph
                            .neighbors(x)
                            .filter(|v| !_E.range().contains(&self._positions[v.index()].index()))
                            .collect();
                        assert_eq!(S, _S);
                        self.refine(S, x, graph, _E);
                        traceln!("P: {:?}", d2(&*self.P));
                    }
                }
            }
        }

        #[allow(non_snake_case)]
        fn splice(P: &mut Vec<Vec<NodeIndex>>, i: usize, A: Vec<NodeIndex>, x: NodeIndex, N: Vec<NodeIndex>) {
            match (A.is_empty(), N.is_empty()) {
                (true, true) => P[i] = vec![x],
                (true, false) => {
                    P[i] = vec![x];
                    P.insert(i + 1, N)
                }
                (false, true) => {
                    P[i] = A;
                    P.insert(i + 1, vec![x]);
                }
                (false, false) => {
                    P[i] = A;
                    P.insert(i + 1, vec![x]);
                    P.insert(i + 2, N)
                }
            }
        }

        fn next_non_singleton(&mut self) -> Option<PartIndex> {
            let mut j = self._non_singleton_idx.index();
            while j + 1 < self._nodes.len() {
                if self._nodes[j].part == self._nodes[j + 1].part {
                    self._non_singleton_idx = NodePos::new(j);
                    return Some(self._nodes[j].part);
                }
                j += 1;
            }
            None
        }

        #[allow(non_snake_case)]
        fn init_partition(&mut self, graph: &UnGraph<(), ()>) -> bool {
            if self.next_non_singleton().is_none() {
                return false;
            }
            // if self.P.iter().all(|p| p.len() <= 1) { return false; }
            if let Some(X) = self.modules.pop_front() {
                let x = X[0];
                self.pivots.push(vec![x]);
                self.first_pivot.insert(X, x);

                {
                    let X = self._modules.pop_front().unwrap();
                    let x_pos: NodePos = X.range().min_by_key(|&pos| self._nodes[pos].node).unwrap().into(); // TODO: any node is okay. only use min to make it identical to other impl.
                    let x = self._nodes[x_pos.index()].node;
                    self._pivots.push(Seq { first: x_pos, len: 1 });
                    self._first_pivot.insert(X, x);
                }
            } else {
                for (i, X) in self.P.iter().enumerate() {
                    if X.len() <= 1 {
                        continue;
                    }
                    let x = self.first_pivot.get(X).copied().unwrap_or(X[0]);
                    let adj: HashSet<_> = graph.neighbors(x).collect();
                    let (A, mut N): (Vec<NodeIndex>, Vec<_>) = X.iter().partition(|&y| *y != x && adj.contains(y));
                    traceln!(
                        "A [x] N : {:?} [{:?}] {:?}",
                        A.iter().map(|u| u.index()).collect::<Vec<_>>(),
                        x.index(),
                        N.iter().map(|u| u.index()).collect::<Vec<_>>()
                    );
                    N.retain(|y| *y != x);
                    Self::splice(&mut self.P, i, A.clone(), x, N.clone());
                    let (S, L) = smaller_larger(A, N);
                    self.center = x;
                    self.pivots.push(S);
                    self.modules.push_back(L);
                    break;
                }

                let X_idx = self.next_non_singleton().unwrap();
                let X = self._parts[X_idx.index()].seq;
                assert!(X.len > 1);

                let x = self._first_pivot.get(&X).copied().unwrap_or_else(|| {
                    let x_pos: NodePos = X.range().min_by_key(|&pos| self._nodes[pos].node).unwrap().into(); // TODO: any node is okay. only use min to make it identical to other impl.
                    self._nodes[x_pos.index()].node
                });

                if true {
                    self._gen.0 += 1;

                    for v in graph.neighbors(x) {
                        let v_pos = self.position(v);
                        if X.contains(v_pos) {
                            self._insert_left(v_pos);
                        }
                    }

                    self._gen.0 += 1;
                    self._insert_left(self.position(x));

                    let x_pos = self.position(x);

                    let [A, N] = [-1_isize, 1_isize].map(|diff| {
                        let pos = x_pos.index().wrapping_add_signed(diff);
                        if pos < self._nodes.len() {
                            let pos = NodePos::new(pos);
                            if X.contains(pos) {
                                return self.part(self.node(pos).part).seq;
                            }
                        }
                        Seq::new(NodePos(0), 0)
                    });

                    let (S, L) = if A.len <= N.len { (A, N) } else { (N, A) };
                    self._center_pos = x_pos;
                    self._pivots.push(S);
                    self._modules.push_back(L);
                } else {
                    let mut w_pos = NodePos::new(X.first.index());
                    //let A_idx = PartIndex::new(self._parts.len());
                    //self._parts.push(Seq { first: X.first, len: 0 });
                    let A_idx = self._new_part(X.first, 0);
                    for v in graph.neighbors(x) {
                        let v_pos = self._positions[v.index()];
                        if X.range().contains(&v_pos.index()) {
                            let w = self._nodes[w_pos.index()].node;
                            self._nodes.swap(v_pos.index(), w_pos.index());
                            self._positions.swap(v.index(), w.index());
                            self._nodes[w_pos.index()].part = A_idx;
                            w_pos.0 += 1;
                        }
                    }
                    self._parts[A_idx.index()].seq.len = (w_pos.index() - X.first.index()) as u32;
                    let w = self._nodes[w_pos.index()].node;
                    let x_pos = self._positions[x.index()];
                    self._nodes.swap(x_pos.index(), w_pos.index());
                    self._positions.swap(x.index(), w.index());
                    self._parts[self._nodes[w_pos.index()].part.index()].seq = Seq { first: w_pos, len: 1 };
                    w_pos.0 += 1;
                    //let N_idx = PartIndex::new(self._parts.len());
                    //self._parts.push(Seq { first: w_pos, len: 0 });
                    let N_idx = self._new_part(w_pos, 0);

                    while w_pos.index() < X.first.index() + X.len as usize {
                        self._nodes[w_pos.index()].part = N_idx;
                        w_pos.0 += 1;
                    }
                    self._parts[N_idx.index()].seq.len =
                        (w_pos.index() - self._parts[N_idx.index()].seq.first.index()) as u32;

                    let A = self._parts[A_idx.index()].seq;
                    let N = self._parts[N_idx.index()].seq;
                    let (S, L) = if A.len <= N.len { (A, N) } else { (N, A) };
                    self._center_pos = self._positions[x.index()];
                    self._pivots.push(S);
                    self._modules.push_back(L);

                    self._remove_if_empty(A_idx);
                    self._remove_if_empty(N_idx);
                }
            }
            self.check_consistency();
            true
        }
    }

    #[cfg(test)]
    mod test {
        use super::factorizing_permutation;
        use common::instances;
        use petgraph::graph::UnGraph;

        #[test]
        fn graph_001() {
            let graph = common::instances::graph_001_10_19();
            let _p = factorizing_permutation(&graph);
            // refine: 2 [0, 1, 6]
            // refine: 3 [0, 5, 6]
            // refine: 4 [6, 0, 5]
            // refine: 5 [3, 4, 6]
            // refine: 2 [1, 3, 4, 0, 6]
            // refine: 1 [2]
            // refine: 1 [2]
            // refine: 6 [4, 3, 2, 9, 7, 8, 5]
            // refine: 1 [2]
            // refine: 5 [3, 6, 4]
            // refine: 6 [2, 5, 8, 7, 4, 9, 3]
            // refine: 7 [9, 8, 6]
            // refine: 3 [5, 6, 2, 4, 0]
            // refine: 4 [6, 5, 2, 3, 0]
            // refine: 8 [9, 7, 6]

            // [[4], [3], [2], [0], [6], [1], [5], [9], [8], [7]]
        }

        #[test]
        fn graph_002() {
            let graph = UnGraph::from_edges([
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 5),
                (4, 5),
                (5, 6),
                (5, 7),
                (6, 7),
                (8, 11),
                (9, 11),
                (10, 11),
            ]);
            let p = factorizing_permutation(&graph);
            let p: Vec<_> = p.iter().map(|u| u.index()).collect();
            println!("{p:?}");
        }

        #[test]
        fn exact_055_4() {
            let graph = UnGraph::from_edges([
                (13, 0),
                (13, 14),
                (13, 10),
                (13, 11),
                (13, 20),
                (13, 15),
                (13, 22),
                (13, 23),
                (13, 12),
                (13, 3),
                (13, 5),
                (13, 2),
                (13, 1),
                (13, 21),
                (13, 7),
                (13, 24),
                (13, 8),
                (13, 25),
                (13, 27),
                (13, 30),
                (0, 14),
                (0, 10),
                (0, 11),
                (0, 20),
                (0, 15),
                (0, 22),
                (0, 23),
                (0, 29),
                (0, 12),
                (0, 3),
                (0, 5),
                (0, 2),
                (0, 32),
                (0, 1),
                (0, 21),
                (0, 7),
                (0, 16),
                (0, 6),
                (0, 31),
                (0, 17),
                (0, 18),
                (0, 19),
                (0, 4),
                (0, 28),
                (0, 26),
                (0, 9),
                (0, 24),
                (0, 8),
                (0, 25),
                (0, 27),
                (14, 10),
                (14, 11),
                (14, 20),
                (14, 15),
                (14, 22),
                (14, 23),
                (14, 12),
                (14, 3),
                (14, 5),
                (14, 2),
                (20, 15),
                (20, 22),
                (20, 23),
                (20, 12),
                (20, 3),
                (20, 5),
                (20, 2),
                (20, 1),
                (20, 21),
                (20, 7),
                (15, 22),
                (15, 23),
                (15, 29),
                (15, 12),
                (15, 3),
                (15, 5),
                (15, 2),
                (15, 1),
                (15, 21),
                (15, 7),
                (22, 23),
                (22, 12),
                (22, 3),
                (22, 5),
                (22, 2),
                (22, 32),
                (22, 1),
                (22, 21),
                (22, 7),
                (22, 16),
                (22, 6),
                (22, 31),
                (22, 17),
                (22, 18),
                (22, 19),
                (22, 4),
                (22, 28),
                (22, 26),
                (22, 9),
                (22, 24),
                (22, 8),
                (22, 25),
                (22, 27),
                (23, 29),
                (23, 12),
                (23, 3),
                (23, 5),
                (23, 2),
                (23, 32),
                (23, 1),
                (23, 21),
                (23, 7),
                (23, 16),
                (23, 6),
                (23, 31),
                (23, 17),
                (23, 18),
                (23, 19),
                (23, 4),
                (23, 28),
                (23, 26),
                (23, 9),
                (23, 24),
                (23, 8),
                (23, 25),
                (23, 27),
                (23, 30),
                (12, 3),
                (12, 5),
                (12, 2),
                (12, 1),
                (12, 21),
                (3, 5),
                (3, 2),
                (3, 32),
                (3, 1),
                (3, 21),
                (3, 7),
                (3, 16),
                (3, 6),
                (3, 31),
                (3, 17),
                (3, 18),
                (3, 19),
                (3, 4),
                (3, 28),
                (3, 26),
                (3, 9),
                (3, 24),
                (3, 8),
                (3, 25),
                (3, 27),
                (3, 30),
                (5, 2),
                (5, 32),
                (5, 1),
                (5, 21),
                (5, 7),
                (5, 16),
                (5, 6),
                (5, 31),
                (5, 17),
                (5, 18),
                (5, 19),
                (5, 4),
                (5, 28),
                (5, 26),
                (5, 9),
                (5, 24),
                (5, 8),
                (5, 25),
                (5, 27),
                (18, 19),
                (18, 4),
                (18, 28),
                (18, 26),
                (18, 9),
                (18, 24),
                (18, 8),
                (18, 25),
                (18, 27),
                (19, 4),
                (9, 24),
                (9, 8),
                (9, 25),
                (9, 27),
                (24, 8),
                (24, 25),
                (24, 27),
                (8, 25),
                (8, 27),
                (8, 30),
                (25, 27),
                (25, 30),
                (27, 30),
            ]);

            let p = factorizing_permutation(&graph);
            let p: Vec<_> = p.iter().map(|u| u.index()).collect();
            println!("{:?}", p);
        }

        #[test]
        fn exact_024() {
            let graph = instances::pace2023_exact_024();
            let p = factorizing_permutation(&graph);
            let p: Vec<_> = p.iter().map(|u| u.index()).collect();
            println!("{:?}", p);
        }
    }
}
