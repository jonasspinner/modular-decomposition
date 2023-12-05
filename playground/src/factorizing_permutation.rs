use petgraph::graph::NodeIndex;


#[macro_export]
macro_rules! traceln {
    ($($x:expr),*) => {
        //println!($($x),*)
    }
}

#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        //print!($($x),*)
    }
}

#[allow(dead_code)]
fn d2<'a, P>(p: P) -> Vec<Vec<usize>>
    where P: IntoIterator<Item=&'a Vec<NodeIndex>>,
{
    p.into_iter().map(|q| d1(q)).collect()
}

fn d1(p: &[NodeIndex]) -> Vec<usize> {
    p.iter().map(|u| u.index()).collect()
}

mod kar19 {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::collections::hash_map::RandomState;
    use petgraph::graph::{NodeIndex, UnGraph};
    #[allow(unused_imports)]
    use super::{d1, d2};

    #[allow(dead_code, non_snake_case)]
    struct State {
        P: Vec<Vec<NodeIndex>>,
        center: NodeIndex,
        pivots: Vec<Vec<NodeIndex>>,
        modules: VecDeque<Vec<NodeIndex>>,
        first_pivot: HashMap::<Vec<NodeIndex>, NodeIndex>,
    }

    #[allow(non_snake_case, dead_code)]
    pub fn symgraph_factorizing_permutation(graph: &UnGraph<(), ()>) -> Vec<NodeIndex> {
        let P = vec![graph.node_indices().collect()];
        let center = NodeIndex::new(0);
        let pivots = vec![];
        let modules = VecDeque::new();
        let first_pivot = HashMap::<Vec<NodeIndex>, NodeIndex>::new();
        let mut state = State { P, center, pivots, modules, first_pivot };

        state.partition_refinement(graph);

        //println!("{:?}", state.first_pivot.iter().map(|(X, v)| (X.iter().map(|x| x.index()).collect::<Vec<_>>(), v.index())).collect::<Vec<_>>());
        state.P.iter().map(|part| part[0]).collect()
    }


    #[allow(non_snake_case)]
    fn smaller_larger(A: Vec<NodeIndex>, B: Vec<NodeIndex>) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
        if A.len() <= B.len() { (A, B) } else { (B, A) }
    }

    impl State {
        #[allow(non_snake_case)]
        fn refine(&mut self, S: HashSet<NodeIndex>, x: NodeIndex) {
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
                if X_a.is_empty() || X.is_empty() { continue; }
                //traceln!("refine:P:0: {:?}", d2(&*self.P));
                self.P[i] = X.clone();
                self.P.insert(i + between as usize, X_a.clone());
                //traceln!("refine:P:1: {:?}", d2(&*self.P));
                self.add_pivot(X, X_a);
                i += 1;
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
        fn partition_refinement(&mut self, graph: &UnGraph<(), ()>) {
            while self.init_partition(graph) {
                while let Some(E) = self.pivots.pop() {
                    traceln!("P: {:?}", d2(&*self.P));
                    traceln!("pivots: {:?}", d2(&self.pivots));
                    traceln!("modules: {:?}", d2(&self.modules));
                    traceln!("pivot: {:?}", d1(&E));
                    let E_h: HashSet<_, RandomState> = HashSet::from_iter(E.clone());
                    for &x in &E {
                        let S = graph.neighbors(x).filter(|v| !E_h.contains(v)).collect();
                        self.refine(S, x);
                        traceln!("P: {:?}", d2(&*self.P));
                    }
                }
            }
        }

        #[allow(non_snake_case)]
        fn splice(P: &mut Vec<Vec<NodeIndex>>, i: usize, A: Vec<NodeIndex>, x: NodeIndex, N: Vec<NodeIndex>) {
            match (A.is_empty(), N.is_empty()) {
                (true, true) => { P[i] = vec![x] }
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

        #[allow(non_snake_case)]
        fn init_partition(&mut self, graph: &UnGraph<(), ()>) -> bool {
            if self.P.iter().all(|p| p.len() <= 1) { return false; }
            if let Some(X) = self.modules.pop_front() {
                let x = X[0];
                self.pivots.push(vec![x]);
                self.first_pivot.insert(X, x);
            } else {
                for (i, X) in self.P.iter().enumerate() {
                    if X.len() <= 1 { continue; }
                    let x = self.first_pivot.get(X).copied().unwrap_or(X[0]);
                    let adj: HashSet<_> = graph.neighbors(x).collect();
                    let (A, mut N): (Vec<NodeIndex>, Vec<_>) = X.iter().partition(|&y| *y != x && adj.contains(y));
                    traceln!("A [x] N : {:?} [{:?}] {:?}", A.iter().map(|u| u.index()).collect::<Vec<_>>(), x.index(), N.iter().map(|u| u.index()).collect::<Vec<_>>());
                    N.retain(|y| *y != x);
                    Self::splice(&mut self.P, i, A.clone(), x, N.clone());
                    let (S, L) = smaller_larger(A, N);
                    self.center = x;
                    self.pivots.push(S);
                    self.modules.push_back(L);
                    break;
                }
            }
            true
        }
    }

    #[cfg(test)]
    mod test {
        use petgraph::graph::UnGraph;
        use crate::factorizing_permutation::kar19::symgraph_factorizing_permutation;

        #[test]
        fn graph_001() {
            let graph = common::instances::graph_001_10_19();
            let _p = symgraph_factorizing_permutation(&graph);
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
        fn exact_055_4() {
            let graph = UnGraph::from_edges([
                (13, 0), (13, 14), (13, 10), (13, 11), (13, 20), (13, 15), (13, 22), (13, 23), (13, 12), (13, 3), (13, 5), (13, 2), (13, 1), (13, 21), (13, 7), (13, 24), (13, 8), (13, 25), (13, 27), (13, 30), (0, 14), (0, 10), (0, 11), (0, 20), (0, 15), (0, 22), (0, 23), (0, 29), (0, 12), (0, 3), (0, 5), (0, 2), (0, 32), (0, 1), (0, 21), (0, 7), (0, 16), (0, 6), (0, 31), (0, 17), (0, 18), (0, 19), (0, 4), (0, 28), (0, 26), (0, 9), (0, 24), (0, 8), (0, 25), (0, 27), (14, 10), (14, 11), (14, 20), (14, 15), (14, 22), (14, 23), (14, 12), (14, 3), (14, 5), (14, 2), (20, 15), (20, 22), (20, 23), (20, 12), (20, 3), (20, 5), (20, 2), (20, 1), (20, 21), (20, 7), (15, 22), (15, 23), (15, 29), (15, 12), (15, 3), (15, 5), (15, 2), (15, 1), (15, 21), (15, 7), (22, 23), (22, 12), (22, 3), (22, 5), (22, 2), (22, 32), (22, 1), (22, 21), (22, 7), (22, 16), (22, 6), (22, 31), (22, 17), (22, 18), (22, 19), (22, 4), (22, 28), (22, 26), (22, 9), (22, 24), (22, 8), (22, 25), (22, 27), (23, 29), (23, 12), (23, 3), (23, 5), (23, 2), (23, 32), (23, 1), (23, 21), (23, 7), (23, 16), (23, 6), (23, 31), (23, 17), (23, 18), (23, 19), (23, 4), (23, 28), (23, 26), (23, 9), (23, 24), (23, 8), (23, 25), (23, 27), (23, 30), (12, 3), (12, 5), (12, 2), (12, 1), (12, 21), (3, 5), (3, 2), (3, 32), (3, 1), (3, 21), (3, 7), (3, 16), (3, 6), (3, 31), (3, 17), (3, 18), (3, 19), (3, 4), (3, 28), (3, 26), (3, 9), (3, 24), (3, 8), (3, 25), (3, 27), (3, 30), (5, 2), (5, 32), (5, 1), (5, 21), (5, 7), (5, 16), (5, 6), (5, 31), (5, 17), (5, 18), (5, 19), (5, 4), (5, 28), (5, 26), (5, 9), (5, 24), (5, 8), (5, 25), (5, 27), (18, 19), (18, 4), (18, 28), (18, 26), (18, 9), (18, 24), (18, 8), (18, 25), (18, 27), (19, 4), (9, 24), (9, 8), (9, 25), (9, 27), (24, 8), (24, 25), (24, 27), (8, 25), (8, 27), (8, 30), (25, 27), (25, 30), (27, 30)
            ]);

            let p : Vec<_> = symgraph_factorizing_permutation(&graph).iter().map(|u| u.index()).collect();
            println!("{:?}", p);
        }
    }
}


pub mod mpv99 {
    use std::collections::{HashMap, VecDeque};
    use petgraph::graph::{NodeIndex, UnGraph};
    use common::make_index;

    make_index!(NodePos);
    make_index!(PartIndex);

    #[derive(Debug)]
    struct Node {
        node: NodeIndex,
        part: PartIndex,
    }

    #[derive(Debug, Clone, Eq, PartialEq, Hash)]
    struct Seq {
        start: NodePos,
        len: u32,
    }

    impl Seq {
        fn empty() -> Self {
            Self { start: NodePos::new(0), len: 0 }
        }
        fn positions(&self) -> impl Iterator<Item=NodePos> {
            (self.start.index()..self.start.index() + self.len()).map(NodePos::new)
        }
        fn is_empty(&self) -> bool { self.len() == 0 }
        fn len(&self) -> usize { self.len as usize }
        fn first(&self) -> NodePos {
            assert!(!self.is_empty());
            self.start
        }
        fn last(&self) -> NodePos {
            assert!(!self.is_empty());
            NodePos::new(self.start.index() + self.len() - 1)
        }
        fn contains(&self, pos: NodePos) -> bool { self.first() <= pos && pos <= self.last() }
    }

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    struct Gen(pub u32);

    #[derive(Debug, Clone)]
    struct Part {
        seq: Seq,
        gen: Gen,
    }

    pub struct State<'a> {
        graph: &'a UnGraph<(), ()>,

        position: Vec<NodePos>,
        nodes: Vec<Node>,
        parts: Vec<Part>,

        removed: Vec<PartIndex>,
        gen: Gen,

        center_pos: NodePos,
        pivots: Vec<Seq>,
        modules: VecDeque<Seq>,
        first_pivot: HashMap<Seq, NodeIndex>,
        singleton_index: usize,
    }

    impl<'a> State<'a> {
        pub fn new(graph: &'a UnGraph<(), ()>) -> Self {
            let size = graph.node_count();
            let gen = Gen(0);
            let mut parts = Vec::with_capacity(size);
            parts.push(Part { seq: Seq { start: NodePos::new(0), len: size as u32 }, gen });
            Self {
                graph,
                position: (0..size).map(NodePos::new).collect(),
                nodes: (0..size)
                    .map(|i| Node {
                        node: NodeIndex::new(i),
                        part: PartIndex::new(0),
                    }).collect(),
                parts,
                removed: Vec::with_capacity(size),
                gen,
                center_pos: NodePos::new(0),
                pivots: Vec::with_capacity(size),
                modules: VecDeque::with_capacity(size),
                first_pivot: HashMap::with_capacity(size),
                singleton_index: 0,
            }
        }
    }

    impl<'a> State<'a> {
        #[allow(dead_code)]
        fn debug_partition(&self) -> Vec<Vec<usize>> {
            let mut res = vec![vec![self.nodes[0].node.index()]];
            let mut i = 1;
            while i < self.nodes.len() {
                if self.nodes[i - 1].part != self.nodes[i].part {
                    res.push(vec![]);
                }
                res.last_mut().unwrap().push(self.nodes[i].node.index());
                i += 1;
            }
            res
        }

        #[allow(non_snake_case)]
        pub fn partition_refinement(&mut self) {
            while self.init_partition() {
                while let Some(pivot) = self.pivots.pop() {
                    traceln!("P: {:?}", self.debug_partition());
                    traceln!("pivots: {:?}", self.pivots.iter().map(|pivot| pivot.positions().map(|pos| self.nodes(pos).node.index()).collect::<Vec<_>>()).collect::<Vec<_>>());
                    traceln!("modules: {:?}", self.modules.iter().map(|module| module.positions().map(|pos| self.nodes(pos).node.index()).collect::<Vec<_>>()).collect::<Vec<_>>());
                    traceln!("pivot: {:?}", pivot.positions().map(|pos| self.nodes(pos).node.index()).collect::<Vec<_>>());
                    for pos in pivot.positions() {
                        let x = self.nodes[pos.index()].node;
                        self.refine(x, pivot.clone());
                        traceln!("P: {:?}", self.debug_partition());
                    }
                }
            }
        }

        pub fn permutation(&self) -> impl Iterator<Item=NodeIndex> + '_ {
            self.nodes.iter().map(|n| n.node)
        }

        #[allow(non_snake_case)]
        fn init_partition(&mut self) -> bool {
            let Some(non_singleton_class) = self.non_singleton_class() else { return false; };

            if let Some(X) = self.modules.pop_front() {
                let x = self.nodes(X.first()).node;
                self.first_pivot.insert(X, x);
            } else {
                let X = non_singleton_class;
                let x = *self.first_pivot.get(&X.seq).unwrap_or(&self.nodes[X.seq.start.index()].node);

                let (X_n, X_a) = self.replace(x, X);

                traceln!("N [x] A : {:?} [{:?}] {:?}", X_n.positions().map(|pos| self.nodes(pos).node.index()).collect::<Vec<_>>(), x.index(), X_a.positions().map(|pos| self.nodes(pos).node.index()).collect::<Vec<_>>());

                self.center_pos = self.position(x);

                let (S, L) = if X_n.len() < X_a.len() { (X_n, X_a) } else { (X_a, X_n) };

                if !S.is_empty() { self.pivots.push(S); }
                if !L.is_empty() { self.modules.push_back(L); }
            }
            true
        }

        fn insert_right(&mut self, part: PartIndex, u: NodeIndex, u_pos: NodePos) {
            let last = self.parts(part).seq.last();
            //println!("{:?} {:?} {:?} {:?}", part.index(), (self.parts(part).seq.start.index(), self.parts(part).seq.len, self.parts(part).gen.0), u, u_pos);
            //println!("{:?}", self.parts(part).seq.positions().map(|u| u.index()).collect::<Vec<_>>());
            //println!("{:?}", self.parts(part).seq.positions().map(|u| self.nodes(u).node.index()).collect::<Vec<_>>());
            //println!("{:?}", self.parts(part).seq.positions().map(|u| self.nodes(u).part.index()).collect::<Vec<_>>());
            let next = (last.index() + 1 != self.nodes.len()).then(|| {
                self.nodes[last.index() + 1].part
            });

            let new_part = if next.is_some_and(|next| self.parts[next.index()].gen.0 == self.gen.0 - 1) {
                next.unwrap()
            } else {
                self.new_part(NodePos::new(last.index() + 1), 0, Gen(self.gen.0 - 1))
            };

            self.nodes.swap(last.index(), u_pos.index());
            self.position[u.index()] = last;
            self.position[self.nodes[u_pos.index()].node.index()] = u_pos;
            self.parts[new_part.index()].seq.start.0 -= 1;
            self.parts[new_part.index()].seq.len += 1;
            self.parts[part.index()].seq.len -= 1;
            self.nodes[last.index()].part = new_part;

            if self.parts(part).seq.is_empty() {
                self.remove_part(part);
                self.parts[new_part.index()].gen.0 = self.gen.0 - 2;
            }

            assert!(self.parts(self.nodes(u_pos).part).seq.contains(u_pos));
            assert!(self.parts(self.nodes(last).part).seq.contains(last));
        }

        fn insert_left(&mut self, part: PartIndex, u: NodeIndex, u_pos: NodePos) {
            let first = self.parts(part).seq.first();
            //println!("{:?} {:?} {:?} {:?}", part.index(), (self.parts(part).seq.start.index(), self.parts(part).seq.len, self.parts(part).gen.0), u, u_pos);
            //println!("{:?}", self.parts(part).seq.positions().map(|u| u.index()).collect::<Vec<_>>());
            //println!("{:?}", self.parts(part).seq.positions().map(|u| self.nodes(u).node.index()).collect::<Vec<_>>());
            //println!("{:?}", self.parts(part).seq.positions().map(|u| self.nodes(u).part.index()).collect::<Vec<_>>());
            let prev = (first.index() != 0).then(|| {
                self.nodes[first.index() - 1].part
            });
            let new_part = if prev.is_some_and(|prev| self.parts[prev.index()].gen == self.gen) {
                prev.unwrap()
            } else {
                self.new_part(first, 0, self.gen)
            };

            self.nodes.swap(first.index(), u_pos.index());
            self.position[u.index()] = first;
            self.position[self.nodes[u_pos.index()].node.index()] = u_pos;
            self.parts[new_part.index()].seq.len += 1;
            self.parts[part.index()].seq.start.0 += 1_u32;
            self.parts[part.index()].seq.len -= 1;
            self.nodes[first.index()].part = new_part;

            if self.parts(part).seq.is_empty() {
                self.remove_part(part);
                self.parts[new_part.index()].gen.0 = self.gen.0 - 2;
            }

            assert!(self.parts(self.nodes(u_pos).part).seq.contains(u_pos));
            assert!(self.parts(self.nodes(first).part).seq.contains(first));
        }

        #[allow(non_snake_case)]
        fn refine(&mut self, y: NodeIndex, Y: Seq) {
            traceln!("refine: {} {:?}", y.index(), self.graph.neighbors(y)
                .filter_map(|u| if !Y.contains(self.position(u)) { Some(u.index())} else { None }).collect::<Vec<_>>());

            let y_pos = self.position(y);
            assert!(Y.contains(y_pos));
            self.gen.0 += 2;

            for u in self.graph.neighbors(y) {
                let u_pos = self.position(u);
                if Y.contains(u_pos) { continue; }

                let part = self.nodes(u_pos).part;
                let X = self.parts(part).seq.clone();

                if self.should_insert_right(&X, y_pos) {
                    self.insert_right(part, u, u_pos);
                } else {
                    self.insert_left(part, u, u_pos);
                }
            }

            for u in self.graph.neighbors(y) {
                let u_pos = self.position(u);
                if Y.contains(u_pos) { continue; }
                let X_idx = self.nodes(u_pos).part;
                let X = self.parts[X_idx.index()].seq.clone();
                let X_gen = self.parts[X_idx.index()].gen;
                if X_gen == self.gen || X_gen.0 + 1 == self.gen.0 {
                    self.parts[X_idx.index()].gen.0 = self.gen.0 - 2;

                    if self.should_insert_right(&X, y_pos) {
                        let prev = self.nodes[X.first().index() - 1].part;
                        let X_other = self.parts(prev).seq.clone();
                        self.add_pivot(&X_other, &X);
                    } else {
                        let next = self.nodes[X.last().index() + 1].part;
                        let X_other = self.parts(next).seq.clone();
                        self.add_pivot(&X_other, &X);
                    }
                }
            }
            self.gen.0 += 1;
        }

        #[allow(non_snake_case)]
        fn add_pivot(&mut self, X: &Seq, X_a: &Seq) {
            traceln!("add_pivot: {:?} {:?}", X.positions().map(|pos| self.nodes[pos.index()].node.index()).collect::<Vec<_>>(), X_a.positions().map(|pos| self.nodes[pos.index()].node.index()).collect::<Vec<_>>());
            let (Z, Z_prime) = if X.len() < X_a.len() { (X.clone(), X_a.clone()) } else { (X_a.clone(), X.clone()) };
            if self.pivots.contains(&X) {
                self.pivots.push(X_a.clone());
            } else {
                self.pivots.push(Z);
                let i = self.modules.iter().position(|Y| Y == X);
                if let Some(i) = i {
                    self.modules[i] = Z_prime;
                } else {
                    self.modules.push_back(Z_prime);
                }
            }
        }

        #[allow(non_snake_case)]
        fn should_insert_right(&self, X: &Seq, y_pos: NodePos) -> bool {
            let (a, b) = (y_pos, self.center_pos);
            let (a, b) = if a < b { (a, b) } else { (b, a) };
            !(a < X.first() && X.first() < b)
        }

        #[allow(non_snake_case)]
        fn replace(&mut self, x: NodeIndex, _X: Part) -> (Seq, Seq) {
            // X
            // X \ {x}, {x}
            // \bar{N}(x) \cap X, N(x) \cap X + {x}
            // \bar{N}(x) \cap X, {x}, N(x) \cap X
            let a_index = self.nodes[self.position[x.index()].index()].part;
            let mut a = self.parts(a_index).seq.clone();
            let b_index = self.new_part(NodePos::new(a.start.index() + a.len()), 0, Gen(0));
            let mut b = self.parts(b_index).seq.clone();

            let mut move_to_end = |this: &mut Self, u: NodeIndex| {
                let u_pos = this.position(u);
                assert!(a.contains(u_pos));

                let last_pos = a.last();
                assert!(a.contains(last_pos));

                let last = this.nodes(last_pos).node;
                this.position.swap(u.index(), last.index());
                this.nodes.swap(u_pos.index(), last_pos.index());

                a.len -= 1;
                b.start.0 -= 1;
                b.len += 1;
                this.nodes[last_pos.index()].part = b_index;

                assert!(b.contains(last_pos));
            };

            move_to_end(self, x);

            for y in self.graph.neighbors(x) {
                if self.nodes(self.position(y)).part == a_index {
                    move_to_end(self, y)
                }
            }

            let res = if b.len() == 1 {
                self.parts[a_index.index()].seq = a.clone();
                self.parts[b_index.index()].seq = b.clone();
                (a, Seq::empty())
            } else {
                let first_pos = b.first();
                let last_pos = b.last();
                let (first, last) = (self.nodes(first_pos).node, x);
                self.nodes.swap(first_pos.index(), last_pos.index());
                self.position.swap(first.index(), last.index());

                let x_pos = first_pos;

                b.start.0 += 1;
                b.len -= 1;
                if a.is_empty() {
                    a.len += 1;
                    self.parts[a_index.index()].seq = a.clone();
                    self.parts[b_index.index()].seq = b.clone();
                    self.nodes[first_pos.index()].part = a_index;
                    (Seq::empty(), b)
                } else {
                    self.parts[a_index.index()].seq = a.clone();
                    self.parts[b_index.index()].seq = b.clone();
                    let c_index = self.new_part(NodePos::new(b.start.index() - 1), 1, Gen(0));
                    self.nodes[x_pos.index()].part = c_index;
                    (a, b)
                }
            };

            //println!("position  {:?}", self.position.iter().map(|p| p.index()).collect::<Vec<_>>());
            //println!("nodes (n) {:?}", self.nodes.iter().map(|n| n.node.index()).collect::<Vec<_>>());
            //println!("nodes (p) {:?}", self.nodes.iter().map(|n| n.part.index()).collect::<Vec<_>>());
            //println!("parts     {:?}", self.parts.iter().map(|p| (p.seq.start.index(), p.seq.len, p.gen.0)).collect::<Vec<_>>());
            //println!("pivots    {:?}", self.pivots);
            //println!("modules   {:?}", self.modules);

            res
        }

        fn only_singleton_classes(&mut self) -> bool {
            let mut i = self.singleton_index;
            while i + 1 < self.nodes.len() && self.nodes[i].part != self.nodes[i + 1].part {
                i += 1;
            }
            self.singleton_index = i;
            self.singleton_index + 1 == self.nodes.len()
        }

        fn non_singleton_class(&mut self) -> Option<Part> {
            if !self.only_singleton_classes() {
                return Some(self.parts[self.nodes[self.singleton_index].part.index()].clone());
            }
            None
        }

        fn position(&self, x: NodeIndex) -> NodePos { self.position[x.index()] }
        fn nodes(&self, pos: NodePos) -> &Node { &self.nodes[pos.index()] }
        fn parts(&self, idx: PartIndex) -> &Part { &self.parts[idx.index()] }
        fn new_part(&mut self, start: NodePos, len: u32, gen:Gen) -> PartIndex {
            let seq = Seq { start, len };
            if let Some(part) = self.removed.pop() {
                self.parts[part.index()] = Part { seq, gen };
                part
            } else {
                let part = PartIndex::new(self.parts.len());
                self.parts.push(Part { seq, gen });
                part
            }
        }
        fn remove_part(&mut self, part: PartIndex) {
            self.removed.push(part);
        }
    }


    #[cfg(test)]
    mod test {
        use petgraph::graph::UnGraph;
        use crate::factorizing_permutation::mpv99::State;

        #[test]
        fn graph_001() {
            let graph = common::instances::graph_001_10_19();

            let mut state = State::new(&graph);
            state.partition_refinement();
        }

        #[test]
        fn exact_055_4() {
            let graph = UnGraph::from_edges([
                (13, 0), (13, 14), (13, 10), (13, 11), (13, 20), (13, 15), (13, 22), (13, 23), (13, 12), (13, 3), (13, 5), (13, 2), (13, 1), (13, 21), (13, 7), (13, 24), (13, 8), (13, 25), (13, 27), (13, 30), (0, 14), (0, 10), (0, 11), (0, 20), (0, 15), (0, 22), (0, 23), (0, 29), (0, 12), (0, 3), (0, 5), (0, 2), (0, 32), (0, 1), (0, 21), (0, 7), (0, 16), (0, 6), (0, 31), (0, 17), (0, 18), (0, 19), (0, 4), (0, 28), (0, 26), (0, 9), (0, 24), (0, 8), (0, 25), (0, 27), (14, 10), (14, 11), (14, 20), (14, 15), (14, 22), (14, 23), (14, 12), (14, 3), (14, 5), (14, 2), (20, 15), (20, 22), (20, 23), (20, 12), (20, 3), (20, 5), (20, 2), (20, 1), (20, 21), (20, 7), (15, 22), (15, 23), (15, 29), (15, 12), (15, 3), (15, 5), (15, 2), (15, 1), (15, 21), (15, 7), (22, 23), (22, 12), (22, 3), (22, 5), (22, 2), (22, 32), (22, 1), (22, 21), (22, 7), (22, 16), (22, 6), (22, 31), (22, 17), (22, 18), (22, 19), (22, 4), (22, 28), (22, 26), (22, 9), (22, 24), (22, 8), (22, 25), (22, 27), (23, 29), (23, 12), (23, 3), (23, 5), (23, 2), (23, 32), (23, 1), (23, 21), (23, 7), (23, 16), (23, 6), (23, 31), (23, 17), (23, 18), (23, 19), (23, 4), (23, 28), (23, 26), (23, 9), (23, 24), (23, 8), (23, 25), (23, 27), (23, 30), (12, 3), (12, 5), (12, 2), (12, 1), (12, 21), (3, 5), (3, 2), (3, 32), (3, 1), (3, 21), (3, 7), (3, 16), (3, 6), (3, 31), (3, 17), (3, 18), (3, 19), (3, 4), (3, 28), (3, 26), (3, 9), (3, 24), (3, 8), (3, 25), (3, 27), (3, 30), (5, 2), (5, 32), (5, 1), (5, 21), (5, 7), (5, 16), (5, 6), (5, 31), (5, 17), (5, 18), (5, 19), (5, 4), (5, 28), (5, 26), (5, 9), (5, 24), (5, 8), (5, 25), (5, 27), (18, 19), (18, 4), (18, 28), (18, 26), (18, 9), (18, 24), (18, 8), (18, 25), (18, 27), (19, 4), (9, 24), (9, 8), (9, 25), (9, 27), (24, 8), (24, 25), (24, 27), (8, 25), (8, 27), (8, 30), (25, 27), (25, 30), (27, 30)
            ]);

            let mut state = State::new(&graph);
            state.partition_refinement();
            let p : Vec<_> = state.permutation().map(|u| u.index()).collect();
            println!("{:?}", p);
        }
    }
}


mod hp10 {
    use petgraph::adj::NodeIndex;

    struct Graph {

    }

    struct Partition {

    }

    struct Seq {

    }

    fn factorizing_permutation(graph: &mut Graph, v: NodeIndex, partition: &mut Partition, X: Seq)  {

    }
}


#[cfg(test)]
mod test {
    use std::time::Instant;
    use petgraph::graph::UnGraph;
    #[allow(unused_imports)]
    use crate::factorizing_permutation::kar19::symgraph_factorizing_permutation;
    use crate::factorizing_permutation::mpv99::State;

    fn run_both(graph: &UnGraph<(), ()>) -> (u128, u128) {
        let now = Instant::now();
        let mut state = State::new(graph);
        state.partition_refinement();
        let t0 = now.elapsed().as_nanos();

        let now = Instant::now();
        //let _p = symgraph_factorizing_permutation(graph);
        let t1 = now.elapsed().as_nanos();
        (t0, t1)
    }

    #[test]
    fn exact_050() {
        let paths = [
            //"../data/02-graphs/pace2023-exact_050",
            //"../data/02-graphs/pace2023-exact_100",
            "data/02-graphs/pace2023-exact_200",
            //"../data/02-graphs/pace2023-exact_200",
        ];
        for path in paths {
            let graph = common::io::read_metis(path).unwrap();
            let (t0, t1) = run_both(&graph);
            println!("{}", path);
            println!("{:?}", t0);
            println!("{:?}", t1);
        }
    }

    #[test]
    fn profile() {
        let graph = common::io::read_metis("data/02-graphs/pace2023-heuristic_100").unwrap();
        for _ in 0..8 {
            let now = Instant::now();
            let mut state = State::new(&graph);
            state.partition_refinement();
            let _p : Vec<_> = state.permutation().collect();
            let t0 = now.elapsed().as_nanos();
            println!("{}", t0);
        }
    }
}


