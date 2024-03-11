use crate::trace;
use common::modular_decomposition::MDNodeKind;
use petgraph::data::{Element, FromElements};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use tracing::instrument;

#[allow(dead_code)]
pub fn modular_decomposition(graph: &UnGraph<(), ()>) -> DiGraph<MDNodeKind, ()> {
    let n = graph.node_count();
    if n == 0 {
        return DiGraph::new();
    }
    if n == 1 {
        return DiGraph::from_elements([Element::Node { weight: MDNodeKind::Vertex(0) }]);
    }

    let p = factorizing_permutation(graph);

    let n = p.len();
    let mut op = vec![0; n];
    op[0] = 1;
    let mut cl = vec![0; n];
    cl[n - 1] = 1;
    let mut lc: Vec<_> = (0..n - 1).collect();
    let mut uc: Vec<_> = (1..n).collect();

    trace!("{:?}", d1(&p));
    trace!("op: {:?}", op);
    trace!("cl: {:?}", cl);

    let neighbors: Vec<HashSet<NodeIndex>> = graph.node_indices().map(|u| graph.neighbors(u).collect()).collect();

    build_parenthesizing(&mut op, &mut cl, &mut lc, &mut uc, &neighbors, &p);

    remove_non_module_dummy_nodes(&mut op, &mut cl, &mut lc, &mut uc);

    create_consecutive_twin_nodes(&mut op, &mut cl, &lc, &uc);

    remove_singleton_dummy_nodes(&mut op, &mut cl);

    let s = build_tree(&op, &cl, &p);

    let t = classify_nodes(&s, &neighbors);

    convert_to_digraph(t)
}

#[instrument(skip_all)]
fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Vec<NodeIndex> {
    factorizing_permutation::factorizing_permutation(graph)
}

#[instrument(skip_all)]
fn build_parenthesizing(
    op: &mut [usize],
    cl: &mut [usize],
    lc: &mut [usize],
    uc: &mut [usize],
    neighbors: &[HashSet<NodeIndex>],
    p: &[NodeIndex],
) {
    let n = p.len();
    for j in 0..n - 1 {
        for i in 0..j {
            let adj_p_i = &neighbors[p[i].index()];
            if adj_p_i.contains(&p[j]) == adj_p_i.contains(&p[j + 1]) {
                continue;
            }
            op[i] += 1;
            cl[j] += 1;
            lc[j] = i;
            break;
        }
        let j = j + 1;
        for i in (j + 1..n).rev() {
            let adj_p_i = &neighbors[p[i].index()];
            if adj_p_i.contains(&p[j - 1]) == adj_p_i.contains(&p[j]) {
                continue;
            }
            op[j] += 1;
            cl[i] += 1;
            uc[j - 1] = i;
            break;
        }
    }
}

#[instrument(skip_all)]
fn create_consecutive_twin_nodes(op: &mut [usize], cl: &mut [usize], lc: &[usize], uc: &[usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);
    let mut l = 0;
    for k in 0..n {
        s.push((k, l));
        l = k;
        s.extend(std::iter::repeat((k, k)).take(op[k]));
        for c in (0..cl[k] + 1).rev() {
            let (j, i) = s.pop().unwrap();

            l = i; // continue twin chain by default
            if i >= j {
                continue;
            }
            if i <= lc[j - 1] && lc[j - 1] < uc[j - 1] && uc[j - 1] <= k {
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
fn remove_singleton_dummy_nodes(op: &mut [usize], cl: &mut [usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);

    for j in 0..n {
        s.extend(std::iter::repeat(j).take(op[j]));
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
fn remove_non_module_dummy_nodes(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);
    for j in 0..n {
        for _ in 0..op[j] {
            s.push(j);
        }
        for _ in 0..cl[j] {
            let i = s.pop().unwrap();
            if i < j {
                let l = (i..j).map(|k| lc[k]).min().unwrap();
                let u = (i..j).map(|k| uc[k]).max().unwrap();
                if i <= l && u <= j {
                    continue;
                }
            }
            op[i] -= 1;
            cl[j] -= 1;
        }
    }
}

#[instrument(skip_all)]
fn build_tree(op: &[usize], cl: &[usize], p: &[NodeIndex]) -> Vec<NodeInProgress> {
    let mut s = vec![vec![]];
    for (j, x) in p.iter().enumerate() {
        for _ in 0..op[j] {
            s.push(vec![]);
        }
        s.last_mut().unwrap().push(NodeInProgress::Node(*x));
        for _ in 0..cl[j] {
            let n = s.pop().unwrap();
            s.last_mut().unwrap().push(NodeInProgress::Vec(n[0].first_leaf(), n));
        }
    }
    s.pop().unwrap()
}

#[derive(PartialEq)]
enum Kind {
    Series,
    Parallel,
    Prime,
}

impl From<Kind> for MDNodeKind {
    fn from(value: Kind) -> Self {
        match value {
            Kind::Series => Self::Series,
            Kind::Parallel => Self::Parallel,
            Kind::Prime => Self::Prime,
        }
    }
}

struct StrongModuleTree {
    kind: Kind,
    nodes: Vec<Node>,
}

impl Debug for StrongModuleTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let parens: [char; 2] = match self.kind {
            Kind::Series => ['(', ')'],
            Kind::Parallel => ['|', '|'],
            Kind::Prime => ['{', '}'],
        };
        write!(f, "{}", parens[0])?;
        for (i, x) in self.nodes.iter().enumerate() {
            write!(f, "{:?}", x)?;
            if i + 1 < self.nodes.len() {
                write!(f, " ")?;
            }
        }
        write!(f, "{}", parens[1])
    }
}

enum Node {
    Node(NodeIndex),
    Tree(StrongModuleTree),
}

impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Node::Node(x) => {
                write!(f, "{:?}", x.index())
            }
            Node::Tree(t) => {
                write!(f, "{:?}", t)
            }
        }
    }
}

enum NodeInProgress {
    Node(NodeIndex),
    Vec(NodeIndex, Vec<NodeInProgress>),
}

impl Debug for NodeInProgress {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeInProgress::Node(x) => {
                write!(f, "{}", x.index())
            }
            NodeInProgress::Vec(_first, v) => f.debug_list().entries(v).finish(),
        }
    }
}

impl NodeInProgress {
    fn first_leaf(&self) -> NodeIndex {
        match self {
            NodeInProgress::Node(x) => *x,
            NodeInProgress::Vec(x, _) => *x,
        }
    }
}

#[instrument(skip_all)]
fn classify_nodes(t: &Vec<NodeInProgress>, neighbors: &[HashSet<NodeIndex>]) -> StrongModuleTree {
    classify_nodes_rec(t, neighbors)
}

fn classify_nodes_rec(t: &Vec<NodeInProgress>, neighbors: &[HashSet<NodeIndex>]) -> StrongModuleTree {
    let n = t.len();

    let nodes: Vec<_> = t.iter().map(|n| n.first_leaf()).collect();

    // This way of counting the edges between children is very inefficient for large
    // inner nodes.
    let mut m = 0;
    for (i, x) in nodes.iter().enumerate() {
        for y in &nodes[i + 1..] {
            m += neighbors[x.index()].contains(y) as usize;
        }
    }

    assert!(2 * m <= n * (n - 1));
    let kind = if m == 0 {
        Kind::Parallel
    } else if 2 * m == n * (n - 1) {
        Kind::Series
    } else {
        Kind::Prime
    };
    let nodes = t
        .iter()
        .map(|x| match x {
            NodeInProgress::Node(x) => Node::Node(*x),
            NodeInProgress::Vec(_, x) => Node::Tree(classify_nodes_rec(x, neighbors)),
        })
        .collect();
    StrongModuleTree { kind, nodes }
}

#[allow(dead_code)]
pub(crate) fn d2<'a, P>(p: P) -> Vec<Vec<usize>>
where
    P: IntoIterator<Item = &'a Vec<NodeIndex>>,
{
    p.into_iter().map(|q| d1(q)).collect()
}

pub(crate) fn d1(p: &[NodeIndex]) -> Vec<usize> {
    p.iter().map(|u| u.index()).collect()
}

fn convert_to_digraph(tree: StrongModuleTree) -> DiGraph<MDNodeKind, ()> {
    let mut md = DiGraph::new();

    let mut stack = vec![(None, Node::Tree(tree))];
    while let Some((parent, node)) = stack.pop() {
        let u = match node {
            Node::Node(vertex) => md.add_node(MDNodeKind::Vertex(vertex.index())),
            Node::Tree(tree) => {
                let u = md.add_node(tree.kind.into());
                for child in tree.nodes {
                    stack.push((Some(u), child));
                }
                u
            }
        };
        if let Some(parent) = parent {
            md.add_edge(parent, u, ());
        }
    }
    md
}

#[allow(non_snake_case)]
mod factorizing_permutation {
    use crate::base::factorizing_permutation::util::{smaller_larger, splice};
    #[allow(unused_imports)]
    use crate::base::{d1, d2};
    #[allow(unused_imports)]
    use crate::trace;
    use petgraph::graph::{NodeIndex, UnGraph};
    use std::collections::hash_map::RandomState;
    use std::collections::{HashMap, HashSet, VecDeque};

    #[allow(dead_code)]
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

    fn refine(
        P: &mut Vec<Vec<NodeIndex>>,
        S: HashSet<NodeIndex>,
        x: NodeIndex,
        center: &NodeIndex,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
    ) {
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
            if X_a.is_empty() || X.is_empty() {
                continue;
            }
            trace!("refine:P:0: {:?}", d2(&*P));
            P[i] = X.clone();
            P.insert(i + between as usize, X_a.clone());
            trace!("refine:P:1: {:?}", d2(&*P));
            add_pivot(X, X_a, pivots, modules);
            i += 1;
        }
    }

    fn add_pivot(
        X: Vec<NodeIndex>,
        X_a: Vec<NodeIndex>,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
    ) {
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

    fn partition_refinement(
        P: &mut Vec<Vec<NodeIndex>>,
        center: &mut NodeIndex,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
        first_pivot: &mut HashMap<Vec<NodeIndex>, NodeIndex>,
        graph: &UnGraph<(), ()>,
    ) {
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

    fn init_partition(
        P: &mut Vec<Vec<NodeIndex>>,
        center: &mut NodeIndex,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
        first_pivot: &mut HashMap<Vec<NodeIndex>, NodeIndex>,
        graph: &UnGraph<(), ()>,
    ) -> bool {
        if P.iter().all(|p| p.len() <= 1) {
            return false;
        }
        if let Some(X) = modules.pop_front() {
            let x = X[0];
            pivots.push(vec![x]);
            first_pivot.insert(X, x);
        } else {
            for (i, X) in P.iter().enumerate() {
                if X.len() <= 1 {
                    continue;
                }
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
            if a.len() <= b.len() {
                (a, b)
            } else {
                (b, a)
            }
        }

        pub(crate) fn splice<T>(vec: &mut Vec<Vec<T>>, i: usize, first: Vec<T>, second: T, third: Vec<T>) {
            match (first.is_empty(), third.is_empty()) {
                (true, true) => vec[i] = vec![second],
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
