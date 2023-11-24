use std::collections::{HashMap, HashSet, VecDeque};
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Formatter};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use common::modular_decomposition::MDNodeKind;


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

#[allow(non_snake_case)]
fn symgraph_factorizing_permutation(graph: &UnGraph<(), ()>, V: Vec<NodeIndex>) -> Vec<NodeIndex> {
    let mut P = vec![V];
    let mut center = NodeIndex::new(0);
    let mut pivots = vec![];
    let mut modules = VecDeque::new();
    let mut first_pivot = HashMap::<Vec<NodeIndex>, NodeIndex>::new();

    partition_refinement(&mut P, &mut center, &mut pivots, &mut modules, &mut first_pivot, &graph);
    P.iter().map(|part| part[0]).collect()
}

#[allow(non_snake_case)]
fn smaller_larger(A: Vec<NodeIndex>, B: Vec<NodeIndex>) -> (Vec<NodeIndex>, Vec<NodeIndex>) {
    if A.len() <= B.len() { (A, B) } else { (B, A) }
}

#[allow(non_snake_case)]
fn refine(P: &mut Vec<Vec<NodeIndex>>,
          S: HashSet<NodeIndex>,
          x: NodeIndex,
          center: &NodeIndex,
          pivots: &mut Vec<Vec<NodeIndex>>,
          modules: &mut VecDeque<Vec<NodeIndex>>) {
    traceln!("refine: {} {:?}", x.index(), S.iter().map(|u| u.index()).collect::<Vec<_>>());
    let mut i = 0_usize.wrapping_sub(1);
    let mut between = false;
    while i.wrapping_add(1) < P.len() {
        i = i.wrapping_add(1);
        let X = &P[i];
        //traceln!("refine:X: {} {:?}", i, d1(&X));
        if X.contains(&center) || X.contains(&x) {
            between = !between;
            continue;
        }
        let (X_a, X): (Vec<_>, Vec<_>) = X.iter().partition(|&y| S.contains(y));
        if X_a.is_empty() || X.is_empty() { continue; }
        traceln!("refine:P:0: {:?}", d2(&*P));
        P[i] = X.clone();
        P.insert(i + between as usize, X_a.clone());
        traceln!("refine:P:1: {:?}", d2(&*P));
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
    traceln!("add_pivot: {:?} {:?}", d1(&X), d1(&X_a));
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
        traceln!("P: {:?}", d2(&*P));
        traceln!("pivots: {:?}", d2(&*pivots));
        traceln!("modules: {:?}", d2(&*modules));
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
            let (A, mut N): (Vec<_>, Vec<_>) = X.into_iter().partition(|&y| *y != x && adj.contains(y));
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


#[allow(non_snake_case)]
fn strong_module_tree(graph: &UnGraph<(), ()>) -> StrongModuleTree {
    let V = graph.node_indices().collect();
    let p = symgraph_factorizing_permutation(graph, V);

    let n = p.len();
    let mut op = vec![0; n];
    op[0] = 1;
    let mut cl = vec![0; n];
    cl[n - 1] = 1;
    let mut lc: Vec<_> = (0..n - 1).collect();
    let mut uc: Vec<_> = (1..n).collect();

    traceln!("{:?}", d1(&p));
    traceln!("op: {:?}", op);
    traceln!("cl: {:?}", cl);

    traceln!("count");
    for j in 0..n - 1 {
        trace!("j={j} :");
        for i in 0..j {
            trace!(" {i}");
            if graph.find_edge(p[i], p[j]).is_some() == graph.find_edge(p[i], p[j + 1]).is_some() &&
                graph.find_edge(p[j], p[i]).is_some() == graph.find_edge(p[j + 1], p[i]).is_some() {
                continue;
            }
            op[i] += 1;
            cl[j] += 1;
            lc[j] = i;
            break;
        }
        traceln!();
        let j = j + 1;
        for i in (j + 1..n).rev() {
            if graph.find_edge(p[i], p[j - 1]).is_some() == graph.find_edge(p[i], p[j]).is_some() &&
                graph.find_edge(p[j - 1], p[i]).is_some() == graph.find_edge(p[j], p[i]).is_some() {
                continue;
            }
            op[j] += 1;
            cl[i] += 1;
            uc[j - 1] = i;
            break;
        }
    }

    traceln!("op: {:?}", op);
    traceln!("cl: {:?}", cl);

    {
        let mut s = vec![];
        traceln!("remove_non_module_dummy_nodes");
        for j in 0..n {
            trace!("j={j} :");
            for _ in 0..op[j] { s.push(j); }
            for _ in 0..cl[j] {
                let i = s.pop().unwrap();
                trace!(" {i}");
                if i < j {
                    let l = (i..j).map(|k| lc[k]).min().unwrap();
                    let u = (i..j).map(|k| uc[k]).max().unwrap();
                    if i <= l && u <= j { continue; }
                }
                op[i] -= 1;
                cl[j] -= 1;
            }
            traceln!();
        }
    }

    traceln!("op: {:?}", op);
    traceln!("cl: {:?}", cl);

    {
        traceln!("create_nodes");
        let mut s = vec![];
        let mut t = vec![];
        let mut l = 0;
        for k in 0..n {
            traceln!("k={k}");
            for _ in 0..op[k] + 1 {
                s.push(k);
                t.push(l);
                l = k;
            }
            for c in (0..cl[k] + 1).rev() {
                let i = t.pop().unwrap();
                let j = s.pop().unwrap();
                traceln!("  {c} {i} {j}");

                l = i;
                if i >= j { continue; }
                if i <= lc[j - 1] && lc[j - 1] < uc[j - 1] && uc[j - 1] <= k {
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

    traceln!("op: {:?}", op);
    traceln!("cl: {:?}", cl);

    {
        let mut s = vec![];
        traceln!("remove_singleton_dummy_nodes");
        for j in 0..n {
            trace!("j={j} :");
            for _ in 0..op[j] { s.push(j); }
            trace!(" {:?} :", s);
            let mut i_ = usize::MAX;
            for _ in 0..cl[j]
            {
                let i = s.pop().unwrap();
                trace!(" {i}");
                if i == i_ {
                    op[i] -= 1;
                    cl[j] -= 1;
                }
                i_ = i;
            }
            traceln!();
        }
    }
    op[0] -= 1;
    cl[n - 1] -= 1;


    traceln!("op: {:?}", op);
    traceln!("cl: {:?}", cl);
    traceln!("p: {:?}", d1(&p));

    let mut s = vec![vec![]];
    for (j, x) in p.iter().enumerate() {
        for _ in 0..op[j] { s.push(vec![]); }
        s.last_mut().unwrap().push(NodeInProgress::Node(*x));
        for _ in 0..cl[j] {
            let n = s.pop().unwrap();
            s.last_mut().unwrap().push(NodeInProgress::Vec(n));
        }
    }

    traceln!("{:?}", s);

    let s = s.pop().unwrap();

    let mut t = classify_nodes(&s, graph);
    delete_weak_modules(&mut t);
    traceln!("{:?}", t);
    t
}

#[derive(PartialEq)]
enum Kind {
    Complete,
    Parallel,
    #[allow(dead_code)]
    Linear,
    Prime,
}

struct StrongModuleTree {
    kind: Kind,
    edge: Option<(bool, bool)>,
    nodes: Vec<Node>,
}

impl Debug for StrongModuleTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let parens: [char; 2] = match self.kind {
            Kind::Complete => { ['(', ')'] }
            Kind::Parallel => { ['|', '|'] }
            Kind::Linear => { ['[', ']'] }
            Kind::Prime => { ['{', '}'] }
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
            Node::Node(x) => { write!(f, "{:?}", x.index()) }
            Node::Tree(t) => { write!(f, "{:?}", t) }
        }
    }
}

enum NodeInProgress {
    Node(NodeIndex),
    Vec(Vec<NodeInProgress>),
}

impl Debug for NodeInProgress {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeInProgress::Node(x) => { write!(f, "{}", x.index()) }
            NodeInProgress::Vec(v) => {
                let mut l = f.debug_list();
                for x in v {
                    l.entry(x);
                }
                l.finish()
            }
        }
    }
}

fn first_leaf(node: &NodeInProgress) -> NodeIndex {
    match node {
        NodeInProgress::Node(x) => { *x }
        NodeInProgress::Vec(x) => { first_leaf(&x[0]) }
    }
}

fn classify_nodes(t: &Vec<NodeInProgress>, graph: &UnGraph<(), ()>) -> StrongModuleTree {
    let n = t.len();

    let nodes: Vec<_> = t.iter().map(first_leaf).collect();

    let mut m = 0;
    for (i, x) in nodes.iter().enumerate() {
        for y in &nodes[i + 1..] {
            m += graph.find_edge(*x, *y).is_some() as usize;
        }
    }

    let edge = Some((false, false));
    let kind = if m == 0 { Kind::Parallel } else if 2 * m == n * (n - 1) { Kind::Complete } else { Kind::Prime };
    let nodes = t.iter().map(|x| match x {
        NodeInProgress::Node(x) => { Node::Node(*x) }
        NodeInProgress::Vec(x) => { Node::Tree(classify_nodes(x, graph)) }
    }).collect();
    StrongModuleTree { kind, edge, nodes }
}

fn _classify_nodes(t: &Vec<NodeInProgress>, graph: &UnGraph<(), ()>) -> StrongModuleTree {
    traceln!("classify_nodes");
    let n = t.len();
    let mut counts = vec![0; n];
    let x = first_leaf(&t[0]);
    let y = first_leaf(&t[1]);
    let mut edge = (graph.find_edge(y, x).is_some(), graph.find_edge(x, y).is_some());

    let mut a = false;
    let mut b = false;
    'outer: for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            let x = first_leaf(&t[i]);
            let y = first_leaf(&t[j]);
            (a, b) = (graph.find_edge(y, x).is_some(), graph.find_edge(x, y).is_some());
            traceln!("{i} {j} {} {} {} {}", x.index(), y.index(), a as i32, b as i32);
            if edge == (a, b) {
                counts[i] += 1;
            } else if edge == (b, a) {
                counts[j] += 1;
            } else {
                break 'outer;
            }
        }
    }
    traceln!("{:?}", counts);
    counts.sort();
    let kind = if a == b && counts.iter().all(|c| *c == n - 1) {
        Kind::Complete
    } else if counts.iter().all(|c| *c == 0) {
        Kind::Parallel
    } else { Kind::Prime };
    if edge.0 & !edge.1 { edge = (edge.1, edge.0); }
    let edge = if kind == Kind::Prime { None } else { Some(edge) };
    let nodes = t.iter().map(|x| match x {
        NodeInProgress::Node(x) => { Node::Node(*x) }
        NodeInProgress::Vec(x) => { Node::Tree(_classify_nodes(x, graph)) }
    }).collect();
    StrongModuleTree { kind, edge, nodes }
}

fn delete_weak_modules(t: &mut StrongModuleTree) {
    let mut i = 0_usize.wrapping_sub(1);
    while i.wrapping_add(1) < t.nodes.len() {
        i = i.wrapping_add(1);
        let x = &mut t.nodes[i];
        let Node::Tree(x) = x else { continue; };
        delete_weak_modules(x);
        if !(t.kind == x.kind && x.kind != Kind::Prime && t.edge == x.edge) { continue; }
        let mut nodes = std::mem::take(&mut x.nodes);
        let x_len = nodes.len();
        t.nodes.append(&mut nodes);
        i += x_len;
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

pub fn modular_decomposition(graph: &UnGraph<(), ()>) -> DiGraph<MDNodeKind, ()> {
    let tree = strong_module_tree(graph);

    let mut md = DiGraph::new();

    let mut stack = vec![(None, Node::Tree(tree))];
    while let Some((parent, node)) = stack.pop() {
        let u = match node {
            Node::Node(vertex) => {
                md.add_node(MDNodeKind::Vertex(vertex.index()))
            }
            Node::Tree(tree) => {
                let kind = match tree.kind {
                    Kind::Complete => { MDNodeKind::Series }
                    Kind::Parallel => { MDNodeKind::Parallel }
                    Kind::Linear => { panic!() }
                    Kind::Prime => { MDNodeKind::Prime }
                };
                let u = md.add_node(kind);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(non_snake_case)]
    #[test]
    fn it_works() {
        let graph = UnGraph::from_edges([(0, 1), (1, 2), (3, 4), (4, 5)]);
        let V = graph.node_indices().collect();
        let _p = symgraph_factorizing_permutation(&graph, V);
    }

    #[allow(non_snake_case)]
    #[test]
    fn ted08_test0() {
        let graph = common::instances::ted08_test0();
        let V = graph.node_indices().collect();
        let p = symgraph_factorizing_permutation(&graph, V);
        println!("{:?}", d1(&p));
    }

    #[test]
    fn ted08_test0_md() {
        let graph = common::instances::ted08_test0();
        let _md = strong_module_tree(&graph);
    }

    #[test]
    fn pace2023_exact053_md() {
        let graph = common::io::read_pace2023("../../hippodrome/instances/pace2023/exact_053.gr").unwrap();
        let _md = strong_module_tree(&graph);
    }
}
