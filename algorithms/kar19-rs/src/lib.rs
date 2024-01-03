mod factorizing_permutation;

use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use common::modular_decomposition::MDNodeKind;
use tracing::{span, Level};
use crate::factorizing_permutation::seq::Permutation;


#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        tracing::trace!($($x),*)
    }
}

#[allow(non_snake_case)]
fn strong_module_tree(graph: &UnGraph<(), ()>) -> StrongModuleTree {
    let p = {
        let _span_ = span!(Level::INFO, "factorizing_permutation").entered();

        factorizing_permutation::seq::factorizing_permutation(graph)
    };

    let neighbors: Vec<HashSet<NodeIndex>> = graph.node_indices()
        .map(|u| {
            graph.neighbors(u).collect()
        }).collect();

    let n = p.len();
    let mut op = vec![0; n];
    op[0] = 1;
    let mut cl = vec![0; n];
    cl[n - 1] = 1;
    let mut lc: Vec<_> = (0..n - 1).collect();
    let mut uc: Vec<_> = (1..n).collect();

    trace!("{:?}", dp(&p));
    trace!("op: {:?}", op);
    trace!("cl: {:?}", cl);

    {
        let _span_ = span!(Level::INFO, "count").entered();
        count_v2(&mut op, &mut cl, &mut lc, &mut uc, &neighbors, &p);
    }

    trace!("op: {:?}", op);
    trace!("cl: {:?}", cl);

    {
        let _span_ = span!(Level::INFO, "remove_non_module_dummy_nodes").entered();
        remove_non_module_dummy_nodes_v2(&mut op, &mut cl, &mut lc, &mut uc);
    }

    trace!("op: {:?}", op);
    trace!("cl: {:?}", cl);

    {
        let _span_ = span!(Level::INFO, "create_nodes").entered();

        let mut s = Vec::with_capacity(n);
        let mut t = Vec::with_capacity(n);
        let mut l = 0;
        for k in 0..n {
            for _ in 0..op[k] + 1 {
                s.push(k);
                t.push(l);
                l = k;
            }
            for c in (0..cl[k] + 1).rev() {
                let i = t.pop().unwrap();
                let j = s.pop().unwrap();

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

    trace!("op: {:?}", op);
    trace!("cl: {:?}", cl);

    {
        let _span_ = span!(Level::INFO, "remove_singleton_dummy_nodes").entered();
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
    }
    op[0] -= 1;
    cl[n - 1] -= 1;


    trace!("op: {:?}", op);
    trace!("cl: {:?}", cl);
    trace!("p: {:?}", dp(&p));

    let mut s = vec![vec![]];
    {
        let _span_ = span!(Level::INFO, "build").entered();

        for (j, x) in p.iter().enumerate() {
            for _ in 0..op[j] { s.push(vec![]); }
            s.last_mut().unwrap().push(NodeInProgress::Node(x));
            for _ in 0..cl[j] {
                let n = s.pop().unwrap();
                s.last_mut().unwrap().push(NodeInProgress::Vec(n[0].first_leaf(), n));
            }
        }
    }

    trace!("{:?}", s);

    let s = s.pop().unwrap();


    let mut t = {
        let _span_ = span!(Level::INFO, "classify_nodes").entered();
        classify_nodes(&s, &neighbors)
    };

    {
        let _span_ = span!(Level::INFO, "delete_weak_modules").entered();
        delete_weak_modules(&mut t);
    }

    trace!("{:?}", t);
    t
}


#[allow(dead_code)]
fn count_v1(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize], neighbors: &[HashSet<NodeIndex>], p: &Permutation) {
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

fn count_v2(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize], neighbors: &[HashSet<NodeIndex>], p: &Permutation) {
    let n = p.len();
    for j in 0..n - 1 {
        let (mut min_i, mut max_i) = (None, None);
        for &x in neighbors[p[j].index()].symmetric_difference(&neighbors[p[j + 1].index()]) {
            let i = p.position(x);
            if i < j && (min_i.map_or(true, |k| k > i)) {
                min_i = Some(i);
            }
            if i > j + 1 && (max_i.map_or(true, |k| k < i)) {
                max_i = Some(i);
            }
        }
        if let Some(i) = min_i {
            assert!(i < j);
            debug_assert_ne!(neighbors[p[i].index()].contains(&p[j]), neighbors[p[i].index()].contains(&p[j + 1]));
            op[i] += 1;
            cl[j] += 1;
            lc[j] = i;
        }
        if let Some(i) = max_i {
            assert!(i > j + 1 && i < n);
            debug_assert_ne!(neighbors[p[i].index()].contains(&p[j]), neighbors[p[i].index()].contains(&p[j + 1]));
            op[j + 1] += 1;
            cl[i] += 1;
            uc[j] = i;
        }
    }
}

#[allow(dead_code)]
fn remove_non_module_dummy_nodes_v1(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);
    for j in 0..n {
        for _ in 0..op[j] { s.push(j); }
        for _ in 0..cl[j] {
            let i = s.pop().unwrap();
            if i < j {
                let l = (i..j).map(|k| lc[k]).min().unwrap();
                let u = (i..j).map(|k| uc[k]).max().unwrap();
                if i <= l && u <= j { continue; }
            }
            op[i] -= 1;
            cl[j] -= 1;
        }
    }
}

fn remove_non_module_dummy_nodes_v2(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize]) {
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

#[derive(PartialEq)]
enum Kind {
    Series,
    Parallel,
    Prime,
}

struct StrongModuleTree {
    kind: Kind,
    nodes: Vec<Node>,
}

impl Debug for StrongModuleTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let parens: [char; 2] = match self.kind {
            Kind::Series => { ['(', ')'] }
            Kind::Parallel => { ['|', '|'] }
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
    Vec(NodeIndex, Vec<NodeInProgress>),
}

impl Debug for NodeInProgress {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeInProgress::Node(x) => { write!(f, "{}", x.index()) }
            NodeInProgress::Vec(_first, v) => {
                f.debug_list().entries(v).finish()
            }
        }
    }
}

impl NodeInProgress {
    fn first_leaf(&self) -> NodeIndex {
        match self {
            NodeInProgress::Node(x) => { *x }
            NodeInProgress::Vec(x, _) => { *x }
        }
    }
}

fn classify_nodes(t: &Vec<NodeInProgress>, neighbors: &[HashSet<NodeIndex>]) -> StrongModuleTree {
    let n = t.len();

    let nodes: Vec<_> = t.iter().map(|n| n.first_leaf()).collect();

    let mut m = 0;
    for (i, x) in nodes.iter().enumerate() {
        for y in &nodes[i + 1..] {
            m += neighbors[x.index()].contains(y) as usize;
        }
    }

    assert!(2 * m <= n * (n - 1));
    let kind = if m == 0 { Kind::Parallel } else if 2 * m == n * (n - 1) { Kind::Series } else { Kind::Prime };
    let nodes = t.iter().map(|x| match x {
        NodeInProgress::Node(x) => { Node::Node(*x) }
        NodeInProgress::Vec(_, x) => { Node::Tree(classify_nodes(x, neighbors)) }
    }).collect();
    StrongModuleTree { kind, nodes }
}

fn delete_weak_modules(t: &mut StrongModuleTree) {
    let mut i = 0_usize.wrapping_sub(1);
    while i.wrapping_add(1) < t.nodes.len() {
        i = i.wrapping_add(1);
        let x = &mut t.nodes[i];
        let Node::Tree(x) = x else { continue; };
        delete_weak_modules(x);
        if !(t.kind == x.kind && x.kind != Kind::Prime) { continue; }
        let mut nodes = std::mem::take(&mut x.nodes);
        let x_len = nodes.len();
        t.nodes.append(&mut nodes);
        i += x_len;
    }
}

#[allow(dead_code)]
pub(crate) fn d2<'a, P>(p: P) -> Vec<Vec<usize>>
    where P: IntoIterator<Item=&'a Vec<NodeIndex>>,
{
    p.into_iter().map(|q| d1(q)).collect()
}

pub(crate) fn d1(p: &[NodeIndex]) -> Vec<usize> {
    p.iter().map(|u| u.index()).collect()
}

pub(crate) fn dp(p: &factorizing_permutation::seq::Permutation) -> Vec<usize> {
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
                    Kind::Series => { MDNodeKind::Series }
                    Kind::Parallel => { MDNodeKind::Parallel }
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
    use crate::factorizing_permutation::kar19::factorizing_permutation;
    use super::*;

    #[allow(non_snake_case)]
    #[test]
    fn it_works() {
        let graph = UnGraph::from_edges([(0, 1), (1, 2), (3, 4), (4, 5)]);
        let p = factorizing_permutation(&graph);
        println!("{:?}", d1(&p));
    }

    #[allow(non_snake_case)]
    #[test]
    fn ted08_test0() {
        let graph = common::instances::ted08_test0();
        let p = factorizing_permutation(&graph);
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

    #[test]
    fn graph_10() {
        let graph = UnGraph::<(), ()>::from_edges([
            (0, 2), (0, 3), (0, 4),
            (1, 2),
            (2, 3), (2, 4), (2, 6),
            (3, 4), (3, 5), (3, 6),
            (4, 5), (4, 6),
            (5, 6),
            (6, 7), (6, 8), (6, 9),
            (7, 8), (7, 9),
            (8, 9)
        ]);
        // println!("{}", graph.edge_count());
        let _md = strong_module_tree(&graph);
    }
}
