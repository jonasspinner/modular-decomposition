use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use common::modular_decomposition::MDNodeKind;
use tracing::instrument;
use crate::{factorizing_permutation, shared, trace};
use crate::factorizing_permutation::kar19;

#[allow(dead_code)]
fn strong_module_tree(graph: &UnGraph<(), ()>) -> StrongModuleTree {
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

    let neighbors: Vec<HashSet<NodeIndex>> = graph.node_indices()
        .map(|u| {
            graph.neighbors(u).collect()
        }).collect();

    count(&mut op, &mut cl, &mut lc, &mut uc, &neighbors, &p);

    remove_non_module_dummy_nodes(&mut op, &mut cl, &mut lc, &mut uc);

    shared::create_nodes(&mut op, &mut cl, &lc, &uc);

    shared::remove_singleton_dummy_nodes(&mut op, &mut cl);

    let s = build(&op, &cl, &p);

    let mut t = classify_nodes(&s, &neighbors);

    delete_weak_modules(&mut t);

    t
}

#[instrument(skip_all)]
fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Vec<NodeIndex> {
    kar19::factorizing_permutation(graph)
}


#[instrument(skip_all)]
fn count(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize], neighbors: &[HashSet<NodeIndex>], p: &[NodeIndex]) {
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
fn remove_non_module_dummy_nodes(op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize]) {
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

#[instrument(skip_all)]
fn build(op: &[usize], cl: &[usize], p: &[NodeIndex]) -> Vec<NodeInProgress> {
    let mut s = vec![vec![]];
    for (j, x) in p.iter().enumerate() {
        for _ in 0..op[j] { s.push(vec![]); }
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


#[instrument(skip_all)]
fn classify_nodes(t: &Vec<NodeInProgress>, neighbors: &[HashSet<NodeIndex>]) -> StrongModuleTree {
    classify_nodes_rec(t, neighbors)
}
fn classify_nodes_rec(t: &Vec<NodeInProgress>, neighbors: &[HashSet<NodeIndex>]) -> StrongModuleTree {
    let n = t.len();

    let nodes: Vec<_> = t.iter().map(|n| n.first_leaf()).collect();

    // This way of counting the edges between children is very inefficient for large inner nodes.
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
        NodeInProgress::Vec(_, x) => { Node::Tree(classify_nodes_rec(x, neighbors)) }
    }).collect();
    StrongModuleTree { kind, nodes }
}

#[instrument(skip_all)]
fn delete_weak_modules(t: &mut StrongModuleTree) {
    delete_weak_modules_rec(t)
}

#[allow(unreachable_code)]
fn delete_weak_modules_rec(t: &mut StrongModuleTree) {
    let mut i = 0_usize.wrapping_sub(1);
    while i.wrapping_add(1) < t.nodes.len() {
        i = i.wrapping_add(1);
        let x = &mut t.nodes[i];
        let Node::Tree(x) = x else { continue; };
        delete_weak_modules_rec(x);
        if !(t.kind == x.kind && x.kind != Kind::Prime) { continue; }

        // TODO: investigate
        // panic!("This case does not seem to occur. Weird.");
        let nodes = std::mem::take(&mut x.nodes);
        let x_len = nodes.len();
        t.nodes.splice(i..i, nodes);
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

#[allow(dead_code)]
pub(crate) fn dp(p: &factorizing_permutation::seq::Permutation) -> Vec<usize> {
    p.iter().map(|u| u.index()).collect()
}

#[allow(dead_code)]
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