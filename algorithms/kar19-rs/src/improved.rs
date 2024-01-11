use std::cmp::Ordering;
use std::collections::HashSet;
use std::iter::zip;
use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
use common::modular_decomposition::MDNodeKind;
use tracing::instrument;
use crate::factorizing_permutation::seq;
use crate::factorizing_permutation::seq::Permutation;
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

    count(graph, &mut op, &mut cl, &mut lc, &mut uc, &p);

    remove_non_module_dummy_nodes(&mut op, &mut cl, &mut lc, &mut uc);

    shared::create_nodes(&mut op, &mut cl, &lc, &uc);

    shared::remove_singleton_dummy_nodes(&mut op, &mut cl);

    build_tree(graph, &op, &cl, &p)
}

#[instrument(skip_all)]
fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Permutation {
    seq::factorizing_permutation(graph)
}

#[instrument(skip_all)]
#[allow(dead_code)]
fn count_with_neighbors(graph: &UnGraph<(), ()>, op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize], p: &Permutation) {
    // NOTE: Instead of collecting neighbors into hashsets, one could
    // + calculate the two hashset of the neighboring nodes in the permutation
    // + use sorted vecs of the node positions of the neighboring nodes and get first different
    //   element from the front and the back.
    let neighbors: Vec<HashSet<NodeIndex>> = graph.node_indices()
        .map(|u| {
            graph.neighbors(u).collect()
        }).collect();

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
        //println!("_ {j} {:?} {:?} {:?}", min_i, max_i, neighbors[p[j].index()].symmetric_difference(&neighbors[p[j + 1].index()]).map(|x| p.position(*x)).collect::<Vec<_>>());
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


#[instrument(skip_all)]
fn count(graph: &UnGraph<(), ()>, op: &mut [usize], cl: &mut [usize], lc: &mut [usize], uc: &mut [usize], p: &Permutation) {
    let get_neighbor_positions = |idx: usize| -> Vec<usize> {
        let mut pos: Vec<_> = graph.neighbors(p[idx]).map(|v| p.position(v)).collect();
        pos.sort();
        pos
    };
    let mut pos_j0;
    let mut pos_j1 = get_neighbor_positions(0);

    fn merge_iter_next<'a>(mut a: impl Iterator<Item=&'a usize>, mut b: impl Iterator<Item=&'a usize>, f: impl Fn(usize, usize) -> Option<usize>) -> Option<usize> {
        loop {
            match (a.next(), b.next()) {
                (Some(i), Some(j)) => {
                    if let Some(k) = f(*i, *j) { break Some(k); }
                }
                (Some(i), None) => break Some(*i),
                (None, Some(j)) => break Some(*j),
                (None, None) => break None,
            }
        }
    }

    let first = |a: &[usize], b: &[usize]| -> Option<usize> {
        merge_iter_next(a.iter(), b.iter(), |i, j| {
            match i.cmp(&j) {
                Ordering::Less => Some(i),
                Ordering::Equal => None,
                Ordering::Greater => Some(j),
            }
        })
    };
    let last = |a: &[usize], b: &[usize]| -> Option<usize> {
        merge_iter_next(a.iter().rev(), b.iter().rev(), |i, j| {
            match i.cmp(&j) {
                Ordering::Less => Some(j),
                Ordering::Equal => None,
                Ordering::Greater => Some(i),
            }
        })
    };

    let n = p.len();
    for j in 0..n - 1 {
        pos_j0 = std::mem::replace(&mut pos_j1, get_neighbor_positions(j + 1));
        let mut min_i = first(&pos_j0, &pos_j1);
        let mut max_i = if min_i.is_some() { last(&pos_j0, &pos_j1) } else { None };
        min_i = min_i.filter(|i| *i < j);
        max_i = max_i.filter(|i| *i > j + 1);
        //println!("{:?} {:?}", pos_j0, pos_j1);
        //println!("# {j} {:?} {:?} {:?}", min_i, max_i, HashSet::<usize>::from_iter(pos_j0).symmetric_difference(&HashSet::<usize>::from_iter(pos_j1.iter().copied())).collect::<Vec<_>>());
        if let Some(i) = min_i {
            assert!(i < j);
            debug_assert_ne!(graph.find_edge(p[i], p[j]).is_some(), graph.find_edge(p[i], p[j + 1]).is_some());
            op[i] += 1;
            cl[j] += 1;
            lc[j] = i;
        }
        if let Some(i) = max_i {
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