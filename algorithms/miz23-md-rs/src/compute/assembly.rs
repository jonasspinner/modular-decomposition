use std::collections::VecDeque;
use crate::compute::{MDComputeNode, Operation};
use crate::forest::{Forest, NodeIdx};
use crate::graph::VertexId;
use crate::set::FastSet;
use crate::trace;

fn determine_left_cocomp_fragments(tree: &Forest<MDComputeNode>, ps: &[NodeIdx], pivot_index: usize) -> Vec<bool> {
    let mut ret = vec![false; ps.len()];
    for i in 1..pivot_index {
        ret[i] = tree[ps[i]].data.comp_number >= 0 && tree[ps[i - 1]].data.comp_number == tree[ps[i]].data.comp_number;
    }
    ret
}

fn determine_right_comp_fragments(tree: &Forest<MDComputeNode>, ps: &[NodeIdx], pivot_index: usize) -> Vec<bool> {
    let mut ret = vec![false; ps.len()];
    for i in pivot_index + 1..ps.len() - 1 {
        ret[i] = tree[ps[i]].data.comp_number >= 0 && tree[ps[i]].data.comp_number == tree[ps[i + 1]].data.comp_number;
    }
    ret
}

fn determine_right_layer_neighbor(tree: &Forest<MDComputeNode>, alpha_list: &[Vec<NodeIdx>], ps: &[NodeIdx], pivot_index: usize) -> Vec<bool> {
    // TODO: The resulting bool vector seems to have at most one index with the value true
    let mut ret = vec![false; ps.len()];
    for i in pivot_index + 1..ps.len() {
        let current_tree = ps[i];
        let current_tree_num = tree[current_tree].data.tree_number;

        'l: for leaf in tree.leaves(current_tree) {
            for &a in &alpha_list[leaf.idx()] {
                if tree[a].data.tree_number > current_tree_num {
                    ret[i] = true;
                    break 'l;
                }
            }
        }
        if ret[i] { break; }
    }
    ret
}

fn compute_fact_perm_edges(tree: &mut Forest<MDComputeNode>, alpha_list: &[Vec<NodeIdx>], ps: &[NodeIdx], pivot_index: usize, vset: &mut FastSet, fp_neighbors: &mut Vec<Vec<VertexId>>) {
    trace!("compute_fact_perm_edges {}", alpha_list.len());
    let k = ps.len();

    fp_neighbors.iter_mut().for_each(|n| n.clear());

    let mut leaves = vec![vec![]; pivot_index];

    for i in 0..k {
        if i < pivot_index {
            leaves[i] = tree.leaves(ps[i]).collect();
            for &leaf in &leaves[i] {
                tree[leaf].data.comp_number = i as _;
            }
        } else {
            for leaf in tree.leaves(ps[i]).collect::<Vec<_>>() {
                tree[leaf].data.comp_number = i as _;
            }
        }
    }

    for i in 0..pivot_index {
        vset.clear();

        for &leaf in &leaves[i] {
            for &a in &alpha_list[leaf.idx()] {
                let j = tree[a].data.comp_number as usize;

                if !vset.get(j) {
                    fp_neighbors[i].push(VertexId::from(j));
                    vset.set(j);
                    if fp_neighbors.len() == k - pivot_index {
                        break; // 'outer_loop_exit; // Note: in C++ actually break inner loop, despite label name
                    }
                }
            }
        }
    }
}

fn compute_mu(_tree: &mut Forest<MDComputeNode>, ps: &[NodeIdx], pivot_index: usize, neighbors: &[Vec<VertexId>]) -> Vec<usize> {
    let mut mu: Vec<usize> = vec![0; ps.len()];

    for (i, m) in mu.iter_mut().enumerate() {
        *m = if i < pivot_index { pivot_index } else { 0 };
    }

    for i in 0..pivot_index {
        for &j in &neighbors[i] {
            let j = j.idx();
            if mu[j] == i { mu[j] = i + 1; }
            if j > mu[i] { mu[i] = j; }
        }
    }

    mu
}

struct DelineateState {
    lb: isize,
    rb: usize,
    left_last_in: usize,
    right_last_in: usize,
}

fn compose_series(lcocomp: &[bool], mu: &[usize], st: &mut DelineateState) -> bool {
    let mut ret = false;
    while 0 <= st.lb && mu[st.lb as usize] <= st.right_last_in && !lcocomp[st.lb as usize] {
        ret = true;

        st.left_last_in = st.lb as usize;
        st.lb -= 1;
    }
    ret
}

fn compose_parallel(rcomp: &[bool], rlayer: &[bool], mu: &[usize], st: &mut DelineateState) -> bool {
    let mut ret = false;
    while st.rb < rcomp.len() as _ && st.left_last_in <= mu[st.rb] && !rcomp[st.rb] && !rlayer[st.rb] {
        ret = true;
        st.right_last_in = st.rb;
        st.rb += 1;
    }
    ret
}

fn compose_prime(lcocomp: &[bool], rcomp: &[bool], rlayer: &[bool], mu: &[usize], st: &mut DelineateState) -> bool {
    let mut left_q = VecDeque::new();
    let mut right_q = VecDeque::new();

    loop {
        left_q.push_back(st.lb);
        st.left_last_in = st.lb as usize;
        st.lb -= 1;
        if !lcocomp[st.left_last_in] { break; }
    }

    while !left_q.is_empty() || !right_q.is_empty() {
        while let Some(current_left) = left_q.pop_front() {
            while st.right_last_in < mu[current_left as usize] {
                loop {
                    right_q.push_back(st.rb);
                    st.right_last_in = st.rb;
                    st.rb += 1;
                    if rlayer[st.right_last_in] { return true; }
                    if !rcomp[st.right_last_in] { break; }
                }
            }
        }
        while let Some(current_right) = right_q.pop_front() {
            while mu[current_right] < st.left_last_in {
                loop {
                    left_q.push_back(st.lb);
                    st.left_last_in = st.lb as usize;
                    st.lb -= 1;
                    if !lcocomp[st.left_last_in] { break; }
                }
            }
        }
    }
    false
}

fn delineate(pivot_index: usize, lcocomp: &[bool], rcomp: &[bool], rlayer: &[bool], mu: &[usize]) -> Vec<(usize, usize)> {
    let mut ret = vec![];

    let mut st = DelineateState { lb: pivot_index as isize - 1, rb: pivot_index + 1, left_last_in: pivot_index, right_last_in: pivot_index };
    let k = lcocomp.len();
    while 0 <= st.lb && st.rb < k {
        if !compose_series(lcocomp, mu, &mut st)
            && !compose_parallel(rcomp, rlayer, mu, &mut st)
            && compose_prime(lcocomp, rcomp, rlayer, mu, &mut st) {
            st.left_last_in = 0;
            st.right_last_in = k - 1;
            st.lb = st.left_last_in as isize - 1;
            st.rb = st.right_last_in + 1;
        }
        ret.push((st.left_last_in, st.right_last_in));
    }
    ret
}

fn assemble_tree(tree: &mut Forest<MDComputeNode>, ps: &[NodeIdx], pivot_index: usize, boundaries: &[(usize, usize)]) -> NodeIdx {
    let k = ps.len();
    let mut lb = pivot_index as isize - 1;
    let mut rb = pivot_index + 1;
    let mut last_module = ps[pivot_index];

    let mut i = 0;

    while 0 <= lb || rb < k {
        let (lbound, rbound) = boundaries.get(i).cloned().unwrap_or((0, k - 1));
        i += 1;

        let new_module = tree.create_node(MDComputeNode::new_operation_node(Operation::Prime));
        tree.move_to(last_module, new_module);

        let mut added_neighbors = false;
        let mut added_non_neighbors = false;

        while lb >= lbound as isize {
            added_neighbors = true;
            tree.move_to(ps[lb as usize], new_module);
            lb -= 1;
        }

        while rb <= rbound {
            added_non_neighbors = true;
            tree.move_to(ps[rb], new_module);
            rb += 1;
        }

        tree[new_module].data.op_type =
            match (added_neighbors, added_non_neighbors) {
                (true, true) => Operation::Prime,
                (true, _) => Operation::Series,
                _ => Operation::Parallel,
            };

        last_module = new_module;
    }

    last_module
}

fn remove_degenerate_duplicates(tree: &mut Forest<MDComputeNode>, index: NodeIdx) {
    let nodes: Vec<NodeIdx> = tree.get_bfs_nodes(index);

    for &it in nodes.iter().rev() {
        if it == index { break; }

        let c = &tree[it];
        let p = &tree[c.parent.unwrap()];
        if c.data.op_type == p.data.op_type && c.data.op_type != Operation::Prime {
            tree.replace_by_children(it);
            tree.remove(it);
        }
    }
}

pub(crate) fn assemble(tree: &mut Forest<MDComputeNode>, alpha_list: &[Vec<NodeIdx>], prob: NodeIdx, fp_neighbors: &mut Vec<Vec<VertexId>>, vset: &mut FastSet) {
    assert!(!tree[prob].is_leaf());

    let mut ps = vec![];

    let current_pivot = tree[prob].data.vertex;
    let mut pivot_index: i32 = -1;

    for p in tree.children(prob) {
        if p.idx() == current_pivot.idx() { pivot_index = ps.len() as _ }
        ps.push(p);
    }

    assert!(pivot_index >= 0);
    let pivot_index = pivot_index as usize;

    let lcocomp = determine_left_cocomp_fragments(tree, &ps, pivot_index);
    let rcomp = determine_right_comp_fragments(tree, &ps, pivot_index);
    let rlayer = determine_right_layer_neighbor(tree, alpha_list, &ps, pivot_index);

    compute_fact_perm_edges(tree, alpha_list, &ps, pivot_index, vset, fp_neighbors);

    let mu = compute_mu(tree, &ps, pivot_index, fp_neighbors);

    let boundaries = delineate(pivot_index, &lcocomp, &rcomp, &rlayer, &mu);

    let root = assemble_tree(tree, &ps, pivot_index, &boundaries);

    remove_degenerate_duplicates(tree, root);

    tree.replace_children(prob, root);
}