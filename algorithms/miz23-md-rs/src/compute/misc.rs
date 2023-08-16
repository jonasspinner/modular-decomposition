use std::ops::Index;
use crate::compute::Operation;
use crate::compute::MDComputeNode;
use crate::forest::{Forest, NodeIdx};
use crate::graph::VertexId;
use crate::set::FastSet;

pub(crate) fn remove_extra_components(tree: &mut Forest<MDComputeNode>, prob: NodeIdx) -> Option<NodeIdx> {
    println!("start: {}", tree.to_string(Some(prob)));

    let mut subprob = tree[prob].first_child;

    while let Some(subprob_idx) = subprob {
        if !(tree.is_valid(subprob_idx) && tree[subprob_idx].data.connected)
        { break; }
        subprob = tree[subprob_idx].right;
    }

    let mut ret = None;
    if let Some(subprob) = subprob.and_then(|idx| { if tree.is_valid(idx) { Some(idx) } else { None } }) {
        ret = tree[subprob].first_child;
        assert!(ret.is_some());
        tree.detach(ret.unwrap());
        assert!(tree[subprob].is_leaf());
        tree.remove(subprob);
    }
    println!("return: {}", tree.to_string(ret));
    return ret;
}

pub(crate) fn remove_layers(tree: &mut Forest<MDComputeNode>, prob: NodeIdx) {
    println!("start: {}", tree.to_string(Some(prob)));
    let mut child = tree[prob].first_child;
    while let Some(c_) = tree.as_valid(child) {
        let next = tree[c_].right;
        assert_eq!(tree[c_].number_of_children(), 1);
        tree.replace(c_, tree[c_].first_child.unwrap());
        tree.remove(c_);
        child = next;
    }
    println!("finish: {}", tree.to_string(Some(prob)));
}

pub(crate) fn complete_alpha_lists(tree: &mut Forest<MDComputeNode>, alpha_list: &mut [Vec<NodeIdx>], vset: &mut FastSet, prob: NodeIdx, leaves: &[NodeIdx]) {
    println!("start: {}", tree.to_string(Some(prob)));

    for &v in leaves {
        for a in alpha_list[v.idx()].iter().copied().collect::<Vec<_>>() {
            assert_ne!(a, v);
            alpha_list[a.idx()].push(v);
        }
    }
    for v in leaves {
        let vs = &mut alpha_list[v.idx()];
        vset.clear();

        let mut len = vs.len();
        let mut i = 0;
        while i < len {
            let a = vs[i].idx();
            if vset.get(a) {
                len -= 1;
                vs[i] = vs[len];
                vs.pop();
            } else {
                vset.set(a);
                i += 1;
            }
        }
    }
}

pub(crate) fn merge_components(tree: &mut Forest<MDComputeNode>, problem: NodeIdx, new_components: Option<NodeIdx>) {
    println!("start: prob={}, new={}", tree.to_string(Some(problem)), tree.to_string(new_components));

    let Some(new_components) = new_components else { return; };
    if !tree.is_valid(new_components) { return; }

    let fc = tree[problem].first_child.unwrap();

    if tree[new_components].data.op_type == Operation::Parallel {
        if tree[fc].data.op_type == Operation::Parallel {
            tree.add_children_from(new_components, fc);
        } else {
            tree.move_to(fc, new_components);
        }
        tree.move_to(new_components, problem);
    } else {
        let new_root = tree.create_node(MDComputeNode::new_operation_node(Operation::Parallel));
        tree.move_to(new_root, problem);
        tree.move_to(new_components, new_root);
        tree.move_to(fc, new_root);
    }
}