use crate::compute::{MDComputeNode, SplitDirection};
use crate::compute::Operation::Prime;
use crate::compute::refinement::group_sibling_nodes::group_sibling_nodes;
use crate::compute::refinement::marking_split_types::{add_split_mark, get_max_subtrees, mark_ancestors_by_split};
use crate::compute::refinement::set_up::{number_by_comp, number_by_tree};
use crate::compute::refinement::utilities::is_root_operator;
use crate::forest::{Forest, NodeIdx};
use crate::graph::VertexId;
use crate::trace;

mod set_up {
    use crate::compute::{MDComputeNode, Operation};
    use crate::forest::Forest;
    use crate::forest::NodeIdx;

    pub(crate) fn number_by_comp(tree: &mut Forest<MDComputeNode>, problem: NodeIdx) {
        let mut comp_number = 0;
        let pivot = tree[problem].data.vertex;
        let mut op_type = Operation::Series;

        for c in tree.children(problem).collect::<Vec<_>>() {
            if c.idx() == pivot.idx() { op_type = Operation::Parallel; }

            if tree[c].data.op_type == op_type {
                for x in tree.children(c).collect::<Vec<_>>() {
                    for y in tree.get_dfs_reverse_preorder_nodes(x) { tree[y].data.comp_number = comp_number; }
                    comp_number += 1;
                }
            } else {
                for y in tree.get_dfs_reverse_preorder_nodes(c) { tree[y].data.comp_number = comp_number; }
                comp_number += 1;
            }
        }
    }

    pub(crate) fn number_by_tree(tree: &mut Forest<MDComputeNode>, problem: NodeIdx) {
        let mut tree_number = 0;
        for c in tree.children(problem).collect::<Vec<_>>() {
            for y in tree.get_dfs_reverse_preorder_nodes(c) { tree[y].data.tree_number = tree_number; }
            tree_number += 1;
        }
    }
}

mod utilities {
    use crate::compute::MDComputeNode;
    use crate::forest::Forest;
    use crate::forest::NodeIdx;

    pub(crate) fn is_root_operator(tree: &Forest<MDComputeNode>, index: NodeIdx) -> bool {
        tree[index].is_root() || !tree[tree[index].parent.unwrap()].data.is_operation_node()
    }
}

mod marking_split_types {
    use crate::compute::{MDComputeNode, SplitDirection};
    use crate::compute::refinement::utilities::is_root_operator;
    use crate::compute::Operation;
    use crate::forest::Forest;
    use crate::forest::NodeIdx;

    pub(crate) fn add_split_mark(tree: &mut Forest<MDComputeNode>, index: NodeIdx, split_type: SplitDirection, should_recurse: bool) {
        if !tree[index].data.is_split_marked(split_type) {
            let p = tree[index].parent.unwrap();
            if tree[p].data.is_operation_node() {
                tree[p].data.increment_num_split_children(split_type);
            }
            tree[index].data.set_split_mark(split_type);
        }

        if !should_recurse || tree[index].data.op_type != Operation::Prime { return; }

        if tree[index].number_of_children() as i32 == tree[index].data.get_num_split_children(split_type) {
            return;
        }

        let mut c = tree[index].first_child;
        while let Some(c_) = tree.as_valid(c) {
            if !tree[c_].data.is_split_marked(split_type) {
                tree[index].data.increment_num_split_children(split_type);
                tree[c_].data.set_split_mark(split_type);
            }
            c = tree[c_].right;
        }
    }

    pub(crate) fn mark_ancestors_by_split(tree: &mut Forest<MDComputeNode>, index: NodeIdx, split_type: SplitDirection) {
        let mut p = tree[index].parent;
        loop {
            let p_ = p.unwrap();
            if tree[p_].data.is_problem_node() { break; }
            if tree[p_].data.is_split_marked(split_type) {
                add_split_mark(tree, p_, split_type, true);
                break;
            }
            add_split_mark(tree, p_, split_type, true);
            p = tree[p_].parent;
        }
    }

    fn is_parent_fully_charged(tree: &Forest<MDComputeNode>, x: NodeIdx) -> bool {
        if is_root_operator(tree, x) { return false; }
        let p = tree[x].parent.unwrap();
        tree[p].number_of_children() == tree[p].data.number_of_marks() as u32
    }

    pub(crate) fn get_max_subtrees(tree: &mut Forest<MDComputeNode>, leaves: &[NodeIdx]) -> Vec<NodeIdx> {
        let mut full_charged : Vec<NodeIdx> = Vec::from(leaves);
        let mut charged = vec![];

        let mut i = 0;
        while i < full_charged.len() {
            let x = full_charged[i];
            if is_root_operator(tree, x) { i+=1; continue; }

            let p = tree[x].parent.unwrap();
            if !tree[p].data.is_marked() { charged.push(p); }
            tree[p].data.add_mark();

            if tree[p].data.num_marks == tree[p].number_of_children() as i32 {
                full_charged.push(p);
            }
            i += 1;
        }

        let mut ret = vec![];
        for x in full_charged {
            if !is_parent_fully_charged(tree, x) { ret.push(x); }
        }
        for x in charged { tree[x].data.clear_marks(); }
        return ret;
    }
}

mod group_sibling_nodes {
    use crate::compute::refinement::utilities::is_root_operator;
    use crate::compute::{MDComputeNode, SplitDirection};
    use crate::compute::Operation;
    use crate::compute::SplitDirection::{Left, Right};
    use crate::forest::Forest;
    use crate::forest::NodeIdx;

    pub(crate) fn group_sibling_nodes(tree: &mut Forest<MDComputeNode>, nodes: &[NodeIdx]) -> Vec<(NodeIdx, bool)> {
        //trace!("group_sibling_nodes start: {}", tree.to_string(Some(tree.get_root(NodeIdx::from(0)))));
        let mut parents = vec![];
        let mut sibling_groups = vec![];

        for &node in nodes {
            if is_root_operator(tree, node) {
                sibling_groups.push((node, false));
            } else {
                tree.make_first_child(node);
                let p = tree[node].parent.unwrap();

                if !tree[p].data.is_marked() { parents.push(p); }
                tree[p].data.add_mark();
            }
        }

        //trace!("group_sibling_nodes mid:   {}", tree.to_string(Some(tree.get_root(NodeIdx::from(0)))));

        for p in parents {
            let c = tree[p].first_child.unwrap();
            let num_marks = tree[p].data.number_of_marks();

            if num_marks == 1 {
                sibling_groups.push((c, false));
            } else {
                let grouped_children = tree.create_node(tree[p].data.clone());
                tree[grouped_children].data.clear_marks();

                for st in [Left, Right] {
                    if tree[grouped_children].data.is_split_marked(st) {
                        tree[p].data.increment_num_split_children(st);
                    }
                }

                let mut c_ = tree[p].first_child;
                for _i in 0..num_marks {
                    let c = c_.unwrap();
                    let next = tree[c].right;
                    tree.move_to(c, grouped_children);

                    for st in [SplitDirection::Left, SplitDirection::Right] {
                        if tree[c].data.is_split_marked(st) {
                            tree[p].data.decrement_num_split_children(st);
                            tree[grouped_children].data.increment_num_split_children(st);
                        }
                    }
                    c_ = next;
                }

                tree.move_to(grouped_children, p);

                sibling_groups.push((grouped_children, tree[grouped_children].data.op_type == Operation::Prime));
            }
            tree[p].data.clear_marks();
        }

        //trace!("group_sibling_nodes end:   {}", tree.to_string(Some(tree.get_root(NodeIdx::from(0)))));

        sibling_groups
    }
}

fn get_split_type(tree: &Forest<MDComputeNode>, index: NodeIdx, refiner: VertexId, pivot: VertexId) -> SplitDirection {
    let pivot = From::from(pivot.idx());
    let refiner = From::from(refiner.idx());
    let pivot_tn = tree[pivot].data.tree_number;
    let refiner_tn = tree[refiner].data.tree_number;
    let current = tree[index].data.tree_number;
    if current < pivot_tn || refiner_tn < current { SplitDirection::Left } else { SplitDirection::Right }
}

fn refine_one_node(tree: &mut Forest<MDComputeNode>, index: NodeIdx, split_type: SplitDirection, new_prime: bool) {
    trace!("refining tree={}, index={}, split_type={:?}, new_prime={}", tree.to_string(Some(index)), index.idx(), split_type, new_prime as i32);
    if is_root_operator(tree, index) { return; }

    let p = tree[index].parent.unwrap();
    let mut new_sibling = None;

    if is_root_operator(tree, p) {
        if split_type == SplitDirection::Left {
            tree.move_to_before(index, p);
        } else {
            tree.move_to_after(index, p);
        }

        for st in [SplitDirection::Left, SplitDirection::Right] {
            if tree[index].data.is_split_marked(st) {
                tree[p].data.decrement_num_split_children(st);
            }
        }

        new_sibling = Some(p);

        if tree[p].has_only_one_child() {
            tree.replace_by_children(p);
            tree.remove(p);
            new_sibling = None;
        }
    } else if tree[p].data.op_type != Prime {
        let replacement = tree.create_node(tree[p].data.clone());
        tree.replace(p, replacement);
        tree.move_to(index, replacement);
        tree.move_to(p, replacement);

        new_sibling = Some(p);

        for st in [SplitDirection::Left, SplitDirection::Right] {
            if tree[index].data.is_split_marked(st) {
                tree[p].data.decrement_num_split_children(st);
                tree[replacement].data.increment_num_split_children(st);
            }
            if tree[p].data.is_split_marked(st) { tree[replacement].data.increment_num_split_children(st) }
        }
    }

    add_split_mark(tree, index, split_type, new_prime);

    mark_ancestors_by_split(tree, index, split_type);

    if let Some(new_sibling) = new_sibling {
        add_split_mark(tree, new_sibling, split_type, true);
    }
}

fn refine_with(tree: &mut Forest<MDComputeNode>, alpha_list: &[Vec<NodeIdx>], refiner: VertexId, pivot: VertexId) {
    let subtree_roots = get_max_subtrees(tree, &alpha_list[refiner.idx()]);

    let sibling_groups = group_sibling_nodes(tree, &subtree_roots);

    trace!("alpha[{}]: {:?}", refiner.idx(), alpha_list[refiner.idx()].iter().map(|n| n.idx()).collect::<Vec<_>>());
    trace!("subtree_roots: {:?}", subtree_roots.iter().map(|n| n.idx()).collect::<Vec<_>>());
    trace!("sibling_groups: {:?}, tree={}", sibling_groups.iter().map(|(n, b)| (n.idx(), *b as i32)).collect::<Vec<_>>(), tree.to_string(tree[NodeIdx::from(pivot.idx())].parent));

    for (index, new_prime) in sibling_groups {
        let split_type = get_split_type(tree, index, refiner, pivot);

        refine_one_node(tree, index, split_type, new_prime);
    }
}

pub(crate) fn refine(tree: &mut Forest<MDComputeNode>, alpha_list: &[Vec<NodeIdx>], problem: NodeIdx, leaves: &[NodeIdx]) {
    trace!("start: {}", tree.to_string(Some(problem)));

    number_by_comp(tree, problem);

    number_by_tree(tree, problem);

    for &v in leaves {
        refine_with(tree, alpha_list, From::from(v.idx()), tree[problem].data.vertex);
    }
}

