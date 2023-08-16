use crate::compute::{MDComputeNode, SplitDirection};
use crate::compute::SplitDirection::{Left, Right};
use crate::forest::{Forest, NodeIdx};

fn promote_one_node(tree: &mut Forest<MDComputeNode>, index: NodeIdx, split_type: SplitDirection) {
    if tree[index].is_leaf() { return; }

    let mut st = vec![];
    st.push((false, index));
    st.push((true, tree[index].first_child.unwrap()));

    while let Some((b, nd)) = st.pop() {
        if b {
            let right = tree[nd].right;
            if let Some(right) = right.and_then(|right| { if tree.is_valid(right) { Some(right) } else { None } }) {
                st.push((true, right));
            }
            if tree[nd].data.is_split_marked(split_type) {
                let p = tree[nd].parent.unwrap();
                assert!(tree.is_valid(p));

                if split_type == Left {
                    tree.move_to_before(nd, p);
                } else {
                    tree.move_to_after(nd, p);
                }
                if tree[nd].has_child() {
                    st.push((false, nd));
                    st.push((true, tree[nd].first_child.unwrap()));
                }
            }
        } else {
            if tree[nd].is_leaf() && tree[nd].data.is_operation_node() {
                tree.remove(nd);
            } else if tree[nd].has_only_one_child() {
                tree.replace_by_children(nd);
                tree.remove(nd);
            }
        }
    }
}

fn promote_one_direction(tree: &mut Forest<MDComputeNode>, index: NodeIdx, split_type : SplitDirection) {
    for c in tree.children(index).collect::<Vec<_>>() {
        promote_one_node(tree, c, split_type);
    }
}

pub(crate) fn promote(tree: &mut Forest<MDComputeNode>, prob: NodeIdx) {
    promote_one_direction(tree, prob, Left);
    promote_one_direction(tree, prob, Right);
}