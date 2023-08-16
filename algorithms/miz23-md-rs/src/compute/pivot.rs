use crate::compute::MDComputeNode;
use crate::forest::{Forest, NodeIdx};
use crate::graph::{Graph, VertexId};

fn is_pivot_layer(tree: &Forest<MDComputeNode>, index: NodeIdx) -> bool {
    let node = &tree[index];
    if let Some(parent) = node.parent {
        if !tree.is_valid(parent) { return false; }
        let parent = &tree[parent];
        return parent.data.is_problem_node() && Some(NodeIdx::from(parent.data.vertex.idx())) == node.first_child;
    } else {
        false
    }
}

fn pull_forward(tree: &mut Forest<MDComputeNode>, v: NodeIdx) {
    println!("pull_forward(): v={}", v.idx());

    let current_layer = tree[v].parent.unwrap();
    assert!(tree.is_valid(current_layer));

    if tree[current_layer].data.connected { return; }
    assert!(tree[current_layer].data.is_problem_node());

    let mut prev_layer = tree[current_layer].left.unwrap();
    assert!(tree.is_valid(prev_layer));

    println!("tree[prev_layer].data.active={}, is_pivot_layer()={}", tree[prev_layer].data.active as i32, is_pivot_layer(tree, prev_layer) as i32);

    if tree[prev_layer].data.active || is_pivot_layer(tree, prev_layer) {
        let new_layer = tree.create_node(MDComputeNode::new_problem_node(true));
        tree.move_to_before(new_layer, current_layer);
        prev_layer = new_layer;
        println!("new layer formed: {}", tree.to_string(tree[prev_layer].parent));
    }

    if tree[prev_layer].data.connected { tree.move_to(v, prev_layer); }
    if tree[current_layer].is_leaf() { tree.remove(current_layer); }
}


pub(crate) fn process_neighbors(graph: &Graph, tree: &mut Forest<MDComputeNode>, alpha_list: &mut [Vec<NodeIdx>], visited: &[bool], pivot: VertexId, current_problem: NodeIdx, neighbor_problem: Option<NodeIdx>) {
    println!("enter: pivot={}, current_prob={}, nbr_prob={:?}", pivot.idx(), current_problem.idx(), neighbor_problem);

    for &neighbor in graph.neighbors(pivot) {
        let neighbor = NodeIdx::from(neighbor.idx());
        if visited[neighbor.idx()] {
            alpha_list[neighbor.idx()].push(NodeIdx::from(pivot.idx()));
        } else if tree[neighbor].parent == Some(current_problem) {
            let neighbor_problem = tree.as_valid(neighbor_problem).unwrap();
            tree.move_to(neighbor, neighbor_problem);
        } else {
            println!("pull_forward(): nbr={}", neighbor.idx());
            pull_forward(tree, neighbor);
        }
    }
}

pub(crate) fn do_pivot(graph: &Graph, tree: &mut Forest<MDComputeNode>, alpha_list: &mut [Vec<NodeIdx>], visited: &[bool], current_problem: NodeIdx, pivot: VertexId) -> NodeIdx {
    let replacement = tree.create_node(tree[current_problem].data.clone());
    tree.swap(current_problem, replacement);
    tree.move_to(current_problem, replacement);
    tree[replacement].data.vertex = pivot;

    tree[current_problem].data.active = false;
    tree[current_problem].data.connected = false;
    tree[current_problem].data.vertex = VertexId::invalid();

    let pivot_problem = tree.create_node(MDComputeNode::new_problem_node(true));
    tree.move_to(pivot_problem, replacement);
    tree.move_to(NodeIdx::from(pivot.idx()), pivot_problem);

    let neighbor_problem = tree.create_node(MDComputeNode::new_problem_node(true));
    tree.move_to(neighbor_problem, replacement);
    process_neighbors(graph, tree, alpha_list, visited, pivot, current_problem, Some(neighbor_problem));


    if tree[current_problem].is_leaf() { tree.remove(current_problem); }

    if tree[neighbor_problem].is_leaf() { tree.remove(neighbor_problem); }

    return replacement;
}