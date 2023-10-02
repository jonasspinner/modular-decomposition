mod promotion;
mod refinement;
mod misc;
mod pivot;
mod assembly;

use std::fmt::{Debug, Formatter};
use crate::graph::VertexId;
use crate::forest::{NodeIdx, Forest};
use crate::graph::Graph;
use crate::trace;


#[derive(PartialEq, Clone, Copy, Debug)]
pub(crate) enum NodeType {
    Vertex,
    Operation,
    Problem,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub(crate) enum Operation {
    Prime,
    Series,
    Parallel,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub(crate) enum SplitDirection {
    None,
    Left,
    Right,
    Mixed,
}

#[derive(Clone)]
pub(crate) struct MDComputeNode {
    pub(crate) node_type: NodeType,
    pub(crate) op_type: Operation,
    split_direction: SplitDirection,
    pub(crate) vertex: VertexId,
    comp_number: i32,
    tree_number: i32,
    num_marks: i32,
    num_left_split_children: i32,
    num_right_split_children: i32,
    active: bool,
    connected: bool,
}

impl Debug for MDComputeNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.node_type {
            NodeType::Vertex => { write!(f, "{}", self.vertex.idx()) }
            NodeType::Operation => {
                match self.op_type {
                    Operation::Prime => { write!(f, "P") }
                    Operation::Series => { write!(f, "J") }
                    Operation::Parallel => { write!(f, "U") }
                }
            }
            NodeType::Problem => {
                if self.vertex.is_invalid() { write!(f, "C-") } else {
                    write!(f, "C{}", self.vertex.idx())
                }
            }
        }
    }
}

impl MDComputeNode {
    fn new(node_type: NodeType) -> Self {
        MDComputeNode {
            node_type,
            op_type: Operation::Prime,
            split_direction: SplitDirection::None,
            vertex: VertexId::invalid(),
            comp_number: -1,
            tree_number: -1,
            num_marks: 0,
            num_left_split_children: 0,
            num_right_split_children: 0,
            active: false,
            connected: false,
        }
    }
    fn new_vertex_node(vertex: VertexId) -> Self {
        let mut result = Self::new(NodeType::Vertex);
        result.vertex = vertex;
        result
    }
    fn new_operation_node(op_type: Operation) -> Self {
        let mut result = Self::new(NodeType::Operation);
        result.op_type = op_type;
        result
    }
    fn new_problem_node(connected: bool) -> Self {
        let mut node = Self::new(NodeType::Problem);
        node.connected = connected;
        node
    }
}

impl MDComputeNode {
    pub(crate) fn is_vertex_node(&self) -> bool { self.node_type == NodeType::Vertex }
    fn is_operation_node(&self) -> bool { self.node_type == NodeType::Operation }
    pub(crate) fn is_problem_node(&self) -> bool { self.node_type == NodeType::Problem }

    fn is_marked(&self) -> bool { self.num_marks > 0 }
    fn add_mark(&mut self) { self.num_marks += 1; }
    fn number_of_marks(&self) -> i32 { self.num_marks }
    fn clear_marks(&mut self) { self.num_marks = 0; }

    fn is_split_marked(&self, split_direction: SplitDirection) -> bool {
        self.split_direction == SplitDirection::Mixed || self.split_direction == split_direction
    }

    fn set_split_mark(&mut self, split_direction: SplitDirection) {
        if self.split_direction == split_direction {
            // already set
        } else if self.split_direction == SplitDirection::None {
            self.split_direction = split_direction;
        } else {
            self.split_direction = SplitDirection::Mixed;
        }
    }

    fn increment_num_split_children(&mut self, split_direction: SplitDirection) {
        if split_direction == SplitDirection::Left {
            self.num_left_split_children += 1;
        } else {
            self.num_right_split_children += 1;
        }
    }

    fn decrement_num_split_children(&mut self, split_direction: SplitDirection) {
        if split_direction == SplitDirection::Left {
            self.num_left_split_children -= 1;
        } else {
            self.num_right_split_children -= 1;
        }
    }

    fn get_num_split_children(&self, split_direction: SplitDirection) -> i32 {
        if split_direction == SplitDirection::Left {
            self.num_left_split_children
        } else { self.num_right_split_children }
    }

    fn clear_num_split_children(&mut self) {
        self.num_left_split_children = 0;
        self.num_right_split_children = 0;
    }

    fn clear(&mut self) {
        self.comp_number = -1;
        self.tree_number = -1;
        // self.num_marks = 0;
        self.split_direction = SplitDirection::None;
        self.clear_num_split_children();
    }
}

pub(crate) fn compute(graph: &Graph) -> (Forest<MDComputeNode>, Option<NodeIdx>) {
    let mut tree = Forest::<MDComputeNode>::new();
    let n = graph.number_of_nodes();
    if n == 0 {
        return (tree, None);
    }

    for u in graph.vertices() {
        tree.create_node(MDComputeNode::new_vertex_node(u));
    }

    let main_problem = tree.create_node(MDComputeNode::new_problem_node(false));

    for idx in (0..graph.number_of_nodes()).rev().map(|i| NodeIdx::new(i)) {
        tree.move_to(idx, main_problem);
    }

    let new_root = implementation::compute(graph, &mut tree, main_problem);

    trace!("result: {}", tree.to_string(Some(new_root)));
    return (tree, Some(new_root));
}

mod implementation {
    use crate::compute::MDComputeNode;
    use crate::compute::misc::{complete_alpha_lists, merge_components, remove_extra_components, remove_layers};
    use crate::compute::promotion::promote;
    use crate::compute::refinement::refine;
    use crate::compute::assembly::assemble;
    use crate::compute::pivot::{do_pivot, process_neighbors};
    use crate::forest::{Forest, NodeIdx};
    use crate::graph::{Graph, VertexId};
    use crate::set::FastSet;
    use crate::trace;

    pub(crate) fn compute(graph: &Graph, tree: &mut Forest<MDComputeNode>, main_problem: NodeIdx) -> NodeIdx {
        trace!("start computer(): {}", tree.to_string(Some(main_problem)));
        let n = graph.number_of_nodes();
        let mut current_problem_ = Some(main_problem);
        let mut alpha_list: Vec<Vec<NodeIdx>> = vec![vec![]; n];
        let mut fp_neighbors: Vec<Vec<VertexId>> = vec![vec![]; n];
        let mut visited = vec![false; n];
        let mut vset = FastSet::new(n);
        let mut result = NodeIdx::from(0);

        #[allow(unused)]
        let mut t = 0;

        while let Some(current_problem) = tree.as_valid(current_problem_) {
            t += 1;
            trace!("main problem ({}): {}", t, tree.to_string(Some(tree.get_root(current_problem))));
            trace!("current problem: {}", tree.to_string(Some(current_problem)));
            #[allow(unused)]
            for i in 0..n {
                trace!("visited [{}]: {:?}", i, visited[i]);
                trace!("alpha [{}]: {:?}", i, alpha_list[i].iter().map(|n| n.idx()).collect::<Vec<_>>());
            }

            tree[current_problem].data.active = true;
            let child = tree[current_problem].first_child.unwrap();

            if !tree[child].data.is_problem_node() {
                visited[child.idx()] = true;

                let child = VertexId::from(child.idx());

                if tree[current_problem].has_only_one_child() {
                    process_neighbors(&graph, tree, &mut alpha_list, &mut visited, child, current_problem, None);
                } else {
                    let pivoted = do_pivot(graph, tree, &mut alpha_list, &mut visited, current_problem, child);

                    current_problem_ = tree[pivoted].first_child;
                    continue;
                }
            } else {
                let extra_components = remove_extra_components(tree, current_problem);
                trace!("extra: {}", tree.to_string(extra_components));

                remove_layers(tree, current_problem);

                let leaves = tree.leaves(current_problem).collect::<Vec<_>>();
                complete_alpha_lists(tree, &mut alpha_list, &mut vset, current_problem, &leaves);

                refine(tree, &alpha_list, current_problem, &leaves);

                promote(tree, current_problem);

                assemble(tree, &alpha_list, current_problem, &mut fp_neighbors, &mut vset);

                for c in tree.get_dfs_reverse_preorder_nodes(tree[current_problem].first_child.unwrap()) {
                    if tree[c].is_leaf() { alpha_list[c.idx()].clear(); }
                    tree[c].data.clear();
                }

                merge_components(tree, current_problem, extra_components);
            }

            result = tree[current_problem].first_child.unwrap();
            current_problem_ = if tree[current_problem].is_last_child() { tree[current_problem].parent } else { tree[current_problem].right };
        }

        let result_parent = tree[result].parent.unwrap();
        tree.detach(result);
        tree.remove(result_parent);

        result
    }
}