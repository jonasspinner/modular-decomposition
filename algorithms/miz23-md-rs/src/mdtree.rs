use std::collections::HashMap;
use crate::compute::{compute, MDComputeNode, Operation};
use crate::forest::{Forest, NodeIdx};
use crate::graph::{Graph, VertexId};

#[allow(unused)]
enum Kind {
    Vertex(VertexId),
    Operation(Operation),
}

struct MDNode {
    kind: Kind,
    vertices_begin: i32,
    vertices_end: i32,
}

#[allow(unused)]
impl MDNode {
    fn new(kind: Kind, vertices_begin: i32, vertices_end: i32) -> Self {
        Self { kind, vertices_begin, vertices_end }
    }
    fn size(&self) -> usize { (self.vertices_end - self.vertices_begin) as usize }
    fn is_vertex_node(&self) -> bool { matches!(self.kind, Kind::Vertex(_)) }
    fn is_operation_node(&self) -> bool { matches!(self.kind, Kind::Operation(_) ) }
    fn is_prime_node(&self) -> bool { matches!(self.kind, Kind::Operation(Operation::Prime)) }
    fn is_series_node(&self) -> bool { matches!(self.kind, Kind::Operation(Operation::Series)) }
    fn is_parallel_node(&self) -> bool { matches!(self.kind, Kind::Operation(Operation::Parallel)) }
}

#[allow(unused)]
pub(crate) struct MDTree {
    tree: Forest<MDNode>,
    root: NodeIdx,
    vertices: Vec<VertexId>,
}

#[allow(unused)]
impl MDTree {
    pub(crate) fn new(graph: &Graph, sorted: bool) -> Option<Self> {
        let (tree, root) = compute(graph);
        root.map(|root| {
            let mut ret = MDTree::from_result(&tree, root);
            if sorted {
                ret.sort();
            }
            ret
        })
    }

    fn from_result(tree: &Forest<MDComputeNode>, root: NodeIdx) -> Self {
        let mut this = MDTree { tree: Forest::new(), root, vertices: vec![] };
        this.vertices = tree.leaves(root).map(|n| VertexId::from(n.idx())).collect();
        this.vertices.reverse();

        let n = this.vertices.len();

        for i in 0..n {
            let i = i as i32;
            this.tree.create_node(MDNode::new(Kind::Vertex(this.vertices[i as usize]), i, i + 1));
        }

        let mut mapping: HashMap<VertexId, usize> = HashMap::new();
        for i in 0..n {
            mapping.insert(this.vertices[i], i);
        }

        let bfs_order = tree.get_bfs_nodes(root);

        for &it in bfs_order.iter().rev() {
            if tree[it].data.is_vertex_node() { continue; };
            if tree[it].data.is_problem_node() { panic!("should not be a problem node"); }

            let children: Vec<_> = tree.children(it).collect();
            let op = tree[it].data.op_type;
            let mut idx_begin = n as i32;
            let mut idx_end = 0;
            for c in &children {
                idx_begin = std::cmp::min(idx_begin, this.tree[mapping[&VertexId::from(c.idx())].into()].data.vertices_begin);
                idx_end = std::cmp::max(idx_end, this.tree[mapping[&VertexId::from(c.idx())].into()].data.vertices_end);
            }

            let node_idx = this.tree.create_node(MDNode::new(Kind::Operation(op), idx_begin, idx_end));
            for c in children.iter().rev() { this.tree.move_to(mapping[&VertexId::from(c.idx())].into(), node_idx); }
            mapping.insert(VertexId::from(it.idx()), node_idx.idx());
        }

        this.root = mapping[&VertexId::from(root.idx())].into();

        this
    }

    fn sort(&mut self) {}
}

