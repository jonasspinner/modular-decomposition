use common::modular_decomposition::MDNodeKind;
use petgraph::graph::DiGraph;
use petgraph::visit::DfsPostOrder;
use petgraph::{Incoming, Outgoing};
use std::cmp::{min, Reverse};

pub fn canonicalize(md_tree: &DiGraph<MDNodeKind, ()>) -> Vec<(u32, u32, MDNodeKind)> {
    let Some(root) = md_tree.externals(Incoming).next() else {
        return vec![];
    };
    let mut dfs = DfsPostOrder::new(md_tree, root);

    let mut info = vec![(u32::MAX, u32::MAX); md_tree.node_count()];

    while let Some(a) = dfs.next(md_tree) {
        let mut min_vertex = u32::MAX;
        let mut num_vertices = 0;
        if let MDNodeKind::Vertex(u) = md_tree[a] {
            min_vertex = u as u32;
            num_vertices = 1;
        }
        for b in md_tree.neighbors_directed(a, Outgoing) {
            let (a_min, a_num) = info[b.index()];
            min_vertex = min(min_vertex, a_min);
            num_vertices += a_num;
        }
        info[a.index()] = (min_vertex, num_vertices);
    }

    let mut result = vec![];
    let mut stack = vec![root];
    while let Some(a) = stack.pop() {
        let (m, n) = info[a.index()];
        result.push((m, n, md_tree[a]));
        let mut children: Vec<_> = md_tree.neighbors_directed(a, Outgoing).collect();
        children.sort_by_key(|c| Reverse(info[c.index()].0));
        stack.extend(children);
    }
    result
}
