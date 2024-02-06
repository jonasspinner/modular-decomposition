use tracing::instrument;
use crate::algos::tree::{Kind, Tree, TreeNode, TreeNodeIndex};
use crate::algos::utils::{add_all_nodes_as_children, build_quotient, check_series_parallel, determine_spine_node_kind, try_trivial_quotient};
use crate::graph::{Graph, NodeIndex};
use crate::ordered_vertex_partition::ovp;
use crate::partition::{Partition, Part};
#[allow(unused)]
use crate::testing::to_vecs;


#[instrument(skip_all)]
#[allow(non_snake_case)]
pub(crate) fn modular_decomposition(graph: &mut Graph) -> Vec<TreeNode> {
    // Let v be the lowest-numbered vertex of G
    // if G has only one vertex then return v
    // else
    // let T' be the modular decomposition of G/P(G,v)
    // foreach member Y of P(G, v) do T_Y = UMD(G|Y)
    // return the composition of T' and {T_Y : Y in P(G,v) }

    if graph.node_count() == 0 { return vec![]; }

    if graph.node_count() == 1 {
        return vec![TreeNode::new(Kind::Vertex(NodeIndex::new(0)))];
    }

    let mut tree = Tree::with_capacity(2 * graph.node_count());
    let root = tree.new_node(Kind::default());

    let mut partition = Partition::new(graph.node_count());

    let mut crossing_edges = vec![];
    let mut map = vec![NodeIndex::new(0); graph.node_count()];
    let mut stack = vec![(Part::new(&partition), root)];

    while let Some((X, X_idx)) = stack.pop() {
        // Handle the subgraph G[X] defined by the part X.
        // At this point there are no edges between X and V \ X.
        assert!(X.len() >= 2);

        if let Some(kind) = check_series_parallel(graph, &X, &partition) {
            add_all_nodes_as_children(X, X_idx, kind, &partition, &mut tree);
            continue;
        }

        let v = X.nodes(&partition).next().unwrap();
        // let v = p.nodes_raw(partition).iter().min_by_key(|n| n.label).unwrap().node;

        // P(G[X], v) = OVP(G[X], [{v}, X - {v}])
        let p = X.separate_single_vertex(&mut partition, v);
        crossing_edges.clear();
        ovp(graph, p, &mut partition, |e| { crossing_edges.extend(e.iter().map(|&(u, v, _)| (u, v))) });

        // case |P(G[X], v)| = 2
        if let Some(ys) = try_trivial_quotient(&crossing_edges, p, &partition, X_idx, &mut tree) {
            stack.push(ys);
            continue;
        }

        // G[X] / P(G[X], v)
        let (mut quotient, v_q, mut ys) = build_quotient(&mut crossing_edges, p, &partition, v, &mut map);

        // MD(G[X] / P(G[X], v))
        chain(&mut quotient, v_q, &mut tree, X_idx);

        tree.for_leaves(X_idx, |(j, v)| {
            ys[v.index()].1 = j;
        });

        // further work on G[Y] for Y in P(G[X], v)
        for (Y, Y_idx) in ys {
            if let Some(v) = Y.try_into_node(&partition) {
                tree[Y_idx].kind = Kind::Vertex(v);
            } else { stack.push((Y, Y_idx)) }
        }
    }
    tree.into_vec()
}

fn chain(graph: &mut Graph, v: NodeIndex, tree: &mut Tree, current: TreeNodeIndex) {
    // Compute the modular decomposition of a nested graph with {v} being an innermost vertex in it.

    // P := OVP(G, ({v}, V(G) - {v});
    // number the vertices of G in order of their appearance in P
    // return ROP(G).

    // P = [V]
    let mut partition = Partition::new(graph.node_count());

    // P = [{v}, V - {v}]
    partition.refine_forward([v]);
    // OVP(G, [{v}, V - {v}])
    ovp(graph, partition.full_sub_partition(), &mut partition, |_| {});

    // reset G and P. keep partition ordering as label
    graph.restore_removed_edges();
    let p = partition.merge_parts_and_label_vertices();

    // ROP(G)
    rop(graph, p, current, &mut partition, tree);
}

#[allow(non_snake_case)]
fn rop(graph: &mut Graph, V: Part, V_idx: TreeNodeIndex, partition: &mut Partition, tree: &mut Tree) {
    // if G has an isolated vertex then let w be that vertex
    // else let w be the highest numbered vertex
    // if G has only one vertex then return {w};
    // else
    //   (X_1, ..., X_k) := OVP(G, ({w}, V(G) - {w}))
    //   Let T be a module tree with one node V(G)
    //   foreach set X_i do Let ROP(G|X_i) be the i_th child of V(G)
    // return T

    assert_eq!(V.len(), graph.node_count());

    let try_single_vertex = |Y: &Part, Y_idx: TreeNodeIndex, partition: &Partition, tree: &mut Tree| -> bool {
        if let Some(v) = Y.try_into_node(partition) {
            tree[Y_idx].kind = Kind::Vertex(v);
            true
        } else { false }
    };

    if try_single_vertex(&V, V_idx, partition, tree) { return; }

    let mut next = Some((V, V_idx));
    while let Some((X, X_idx)) = next.take() {
        assert!(X.len() >= 2);

        // choose w
        let w = rop_find_w(graph, &X, partition);
        // [{w}, X - {v}]
        let subpartition = X.separate_single_vertex(partition, w);

        // OVP(G[X], [{w}, X - {w}])
        let mut num_edges = 0;
        ovp(graph, subpartition, partition, |e| { num_edges += e.len() });

        let ys: Vec<_> = subpartition.parts(partition).collect();
        tree[X_idx].kind = determine_spine_node_kind(&ys, num_edges);

        // recurse on G[Y] for Y in OVP(G[X], [{w},  X - {w}])
        // exactly one child of X in MD(G[X]) is a non-singleton node
        for Y in ys {
            let Y_idx = tree.new_node(Kind::default());
            tree[X_idx].children.push(Y_idx);
            if !try_single_vertex(&Y, Y_idx, partition, tree) {
                next = Some((Y, Y_idx));
            }
        }
    }
}


#[allow(non_snake_case)]
fn rop_find_w(graph: &mut Graph, X: &Part, partition: &mut Partition) -> NodeIndex {
    // If G has a isolated vertex let w be that vertex
    // else let w be the highest numbered vertex
    let mut max = NodeIndex::invalid();
    let mut max_label = 0;
    for u in X.nodes_raw(partition) {
        if graph.degree(u.node) == 0 { return u.node; }
        if !max.is_valid() || u.label > max_label {
            max = u.node;
            max_label = u.label;
        }
    }
    max
}


pub(crate) mod tree {
    use std::ops::{Index, IndexMut};
    use common::make_index;
    use common::modular_decomposition::MDNodeKind;
    use crate::graph::NodeIndex;
    make_index!(pub(crate) TreeNodeIndex);

    #[derive(Debug, Copy, Clone, PartialEq)]
    pub(crate) enum Kind {
        Prime,
        Series,
        Parallel,
        Vertex(NodeIndex),
    }

    impl From<Kind> for MDNodeKind {
        fn from(value: Kind) -> Self {
            match value {
                Kind::Prime => { MDNodeKind::Prime }
                Kind::Series => { MDNodeKind::Series }
                Kind::Parallel => { MDNodeKind::Parallel }
                Kind::Vertex(u) => { MDNodeKind::Vertex(u.index()) }
            }
        }
    }

    impl Default for Kind {
        fn default() -> Self {
            Kind::Vertex(NodeIndex::invalid())
        }
    }

    #[derive(Debug, Default)]
    pub(crate) struct TreeNode {
        pub(crate) kind: Kind,
        pub(crate) children: Vec<TreeNodeIndex>,
    }

    impl TreeNode {
        pub(crate) fn new(kind: Kind) -> Self {
            Self { kind, children: vec![] }
        }
    }

    pub(crate) struct Tree {
        nodes: Vec<TreeNode>,
    }

    impl Tree {
        pub(crate) fn with_capacity(capacity: usize) -> Self {
            Tree { nodes: Vec::with_capacity(capacity) }
        }

        pub(crate) fn new_node(&mut self, kind: Kind) -> TreeNodeIndex {
            let n = self.nodes.len();
            self.nodes.push(TreeNode::new(kind));
            TreeNodeIndex::new(n)
        }

        pub(crate) fn for_leaves(&self, root: TreeNodeIndex, mut f: impl FnMut((TreeNodeIndex, NodeIndex))) {
            let mut stack = vec![root];
            while let Some(node) = stack.pop() {
                if let Kind::Vertex(u) = self[node].kind {
                    assert!(u.is_valid());
                    f((node, u));
                } else {
                    stack.extend(self[node].children.iter().copied());
                }
            }
        }
        pub(crate) fn into_vec(self) -> Vec<TreeNode> { self.nodes }
    }

    impl Index<TreeNodeIndex> for Tree {
        type Output = TreeNode;

        fn index(&self, index: TreeNodeIndex) -> &Self::Output {
            &self.nodes[index.index()]
        }
    }

    impl IndexMut<TreeNodeIndex> for Tree {
        fn index_mut(&mut self, index: TreeNodeIndex) -> &mut Self::Output {
            &mut self.nodes[index.index()]
        }
    }
}

mod utils {
    use crate::algos::tree::{Kind, Tree, TreeNodeIndex};
    use crate::algos::tree::Kind::{Prime, Series, Parallel, Vertex};
    use crate::graph::{Graph, NodeIndex};
    use crate::partition::{Part, Partition, SubPartition};

    pub(crate) fn determine_spine_node_kind(xs: &[Part], num_edges: usize) -> Kind {
        // This is used for determining node kinds for MD(G/P(G,v)).
        // If a node is a parallel or series node, it has exactly two children.
        // The smallest prime node has at least four children (P_4)
        assert!((xs.len() == 2) || (xs.len() >= 4));
        if xs.len() > 2 { Prime } else if num_edges == 0 { Parallel } else { Series }
    }

    /// Try to apply the case that no further parts have been split, i.e. |P(G[X], v)| = 2.
    /// This is not necessary for correctness, but avoids building tiny quotient graphs.
    #[allow(non_snake_case)]
    pub(crate) fn try_trivial_quotient(crossing_edges: &[(NodeIndex, NodeIndex)], p: SubPartition, partition: &Partition, current: TreeNodeIndex, tree: &mut Tree) -> Option<(Part, TreeNodeIndex)> {
        let nodes = partition.nodes(&p);
        assert_ne!(nodes[0].part, nodes[1].part, "The subpartition started as [{{v}}, X - {{v}}].");
        if nodes[1].part != nodes[nodes.len() - 1].part {
            return None;
        }
        // We have the case that P(G[X], v) = [{v}, X - {v}]
        let v = nodes[0].node;
        let B = partition.part_by_index(nodes[1].part);
        tree[current].kind = if crossing_edges.is_empty() { Parallel } else { Series };
        let A_idx = tree.new_node(Vertex(v));
        let B_idx = tree.new_node(Kind::default());
        tree[current].children.extend([A_idx, B_idx]);
        assert!(B.len() > 1, "The case |X| = 2 should be covered earlier. |X| > 2 and |P(G[X], v)| = 2 imply |p1| = |X - {{v}}| > 1.");
        Some((B, B_idx))
    }

    #[allow(non_snake_case)]
    pub(crate) fn build_quotient(crossing_edges: &mut Vec<(NodeIndex, NodeIndex)>,
                                 p: SubPartition, partition: &Partition,
                                 inner_vertex: NodeIndex, map: &mut [NodeIndex])
                                 -> (Graph, NodeIndex, Vec<(Part, TreeNodeIndex)>) {
        // map parts Y_1,...,Y_k to indices 1,...,k
        let ys: Vec<_> = p.part_indices(partition)
            .enumerate()
            .map(|(node_idx, Y_idx)| {
                let Y = partition.part_by_index(Y_idx);
                let node_idx = node_idx.into();
                for u in Y.nodes(partition) {
                    map[u.index()] = node_idx;
                }
                (partition.part_by_index(Y_idx), TreeNodeIndex::invalid())
            }).collect();

        for (u, v) in crossing_edges.iter_mut() {
            *u = map[u.index()];
            *v = map[v.index()];
        }
        crossing_edges.sort_unstable();
        crossing_edges.dedup();
        (Graph::from_edges(ys.len(), crossing_edges.iter().copied()), map[inner_vertex.index()], ys)
    }

    /// Check if the nodes of the part represent a series or parallel module and return its kind if
    /// that's the case
    #[allow(non_snake_case)]
    pub(crate) fn check_series_parallel(graph: &Graph, X: &Part, partition: &Partition) -> Option<Kind> {
        assert!(X.len() >= 2);
        if graph.edge_count() == 0 { return Some(Parallel); }
        let mut nodes = X.nodes(partition);
        let d = graph.degree(nodes.next().unwrap());
        if (d == 0 || d == X.len() - 1) && nodes.all(|v| graph.degree(v) == d) {
            Some(if d == 0 { Parallel } else { Series })
        } else {
            None
        }
    }

    /// Assign the given kind to the current node and add all nodes in the part as its children.
    #[allow(non_snake_case)]
    pub(crate) fn add_all_nodes_as_children(X: Part, X_idx: TreeNodeIndex, kind: Kind, partition: &Partition, tree: &mut Tree) {
        tree[X_idx].kind = kind;
        for v in X.nodes(partition) {
            let c = tree.new_node(Vertex(v));
            tree[X_idx].children.push(c);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::algos::{chain, Kind};
    use crate::algos::tree::{Tree, TreeNodeIndex};
    use crate::graph::{Graph, NodeIndex};
    use crate::testing::ted08_test0_graph;

    #[test]
    fn modular_decomposition() {
        let mut graph = ted08_test0_graph();

        let tree = super::modular_decomposition(&mut graph);

        println!("digraph {{");
        for (i, node) in tree.iter().enumerate() {
            match node.kind {
                Kind::Vertex(u) => {
                    println!("  {} [label=\"{}:{:?}\"]", i, i, u.index());
                }
                kind => {
                    println!("  {} [label=\"{}:{:?}\"]", i, i, kind);
                }
            }
        }
        for i in 0..tree.len() {
            let i = TreeNodeIndex::new(i);
            for &j in &tree[i.index()].children {
                println!("  {} -> {}", i.index(), j.index());
            }
        }
        println!("}}");
    }

    #[test]
    fn ted08_test0_v0_quotient() {
        let graph = Graph::from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)].map(|(u, v)| (NodeIndex::new(u), NodeIndex::new(v))));

        {
            let mut graph = graph.clone();
            let mut tree = Tree::with_capacity(0);
            let root = tree.new_node(Kind::default());
            chain(&mut graph, NodeIndex::new(0), &mut tree, root);
            println!("{:?}", tree.into_vec());
        }

        {
            let mut graph = graph.clone();
            let mut tree = Tree::with_capacity(0);
            let root = tree.new_node(Kind::default());
            chain(&mut graph, NodeIndex::new(1), &mut tree, root);
            println!("{:?}", tree.into_vec());
        }

        {
            let mut graph = graph.clone();
            let mut tree = Tree::with_capacity(0);
            let root = tree.new_node(Kind::default());
            chain(&mut graph, NodeIndex::new(2), &mut tree, root);
            println!("{:?}", tree.into_vec());
        }

        {
            let mut graph = graph.clone();
            let mut tree = Tree::with_capacity(0);
            let root = tree.new_node(Kind::default());
            chain(&mut graph, NodeIndex::new(3), &mut tree, root);
            println!("{:?}", tree.into_vec());
        }

        {
            let mut graph = graph.clone();
            let mut tree = Tree::with_capacity(0);
            let root = tree.new_node(Kind::default());
            chain(&mut graph, NodeIndex::new(4), &mut tree, root);
            println!("{:?}", tree.into_vec());
        }
    }
}