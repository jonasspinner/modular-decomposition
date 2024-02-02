use tracing::instrument;
use crate::algos::tree::{for_leaves, Kind, TreeNode, TreeNodeIndex};
use crate::algos::utils::{add_all_nodes_as_children, build_quotient, check_series_parallel, determine_spine_node_kind, try_trivial_quotient};
use crate::graph::{Graph, NodeIndex};
use crate::ordered_vertex_partition::ovp;
use crate::partition::{Partition, Part};
#[allow(unused)]
use crate::testing::to_vecs;


#[instrument(skip_all)]
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

    let mut tree = Vec::with_capacity(2 * graph.node_count());
    let root = TreeNodeIndex::new(0);
    tree.push(TreeNode::default());

    let mut partition = Partition::new(graph.node_count());

    let mut removed = vec![];
    let mut stack = vec![(Part::new(&partition), root)];

    while let Some((p, current)) = stack.pop() {
        // We handle to subgraph G[X] with the part p = X.
        // At this point there are no edges from X to V \ X.
        assert!(p.len() >= 2);

        if let Some(kind) = check_series_parallel(graph, &p, &partition) {
            add_all_nodes_as_children(kind, p, &partition, &mut tree, current);
            continue;
        }

        let v = p.nodes(&partition).next().unwrap();
        // let v = p.nodes_raw(partition).iter().min_by_key(|n| n.label).unwrap().node;

        // P(G[X], v) = OVP(G[X], [{v}, X - {v}])
        let p = p.separate_single_vertex(&mut partition, v);
        removed.clear();
        ovp(graph, p, &mut partition, |e| { removed.extend(e.iter().map(|&(u, v, _)| (u, v))) });

        // case |P(G[X], v)| = 2
        if let Some(ys) = try_trivial_quotient(&removed, p, &partition, current, &mut tree) {
            stack.push(ys);
            continue;
        }

        // G[X] / P(G[X], v)
        let (mut quotient, v_q, mut ys) = build_quotient(&mut removed, p, &partition, v);

        // MD(G[X] / P(G[X], v))
        chain(&mut quotient, v_q, &mut tree, current);

        for_leaves(&tree, current, |(j, v)| {
            ys[v.index()].1 = j;
        });

        // Further work on G[Y] for Y in P(G[X], v)
        for (part, idx) in ys {
            if let Some(v) = part.try_into_node(&partition) {
                tree[idx.index()].kind = Kind::Vertex(v);
            } else { stack.push((part, idx)) }
        }
    }
    tree
}

fn chain(graph: &mut Graph, v: NodeIndex, tree: &mut Vec<TreeNode>, current: TreeNodeIndex) {
    // Compute the modular decomposition of a nested graph with {v} being an innermost vertex in it.

    // P := OVP(G, ({v}, V(G) - {v});
    // number the vertices of G in order of their appearance in P
    // return ROP(G).

    let mut partition = Partition::new(graph.node_count());

    partition.refine_forward([v]);
    ovp(graph, partition.full_sub_partition(), &mut partition, |_| {});

    graph.restore_removed_edges();
    let p = partition.merge_parts_and_label_vertices();

    rop(graph, p, &mut partition, tree, current);
}

fn rop(graph: &mut Graph, p: Part, partition: &mut Partition, tree: &mut Vec<TreeNode>, current: TreeNodeIndex) {
    // if G has an isolated vertex then let w be that vertex
    // else let w be the highest numbered vertex
    // if G has only one vertex then return {w};
    // else
    //   (X_1, ..., X_k) := OVP(G, ({w}, V(G) - {w}))
    //   Let T be a module tree with one node V(G)
    //   foreach set X_i do Let ROP(G|X_i) be the i_th child of V(G)
    // return T

    let try_single_vertex = |part: &Part, partition: &Partition, idx: TreeNodeIndex, tree: &mut Vec<TreeNode>| -> bool {
        if let Some(v) = part.try_into_node(partition) {
            tree[idx.index()].kind = Kind::Vertex(v);
            true
        } else { false }
    };

    if try_single_vertex(&p, partition, current, tree) { return; }

    let mut stack = vec![(p, current)];
    while let Some((part, current)) = stack.pop() {
        assert!(part.len() >= 2);

        let w = rop_find_w(graph, &part, partition);
        let subpartition = part.separate_single_vertex(partition, w);

        let mut num_edges = 0;
        ovp(graph, subpartition, partition, |e| { num_edges += e.len() });

        let xs: Vec<_> = subpartition.parts(partition).collect();
        let kind = determine_spine_node_kind(&xs, num_edges);
        tree[current.index()].kind = kind;

        for x in xs {
            let t_x = TreeNodeIndex::new(tree.len());
            tree.push(TreeNode::default());
            tree[current.index()].children.push(t_x);
            if !try_single_vertex(&x, partition, t_x, tree) {
                stack.push((x, t_x));
            }
        }
    }
}


fn rop_find_w(graph: &mut Graph, p: &Part, partition: &mut Partition) -> NodeIndex {
    let mut max = NodeIndex::invalid();
    let mut max_label = 0;
    for u in p.nodes_raw(partition) {
        if graph.degree(u.node) == 0 {
            return u.node;
        }
        if !max.is_valid() || u.label > max_label {
            max = u.node;
            max_label = u.label;
        }
    }
    max
}


pub(crate) mod tree {
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

    #[derive(Debug)]
    pub(crate) struct TreeNode {
        pub(crate) kind: Kind,
        pub(crate) children: Vec<TreeNodeIndex>,
    }

    impl TreeNode {
        pub(crate) fn new(kind: Kind) -> Self {
            Self { kind, children: vec![] }
        }
    }

    impl Default for TreeNode {
        fn default() -> Self {
            Self { kind: Kind::Vertex(NodeIndex::invalid()), children: vec![] }
        }
    }

    pub(crate) fn for_leaves(tree: &[TreeNode], root: TreeNodeIndex, mut f: impl FnMut((TreeNodeIndex, NodeIndex))) {
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if let Kind::Vertex(u) = tree[node.index()].kind {
                assert!(u.is_valid());
                f((node, u));
            } else {
                stack.extend(tree[node.index()].children.iter().copied());
            }
        }
    }
}

mod utils {
    use std::collections::HashMap;
    use crate::algos::tree::{Kind, TreeNode, TreeNodeIndex};
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
    pub(crate) fn try_trivial_quotient(removed: &[(NodeIndex, NodeIndex)], p: SubPartition, partition: &Partition, current: TreeNodeIndex, tree: &mut Vec<TreeNode>) -> Option<(Part, TreeNodeIndex)> {
        let nodes = partition.nodes(&p);
        assert_ne!(nodes[0].part, nodes[1].part, "The subpartition started as [{{v}}, X - {{v}}].");
        if nodes[1].part != nodes[nodes.len() - 1].part {
            return None;
        }
        let v = nodes[0].node;
        let p1 = partition.part_by_index(nodes[1].part);
        let kind = if removed.is_empty() { Parallel } else { Series };
        tree[current.index()].kind = kind;
        let n = tree.len();
        for i in [0, 1] {
            tree.push(TreeNode::new(Vertex(v)));
            tree[current.index()].children.push(TreeNodeIndex::new(n + i));
        }
        assert!(p1.len() > 1, "The case |X| = 2 should be covered earlier. |X| > 2 and |P(G[X], v)| = 2 imply |p1| = |X - {{v}}| > 1.");
        Some((p1, TreeNodeIndex::new(n + 1)))
    }

    pub(crate) fn build_quotient(removed: &mut Vec<(NodeIndex, NodeIndex)>,
                                 p: SubPartition, partition: &Partition,
                                 inner_vertex: NodeIndex) -> (Graph, NodeIndex, Vec<(Part, TreeNodeIndex)>) {
        let mut new_part_ids = HashMap::with_hasher(nohash_hasher::BuildNoHashHasher::<u32>::default());
        let mut ys = vec![];
        let mut n_quotient = NodeIndex::new(0);
        for part in p.part_indices(partition) {
            new_part_ids.insert(part.index() as u32, n_quotient);
            ys.push((partition.part_by_index(part), TreeNodeIndex::invalid()));
            n_quotient = (u32::from(n_quotient) + 1).into();
        }

        let map = |x: NodeIndex| {
            *new_part_ids.get(&(partition.part_by_node(x).index() as u32)).unwrap()
        };

        for (u, v) in removed.iter_mut() {
            *u = map(*u);
            *v = map(*v);
        }
        removed.sort();
        removed.dedup();
        (Graph::from_edges(n_quotient.index(), removed.iter().copied()), map(inner_vertex), ys)
    }

    /// Check if the nodes of the part represent a series or parallel module and return its kind if
    /// that's the case
    pub(crate) fn check_series_parallel(graph: &Graph, part: &Part, partition: &Partition) -> Option<Kind> {
        assert!(part.len() >= 2);
        if graph.edge_count() == 0 { return Some(Parallel); }
        let mut nodes = part.nodes(partition);
        let v0 = nodes.next().unwrap();
        let d0 = graph.degree(v0);
        if (d0 == 0 || d0 == part.len() - 1) && nodes.all(|v| graph.degree(v) == d0) {
            let kind = if d0 == 0 { Parallel } else { Series };
            Some(kind)
        } else {
            None
        }
    }

    /// Assign the given kind to the current node and add all nodes in the part as its children.
    pub(crate) fn add_all_nodes_as_children(kind: Kind, part: Part, partition: &Partition, tree: &mut Vec<TreeNode>, current: TreeNodeIndex) {
        tree.reserve(part.len());
        tree[current.index()].kind = kind;
        let tree_len = tree.len();
        tree[current.index()].children.extend((tree_len..tree_len + part.len()).map(TreeNodeIndex::new));
        for v in part.nodes(partition) {
            tree.push(TreeNode::new(Vertex(v)));
        }
    }
}

#[cfg(test)]
mod test {
    use crate::algos::{chain, Kind};
    use crate::algos::tree::{TreeNode, TreeNodeIndex};
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
            let mut tree = vec![TreeNode::default()];
            chain(&mut graph, NodeIndex::new(0), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode::default()];
            chain(&mut graph, NodeIndex::new(1), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode::default()];
            chain(&mut graph, NodeIndex::new(2), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode::default()];
            chain(&mut graph, NodeIndex::new(3), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode::default()];
            chain(&mut graph, NodeIndex::new(4), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }
    }
}