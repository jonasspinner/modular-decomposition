use std::collections::HashMap;
use common::make_index;
use crate::seq::algos::Kind::{Parallel, Prime, Series, UnderConstruction};
use crate::seq::graph::{Graph, NodeIndex};
use crate::seq::ordered_vertex_partition::ovp;
use crate::seq::partition::{Partition, SubPartition, Node, Part};
use crate::seq::testing::to_vecs;
use crate::trace;

fn rop_find_w(graph: &mut Graph, p: &Part, partition: &mut Partition) -> NodeIndex {
    let mut max = NodeIndex::invalid();
    let mut max_label = 0;
    for u in p.nodes_raw(partition) {
        if graph.incident_edges(u.node).next().is_none() {
            trace!("isolated vertex {}", u.node.index());
            return u.node;
        }
        if !max.is_valid() || u.label > max_label {
            max = u.node;
            max_label = u.label;
        }
    }
    trace!("max label vertex {} with label {}", max.index(), max_label);
    max
}

make_index!(pub(crate) TreeNodeIndex);

#[derive(Debug)]
pub(crate) enum Kind {
    Prime,
    Series,
    Parallel,
    Vertex(NodeIndex),
    UnderConstruction,
}

#[derive(Debug)]
pub(crate) struct TreeNode {
    pub(crate) kind: Kind,
    pub(crate) children: Vec<TreeNodeIndex>,
}

impl Default for TreeNode {
    fn default() -> Self {
        Self { kind: Kind::UnderConstruction, children: vec![] }
    }
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

    if p.len() == 1 {
        let w = partition.nodes(&p.into())[0].node;
        tree[current.index()].kind = Kind::Vertex(w);
        return;
    }

    let mut removed = vec![];

    let w = rop_find_w(graph, &p, partition);
    let p = p.singleton(partition, w);

    trace!("rop:partition:0 {:?}", to_vecs(&p.clone(), partition));

    ovp::<false>(graph, p.clone(), partition, &mut removed);

    trace!("rop:partition:1 {:?}", to_vecs(&p.clone(), partition));
    trace!("rop:removed:len {}", removed.len());

    let xs: Vec<_> = p.parts(partition).collect();
    trace!("rop:parts:len {}", xs.len());
    let series_case = xs.len() * (xs.len() - 1) / 2;
    let kind = if removed.len() == series_case {
        Series
    } else if removed.is_empty() { Parallel } else { Prime };
    trace!("rop:kind {:?}", kind);
    tree[current.index()].kind = kind;
    for x in xs {
        let t_x = TreeNodeIndex::new(tree.len());
        tree.push(TreeNode { kind: UnderConstruction, children: vec![] });
        tree[current.index()].children.push(t_x);
        rop(graph, x, partition, tree, t_x);
    }
}

fn chain(graph: &mut Graph, v: NodeIndex, tree: &mut Vec<TreeNode>, current: TreeNodeIndex) {
    // P := OVP(G, ({v}, V(G) - {v});
    // number the vertices of G in order of their appearance in P
    // return ROP(G).
    ;
    let mut partition = Partition::new(graph.node_count());
    let p = SubPartition::new(&partition);

    partition.refine_forward([v]);
    let mut graph_clone = graph.clone();
    ovp::<false>(&mut graph_clone, p.clone(), &mut partition, &mut Vec::new());

    //graph.restore_removed_edges(); FIXME

    trace!("chain:partition {:?}", to_vecs(&p.clone(), &partition));

    let p = partition.merge_all_parts();

    trace!("chain:merge {:?}", p.nodes_raw(&partition).iter().map(|u| (u.node.index(), u.label)).collect::<Vec<_>>());

    rop(graph, p, &mut partition, tree, current);
}


fn collect_leaves(tree: &[TreeNode], root: TreeNodeIndex) -> Vec<(TreeNodeIndex, NodeIndex)> {
    if let Kind::Vertex(u) = tree[root.index()].kind {
        return vec![(root, u)];
    }
    let mut res = vec![];
    for &child in &tree[root.index()].children {
        res.append(&mut collect_leaves(tree, child));
    }
    res
}

pub(crate) fn modular_decomposition(graph: &mut Graph, p: Part, partition: &mut Partition, tree: &mut Vec<TreeNode>, current: TreeNodeIndex) {
    trace!("modular_decomposition:parts:0  {:?}", to_vecs(&p.clone().into(), partition));
    // Let v be the lowest-numbered vertex of G
    // if G has only one vertex then return v
    // else
    // let T' be the modular decomposition of G/P(G,v)
    // foreach member Y of P(G, v) do T_Y = UMD(G|Y)
    // return the composition of T' and {T_Y : Y in P(G,v) }
    let mut removed = vec![];

    let v = p.nodes_raw(partition).iter().min_by_key(|n| n.label).unwrap().node;
    if p.len() == 1 {
        tree[current.index()].kind = Kind::Vertex(v);
        return;
    }


    let p = p.singleton(partition, v);
    ovp::<true>(graph, p.clone(), partition, &mut removed);


    let mut new_part_ids = HashMap::new();
    let mut ys_ = vec![];
    let mut i = NodeIndex::new(0);
    for part in p.part_indices(partition) {
        new_part_ids.insert(part, i);
        ys_.push((partition.part_by_index(part), TreeNodeIndex::invalid()));
        i = (u32::from(i) + 1).into();
    }

    for (u, v) in &mut removed {
        *u = *new_part_ids.get(&partition.part_by_node(*u)).unwrap();
        *v = *new_part_ids.get(&partition.part_by_node(*v)).unwrap();
    }
    removed.sort();
    removed.dedup();

    let mut quotient = Graph::from_edges(i.index(), removed);
    let v_q = *new_part_ids.get(&partition.part_by_node(v)).unwrap();

    chain(&mut quotient, v_q, tree, current);

    for (j, v) in collect_leaves(tree, current) {
        assert!(matches!(tree[j.index()].kind, Kind::Vertex(_)));
        ys_[v.index()].1 = j;
        tree[j.index()].kind = Kind::UnderConstruction;
    }

    trace!("modular_decomposition:parts:1 {:?}", ys_.iter().map(|(y, j)| (j.index(), y.nodes(partition).map(|u| u.index()).collect::<Vec<_>>())).collect::<Vec<_>>());


    for (y, j) in ys_ {
        assert!(y.len() < p.len());
        modular_decomposition(graph, y, partition, tree, j);
    }
}


mod test {
    use std::slice::SliceIndex;
    use crate::seq::algos::{chain, Kind, TreeNode, TreeNodeIndex};
    use crate::seq::graph::{Graph, NodeIndex};
    use crate::seq::partition::{Part, Partition};
    use crate::seq::testing::ted08_test0_graph;

    #[test]
    fn modular_decomposition() {
        let mut graph = ted08_test0_graph();
        let original_graph = graph.clone();

        let mut partition = Partition::new(graph.node_count());
        let p = Part::new(&partition);
        let mut tree = vec![];

        let root = TreeNodeIndex::new(0);
        tree.push(TreeNode { kind: Kind::UnderConstruction, children: vec![] });
        super::modular_decomposition(&mut graph, p, &mut partition, &mut tree, root);

        println!("digraph {{");
        for i in 0..tree.len() {
            match tree[i].kind {
                Kind::Vertex(u) => {
                    println!("  {} [label=\"{}:{:?}\"]", i, i, u.index());
                }
                _ => {
                    println!("  {} [label=\"{}:{:?}\"]", i, i, tree[i].kind);
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
            let mut tree = vec![TreeNode { kind: Kind::UnderConstruction, children: vec![] }];
            chain(&mut graph, NodeIndex::new(0), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode { kind: Kind::UnderConstruction, children: vec![] }];
            chain(&mut graph, NodeIndex::new(1), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode { kind: Kind::UnderConstruction, children: vec![] }];
            chain(&mut graph, NodeIndex::new(2), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode { kind: Kind::UnderConstruction, children: vec![] }];
            chain(&mut graph, NodeIndex::new(3), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }

        {
            let mut graph = graph.clone();
            let mut tree = vec![TreeNode { kind: Kind::UnderConstruction, children: vec![] }];
            chain(&mut graph, NodeIndex::new(4), &mut tree, TreeNodeIndex::new(0));
            println!("{:?}", tree);
        }
    }
}