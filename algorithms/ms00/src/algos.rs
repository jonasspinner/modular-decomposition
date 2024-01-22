use std::collections::HashMap;
use tracing::instrument;
use common::make_index;
use crate::algos::Kind::{Parallel, Prime, Series};
use crate::graph::{Graph, NodeIndex};
use crate::ordered_vertex_partition::ovp;
use crate::partition::{Partition, Part, SubPartition};
#[allow(unused)]
use crate::testing::to_vecs;
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

#[derive(Debug, PartialEq)]
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

fn determine_node_kind(xs: &[Part], num_edges: usize) -> Kind {
    let num_edges = num_edges * 2; // we count edges twice in the following calculations

    let (num_nodes, max_num_intra_cluster_edges) = xs.iter()
        .fold((0, 0), |(n1, n2), x| (n1 + x.len(), n2 + x.len() * x.len()));
    let max_num_edges = num_nodes.pow(2);
    let max_num_inter_cluster_edges = max_num_edges - max_num_intra_cluster_edges;
    assert!(num_edges <= max_num_inter_cluster_edges);

    if num_edges == max_num_inter_cluster_edges {
        Series
    } else if num_edges == 0 {
        Parallel
    } else {
        Prime
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

    let mut stack = vec![(p, current)];
    while let Some((part, current)) = stack.pop() {
        if part.len() == 1 {
            let w = partition.nodes(&part.into())[0].node;
            tree[current.index()].kind = Kind::Vertex(w);
            continue;
        }

        let w = rop_find_w(graph, &part, partition);
        let subpartition = part.singleton(partition, w);
        assert_eq!(subpartition.parts(partition).count(), 2);

        let mut num_edges = 0;
        ovp(graph, subpartition.clone(), partition, |e| { num_edges += e.len() });

        let xs: Vec<_> = subpartition.parts(partition).collect();
        tree[current.index()].kind = determine_node_kind(&xs, num_edges);

        for x in xs {
            let t_x = TreeNodeIndex::new(tree.len());
            tree.push(TreeNode::default());
            tree[current.index()].children.push(t_x);
            stack.push((x, t_x));
        }
    }
}

fn chain(graph: &mut Graph, v: NodeIndex, tree: &mut Vec<TreeNode>, current: TreeNodeIndex) {
    // P := OVP(G, ({v}, V(G) - {v});
    // number the vertices of G in order of their appearance in P
    // return ROP(G).
    let mut partition = Partition::new(graph.node_count());

    partition.refine_forward([v]);
    ovp(graph, partition.full_sub_partition(), &mut partition, |_| {});

    graph.restore_removed_edges();

    trace!("chain:partition {:?}", to_vecs(&partition.full_sub_partition(), &partition));

    let p = partition.merge_all_parts();

    trace!("chain:merge {:?}", p.nodes_raw(&partition).iter().map(|u| (u.node.index(), u.label)).collect::<Vec<_>>());

    rop(graph, p, &mut partition, tree, current);
}


fn for_leaves(tree: &[TreeNode], root: TreeNodeIndex, mut f: impl FnMut((TreeNodeIndex, NodeIndex))) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if let Kind::Vertex(u) = tree[node.index()].kind {
            f((node, u));
        } else {
            stack.extend(tree[node.index()].children.iter().copied());
        }
    }
}

fn build_quotient(removed: &mut Vec<(NodeIndex, NodeIndex)>,
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

#[instrument(skip_all)]
pub(crate) fn modular_decomposition(graph: &mut Graph, p0: Part, partition: &mut Partition, tree: &mut Vec<TreeNode>, current: TreeNodeIndex) {
    // Let v be the lowest-numbered vertex of G
    // if G has only one vertex then return v
    // else
    // let T' be the modular decomposition of G/P(G,v)
    // foreach member Y of P(G, v) do T_Y = UMD(G|Y)
    // return the composition of T' and {T_Y : Y in P(G,v) }

    if graph.node_count() == 0 { return; }

    let mut removed = vec![];

    let mut stack = vec![(p0, current)];
    while let Some((p, current)) = stack.pop() {
        removed.clear();

        let v = p.nodes_raw(partition).iter().min_by_key(|n| n.label).unwrap().node;
        if p.len() == 1 {
            tree[current.index()].kind = Kind::Vertex(v);
            continue;
        }

        let p = p.singleton(partition, v);
        ovp(graph, p.clone(), partition, |e| { removed.extend(e.iter().map(|&(u, v, _)| (u, v))) });

        let (mut quotient, v_q, mut ys) = build_quotient(&mut removed, p.clone(), partition, v);

        chain(&mut quotient, v_q, tree, current);

        for_leaves(tree, current, |(j, v)| {
            ys[v.index()].1 = j;
        });

        stack.extend(ys);
    }
}


#[cfg(test)]
mod test {
    use crate::algos::{chain, Kind, TreeNode, TreeNodeIndex};
    use crate::graph::{Graph, NodeIndex};
    use crate::partition::{Part, Partition};
    use crate::testing::ted08_test0_graph;

    #[test]
    fn modular_decomposition() {
        let mut graph = ted08_test0_graph();

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