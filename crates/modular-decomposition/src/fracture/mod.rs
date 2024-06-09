mod factorizing_permutation;

use crate::fracture::factorizing_permutation::{factorizing_permutation, Permutation};
use crate::md_tree::{MDTree, ModuleKind, NodeIndex, NullGraphError};
use crate::segmented_stack::SegmentedStack;
use petgraph::graph::DiGraph;
use petgraph::visit::{GraphProp, IntoNeighbors, NodeCompactIndexable};
use petgraph::Undirected;
use tracing::{info, instrument};

/// Computes the modular decomposition of the graph.
///
/// # Errors
///
/// Returns a `NullGraphError` if the input graph does not contain any nodes or edges.
#[instrument(skip_all)]
pub fn modular_decomposition<G>(graph: G) -> Result<MDTree<G::NodeId>, NullGraphError>
where
    G: NodeCompactIndexable + IntoNeighbors + GraphProp<EdgeType = Undirected>,
{
    let n = graph.node_bound();
    if n == 0 {
        return Err(NullGraphError);
    }
    if n == 1 {
        let mut tree = DiGraph::new();
        tree.add_node(ModuleKind::Node(graph.from_index(0)));
        return MDTree::from_digraph(tree);
    }

    let p = factorizing_permutation(graph);

    let (mut op, mut cl, mut lc, mut uc) = init_parenthesizing(n);

    build_parenthesizing(graph, &mut op, &mut cl, &mut lc, &mut uc, &p);

    remove_non_module_dummy_nodes(&mut op, &mut cl, &lc, &uc);

    create_consecutive_twin_nodes(&mut op, &mut cl, &lc, &uc);

    let tree = build_tree(graph, &op, &cl, &p);

    info!(number_of_nodes = tree.node_count(), number_of_inner_nodes = tree.node_count() - n);

    MDTree::from_digraph(tree)
}

pub(crate) fn init_parenthesizing(n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut op = vec![0; n];
    op[0] = 1;
    let mut cl = vec![0; n];
    cl[n - 1] = 1;
    let lc = (0..n - 1).map(|i| i as u32).collect();
    let uc = (1..n).map(|i| i as u32).collect();
    (op, cl, lc, uc)
}

#[instrument(skip_all)]
pub(crate) fn build_parenthesizing<G>(
    graph: G,
    op: &mut [u32],
    cl: &mut [u32],
    lc: &mut [u32],
    uc: &mut [u32],
    p: &Permutation,
) where
    G: NodeCompactIndexable + IntoNeighbors + GraphProp<EdgeType = Undirected>,
{
    fn next_unequal<'a>(
        mut a: impl Iterator<Item = &'a u32>,
        mut b: impl Iterator<Item = &'a u32>,
        g: impl Fn(u32, u32) -> u32,
    ) -> Option<u32> {
        loop {
            match (a.next(), b.next()) {
                (Some(i), Some(j)) => {
                    if i != j {
                        break Some(g(*i, *j));
                    }
                }
                (Some(i), None) => break Some(*i),
                (None, Some(j)) => break Some(*j),
                (None, None) => break None,
            }
        }
    }

    let first =
        |a: &[u32], b: &[u32]| -> Option<usize> { next_unequal(a.iter(), b.iter(), u32::min).map(|i| i as usize) };
    let last = |a: &[u32], b: &[u32]| -> Option<usize> {
        next_unequal(a.iter().rev(), b.iter().rev(), u32::max).map(|i| i as usize)
    };

    // To find the first and last vertices (in respect to the permutation) which are
    // not in both neighborhoods of consecutive vertices, the sorted
    // neighborhoods are traversed from the beginning and end, stopping when
    // they first disagree. Every neighborhood is traversed at most twice, once
    // for each adjacent vertex in the permutation. This results in a total O(n
    // + m) running time.

    let add_neighbor_positions = |positions: &mut Vec<u32>, i: usize| {
        positions.clear();
        positions.extend(
            graph
                .neighbors(graph.from_index(p[i].index()))
                .map(|v| p.position(NodeIndex::new(graph.to_index(v))) as u32),
        );
        positions.sort_unstable();
    };

    let mut pos_j0 = Vec::new();
    let mut pos_j1 = Vec::new();
    add_neighbor_positions(&mut pos_j1, 0);

    let has_edge = |u: NodeIndex, v: NodeIndex| -> bool {
        graph.neighbors(graph.from_index(u.index())).any(|w| graph.to_index(w) == v.index())
    };

    let n = p.len();
    for j in 0..n - 1 {
        std::mem::swap(&mut pos_j0, &mut pos_j1);
        add_neighbor_positions(&mut pos_j1, j + 1);

        let mut lc_idx = first(&pos_j0, &pos_j1);
        let mut uc_idx = if lc_idx.is_some() { last(&pos_j0, &pos_j1) } else { None };
        lc_idx = lc_idx.filter(|i| *i < j);
        uc_idx = uc_idx.filter(|i| *i > j + 1);

        if let Some(i) = lc_idx {
            assert!(i < j);
            debug_assert_ne!(has_edge(p[i], p[j]), has_edge(p[i], p[j + 1]));
            op[i] += 1;
            cl[j] += 1;
            lc[j] = i as u32;
        }
        if let Some(i) = uc_idx {
            assert!(i > j + 1 && i < n);
            debug_assert_ne!(has_edge(p[i], p[j]), has_edge(p[i], p[j + 1]));
            op[j + 1] += 1;
            cl[i] += 1;
            uc[j] = i as u32;
        }
    }
}

#[instrument(skip_all)]
pub(crate) fn remove_non_module_dummy_nodes(op: &mut [u32], cl: &mut [u32], lc: &[u32], uc: &[u32]) {
    #[derive(Clone, Debug)]
    struct Info {
        first_vertex: u32,
        last_vertex: u32,
        first_cutter: u32,
        last_cutter: u32,
    }
    impl Info {
        fn new(first_vertex: u32, last_vertex: u32, first_cutter: u32, last_cutter: u32) -> Self {
            Self { first_vertex, last_vertex, first_cutter, last_cutter }
        }
    }
    fn handle_vertex_node(position: usize) -> Info {
        let pos = position as u32;
        Info::new(pos, pos, pos, pos)
    }

    let handle_inner_node = |op: &mut [u32], cl: &mut [u32], nodes: &[Info]| -> Info {
        let k = nodes.len();

        debug_assert!((0..k - 1).all(|i| nodes[i].last_vertex + 1 == nodes[i + 1].first_vertex));

        let first_vertex = nodes[0].first_vertex;
        let last_vertex = nodes[k - 1].last_vertex;
        let first_cutter = u32::min(
            first_vertex,
            nodes[0..k - 1]
                .iter()
                .map(|n| lc[n.last_vertex as usize])
                .chain(nodes.iter().map(|n| n.first_cutter))
                .min()
                .unwrap_or(first_vertex),
        );
        let last_cutter = u32::max(
            last_vertex,
            nodes[0..k - 1]
                .iter()
                .map(|n| uc[n.last_vertex as usize])
                .chain(nodes.iter().map(|n| n.last_cutter))
                .max()
                .unwrap_or(last_vertex),
        );

        let info = Info::new(first_vertex, last_vertex, first_cutter, last_cutter);
        // Check if we are at a genuine node, i.e. the node of the fracture tree
        // represents a module.
        if first_vertex < last_vertex && first_vertex <= first_cutter && last_cutter <= last_vertex {
            return info;
        }
        // As this is a post-order traversal, we already walked the parenthesis up to
        // the last vertex and we can change the parenthesis expression without
        // affecting the remaining traversal.
        // Removing the pair parenthesis deletes the node from the fracture tree.
        op[info.first_vertex as usize] -= 1;
        cl[info.last_vertex as usize] -= 1;
        info
    };

    let mut s = SegmentedStack::with_capacity(op.len());
    // We do a post-order traversal of the fracture tree induced by the
    // parenthesized permutation.
    for j in 0..op.len() {
        s.extend(op[j] as _);
        s.add(handle_vertex_node(j));
        for _ in 0..cl[j] {
            let info = handle_inner_node(op, cl, s.pop());
            s.add(info);
        }
    }
    assert!(s.is_empty());
    debug_assert_eq!(op.iter().sum::<u32>(), cl.iter().sum());
}

#[instrument(skip_all)]
pub(crate) fn create_consecutive_twin_nodes(op: &mut [u32], cl: &mut [u32], lc: &[u32], uc: &[u32]) {
    let handle_inner_node = |op: &mut [u32], cl: &mut [u32], nodes: &[(u32, u32)]| -> (u32, u32) {
        if nodes.len() == 1 {
            return nodes[0];
        }
        let mut add_brackets = |start: &mut Option<u32>, end: u32| {
            if let Some(start) = start.take() {
                op[start as usize] += 1;
                cl[end as usize] += 1;
            }
        };
        let mut start = None;
        for (&(first_vertex, end), &(_, last_vertex)) in nodes.iter().zip(nodes.iter().skip(1)) {
            if first_vertex <= lc[end as usize] && uc[end as usize] <= last_vertex {
                start = start.or(Some(first_vertex));
            } else {
                add_brackets(&mut start, end)
            }
        }
        let ((first_vertex, _), (_, last_vertex)) = (nodes[0], nodes[nodes.len() - 1]);
        add_brackets(&mut start, last_vertex);
        (first_vertex, last_vertex)
    };

    let mut s = SegmentedStack::with_capacity(op.len());
    for j in 0..op.len() {
        s.extend(op[j] as _);
        s.add((j as u32, j as u32));
        for _ in 0..cl[j] {
            let info = handle_inner_node(op, cl, s.pop());
            s.add(info);
        }
    }
    assert!(s.is_empty());
    debug_assert_eq!(op.iter().sum::<u32>(), cl.iter().sum());
}

#[instrument(skip_all)]
pub(crate) fn build_tree<G>(graph: G, op: &[u32], cl: &[u32], p: &Permutation) -> DiGraph<ModuleKind<G::NodeId>, ()>
where
    G: NodeCompactIndexable + IntoNeighbors + GraphProp<EdgeType = Undirected>,
{
    // Do a post-order traversal of the fracture tree (represented by op, cl and p),
    // skipping nodes with a single child, determining the module type of the
    // others and adding it to the modular decomposition tree.
    let n = graph.node_bound();

    let degrees: Vec<u32> =
        (0..graph.node_bound()).map(|i| graph.from_index(i)).map(|u| graph.neighbors(u).count() as _).collect();

    let handle_vertex_node =
        |t: &mut DiGraph<ModuleKind<G::NodeId>, ()>, x: NodeIndex| -> (NodeIndex, petgraph::graph::NodeIndex) {
            (x, t.add_node(ModuleKind::Node(graph.from_index(x.index()))))
        };

    let mut marked = vec![0; n];
    let mut gen = 0;

    let mut determine_node_kind = |t: &mut DiGraph<ModuleKind<G::NodeId>, ()>,
                                   nodes: &[(NodeIndex, petgraph::graph::NodeIndex)]|
     -> (NodeIndex, ModuleKind<G::NodeId>) {
        // Calculate the degrees between children of a module.
        // Every module keeps a graph node as a representative. Mark the representatives
        // of the children. For each representative, iterate over its neighbors.
        // If a neighbor is also a representative, increment the degree.

        gen += 1;
        for (x, _) in nodes {
            marked[x.index()] = gen;
        }

        let quotient_degree = |x: NodeIndex| -> usize {
            graph
                .neighbors(graph.from_index(x.index()))
                .filter(|w: &G::NodeId| marked[graph.to_index(*w)] == gen)
                .count()
        };
        // Choosing the node with the minimum degree ensures a linear total running
        // time. Let n be the number of leaves corresponding to the vertices of
        // the graph. In the worst case, every inner node has exactly two
        // children and we have at most n - 1 inner nodes. For each inner node,
        // we can choose one of the leaves of its subtree, such that every inner
        // node gets assigned a unique vertex as a representative. Leaving one vertex
        // unassigned in the tree. Every inner node has work in order of the number of
        // its children and the degree of its representative in the original
        // graph. The work for the children is O(n) in total and the work for
        // the representatives is O(m) in total. By choosing the representatives
        // to be the vertex with the lowest degree in the subtree, the work is
        // bounded by the previous assignment of representatives. This proves a total
        // running time of O(n + m) with this strategy.
        let &(y, _) = nodes.iter().min_by_key(|(x, _)| degrees[x.index()]).unwrap();

        // It is enough to look at a single vertex and its degree in the quotient graph.
        // Let y be one of the vertices and k be the number of vertices.
        // If the quotient has only two vertices, it is either series or parallel and we
        // can distinguish these case by calculating the degree of one of the
        // vertices. Now assume there are at least three vertices. If y does not
        // have degree 0 or k - 1, the graph cannot be series or parallel. If y
        // does have degree 0, then y and the other vertices would be the
        // children of a parallel node. It y does have degree k - 1,
        // then they would be the children of a series node.
        let d0 = quotient_degree(y);
        let kind = if d0 == 0 {
            ModuleKind::Parallel
        } else if d0 == (nodes.len() - 1) {
            ModuleKind::Series
        } else {
            ModuleKind::Prime
        };

        if kind != ModuleKind::Prime {
            debug_assert!(nodes.iter().map(|(y, _)| quotient_degree(*y)).all(|d| d == d0));
            debug_assert!(nodes.iter().all(|(_, u)| t[*u] != kind));
        }

        (y, kind)
    };

    let mut handle_inner_node = |t: &mut DiGraph<ModuleKind<G::NodeId>, ()>,
                                 nodes: &[(NodeIndex, petgraph::graph::NodeIndex)]|
     -> (NodeIndex, petgraph::graph::NodeIndex) {
        if nodes.len() == 1 {
            return nodes[0];
        }
        assert!(nodes.len() > 1);
        let (y, kind) = determine_node_kind(t, nodes);
        let idx = t.add_node(kind);
        for (_, u) in nodes {
            t.add_edge(idx, *u, ());
        }
        (y, idx)
    };

    let mut t = DiGraph::with_capacity(n + 256, n + 256);

    // We keep track of (y, u) where u is a node in the tree and y is a
    // representative of the vertices in the subtree of u, i.e. one of the its
    // leaves.
    let mut s = SegmentedStack::with_capacity(n);
    // We do a post-order traversal of the fracture tree induced by the
    // parenthesized permutation.
    for (j, x) in p.iter().enumerate() {
        s.extend(op[j] as _);
        s.add(handle_vertex_node(&mut t, x));
        for _ in 0..cl[j] {
            let (x, idx) = handle_inner_node(&mut t, s.pop());
            s.add((x, idx));
        }
    }
    assert!(s.is_empty());
    t
}

#[cfg(test)]
mod test {
    use crate::fracture::factorizing_permutation::Permutation;
    use crate::fracture::{
        build_parenthesizing, create_consecutive_twin_nodes, init_parenthesizing, modular_decomposition,
        remove_non_module_dummy_nodes,
    };
    use petgraph::dot::{Config, Dot};
    use petgraph::graph::UnGraph;
    use petgraph::visit::NodeIndexable;

    fn print_parenthesis(op: &[u32], cl: &[u32], permutation: &Permutation) {
        let n = op.len();
        for j in 0..n {
            for _ in 0..op[j] {
                print!("(");
            }
            print!("{}", permutation[j].index());
            for _ in 0..cl[j] {
                print!(")");
            }
        }
        println!();
    }

    #[test]
    fn parenthesized_factorizing_permutation() {
        let graph: UnGraph<(), ()> = UnGraph::from_edges([
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
            (5, 6),
        ]);
        let n = graph.node_bound();
        let permutation = Permutation::new_identity(n);

        let (mut op, mut cl, mut lc, mut uc) = init_parenthesizing(n);

        build_parenthesizing(&graph, &mut op, &mut cl, &mut lc, &mut uc, &permutation);

        assert_eq!(op, [3, 0, 0, 0, 2, 1, 0]);
        assert_eq!(cl, [0, 1, 0, 0, 1, 2, 2]);

        print_parenthesis(&op, &cl, &permutation);

        remove_non_module_dummy_nodes(&mut op, &mut cl, &lc, &uc);

        assert_eq!(op, [3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(cl, [0, 1, 0, 0, 0, 0, 2]);

        print_parenthesis(&op, &cl, &permutation);

        create_consecutive_twin_nodes(&mut op, &mut cl, &lc, &uc);

        assert_eq!(op, [5, 0, 0, 0, 0, 0, 0]);
        assert_eq!(cl, [0, 2, 0, 1, 0, 0, 2]);

        print_parenthesis(&op, &cl, &permutation);

        let tree = modular_decomposition(&graph).map(|tree| tree.into_digraph()).unwrap_or_default();
        println!("{:?}", Dot::with_config(&tree, &[Config::EdgeNoLabel]));
    }
}
