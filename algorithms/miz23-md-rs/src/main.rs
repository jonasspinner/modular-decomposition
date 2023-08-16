mod forest;
mod mdtree;
mod compute;
mod graph;
mod set;

use std::vec::Vec;
use crate::forest::Forest;
use crate::graph::{Graph, VertexId};


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ModuleKind {
    Prime,
    Parallel,
    Series,
    Leaf(VertexId),
}

fn example_tree() -> Forest<ModuleKind> {
    let mut tree = Forest::new();

    let prime_1 = tree.create_node(ModuleKind::Prime);
    let prime_2 = tree.create_node(ModuleKind::Prime);
    let series_1 = tree.create_node(ModuleKind::Series);
    let series_2 = tree.create_node(ModuleKind::Series);
    let series_3 = tree.create_node(ModuleKind::Series);
    let series_4 = tree.create_node(ModuleKind::Series);
    let parallel_1 = tree.create_node(ModuleKind::Parallel);
    let parallel_2 = tree.create_node(ModuleKind::Parallel);
    let parallel_3 = tree.create_node(ModuleKind::Parallel);

    let a = tree.create_node(ModuleKind::Leaf(0.into()));
    let b = tree.create_node(ModuleKind::Leaf(1.into()));
    let c = tree.create_node(ModuleKind::Leaf(2.into()));
    let d = tree.create_node(ModuleKind::Leaf(3.into()));
    let e = tree.create_node(ModuleKind::Leaf(4.into()));
    let f = tree.create_node(ModuleKind::Leaf(5.into()));
    let g = tree.create_node(ModuleKind::Leaf(6.into()));
    let h = tree.create_node(ModuleKind::Leaf(7.into()));
    let i = tree.create_node(ModuleKind::Leaf(8.into()));
    let j = tree.create_node(ModuleKind::Leaf(9.into()));
    let k = tree.create_node(ModuleKind::Leaf(10.into()));
    let l = tree.create_node(ModuleKind::Leaf(11.into()));
    let m = tree.create_node(ModuleKind::Leaf(12.into()));
    let n = tree.create_node(ModuleKind::Leaf(13.into()));
    let o = tree.create_node(ModuleKind::Leaf(14.into()));
    let p = tree.create_node(ModuleKind::Leaf(15.into()));
    let q = tree.create_node(ModuleKind::Leaf(16.into()));
    let r = tree.create_node(ModuleKind::Leaf(17.into()));
    let x = tree.create_node(ModuleKind::Leaf(18.into()));

    tree.add_child(prime_1, r);
    tree.add_child(prime_1, series_1);
    tree.add_child(series_1, parallel_1);
    tree.add_child(parallel_1, n);
    tree.add_child(parallel_1, p);
    tree.add_child(series_1, m);
    tree.add_child(prime_1, q);
    tree.add_child(prime_1, a);
    tree.add_child(prime_1, parallel_2);
    tree.add_child(parallel_2, b);
    tree.add_child(parallel_2, j);
    tree.add_child(parallel_2, series_2);
    tree.add_child(series_2, k);
    tree.add_child(series_2, l);
    tree.add_child(parallel_2, prime_2);
    tree.add_child(prime_2, series_3);
    tree.add_child(series_3, h);
    tree.add_child(series_3, f);
    tree.add_child(series_3, g);
    tree.add_child(prime_2, i);
    tree.add_child(prime_2, parallel_3);
    tree.add_child(parallel_3, series_4);
    tree.add_child(series_4, e);
    tree.add_child(series_4, d);
    tree.add_child(parallel_3, c);
    tree.add_child(prime_2, x);
    tree
}

fn example_graph() -> Graph {
    let mut graph = Graph::new(18);
    graph.add_edge(0, 4);
    graph.add_edge(1, 3);
    graph.add_edge(1, 4);
    graph.add_edge(1, 5);
    graph.add_edge(2, 3);
    graph.add_edge(2, 4);
    graph.add_edge(2, 5);
    graph.add_edge(3, 4);
    graph.add_edge(3, 5);
    graph.add_edge(4, 5);
    graph.add_edge(5, 14);
    graph.add_edge(5, 15);
    graph.add_edge(5, 16);
    graph.add_edge(5, 17);
    graph.add_edge(5, 11);
    graph.add_edge(5, 12);
    graph.add_edge(5, 10);
    graph.add_edge(5, 13);
    graph.add_edge(5, 6);
    graph.add_edge(5, 7);
    graph.add_edge(5, 8);
    graph.add_edge(5, 9);
    graph.add_edge(8, 9);
    graph.add_edge(10, 11);
    graph.add_edge(10, 12);
    graph.add_edge(10, 13);
    graph.add_edge(11, 12);
    graph.add_edge(11, 13);
    graph.add_edge(12, 13);
    graph.add_edge(13, 14);
    graph.add_edge(13, 15);
    graph.add_edge(13, 16);
    graph.add_edge(14, 15);
    graph.add_edge(14, 17);
    graph.add_edge(15, 17);
    graph.add_edge(16, 17);
    graph
}


fn main() {
    // let tree = example_tree()
    // println!("{:?}", tree.pre_order_node_indices(prime_1).collect::<Vec<_>>());

    let mut graph = Graph::new(5);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(2, 3);
    graph.add_edge(4, 1);
    graph.add_edge(4, 2);

    let graph = example_graph();

    let result = compute::compute(&graph);

}
