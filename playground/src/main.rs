mod modular_partition;
mod standard_partition_data_structure;

use std::fmt::{Debug};
use std::error::Error;
use petgraph::adj::{IndexType};
use petgraph::graph::{UnGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeCount};
use crate::modular_partition::modular_partition;

fn small_graph() -> UnGraph<(), ()> {
    let mut graph = UnGraph::<(), ()>::new_undirected();
    graph.extend_with_edges([
        (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 5), (4, 6),
        (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 7), (6, 8), (6, 9), (6, 10),
        (7, 8), (7, 9), (7, 10),
        (8, 9), (8, 10),
    ]);
    graph
}


fn main() {
    let graph = small_graph();
    let partition = vec![vec![5, 6], vec![0, 1, 2, 3, 4, 7, 8, 9, 10]];
    //let partition = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]];

    let partition = vec![vec![3, 4], vec![0, 1, 2, 5, 6, 7, 8, 9, 10]];

    let (time, Q) = modular_partition(partition.clone(), &graph);

    println!("{:?}", partition);
    println!("{:?}", Q);

    /*
    let mut partition = StandardPartitionDataStructure::new(8);
    partition.refine([2, 3, 4, 5]);
    partition.refine([4, 5, 6, 7]);

    fs::write(format!("partition.dot"), partition.to_dot()).unwrap();
    */
}
