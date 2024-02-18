mod modular_partition;
mod lexbfs;

mod splitters;
mod list;
mod partition_refinement;
mod modular_functions;
mod factorizing_permutation;

use petgraph::dot::{Config, Dot};
use petgraph::graph::UnGraph;
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


#[allow(non_snake_case)]
fn main() {
    let graph = small_graph();
    let _partition = vec![vec![5, 6], vec![0, 1, 2, 3, 4, 7, 8, 9, 10]];
    //let partition = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]];

    let partition = vec![vec![3, 4], vec![0, 1, 2, 5, 6, 7, 8, 9, 10]];

    let Q = modular_partition(&partition, &graph);

    println!("{:?}", partition);
    println!("{:?}", Q);

    /*
    let mut partition = StandardPartitionDataStructure::new(8);
    partition.refine([2, 3, 4, 5]);
    partition.refine([4, 5, 6, 7]);

    fs::write(format!("partition.dot"), partition.to_dot()).unwrap();
    */

    //println!("{:?}", Dot::with_config(&graph, &[Config::NodeIndexLabel, Config::EdgeNoLabel]));


    let mut graph = small_graph();

    let mut partition = crate::partition_refinement::Partition::new(graph.node_count());
    for u in graph.node_indices() {
        partition.refine(graph.neighbors(u).map(|v| v.index() as _));
    }
    println!("{:?}", partition);

    let mut keep = vec![false; graph.node_count()];
    for mut part in partition.parts() {
        let u = part.next().unwrap();
        keep[u as usize] = true;
    }
    graph.retain_nodes(|_, u| { keep[u.index()] });

    let mut partition = partition_refinement::Partition::new(graph.node_count());
    for u in graph.node_indices() {
        partition.refine(graph.neighbors(u).chain(std::iter::once(u)).map(|v| v.index() as _));
    }
    println!("{:?}", partition);

    let mut keep = vec![false; graph.node_count()];
    for mut part in partition.parts() {
        let u = part.next().unwrap();
        keep[u as usize] = true;
    }
    graph.retain_nodes(|_, u| { keep[u.index()] });

    println!("{:?}", Dot::with_config(&graph, &[Config::NodeIndexLabel, Config::EdgeNoLabel]));
}


mod test {
    #[test]
    fn test() {
        struct PartitioningIterator<'a, T, P> {
            elements: &'a mut [T],
            end: &'a mut usize,
            i: usize,
            j: usize,
            predicate: P,
        }
        impl<'a, T, P> PartitioningIterator<'a, T, P> {
            fn new(elements: &'a mut [T], end: &'a mut usize, predicate: P) -> Self {
                Self { elements, end, i: 0, j: 0, predicate }
            }
        }
        impl<'a, T, P> Iterator for PartitioningIterator<'a, T, P>
            where P: FnMut(&T) -> bool
        {
            type Item = &'a T;

            fn next(&mut self) -> Option<Self::Item> {
                todo!()
            }
        }

        fn partition_in_place<T, P>(elements: &mut [T], mut predicate: P) -> usize
            where P: FnMut(&T) -> bool {
            let mut j = 0;
            for i in 0..elements.len() {
                if predicate(&elements[i]) {
                    elements.swap(i, j);
                    j += 1;
                }
            }
            j
        }

        let mut v = [0, 1, 2, 3, 4];
        let mut end = v.len();

        println!("{:?}", &v[..end]);
        end = partition_in_place(&mut v[..end], |x| { *x % 2 == 0 });
        println!("{:?}", &v[..end]);
        end = partition_in_place(&mut v[..end], |x| { *x >= 2 });
        println!("{:?}", &v[..end]);
        end = partition_in_place(&mut v[..end], |x| { *x <= 2 });
        println!("{:?}", &v[..end]);
    }
}