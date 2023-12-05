use std::collections::{HashSet, VecDeque};
use std::mem::replace;
use petgraph::{Graph, Undirected};
use petgraph::adj::IndexType;
use petgraph::graph::NodeIndex;


#[allow(non_snake_case, dead_code)]
fn overlap(A: impl IntoIterator<Item=u32>, B: impl IntoIterator<Item=u32>) -> bool {
    let A: HashSet<u32> = A.into_iter().collect();
    let B: HashSet<u32> = B.into_iter().collect();
    return !A.is_disjoint(&B) && !A.is_subset(&B) && !B.is_subset(&A);
}


#[allow(non_snake_case)]
pub(crate) fn modular_partition<N, E, Ix>(partition: &Vec<Vec<u32>>, graph: &Graph<N, E, Undirected, Ix>) -> Vec<Vec<u32>>
    where Ix: IndexType {
    let (z_index, _) = partition.iter().enumerate().fold(None, |max, (i, part)| {
        match max {
            None => Some((i, part.len())),
            Some((_, len)) if len < part.len() => Some((i, part.len())),
            Some(max) => Some(max)
        }
    }).unwrap();

    // Let Z be the largest part of P
    // Q <- P; K <- { Z }; L <- { X | X ≠ Z, X ∈ P }
    let mut Q = partition.clone();
    let mut L = partition.clone();
    let mut K = VecDeque::new();
    K.push_front(L.remove(z_index));

    // While L ∪ K ≠ ∅
    loop {
        println!("K = {:?}, L = {:?}, Q = {:?}", K, L, Q);
        // If there exists X ∈ L
        let (S, X) = if let Some(X) = L.pop() {
            // S <- X and L <- L \ { X }
            (X.clone(), X)
        } else if let Some(X) = K.pop_front() {
            // Let X be the first part of K and x arbitrarily selected in X
            let x = *X.last().unwrap();
            // S <- { x } and K <- K \ { X }
            (vec![x], X)
        } else { break; };

        println!("S = {:?}, X = {:?}", S, X);

        // For each x ∈ S
        for &x in &S {
            let N_x = graph.neighbors(NodeIndex::new(x as usize)).map(|i| i.index() as u32).collect::<HashSet<_>>();
            println!("\tx = {}, N(x) = {:?}", x, N_x.iter().collect::<Vec<_>>());
            let old_Q_len = Q.len();

            // For each part Y ≠ X such that N(x) ⊥ Y
            for Y_idx in 0..old_Q_len {
                let Y = &Q[Y_idx];
                println!("\t\tY={:?}", Y);
                if X != *Y {
                    if N_x.iter().all(|y| Y.contains(y)) {
                        println!("\t\t\tY ≠ X, N(x) ⊆ Y");
                        continue;
                    }
                    let (Y_1, Y_2): (Vec<u32>, Vec<u32>) = Y.iter().partition(|y| N_x.contains(y));
                    if Y_1.is_empty() {
                        println!("\t\t\tY ≠ X, Y ∩ N(x) = ∅");
                        continue;
                    }
                    if Y_2.is_empty() {
                        println!("\t\t\tY ≠ X, Y ⊆ N(x)");
                        continue;
                    }
                    println!("\t\t\tY ⊥ N(x):\n\t\t\t\tY_1 = Y ∩ N(x) = {:?},\n\t\t\t\tY_2 = Y \\ N(x) = {:?},\n\t\t\t\t      N(x) \\ Y = {:?}", Y_1, Y_2, N_x.difference(&Y.iter().copied().collect::<HashSet<_>>()));

                    // Replace in Q, Y by Y_1 = Y ∩ N(x) and Y_2 = Y \ N(x)
                    println!("\t\t\tReplace in Q, Y by Y_1 and Y_2, i.e. {:?} by {:?} and {:?}", Y, Y_1, Y_2);
                    let Y = replace(&mut Q[Y_idx], Y_1.clone());
                    Q.push(Y_2.clone());

                    // Let Y_min (resp. Y_max) be the smallest part (resp. largest) among Y_1 and Y_2
                    let (Y_min, Y_max) = if Y_1.len() < Y_2.len() { (Y_1, Y_2) } else { (Y_2, Y_1) };

                    // If Y ∈ L
                    if let Some(idx) = L.iter().position(|X| *X == Y) {
                        println!("\t\t\tY ∈ L:");
                        // L <- L ∪ { Y_min, Y_max } \ { Y }
                        println!("\t\t\t\tL <- L ∪ {{ Y_min, Y_max }} \\ {{ Y }}");
                        L[idx] = Y_min;
                        L.push(Y_max);
                    } else {
                        println!("\t\t\tY ∉ L:");
                        // L <- L ∪ { Y_min }
                        println!("\t\t\t\tL <- L ∪ {{ Y_min }}");
                        L.push(Y_min);
                        // If Y ∈ K
                        if let Some(idx) = K.iter().position(|X| *X == Y) {
                            println!("\t\t\t\tY ∈ K");
                            // Replace Y by Y_max in K
                            println!("\t\t\t\t\tReplace Y by Y_max in K, i.e. {:?} by {:?}", Y, Y_max);
                            K[idx] = Y_max;
                        } else {
                            println!("\t\t\t\tY ∉ K");
                            // Add Y_max at the end of K
                            println!("\t\t\t\t\tAdd Y_max at the end of K");
                            K.push_back(Y_max);
                        }
                    }
                } else {
                    println!("\t\t\tX = Y");
                }
            }
        }
    }

    println!("Q = {:?}", Q);

    Q
}

mod test {
    use petgraph::graph::NodeIndex;
    use crate::modular_partition::modular_partition;

    fn canonicalize(partition: Vec<Vec<u32>>) -> Vec<Vec<u32>> {
        let mut partition = partition;
        partition.iter_mut().for_each(|part| part.sort());
        partition.sort_by_key(|part| part[0]);
        partition
    }

    #[test]
    fn basic() {
        let graph = common::instances::ted08_test0();
        // [[6], [2, 3, 4], [15], [17, 9, 10, 11, 1], [13, 12, 14], [16], [5, 7], [8], [0]]
        // [[0], [1], [2, 3, 4], [5, 7], [6], [8], [9], [10, 11], [12, 13, 14], [15], [16], [17]]
        let partition = vec![vec![0, 5, 7, 8], vec![6], vec![1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17]];

        let modular_partition = modular_partition(&partition, &graph);
        assert_eq!(canonicalize(modular_partition), [vec![0], vec![1], vec![2, 3, 4], vec![5, 7], vec![6], vec![8], vec![9], vec![10, 11], vec![12, 13, 14], vec![15], vec![16], vec![17]]);
    }

    #[test]
    fn ms00_fig1() {
        let graph = common::instances::ms00_fig1();
        let init = |x: u32| -> Vec<Vec<u32>> {
            let neighbors: Vec<_> = graph.neighbors(NodeIndex::new(x as usize)).map(|u| u.index() as u32).collect();
            let non_neighbors = graph.node_indices().filter_map(|u| {
                let u = u.index() as u32;
                if u != x && !neighbors.contains(&u) { Some(u) } else { None }
            }).collect();
            vec![neighbors, vec![x], non_neighbors]
        };
        {
            let partition = init(0);
            let modular_partition = modular_partition(&partition, &graph);
            assert_eq!(canonicalize(modular_partition), vec![vec![0], vec![1], vec![2], vec![3, 4], vec![5], vec![6, 7], vec![8, 9, 10, 11]]);
        }
        {
            let partition = init(1);
            let modular_partition = modular_partition(&partition, &graph);
            assert_eq!(canonicalize(modular_partition), vec![vec![0], vec![1], vec![2], vec![3, 4], vec![5], vec![6, 7], vec![8, 9, 10, 11]]);
        }
        {
            let partition = init(2);
            let modular_partition = modular_partition(&partition, &graph);
            assert_eq!(canonicalize(modular_partition), vec![vec![0, 1], vec![2], vec![3, 4], vec![5], vec![6, 7], vec![8, 9, 10, 11]]);
        }
        {
            let partition = init(3);
            let modular_partition = modular_partition(&partition, &graph);
            assert_eq!(canonicalize(modular_partition), vec![vec![0, 1, 2], vec![3], vec![4], vec![5], vec![6, 7], vec![8, 9, 10, 11]]);
        }
        {
            let partition = init(8);
            let modular_partition = modular_partition(&partition, &graph);
            assert_eq!(canonicalize(modular_partition), vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![8], vec![9, 10], vec![11]]);
        }
    }
}