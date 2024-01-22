use std::collections::{HashSet, VecDeque};
use std::mem::replace;
use petgraph::{Graph, Undirected};
use petgraph::adj::IndexType;
use petgraph::graph::NodeIndex;


macro_rules! traceln {
    ($($x:expr),*) => {
        // println!($($x),*)
    }
}

#[allow(non_snake_case, dead_code)]
fn overlap(A: impl IntoIterator<Item=u32>, B: impl IntoIterator<Item=u32>) -> bool {
    let A: HashSet<u32> = A.into_iter().collect();
    let B: HashSet<u32> = B.into_iter().collect();
    !A.is_disjoint(&B) && !A.is_subset(&B) && !B.is_subset(&A)
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
        traceln!("K = {:?}, L = {:?}, Q = {:?}", K, L, Q);
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

        traceln!("S = {:?}, X = {:?}", S, X);

        // For each x ∈ S
        for &x in &S {
            let N_x = graph.neighbors(NodeIndex::new(x as usize)).map(|i| i.index() as u32).collect::<HashSet<_>>();
            traceln!("\tx = {}, N(x) = {:?}", x, N_x.iter().collect::<Vec<_>>());

            // For each part Y ≠ X such that N(x) ⊥ Y
            let mut Y_idx = 0;
            while Y_idx < Q.len() {
                let Y = &Q[Y_idx];
                Y_idx += 1;
                traceln!("\t\tY={:?}", Y);
                if X != *Y {
                    if N_x.iter().all(|y| Y.contains(y)) {
                        traceln!("\t\t\tY ≠ X, N(x) ⊆ Y");
                        // NOTE: This is different to the paper.
                        //continue;
                    }
                    let (Y_1, Y_2): (Vec<u32>, Vec<u32>) = Y.iter().partition(|y| N_x.contains(y));
                    if Y_1.is_empty() {
                        traceln!("\t\t\tY ≠ X, Y ∩ N(x) = ∅");
                        continue;
                    }
                    if Y_2.is_empty() {
                        traceln!("\t\t\tY ≠ X, Y ⊆ N(x)");
                        continue;
                    }
                    traceln!("\t\t\tY ⊥ N(x):\n\t\t\t\tY_1 = Y ∩ N(x) = {:?},\n\t\t\t\tY_2 = Y \\ N(x) = {:?},\n\t\t\t\t      N(x) \\ Y = {:?}", Y_1, Y_2, N_x.difference(&Y.iter().copied().collect::<HashSet<_>>()));

                    // Replace in Q, Y by Y_1 = Y ∩ N(x) and Y_2 = Y \ N(x)
                    traceln!("\t\t\tReplace in Q, Y by Y_1 and Y_2, i.e. {:?} by {:?} and {:?}", Y, Y_1, Y_2);
                    let Y = replace(&mut Q[Y_idx - 1], Y_1.clone());
                    Q.insert(Y_idx, Y_2.clone());
                    Y_idx += 1;

                    // Let Y_min (resp. Y_max) be the smallest part (resp. largest) among Y_1 and Y_2
                    let (Y_min, Y_max) = if Y_1.len() < Y_2.len() { (Y_1, Y_2) } else { (Y_2, Y_1) };

                    // If Y ∈ L
                    if let Some(idx) = L.iter().position(|X| *X == Y) {
                        traceln!("\t\t\tY ∈ L:");
                        // L <- L ∪ { Y_min, Y_max } \ { Y }
                        traceln!("\t\t\t\tL <- L ∪ {{ Y_min, Y_max }} \\ {{ Y }}");
                        L[idx] = Y_min;
                        L.push(Y_max);
                    } else {
                        traceln!("\t\t\tY ∉ L:");
                        // L <- L ∪ { Y_min }
                        traceln!("\t\t\t\tL <- L ∪ {{ Y_min }}");
                        L.push(Y_min);
                        // If Y ∈ K
                        if let Some(idx) = K.iter().position(|X| *X == Y) {
                            traceln!("\t\t\t\tY ∈ K");
                            // Replace Y by Y_max in K
                            traceln!("\t\t\t\t\tReplace Y by Y_max in K, i.e. {:?} by {:?}", Y, Y_max);
                            K[idx] = Y_max;
                        } else {
                            traceln!("\t\t\t\tY ∉ K");
                            // Add Y_max at the end of K
                            traceln!("\t\t\t\t\tAdd Y_max at the end of K");
                            K.push_back(Y_max);
                        }
                    }
                } else {
                    traceln!("\t\t\tX = Y");
                }
            }
        }
    }

    traceln!("Q = {:?}", Q);

    Q
}

#[cfg(test)]
#[allow(non_snake_case)]
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
        // ( 16 15 0 ( 9 1 ( 10 11 ) ( 8 18 ( 2 ( 3 4 ) ) ( 5 6 7 ) ) ) ( 12 ( 13 14 ) ) )

        let mut counter = crate::splitters::counting::CountingSplitters::new();
        let mut is_modular_partition = |p: &[Vec<u32>]| -> bool {
            for X in p {
                let s: Vec<_> = counter.splitters(&graph, X.iter().copied()).map(|u| u.index()).collect();
                if !s.is_empty() {
                    println!("{:?}: {:?}", X, s);
                    return false;
                }
            }
            true
        };

        // [15, 3, 4, 2, 17, 8, 5, 6, 7, 11, 10, 9, 1, 14, 13, 12, 0, 16]
        // ( 15 ( ( ( ( ( 3 4 ) 2 ) 17 8 ( 5 6 7 ) ) ( 11 10 ) ) 9 1 ) ( ( 14 13 ) 12 ) 0 16 )
        let partition = vec![vec![0], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17], vec![12, 13, 14], vec![15], vec![16]];
        let result = canonicalize(modular_partition(&partition, &graph));
        assert_eq!(result, partition);
        assert!(is_modular_partition(&result));

        // println!("{:?}", Dot::with_config(&graph, &[Config::NodeIndexLabel, Config::EdgeNoLabel]));

        let partition = vec![vec![0, 5, 7, 8], vec![6], vec![1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17]];
        let result = canonicalize(modular_partition(&partition, &graph));
        assert_eq!(result, [vec![0], vec![1, 9, 10, 11], vec![2, 3, 4], vec![5, 7], vec![6], vec![8], vec![12, 13, 14], vec![15], vec![16], vec![17]]);
        assert!(is_modular_partition(&result));

        let partition = vec![vec![0], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17], vec![16]];
        let result = canonicalize(modular_partition(&partition, &graph));
        assert_eq!(result, [vec![0], vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 17], vec![12, 13, 14], vec![15], vec![16]]);
        assert!(is_modular_partition(&result));

        let partition = vec![vec![0, 6, 7, 8], vec![5], vec![1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17]];
        let result = canonicalize(modular_partition(&partition, &graph));
        assert_eq!(result, [vec![0], vec![1, 9, 10, 11], vec![2, 3, 4], vec![5], vec![6, 7], vec![8], vec![12, 13, 14], vec![15], vec![16], vec![17]]);
        assert!(is_modular_partition(&result));

        for i in 0..graph.node_count() {
            let mut all: Vec<_> = (0..graph.node_count()).map(|u| u as u32).collect();
            let single = all.remove(i);
            let partition = vec![vec![single], all];
            let result = modular_partition(&partition, &graph);
            assert!(is_modular_partition(&result));
        }
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