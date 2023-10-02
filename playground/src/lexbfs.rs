mod rou21 {
    use petgraph::graph::{IndexType, node_index, UnGraph};
    use crate::list::{List, ListHandle};

    pub(crate) fn lexbfs<N, E, Ix>(graph: UnGraph<N, E, Ix>) -> Vec<u32>
        where Ix: IndexType {
        // Based on: https://github.com/ArthurRouquan/LexBFS/blob/41f86fe120764b450f596760bf35a803baacdf2b/src/lexbfs.hpp
        let mut ordering: Vec<u32> = (0..graph.node_count() as u32).collect();

        struct Interval {
            start: u32,
        }

        let mut intervals = List::new();
        intervals.push_back(Interval { start: 0 });
        intervals.push_back(Interval { start: ordering.len() as _ });

        struct VertexInfo {
            order: u32,
            interval: ListHandle,
            reached: bool,
        }

        let mut vertices_info: Vec<_> = (0..graph.node_count()).map(|i| VertexInfo { order: i as _, interval: intervals.handle_front().unwrap(), reached: false }).collect();

        let pop_front = |intervals: &mut List<Interval>, interval: ListHandle| {
            let next = intervals.next(interval).unwrap();
            if intervals[interval].start + 1 != intervals[next].start {
                intervals[interval].start += 1;
            } else {
                intervals.remove(interval);
            }
        };

        for i in 0..ordering.len() {
            let pivot = ordering[i] as usize;
            vertices_info[pivot].reached = true;

            pop_front(&mut intervals, vertices_info[pivot].interval);

            let mut neighbors: Vec<_> = graph.neighbors(node_index(pivot)).map(|n| n.index()).collect();
            neighbors.sort();
            for neighbor in neighbors {
                let VertexInfo { order, interval, reached } = vertices_info[neighbor];
                if reached { continue; }

                let start_order = std::mem::replace(&mut vertices_info[ordering[intervals[interval].start as usize] as usize].order, order);
                vertices_info[neighbor.index()].order = start_order;
                ordering.swap(order as usize, start_order as usize);


                if Some(interval) == intervals.handle_front() {
                    intervals.push_front(Interval { start: order });
                }

                vertices_info[neighbor.index()].interval = intervals.prev(interval).unwrap();
                pop_front(&mut intervals, interval);
            }
        }

        ordering
    }

    mod test {
        use petgraph::graph::UnGraph;
        use super::*;

        #[test]
        fn simple_graph() {
            let graph = UnGraph::<(), (), u32>::from_edges([(0, 1), (1, 2), (2, 3)]);

            assert_eq!(lexbfs(graph), [0, 1, 2, 3]);
        }

        #[test]
        fn toy() {
            let graph = UnGraph::<(), (), u32>::from_edges([
                (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]);

            assert_eq!(lexbfs(graph), [0, 1, 2, 3, 4, 5]);
        }
    }
}


mod partition_refinement {
    use petgraph::graph::{IndexType, NodeIndex, UnGraph};
    use crate::partition_refinement::Partition;

    pub(crate) fn lexbfs<N, E, Ix>(graph: &UnGraph<N, E, Ix>) -> Vec<u32>
        where Ix: IndexType
    {
        let n = graph.node_count();
        let mut partition = Partition::new(n);
        for u in (0..n).map(|i| NodeIndex::new(i)) {
            partition.refine([u.index() as u32]);
            partition.refine(graph.neighbors(u).map(|v| v.index() as u32));
        }
        partition.elements().collect::<Vec<_>>()
    }

    pub(crate) fn lexbfs_plus<N, E, Ix>(graph: &UnGraph<N, E, Ix>, ordering: &[NodeIndex<Ix>]) -> Vec<NodeIndex<Ix>>
        where Ix: IndexType
    {
        let n = graph.node_count();
        let mut partition = Partition::new(n);
        let mut neighbors = vec![];
        let mut ordering_inverse = vec![0; ordering.len()];
        for (i, u) in ordering.iter().enumerate() {
            ordering_inverse[u.index()] = (n - i) as u32;
        }
        for u in ordering {
            neighbors.clear();
            neighbors.extend(graph.neighbors(*u));
            neighbors.sort_by_key(|v| { ordering_inverse[v.index()] });

            partition.refine([u.index() as u32]);
            partition.refine(neighbors.iter().map(|v| v.index() as u32));
        }
        partition.elements().map(|u| NodeIndex::new(u as _)).collect::<Vec<_>>()
    }

    mod test {
        use petgraph::graph::UnGraph;
        use super::*;

        #[test]
        fn simple_graph() {
            let graph = UnGraph::<(), (), u32>::from_edges([(0, 1), (1, 2), (2, 3)]);

            assert_eq!(lexbfs(&graph), [0, 1, 2, 3]);
        }

        #[test]
        fn toy() {
            let graph = UnGraph::<(), (), u32>::from_edges([
                (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]);

            // 0 - 1
            // | / |
            // 2 - 3
            // | / |
            // 4 - 5

            assert_eq!(lexbfs(&graph), [0, 1, 2, 3, 4, 5]);
        }

        #[test]
        fn lexbfs_plus_works() {
            let graph = UnGraph::<(), (), u32>::from_edges([
                (0, 1), (0, 4), (0, 5), (1, 4), (1, 5), (2, 3), (2, 5), (2, 7), (3, 5), (3, 7), (4, 5), (4, 7), (5, 7), (6, 7)]);

            let sigma_0 = [0, 1, 2, 3, 4, 5, 6, 7].map(NodeIndex::new);
            let sigma_1 = lexbfs_plus(&graph, &sigma_0);
            assert_eq!(sigma_1.iter().map(|v| v.index() as u32).collect::<Vec<_>>(), [7, 6, 5, 4, 3, 2, 0, 1]);
        }
    }
}