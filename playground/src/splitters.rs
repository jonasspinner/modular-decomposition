use petgraph::adj::IndexType;
use petgraph::graph::{NodeIndex, UnGraph};

#[allow(dead_code)]
fn is_splitter<N, E, Ix>(graph: &UnGraph<N, E, Ix>, set: &[NodeIndex<Ix>], z: NodeIndex<Ix>) -> bool
    where Ix: IndexType
{
    for &x in set {
        for &y in set {
            if graph.contains_edge(x, z) && !graph.contains_edge(y, z) {
                return true;
            }
        }
    }
    false
}

mod hashset {
    use std::collections::HashSet;
    use petgraph::graph::{IndexType, NodeIndex, UnGraph};

    pub(crate) fn splitters<N, E, Ix, S>(graph: &UnGraph<N, E, Ix>, set: S) -> impl Iterator<Item=NodeIndex<Ix>>
        where Ix: IndexType,
              S: IntoIterator,
              S::Item: Into<NodeIndex<Ix>>,
              S::IntoIter: Clone,
    {
        let mut iter = set.into_iter().map(Into::into);
        let set: HashSet<_> = iter.clone().collect();

        let start: HashSet<_> = iter.next().map(|u| graph.neighbors(u).collect()).unwrap_or_default();

        let mut union = start.clone();
        let mut intersection = start;

        while let Some(u) = iter.next() {
            union.extend(graph.neighbors(u));
            intersection = graph.neighbors(u).filter(|v| intersection.contains(v)).collect();
        }
        union.into_iter().filter(move |u| !intersection.contains(u) && !set.contains(u))
    }
}

mod counting {
    use std::iter::FusedIterator;
    use petgraph::graph::{IndexType, NodeIndex, UnGraph};

    pub(crate) fn splitters<N, E, Ix, S>(graph: &UnGraph<N, E, Ix>, set: S, counts: &mut [u32], nodes: &mut Vec<NodeIndex<Ix>>)
        where Ix: IndexType,
              S: IntoIterator,
              S::Item: Into<NodeIndex<Ix>>,
    {
        assert!(graph.node_count() <= counts.len());
        assert!(graph.node_count() < u32::MAX as usize);
        assert!(nodes.is_empty());

        let mut num_nodes_in_set = 0;
        for u in set {
            num_nodes_in_set += 1;
            let u = u.into();

            let count = std::mem::replace(&mut counts[u.index()], u32::MAX);
            if count == 0 { nodes.push(u); }

            for v in graph.neighbors(u) {
                let count = counts[v.index()];
                counts[v.index()] = count.saturating_add(1);
                if count == 0 { nodes.push(v); }
            }
        }

        nodes.retain(|u| {
            let count = std::mem::replace(&mut counts[u.index()], 0);
            count < num_nodes_in_set
        });
    }

    #[derive(Default)]
    pub(crate) struct CountingSplitters<Ix: IndexType> {
        counts: Vec<u32>,
        nodes: Vec<NodeIndex<Ix>>,
    }

    impl<Ix: IndexType> CountingSplitters<Ix> {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        pub(crate) fn splitters<'a, N, E, S>(&'a mut self, graph: &UnGraph<N, E, Ix>, set: S) -> impl Iterator<Item=NodeIndex<Ix>> + 'a
            where S: IntoIterator,
                  S::Item: Into<NodeIndex<Ix>>,
        {
            self.counts.resize(graph.node_count(), 0);
            splitters(graph, set, &mut self.counts, &mut self.nodes);
            SplittersIterator { nodes: &mut self.nodes, idx: 0 }
        }
    }

    struct SplittersIterator<'a, Ix: IndexType> {
        nodes: &'a mut Vec<NodeIndex<Ix>>,
        idx: usize,
    }

    impl<'a, Ix: IndexType> Iterator for SplittersIterator<'a, Ix> {
        type Item = NodeIndex<Ix>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.idx < self.nodes.len() {
                let node = self.nodes[self.idx];
                self.idx += 1;
                Some(node)
            } else {
                self.nodes.clear();
                None
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = self.nodes.len() - self.idx;
            (len, Some(len))
        }
    }

    impl<'a, Ix: IndexType> FusedIterator for SplittersIterator<'a, Ix> {}

    impl<'a, Ix: IndexType> ExactSizeIterator for SplittersIterator<'a, Ix> {}
}

mod test {
    use std::collections::HashSet;
    use petgraph::graph::{NodeIndex, UnGraph};

    fn example_graph() -> UnGraph<(), ()> {
        let mut graph = UnGraph::new_undirected();
        graph.extend_with_edges([
            (0, 4),
            (1, 3), (1, 4), (1, 5),
            (2, 3), (2, 4), (2, 5),
            (3, 4), (3, 5),
            (4, 5),
            (5, 14), (5, 15), (5, 16), (5, 17), (5, 11), (5, 12), (5, 10), (5, 13), (5, 6), (5, 7), (5, 8), (5, 9),
            (8, 9),
            (10, 11), (10, 12), (10, 13),
            (11, 12), (11, 13),
            (12, 13),
            (13, 14), (13, 15), (13, 16),
            (14, 15), (14, 17),
            (15, 17),
            (16, 17),
        ]);
        graph
    }

    fn to_set<S>(elements: S) -> HashSet<NodeIndex<u32>>
        where S: IntoIterator, S::Item: Into<NodeIndex<u32>>
    { elements.into_iter().map(Into::into).collect() }

    #[test]
    fn toy() {
        use super::hashset::splitters;
        let graph = UnGraph::<(), (), u32>::from_edges([
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]);

        assert_eq!(to_set(splitters(&graph, [0, 1])), to_set([3]));
    }

    #[test]
    fn hashset_example() {
        use super::hashset::splitters;
        let graph = example_graph();

        assert_eq!(to_set(splitters(&graph, Vec::<u32>::new())), HashSet::new());
        for u in graph.node_indices() {
            assert_eq!(to_set(splitters(&graph, [u])), HashSet::new());
        }

        assert_eq!(to_set(splitters(&graph, [8, 9])), HashSet::new());
        assert_eq!(to_set(splitters(&graph, [1, 2])), HashSet::new());
        assert_eq!(to_set(splitters(&graph, [14, 15])), HashSet::new());
        assert_eq!(to_set(splitters(&graph, [14, 15, 16])), HashSet::new());
        assert_eq!(to_set(splitters(&graph, [10, 11, 12])), HashSet::new());

        assert_eq!(to_set(splitters(&graph, [0, 1])), to_set([3, 5]));
        assert_eq!(to_set(splitters(&graph, [0, 1, 2])), to_set([3, 5]));
        assert_eq!(to_set(splitters(&graph, [0, 1, 2, 3])), to_set([5]));
        assert_eq!(to_set(splitters(&graph, [0, 1, 2, 3, 4])), to_set([5]));
        assert_eq!(to_set(splitters(&graph, [0, 1, 2, 3, 4, 5])), to_set([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]));
    }

    #[test]
    fn counting_raw_example() {
        use super::counting::splitters;
        let graph = example_graph();

        let mut counts = vec![0; graph.node_count()];
        let mut nodes = vec![];

        splitters(&graph, Vec::<u32>::new(), &mut counts, &mut nodes);
        assert_eq!(nodes, []);

        for u in graph.node_indices() {
            splitters(&graph, [u], &mut counts, &mut nodes);
            assert_eq!(nodes, []);
        }

        splitters(&graph, [8, 9], &mut counts, &mut nodes);
        assert_eq!(nodes, []);
        splitters(&graph, [1, 2], &mut counts, &mut nodes);
        assert_eq!(nodes, []);
        splitters(&graph, [14, 15], &mut counts, &mut nodes);
        assert_eq!(nodes, []);
        splitters(&graph, [14, 15, 16], &mut counts, &mut nodes);
        assert_eq!(nodes, []);
        splitters(&graph, [10, 11, 12], &mut counts, &mut nodes);
        assert_eq!(nodes, []);

        splitters(&graph, [0, 1], &mut counts, &mut nodes);
        assert_eq!(to_set(nodes.iter().copied()), to_set([3, 5]));
        nodes.clear();
        splitters(&graph, [0, 1, 2], &mut counts, &mut nodes);
        assert_eq!(to_set(nodes.iter().copied()), to_set([3, 5]));
        nodes.clear();
        splitters(&graph, [0, 1, 2, 3], &mut counts, &mut nodes);
        assert_eq!(to_set(nodes.iter().copied()), to_set([5]));
        nodes.clear();
        splitters(&graph, [0, 1, 2, 3, 4], &mut counts, &mut nodes);
        assert_eq!(to_set(nodes.iter().copied()), to_set([5]));
        nodes.clear();
        splitters(&graph, [0, 1, 2, 3, 4, 5], &mut counts, &mut nodes);
        assert_eq!(to_set(nodes.iter().copied()), to_set([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]));
        nodes.clear();
    }

    #[test]
    fn counting_example() {
        use super::counting::CountingSplitters;
        let graph = example_graph();

        let mut counts = CountingSplitters::new();

        assert_eq!(to_set(counts.splitters(&graph, Vec::<u32>::new())), HashSet::new());
        for u in graph.node_indices() {
            assert_eq!(to_set(counts.splitters(&graph, [u])), HashSet::new());
        }

        assert_eq!(to_set(counts.splitters(&graph, [8, 9])), HashSet::new());
        assert_eq!(to_set(counts.splitters(&graph, [1, 2])), HashSet::new());
        assert_eq!(to_set(counts.splitters(&graph, [14, 15])), HashSet::new());
        assert_eq!(to_set(counts.splitters(&graph, [14, 15, 16])), HashSet::new());
        assert_eq!(to_set(counts.splitters(&graph, [10, 11, 12])), HashSet::new());

        assert_eq!(to_set(counts.splitters(&graph, [0, 1])), to_set([3, 5]));
        assert_eq!(to_set(counts.splitters(&graph, [0, 1, 2])), to_set([3, 5]));
        assert_eq!(to_set(counts.splitters(&graph, [0, 1, 2, 3])), to_set([5]));
        assert_eq!(to_set(counts.splitters(&graph, [0, 1, 2, 3, 4])), to_set([5]));
        assert_eq!(to_set(counts.splitters(&graph, [0, 1, 2, 3, 4, 5])), to_set([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]));
    }
}