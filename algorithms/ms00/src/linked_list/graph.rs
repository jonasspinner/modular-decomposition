use common::make_index;



make_index!(pub(crate) NodeIndex);
make_index!(pub(crate) EdgeIndex);

#[derive(Default, Debug)]
struct Node {
    edges: EdgeIndex,
    removed_edges: EdgeIndex,
}

#[derive(Copy, Clone, Default, Debug)]
struct Edge {
    x: NodeIndex,
    y: NodeIndex,
    next: EdgeIndex,
    prev: EdgeIndex,
    twin: EdgeIndex,
}

#[derive(Default)]
pub struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Graph {
    fn with_capacity(n: usize, m: usize) -> Self {
        Self { nodes: Vec::with_capacity(n), edges: Vec::with_capacity(m) }
    }
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }
    fn edge_count(&self) -> usize {
        self.edges.len()
    }
    fn add_node(&mut self) -> NodeIndex {
        let i = self.nodes.len();
        self.nodes.push(Node { edges: EdgeIndex::invalid(), removed_edges: EdgeIndex::invalid() });
        NodeIndex::new(i)
    }
    fn add_edge(&mut self, a: NodeIndex, b: NodeIndex) -> EdgeIndex {
        let ab_index = EdgeIndex::new(self.edges.len());
        let ba_index = EdgeIndex::new(self.edges.len() + 1);
        let mut ab = Edge {
            x: a,
            y: b,
            next: ab_index,
            prev: ab_index,
            twin: ba_index,
        };
        let mut ba = Edge {
            x: b,
            y: a,
            next: ba_index,
            prev: ba_index,
            twin: ab_index,
        };

        let mut insert = |x: NodeIndex, xy: &mut Edge, xy_index| {
            let x_first = std::mem::replace(&mut self.nodes[x.index()].edges, xy_index);
            if x_first.is_valid() {
                let x_last = std::mem::replace(&mut self.edges[x_first.index()].prev, xy_index);
                assert!(x_last.is_valid());
                xy.next = x_first;
                xy.prev = x_last;
                self.edges[x_last.index()].next = xy_index;
            }
        };

        insert(a, &mut ab, ab_index);
        insert(b, &mut ba, ba_index);

        self.edges.push(ab);
        self.edges.push(ba);
        ab_index
    }

    fn remove_edge(&mut self, edge: EdgeIndex) {
        let ab_index = edge;
        let Edge { x: a, y: b, twin: ba_index, .. } = self.edges[ab_index.index()];

        let mut remove = |xy_index: EdgeIndex| {
            let Edge { prev, next, .. } = self.edges[xy_index.index()];
            self.edges[prev.index()].next = next;
            self.edges[next.index()].prev = prev;
        };

        remove(ab_index);
        remove(ba_index);

        let mut insert = |x: NodeIndex, xy_index| {
            let x_first = std::mem::replace(&mut self.nodes[x.index()].removed_edges, xy_index);
            if x_first.is_valid() {
                let x_last = std::mem::replace(&mut self.edges[x_first.index()].prev, xy_index);
                assert!(x_last.is_valid());
                self.edges[xy_index.index()].next = x_first;
                self.edges[xy_index.index()].prev = x_last;
                self.edges[x_last.index()].next = xy_index;
            }
        };

        insert(a, ab_index);
        insert(b, ba_index);
    }
}

pub(crate) trait IntoEdge {
    fn into_edge(self) -> (NodeIndex, NodeIndex);
}

impl<T: IntoEdge + Copy> IntoEdge for &T {
    fn into_edge(self) -> (NodeIndex, NodeIndex) {
        (*self).into_edge()
    }
}

impl<T: Into<NodeIndex>> IntoEdge for (T, T) {
    fn into_edge(self) -> (NodeIndex, NodeIndex) {
        (self.0.into(), self.1.into())
    }
}

impl Graph {
    pub(crate) fn from_edges<I>(n: usize, iterable: I) -> Self
        where
            I: IntoIterator + Copy,
            I::Item: IntoEdge,
    {
        let mut edge_counts: Vec<u32> = vec![0; n + 1];
        let mut m = 0;
        for (a, b) in iterable.into_iter().map(IntoEdge::into_edge) {
            edge_counts[a.index() + 1] += 1;
            edge_counts[b.index() + 1] += 1;
            m += 2;
        }

        for i in 1..n + 1 {
            edge_counts[i] += edge_counts[i - 1];
        }

        let nodes: Vec<_> = edge_counts[..n]
            .iter()
            .map(|&i| Node { edges: i.into(), removed_edges: EdgeIndex::invalid() }).collect();

        let mut edges = vec![Edge::default(); m];
        for (a, b) in iterable.into_iter().map(IntoEdge::into_edge) {
            let ab_index: EdgeIndex = edge_counts[a.index()].into();
            let ba_index: EdgeIndex = edge_counts[b.index()].into();
            let ab = Edge {
                x: a,
                y: b,
                next: EdgeIndex::new(ab_index.index().saturating_add(1)),
                prev: EdgeIndex::new(ab_index.index().saturating_sub(1)),
                twin: ba_index,
            };
            let ba = Edge {
                x: b,
                y: a,
                next: EdgeIndex::new(ba_index.index().saturating_add(1)),
                prev: EdgeIndex::new(ba_index.index().saturating_sub(1)),
                twin: ab_index,
            };
            edges[ab_index.index()] = ab;
            edges[ba_index.index()] = ba;
            edge_counts[a.index()] += 1;
            edge_counts[b.index()] += 1;
        }

        for i in 0..n - 1 {
            let first = nodes[i].edges;
            let last = EdgeIndex::new(nodes[i + 1].edges.index() - 1);
            edges[first.index()].prev = last;
            edges[last.index()].next = first;
        }

        let first = nodes[n - 1].edges;
        let last = EdgeIndex::new(edges.len() - 1);
        edges[first.index()].prev = last;
        edges[last.index()].next = first;

        Self { nodes, edges }
    }

    pub(crate) fn remove_edges<I>(&mut self, iterator: I)
        where
            I: IntoIterator<Item=EdgeIndex>,
    {
        for e in iterator {
            let (uv, vu) = (e, self.edges[e.index()].twin);
            for e in [uv, vu] {
                let Edge { prev, next, x, .. } = self.edges[e.index()];
                self.edges[prev.index()].next = next;
                self.edges[next.index()].prev = prev;
                if self.nodes[x.index()].edges == e {
                    self.nodes[x.index()].edges = if next == e { EdgeIndex::invalid() } else { next };
                }
            }
        }
    }
}

pub(crate) struct Neighbors<'a> {
    graph: &'a Graph,
    start: EdgeIndex,
    current: EdgeIndex,
}

impl<'a> Neighbors<'a> {
    fn new(graph: &'a Graph, start: EdgeIndex) -> Self {
        Self { graph, start, current: start }
    }
}

impl<'a> Iterator for Neighbors<'a> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_valid() {
            let Edge { y, next, .. } = self.graph.edges[self.current.index()];
            self.current = if next == self.start { EdgeIndex::invalid() } else { next };
            Some(y)
        } else {
            None
        }
    }
}

impl Graph {
    pub(crate) fn neighbors(&self, a: NodeIndex) -> Neighbors {
        Neighbors::new(self, self.nodes[a.index()].edges)
    }
}

pub(crate) struct IncidentEdgeIndices<'a> {
    graph: &'a Graph,
    start: EdgeIndex,
    next: EdgeIndex,
    //node: NodeIndex,
}

impl<'a> IncidentEdgeIndices<'a> {
    fn new(graph: &'a Graph, node: NodeIndex) -> Self {
        let start = graph.nodes[node.index()].edges;
        Self { graph, start, next: start }
    }
}

impl<'a> Iterator for IncidentEdgeIndices<'a> {
    type Item = (NodeIndex, EdgeIndex);

    fn next(&mut self) -> Option<Self::Item> {
        self.next.is_valid().then(|| {
            let Edge { y, next, .. } = self.graph.edges[self.next.index()];
            let curr = self.next;
            self.next = if next == self.start { EdgeIndex::invalid() } else { next };
            (y, curr)
        })
    }
}

impl Graph {
    pub(crate) fn incident_edges(&self, a: NodeIndex) -> IncidentEdgeIndices {
        IncidentEdgeIndices::new(self, a)
    }
}


mod test {
    use petgraph::graph::UnGraph;
    use petgraph::visit::{EdgeRef, NodeCount};
    use common::instances::ted08_test0;
    use common::set::FastResetBitSet;
    use crate::linked_list::graph::{Graph, NodeIndex};

    fn to_graph(graph: &UnGraph<(), ()>) -> Graph {
        let n = graph.node_count();
        let edges: Vec<_> = graph.edge_references().map(|e| (e.source().index(), e.target().index())).collect();
        Graph::from_edges(n, &edges)
    }

    #[test]
    fn from_edges() {
        let test_graph = ted08_test0();
        let n = test_graph.node_count();
        let edges: Vec<_> = test_graph.edge_references().map(|e| (e.source().index(), e.target().index())).collect();

        let graph = Graph::from_edges(n, &edges);

        println!("{:?}", graph.neighbors(NodeIndex::new(2)).collect::<Vec<_>>())
    }


    #[test]
    fn edges_across() {
        #[allow(non_snake_case)]

            let X = [2, 8].map(NodeIndex);

        let graph = to_graph(&ted08_test0());

        let mut is_in_x: FastResetBitSet = FastResetBitSet::new(graph.node_count());

        for &x in &X {
            is_in_x.set(x);
        }

        for &x in &X {
            for y in graph.neighbors(x) {
                if !is_in_x.get(y) {
                    println!("{:?}", (x, y));
                }
            }
        }
    }
}