use std::iter::FusedIterator;

#[derive(Clone, Copy, PartialEq, Debug, Eq, Hash)]
pub(crate) struct VertexId(u32);

impl VertexId {
    pub(crate) fn idx(&self) -> usize { self.0 as _ }
}

impl VertexId {
    pub(crate) fn invalid() -> VertexId { VertexId(u32::MAX) }
    pub(crate) fn is_invalid(self) -> bool { self.0 == u32::MAX }
}

impl From<usize> for VertexId {
    fn from(value: usize) -> Self {
        assert!(value < u32::MAX as _);
        VertexId(value as _)
    }
}


pub(crate) struct Graph {
    adj: Vec<Vec<VertexId>>,
    num_edges: usize,
}

impl Graph {
    pub(crate) fn new(number_of_nodes: usize) -> Self {
        Self { adj: vec![vec![]; number_of_nodes], num_edges: 0 }
    }
    pub(crate) fn number_of_nodes(&self) -> usize {
        self.adj.len()
    }
    fn number_of_edges(&self) -> usize {
        self.num_edges
    }
}

#[allow(unused)]
impl Graph {
    pub(crate) fn neighbors(&self, u: VertexId) -> &[VertexId] {
        &self.adj[u.idx()]
    }
    fn degree(&self, u: VertexId) -> usize {
        self.adj[u.idx()].len()
    }
    fn has_vertex(&self, u: VertexId) -> bool {
        u.idx() < self.adj.len()
    }
    fn has_edge(&self, mut u: VertexId, mut v: VertexId) -> bool {
        if !self.has_vertex(u) || !self.has_vertex(v) || u == v { return false; }
        if self.degree(u) > self.degree(v) { std::mem::swap(&mut u, &mut v); }
        self.neighbors(u).contains(&v)
    }
    pub(crate) fn add_edge<V>(&mut self, u: V, v: V) where V: Into<VertexId> {
        let u = u.into();
        let v = v.into();
        assert!(self.has_vertex(u));
        assert!(self.has_vertex(v));
        assert_ne!(u, v);
        self.adj[u.idx()].push(v);
        self.adj[v.idx()].push(u);
        self.num_edges += 1;
    }
}

pub(crate) struct EdgeIter<'a> {
    graph: &'a Graph,
    u: VertexId,
    v_idx: usize,
}

impl<'a> EdgeIter<'a> {
    fn new(graph: &'a Graph) -> Self {
        EdgeIter { graph, u: VertexId(0), v_idx: 0 }
    }
}

impl Iterator for EdgeIter<'_> {
    type Item = (VertexId, VertexId);

    fn next(&mut self) -> Option<Self::Item> {
        while self.u.idx() < self.graph.adj.len() {
            let neighbors = self.graph.neighbors(self.u);
            for (i, &v) in neighbors.iter().enumerate().skip(self.v_idx) {
                if self.u.idx() < v.idx() {
                    self.v_idx = i + 1;
                    return Some((self.u, v));
                }
            }
            self.u = VertexId::from(self.u.idx() + 1);
            self.v_idx = 0;
        }
        None
    }
}

impl FusedIterator for EdgeIter<'_> {}


impl Graph {
    pub(crate) fn vertices(&self) -> impl Iterator<Item=VertexId> {
        (0..self.number_of_nodes()).map(|idx| VertexId::from(idx))
    }
    pub(crate) fn edges(&self) -> EdgeIter<'_> {
        EdgeIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_graph() -> Graph {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(1, 3);
        graph
    }

    #[test]
    fn basic() {
        let graph = init_graph();
        assert_eq!(graph.number_of_nodes(), 5);
        assert_eq!(graph.number_of_edges(), 5);

        assert_eq!(graph.vertices().collect::<Vec<_>>(), [0, 1, 2, 3, 4].iter().map(|&i| VertexId(i)).collect::<Vec<_>>());
        assert_eq!(graph.edges().collect::<Vec<_>>(), [(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)].iter().map(|&(i, j)| (VertexId(i), VertexId(j))).collect::<Vec<_>>());
    }
}