use std::cmp::min;
use std::iter::{FusedIterator, Step};
use std::slice::SliceIndex;
use common::make_index;



make_index!(pub(crate) NodeIndex);
make_index!(pub(crate) EdgeIndex);


impl Step for EdgeIndex {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        end.index().checked_sub(start.index())
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        Some(EdgeIndex::new(start.index() + count))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        start.index().checked_sub(count).map(|e| e.into())
    }
}

#[derive(Default, Debug, Copy, Clone)]
struct Node {
    start: EdgeIndex,
    end: EdgeIndex,
}

#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct Edge {
    pub(crate) head: NodeIndex,
    twin: EdgeIndex,
    deleted: bool,
}


#[derive(Clone)]
pub(crate) struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Graph {
    pub(crate) fn from_edges<I>(n: usize, iterable: I) -> Self
        where
            I: IntoIterator<Item=(NodeIndex, NodeIndex)> + Clone,
    {
        let mut nodes = vec![Node { start: EdgeIndex::new(0), end: EdgeIndex::new(0) }; n + 1];
        let mut m = 0;
        for (a, b) in iterable.clone() {
            nodes[a.index() + 1].end.0 += 1;
            nodes[b.index() + 1].end.0 += 1;
            m += 2;
        }

        for i in 1..n + 1 {
            nodes[i].end.0 += nodes[i - 1].end.0;
            nodes[i].start = nodes[i].end;
        }

        let mut edges = vec![Edge { head: NodeIndex::invalid(), twin: EdgeIndex::invalid(), deleted: false }; m];
        for (a, b) in iterable {
            let ab = nodes[a.index()].end;
            let ba = nodes[b.index()].end;
            edges[ab.index()] = Edge { head: b, twin: ba, deleted: false };
            edges[ba.index()] = Edge { head: a, twin: ab, deleted: false };
            nodes[a.index()].end.0 += 1;
            nodes[b.index()].end.0 += 1;
        }

        //if n >= 4 { println!("n={:8} m={:12}", n, m); }

        Self { nodes, edges }
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len().saturating_sub(1)
    }

    pub(crate) fn node_indices(&self) -> impl Iterator<Item=NodeIndex> {
        (0..self.node_count()).map(NodeIndex::new)
    }

    pub(crate) fn edge_references(&self) -> impl Iterator<Item=(NodeIndex, NodeIndex, EdgeIndex)> + '_ {
        self.node_indices()
            .flat_map(|u|
                self.incident_edges(u)
                    .map(move |(v, e)| (u, v, e)))
    }

    pub(crate) fn remove_edges<I>(&mut self, edges: I)
        where
            I: IntoIterator<Item=EdgeIndex> + Clone,
    {
        let move_edge = |x: NodeIndex, xy: EdgeIndex, nodes: &mut [Node], edges: &mut [Edge]| {
            let Node { end, start } = nodes[x.index()];
            let next_start = nodes[x.index() + 1].start;

            if let Some(e) = edges[start.index()..end.index()]
                .iter().rposition(|e| !e.deleted)
                .map(|i| (start.index() + i).into()) {
                if xy < e {
                    let (a, b): (_, EdgeIndex) = (xy, e);
                    let a_twin = edges[a.index()].twin;
                    let b_twin = edges[b.index()].twin;
                    edges.swap(a.index(), b.index());
                    edges[a_twin.index()].twin = b;
                    edges[b_twin.index()].twin = a;
                    nodes[x.index()].end = b;
                } else {
                    assert_ne!(e, xy);
                    let e = EdgeIndex::new(e.index() + 1);
                    assert!(e < next_start);
                    assert!(edges[e.index()].deleted);
                    nodes[x.index()].end = e;
                }
            } else {
                nodes[x.index()].end = start;
            }
        };

        for uv in edges.clone() {
            let vu = self.edges[uv.index()].twin;
            assert!(!self.edges[uv.index()].deleted);
            assert!(!self.edges[vu.index()].deleted);

            self.edges[uv.index()].deleted = true;
            self.edges[vu.index()].deleted = true;
        }

        for uv in edges {
            let Edge { head: v, twin: vu, .. } = self.edges[uv.index()];
            let u = self.edges[vu.index()].head;

            move_edge(u, uv, &mut self.nodes, &mut self.edges);
            move_edge(v, vu, &mut self.nodes, &mut self.edges);
        }
    }

    pub(crate) fn restore_removed_edges(&mut self) {
        for i in 0..self.nodes.len() - 1 {
            let (a, b) = (self.nodes[i].end.index(), self.nodes[i + 1].start.index());
            for e in &mut self.edges[a..b] { e.deleted = false; }
            self.nodes[i].end = self.nodes[i + 1].start;
        }
    }

    fn find_edge(&self, u: NodeIndex, v: NodeIndex) -> Option<EdgeIndex> {
        let Node { start, end } = self.nodes[u.index()];
        self.edges[start.index()..end.index()].iter().position(|e| e.head == v).map(|pos| EdgeIndex::new(start.index() + pos))
    }

    #[cfg(test)]
    fn check(&self) -> bool {
        for u in 0..self.nodes.len() - 1 {
            if !(self.nodes[u].start.0 <= self.nodes[u].end.0 && self.nodes[u].end.0 <= self.nodes[u + 1].start.0) {
                panic!("u={}  [{} {}] [{} ..]", u, self.nodes[u].start.0, self.nodes[u].end.0, self.nodes[u + 1].start.0);
            }
            for (v, e) in self.incident_edges(u.into()) {
                if !(self.nodes[u].start <= e && e < self.nodes[u].end) {
                    panic!("u={}  {} e={} {}", u, self.nodes[u].start.0, e, self.nodes[u].end.0);
                }
                let v = v.index();
                let e_twin = self.edges[e.index()].twin;
                if !(self.nodes[v].start <= e_twin && e_twin < self.nodes[v].end) {
                    panic!("v={}  {} e={} {}", v, self.nodes[v].start.0, e_twin, self.nodes[v].end.0);
                }
            }
        }
        for e in 0..self.edges.len() {
            let e = EdgeIndex::new(e);
            if self.edges[self.edges[e.index()].twin.index()].twin != e {
                return false;
            }
        }
        true
    }
}

impl Graph {
    pub(crate) fn incident_edges_raw(&self, node: NodeIndex) -> &[Edge] {
        let Node { start, end } = self.nodes[node.index()];
        &self.edges[start.index()..end.index()]
    }
}

pub(crate) struct IncidentEdges<'edges> {
    edges: &'edges [Edge],
    next: EdgeIndex,
    end: EdgeIndex,
}

impl<'edges> Iterator for IncidentEdges<'edges> {
    type Item = (NodeIndex, EdgeIndex);

    fn next(&mut self) -> Option<Self::Item> {
        (self.next != self.end).then(|| {
            let e = self.next;
            self.next.0 += 1;
            assert!(!self.edges[e.index()].deleted);
            (self.edges[e.index()].head, e)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end.0 - self.next.0) as usize;
        (len, Some(len))
    }
}

impl FusedIterator for IncidentEdges<'_> {}

impl ExactSizeIterator for IncidentEdges<'_> {}


impl Graph {
    pub(crate) fn incident_edges(&self, node: NodeIndex) -> IncidentEdges {
        let Node { start: next, end } = self.nodes[node.index()];
        IncidentEdges { edges: &self.edges, next, end }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::visit::{EdgeRef, IntoEdgeReferences, NodeCount};
    use common::instances::ted08_test0;

    fn ted08_test0_graph() -> Graph {
        let test0 = ted08_test0();
        let edges = test0.edge_references()
            .map(|e| -> (NodeIndex, NodeIndex) { (e.source().index().into(), e.target().index().into()) });
        Graph::from_edges(test0.node_count(), edges)
    }

    #[test]
    fn remove_edges() {
        let mut graph = ted08_test0_graph();

        assert!(graph.check());

        println!("{:?}",
                 graph.nodes.iter()
                     .enumerate()
                     .map(|(i, n)|
                         (i, n.start.index(), n.end.index())).collect::<Vec<_>>());
        println!("{:?}",
                 graph.edges.iter()
                     .enumerate()
                     .map(|(i, e)|
                         (i, graph.edges[e.twin.index()].head.index(), e.head.index(), e.twin.index())).collect::<Vec<_>>());


        let edges: Vec<_> = graph.incident_edges(8_u32.into()).map(|(_, e)| e).collect();
        println!("{:?}", graph.incident_edges(8_u32.into()).map(|(v, _)| (8, v.index())).collect::<Vec<_>>());
        graph.remove_edges(edges.iter().copied());

        assert!(graph.check());

        println!("{:?}",
                 graph.nodes.iter()
                     .enumerate()
                     .map(|(i, n)|
                         (i, n.start.index(), n.end.index())).collect::<Vec<_>>());
        println!("{:?}",
                 graph.edges.iter()
                     .enumerate()
                     .map(|(i, e)|
                         (i, graph.edges[e.twin.index()].head.index(), e.head.index(), e.twin.index())).collect::<Vec<_>>());

        let edges: Vec<_> = graph.incident_edges(0_u32.into()).map(|(_, e)| e).collect();
        println!("{:?}", graph.incident_edges(0_u32.into()).map(|(v, _)| (0, v.index())).collect::<Vec<_>>());
        graph.remove_edges(edges.iter().copied());

        assert!(graph.check());

        for u in 0..graph.nodes.len() - 1 {
            for e in graph.nodes[u].start.0..graph.nodes[u].end.0 {
                println!("{} {}", u, graph.edges[e as usize].head.0);
            }
        }
    }

    #[test]
    fn removed_edges_problem() {
        let mut graph = ted08_test0_graph();
        let edges = [
            (6_u32, 0_u32), (6, 5), (6, 7), (6, 8), (8, 2), (8, 3), (8, 4), (0, 1), (0, 2), (0, 3), (0, 4), (0, 17), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (16, 15), (15, 14), (15, 12), (15, 13)]
            .map(|(u, v)| graph.find_edge(u.into(), v.into()).unwrap());

        graph.remove_edges(edges);

        assert!(graph.check());
    }

    #[test]
    fn remove_edges_and_restore() {
        let mut graph = Graph::from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)].map(|(u, v)| (NodeIndex::new(u), NodeIndex::new(v))));

        for e in [vec![(0_u32, 1_u32)], vec![(1, 3), (1, 2)], vec![(4, 3)], vec![(2, 3)]] {
            let e: Vec<_> = e.iter().map(|&(u, v)| graph.find_edge(u.into(), v.into()).unwrap()).collect();
            graph.remove_edges(e.iter().copied());
        }

        graph.restore_removed_edges();
        for u in 0..graph.node_count() {
            println!("{u}: {:?}", graph.incident_edges(u.into()).map(|(v, _)| v.index()).collect::<Vec<_>>());
        }
    }
}