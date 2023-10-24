use std::collections::HashSet;
use petgraph::visit::EdgeRef;
use common::instances::ted08_test0;
use crate::seq::graph::{Graph, NodeIndex};
use crate::seq::partition::{Partition, SubPartition};

pub(crate) fn splitters<'a>(graph: &Graph, set: &'a [NodeIndex]) -> impl Iterator<Item=NodeIndex> + 'a {
    let mut iter = set.iter();
    let set: HashSet<_> = set.iter().copied().collect();

    let start: HashSet<_> = iter.next().map(|&u| graph.incident_edges(u).map(|(v, _)| v).collect()).unwrap_or_default();

    let mut union = start.clone();
    let mut intersection = start;

    for u in iter {
        union.extend(graph.incident_edges(*u).map(|(v, _)| v));
        intersection = graph.incident_edges(*u).map(|(v, _)| v).filter(|v| intersection.contains(v)).collect();
    }
    union.into_iter().filter(move |u| !intersection.contains(u) && !set.contains(u))
}


pub(crate) fn to_vecs(p: &SubPartition, partition: &Partition) -> Vec<Vec<u32>> {
    p.part_indices(partition).map(|part| partition.elements(part).map(|i| i.index() as u32).collect()).collect()
}


pub(crate) fn ted08_test0_graph() -> Graph {
    let test0 = ted08_test0();
    let edges = test0.edge_references()
        .map(|e| -> (NodeIndex, NodeIndex) { (e.source().index().into(), e.target().index().into()) });
    Graph::from_edges(test0.node_count(), edges)
}