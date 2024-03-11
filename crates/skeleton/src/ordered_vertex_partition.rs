use crate::graph::{EdgeIndex, Graph, NodeIndex};
use crate::partition::{divide, PartIndex, Partition, SubPartition};

pub(crate) fn ovp<F>(graph: &mut Graph, p: SubPartition, partition: &mut Partition, mut f: F)
where
    F: FnMut(&[(NodeIndex, NodeIndex, EdgeIndex)]),
{
    let add_if_more_than_one_part = |stack: &mut Vec<SubPartition>, partition: &Partition, p: SubPartition| {
        if has_at_least_two_parts(&p, partition) {
            stack.push(p)
        }
    };

    let mut stack = vec![];
    add_if_more_than_one_part(&mut stack, partition, p);

    while let Some(p) = stack.pop() {
        let (q, q_prime) = split(graph, p, partition, &mut f);

        add_if_more_than_one_part(&mut stack, partition, q_prime);
        add_if_more_than_one_part(&mut stack, partition, q);
    }
}

fn has_at_least_two_parts(p: &SubPartition, partition: &Partition) -> bool {
    let nodes = partition.nodes(p);
    nodes[0].part != nodes[nodes.len() - 1].part
}

fn split<F>(graph: &mut Graph, p: SubPartition, partition: &mut Partition, f: &mut F) -> (SubPartition, SubPartition)
where
    F: FnMut(&[(NodeIndex, NodeIndex, EdgeIndex)]),
{
    let (x, q, q_prime, dir) = divide(p, partition);

    let mut edges = crossing_edges(graph, x, partition);

    if edges.is_empty() {
        return (q, q_prime);
    }

    graph.remove_edges(edges.iter().map(|e| e.2));

    for (_u, pivots) in group_by(&mut edges, |e| e.0) {
        partition.refine(dir, pivots.iter().map(|e| e.1));
    }

    for (_u, pivots) in group_by(&mut edges, |e| e.1) {
        partition.refine(dir.reversed(), pivots.iter().map(|e| e.0));
    }

    f(&edges);

    (q, q_prime)
}

fn crossing_edges(graph: &mut Graph, x: PartIndex, partition: &Partition) -> Vec<(NodeIndex, NodeIndex, EdgeIndex)> {
    let mut edges = vec![];
    for u in partition.elements(x) {
        for (v, i) in graph.incident_edges(u) {
            if partition.part_by_node(v) != x {
                edges.push((u, v, i));
            }
        }
    }
    edges
}

fn group_by<'a, T, K, F>(elements: &'a mut [T], f: F) -> impl Iterator<Item = (K, &[T])> + 'a
where
    F: Fn(&T) -> K + 'a,
    K: Ord + 'a,
{
    elements.sort_unstable_by_key(|e| f(e));

    struct GroupBy<'b, T, K: Eq, F: Fn(&T) -> K> {
        elements: &'b [T],
        start: usize,
        f: F,
    }
    impl<'b, T, K: Eq, F: Fn(&T) -> K> Iterator for GroupBy<'b, T, K, F> {
        type Item = (K, &'b [T]);

        fn next(&mut self) -> Option<Self::Item> {
            if self.start < self.elements.len() {
                let (start, mut end) = (self.start, self.start + 1);
                let key = (self.f)(&self.elements[start]);
                while end < self.elements.len() && (self.f)(&self.elements[end]) == key {
                    end += 1;
                }
                self.start = end;
                return Some((key, &self.elements[start..end]));
            }
            None
        }
    }
    GroupBy { elements, start: 0, f }
}

#[cfg(test)]
mod test {
    use crate::partition::Partition;
    use crate::testing::{splitters, ted08_test0_graph, to_vecs};

    #[test]
    fn partition() {
        let mut graph = ted08_test0_graph();
        let original_graph = graph.clone();

        let mut partition = Partition::new(graph.node_count());
        partition.refine_forward([6_u32]);

        let mut removed = vec![];
        super::ovp(&mut graph, partition.full_sub_partition(), &mut partition, |e| {
            removed.extend(e.iter().map(|&(u, v, _)| (u, v)))
        });

        let p = partition.full_sub_partition();
        println!("{:?}", to_vecs(&p, &partition));
        println!(
            "removed: {:?}.map(|(u,v)| (NodeIndex::new(u), NodeIndex::new(v)));",
            removed.iter().map(|(u, v)| (u.index(), v.index())).collect::<Vec<_>>()
        );

        for part in p.part_indices(&partition) {
            let part: Vec<_> = partition.elements(part).collect();
            let s: Vec<_> = splitters(&original_graph, &part).collect();
            println!("{:?} {:?}", part.iter().map(|u| u.index()).collect::<Vec<_>>(), s);
        }
    }
}
