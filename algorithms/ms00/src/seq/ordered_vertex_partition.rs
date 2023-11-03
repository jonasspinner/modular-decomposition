use std::ops::Sub;
use std::slice::Iter;
use crate::seq::graph::{EdgeIndex, Graph, NodeIndex};
use crate::seq::partition::{Dir, divide, PartIndex, Partition, SubPartition};
use crate::trace;

fn crossing_edges(graph: &mut Graph, x: PartIndex, partition: &Partition) -> Vec<(NodeIndex, NodeIndex, EdgeIndex)> {
    let mut e = vec![];
    for u in partition.elements(x) {
        for (v, i) in graph.incident_edges(u) {
            if partition.part_by_node(v) != x {
                e.push((u, v, i));
            }
        }
    }
    e
}

fn group_by<'a, T, K, F>(elements: &'a mut [T], f: F) -> impl Iterator<Item=(K, &[T])> + 'a
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
                while end < self.elements.len() &&
                    (self.f)(&self.elements[end]) == key {
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


//     assert!(edges.iter().all(|(_, _, i)| i.is_valid()));
//     println!("{:?}", edges.iter().map(|(u, v, _)| (u.index(), v.index())).collect::<Vec<_>>());
fn split<F>(graph: &mut Graph, p: SubPartition, partition: &mut Partition, f: &mut F) -> (SubPartition, SubPartition)
    where F: FnMut(&[(NodeIndex, NodeIndex, EdgeIndex)])
{
    let (x, q, q_prime, dir) = divide(p, partition);

    //trace!("split: {:?} {:?} {:?} {:?}", x.index(), q, q_prime, dir);

    let mut edges = crossing_edges(graph, x, partition);

    trace!("split:edges {:?}", edges.iter().map(|&(u,v,_)| (u.index(), v.index())).collect::<Vec<_>>());

    graph.remove_edges(edges.iter().map(|e| e.2));

    for (_u, pivots) in group_by(&mut edges, |e| { e.0 }) {
        //println!("refine {} {:?} {:?}", u.index(), dir, pivots.iter().map(|e| e.1.index()).collect::<Vec<_>>());
        partition.refine(dir, pivots.iter().map(|e| e.1));
    }

    for (_u, pivots) in group_by(&mut edges, |e| { e.1 }) {
        //println!("refine {} {:?} {:?}", u.index(), dir.reversed(), pivots.iter().map(|e| e.0.index()).collect::<Vec<_>>());
        partition.refine(dir.reversed(), pivots.iter().map(|e| e.0));
    }

    //edges.sort_by_key(|(u, v, _)| { (partition.part_by_node(*u), partition.part_by_node(*v)) });
    //edges.dedup_by_key(|(u, v, _)| { (partition.part_by_node(*u), partition.part_by_node(*v)) });
    f(&edges);

    (q, q_prime)
}

enum NumParts {
    Zero,
    One,
    AtLeastTwo,
}

fn number_of_parts(p: &SubPartition, partition: &Partition) -> NumParts {
    let mut parts = p.part_indices(partition);
    let Some(_) = parts.next() else { return NumParts::Zero; };
    let Some(_) = parts.next() else { return NumParts::One; };
    NumParts::AtLeastTwo
}

pub(crate) fn ovp<F>(graph: &mut Graph, partition: &mut Partition, mut f: F)
    where F: FnMut(&[(NodeIndex, NodeIndex, EdgeIndex)])
{
    //println!("partition: {:?}  {} nodes,  {} parts", p, partition.nodes(&p).len(), p.part_indices(partition).count());
    //println!("partition: {:?}", p.part_indices(partition).map(|part| partition.elements(part).map(|i| i.index() as u32).collect::<Vec<_>>()).collect::<Vec<_>>());

    let mut queue = vec![SubPartition::new(partition)];

    while let Some(p) = queue.pop() {
        match number_of_parts(&p, partition) {
            NumParts::Zero => { panic!(); }
            NumParts::One => { continue; }
            NumParts::AtLeastTwo => {}
        }

        debug_assert!(p.part_indices(partition).count() >= 2);

        //let num_removed_edges_before = removed.len();
        let (q, q_prime) = split(graph, p, partition, &mut f);

        //println!("e'      = {:?}", removed[num_removed_edges_before..].iter().map(|(u, v)| (u.index(), v.index())).collect::<Vec<_>>());
        //println!("q       = {:?}", q.part_indices(partition).map(|i| partition.elements(i).map(|j| j.index()).collect::<Vec<_>>()).collect::<Vec<_>>());
        //println!("q_prime = {:?}", q_prime.part_indices(partition).map(|i| partition.elements(i).map(|j| j.index()).collect::<Vec<_>>()).collect::<Vec<_>>());

        queue.push(q_prime);
        queue.push(q);
    }
}


mod test {
    use std::path::Path;
    use petgraph::visit::EdgeRef;
    use common::io::read_pace2023;
    use crate::seq::graph::{Graph, NodeIndex};
    use crate::seq::partition::Partition;
    use crate::seq::testing::{splitters, ted08_test0_graph, to_vecs};

    #[test]
    fn partition() {
        let mut graph = ted08_test0_graph();
        let original_graph = graph.clone();

        let mut partition = Partition::new(graph.node_count());
        partition.refine_forward([6_u32]);

        let mut removed = vec![];
        super::ovp(&mut graph, &mut partition, |e| { removed.extend(e.iter().map(|&(u, v, _)| (u, v))) });

        let p = partition.full_sub_partition();
        println!("{:?}", to_vecs(&p, &partition));
        println!("removed: {:?}.map(|(u,v)| (NodeIndex::new(u), NodeIndex::new(v)));", removed.iter().map(|(u, v)| (u.index(), v.index())).collect::<Vec<_>>());

        for part in p.part_indices(&partition) {
            let part: Vec<_> = partition.elements(part).collect();
            let s: Vec<_> = splitters(&original_graph, &part).collect();
            println!("{:?} {:?}", part.iter().map(|u| u.index()).collect::<Vec<_>>(), s);
        }
    }

    #[test]
    fn test_file() {
        //for i in [40, 50, 106, 146] {
        for i in 1..=198 {
            let path = format!("../../hippodrome/instances/pace2023/exact_{i:0>3}.gr");
            let path = Path::new(&path);
            let mut graph = {
                let test0 = read_pace2023(path).unwrap();
                let edges = test0.edge_references()
                    .map(|e| -> (NodeIndex, NodeIndex) { (e.source().index().into(), e.target().index().into()) });
                Graph::from_edges(test0.node_count(), edges)
            };

            let original_graph = graph.clone();

            let mut partition = Partition::new(graph.node_count());
            partition.refine_forward([0_u32]);

            super::ovp(&mut graph, &mut partition, |_| {});

            let p = partition.full_sub_partition();
            for part in p.part_indices(&partition) {
                let elements: Vec<_> = partition.elements(part).collect();
                assert_eq!(splitters(&original_graph, elements.as_slice()).count(), 0);
            }

            let part_sizes: Vec<_> = p.parts(&partition).map(|s| s.len()).collect();
            let part_size_max = part_sizes.iter().copied().max().unwrap_or(0);
            let part_size_arithmetic_mean = part_sizes.iter().map(|c| *c as f64).sum::<f64>() / (part_sizes.len() as f64);
            let part_size_harmonic_mean = (part_sizes.len() as f64) / part_sizes.iter().map(|c| 1.0 / (*c as f64)).sum::<f64>();
            //println!("{i} {:?}", to_vecs(p, &partition));
            println!("{i:<3} {:>6} {:>6} {:8.2} {:8.2}", graph.node_count(), part_size_max, part_size_arithmetic_mean, part_size_harmonic_mean);
        }
    }
}