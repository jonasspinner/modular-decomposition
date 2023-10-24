use std::iter::{from_fn, Map};
use std::slice::Iter;
use common::linked_list::{concat, cut, CyclicLinkedList, CyclicLinkedListNode, elements, move_before};
use common::make_index;
use common::set::FastResetBitSet;
use crate::linked_list::graph::{EdgeIndex, Graph, NodeIndex};

make_index!(pub(crate) PartIndex);

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
struct Gen(pub u32);

#[derive(Debug, Clone)]
struct Node {
    next: NodeIndex,
    prev: NodeIndex,
    part: PartIndex,
}

#[derive(Debug, Clone)]
struct Part {
    next: PartIndex,
    prev: PartIndex,
    first: NodeIndex,
    gen: Gen,
}

#[derive(Debug, Clone)]
struct Partition {
    first: PartIndex,
}

pub(crate) struct PartitionArena {
    nodes: Vec<Node>,
    parts: Vec<Part>,
    removed: Vec<PartIndex>,
    gen: Gen,
}

impl CyclicLinkedListNode for Node {
    type Index = NodeIndex;
    fn prev(&self) -> Self::Index { self.prev }
    fn prev_mut(&mut self) -> &mut Self::Index { &mut self.prev }
    fn next(&self) -> Self::Index { self.next }
    fn next_mut(&mut self) -> &mut Self::Index { &mut self.next }
    fn prev_next(&self) -> (Self::Index, Self::Index) { (self.prev, self.next) }
}

impl CyclicLinkedListNode for Part {
    type Index = PartIndex;
    fn prev(&self) -> Self::Index { self.prev }
    fn prev_mut(&mut self) -> &mut Self::Index { &mut self.prev }
    fn next(&self) -> Self::Index { self.next }
    fn next_mut(&mut self) -> &mut Self::Index { &mut self.next }
    fn prev_next(&self) -> (Self::Index, Self::Index) { (self.prev, self.next) }
}

impl CyclicLinkedList for Part {
    type Node = Node;
    fn head(&self) -> NodeIndex { self.first }
    fn head_mut(&mut self) -> &mut NodeIndex { &mut self.first }
}

impl CyclicLinkedList for Partition {
    type Node = Part;
    fn head(&self) -> PartIndex { self.first }
    fn head_mut(&mut self) -> &mut PartIndex { &mut self.first }
}

impl PartitionArena {
    fn new(size: usize) -> (Self, Partition) {
        let gen = Gen(0);
        if size == 0 {
            return (Self { nodes: vec![], parts: vec![], removed: vec![], gen }, Partition { first: PartIndex::invalid() });
        }
        let mut nodes = Vec::with_capacity(size);
        let part = PartIndex::new(0);
        let parts = vec![Part {
            next: part,
            prev: part,
            first: NodeIndex::new(0),
            gen,
        }];
        let first = NodeIndex::new(0);
        nodes.push(Node { next: 1_usize.into(), prev: (size - 1).into(), part });
        for i in 1..size {
            nodes.push(Node { next: (i + 1).into(), prev: (i - 1).into(), part });
        }
        nodes[size - 1].next = first;
        (Self { nodes, parts, removed: vec![], gen }, Partition { first: PartIndex::new(0) })
    }

    fn from_iter<I>(size: usize, elements: I) -> (Self, Partition)
        where I: IntoIterator,
              I::Item: Into<NodeIndex> {
        let gen = Gen(0);
        let first_part = PartIndex::new(0);
        let mut nodes = vec![Node { prev: NodeIndex::invalid(), next: NodeIndex::invalid(), part: first_part }; size];
        let mut elements = elements.into_iter();
        let first = elements.next().unwrap().into();
        let part = Part {
            next: first_part,
            prev: first_part,
            first,
            gen,
        };
        nodes[first.index()] = Node { prev: first, next: first, part: first_part };
        let mut prev = first;
        for element in elements {
            let element: NodeIndex = element.into();
            nodes[prev.index()].next = element;
            nodes[element.index()].prev = prev;
            prev = element;
        }
        nodes[prev.index()].next = first;
        nodes[first.index()].prev = prev;
        (Self {
            nodes,
            parts: vec![part],
            removed: vec![],
            gen,
        }, Partition { first: first_part })
    }

    fn new_part(&mut self, first: NodeIndex, prev: PartIndex, next: PartIndex, gen: Gen) -> PartIndex {
        let idx = if let Some(idx) = self.removed.pop() {
            let part = &mut self.parts[idx.index()];
            part.first = first;
            part.prev = prev;
            part.next = next;
            part.gen = gen;
            idx
        } else {
            let idx = PartIndex::new(self.parts.len());
            self.parts.push(Part { first, prev, next, gen });
            idx
        };
        if prev.is_valid() { self.parts[prev.index()].next = idx; } else {
            // self.first_part = idx;
            todo!()
        }
        if next.is_valid() { self.parts[next.index()].prev = idx; }
        idx
    }
    fn remove_part(&mut self, part: PartIndex) {
        cut(&mut self.parts, part);
        self.parts[part.index()].first = Default::default();
        self.removed.push(part)
    }
}

impl Partition {
    fn refine_forward<I>(&mut self, arena: &mut PartitionArena, pivots: I) where
        I: IntoIterator,
        I::Item: Into<NodeIndex>,
    {
        arena.gen.0 += 1;
        for pivot in pivots {
            let pivot: NodeIndex = pivot.into();
            //println!("{:?}", pivot.index());
            let Node { next: node_next, part, .. } = arena.nodes[pivot.index()];
            let Part { first, prev, .. } = arena.parts[part.index()];

            if arena.parts[part.index()].gen == arena.gen { continue; }

            let prev_part_is_appendable = arena.parts[prev.index()].gen == arena.gen && self.first != part;
            let pivot_is_first = first == pivot;
            let pivot_part_is_singleton = pivot_is_first && arena.nodes[node_next.index()].part != part;


            // There are several different cases. A part is called `appendable` if we already added
            // an element from the current pivot set and it is possible for another element to be
            // added to this new part. An appendable part is marked with "*" and a part which is
            // definitely not appendable is marked with "'".
            // Note, the part of the pivot element is never appendable.
            // We branch on whether the previous part is appendable ([.. a]*) or not ([.. a]').
            // The case where the pivot is the first element of the first part is treated as if the
            // previous part is not appendable.
            if pivot_part_is_singleton {
                // 1:  [.. a]' [x]' [b ..] -> [.. a]' [x]' [b ..]
                // 2:  [.. a]* [x]' [b ..] -> [.. a    x]' [b ..]
                if prev_part_is_appendable {
                    //println!("{} 2", pivot.index());
                    // Case 2.
                    // If the current pivot is the last pivot element of its part and the previous
                    // part is appendable, all elements of the original part (before the call to
                    // refine) were pivots and any element in the next part should not be appended
                    // to it. Therefore, mark it as not appendable.
                    arena.nodes[pivot.index()].part = prev;
                    arena.remove_part(part);
                    arena.parts[prev.index()].gen.0 = arena.gen.0 - 1;
                } else {
                    //println!("{} 1", pivot.index());
                }
                debug_assert_eq!(arena.nodes[pivot.index()].next, node_next);
            } else {
                /*
                let v = self.parts(arena)
                    .map(|part|
                        (arena.parts[part.index()].gen,
                         arena.parts[part.index()].gen == arena.gen,
                         part.elements(arena)
                             .map(|i| i.index() as u32)
                             .collect::<Vec<_>>()))
                    .collect::<Vec<_>>();
                 */
                // Append to the previous part if it is appendable (cases 4 and 6), otherwise create
                // a new part for the pivot (cases 3 and 5).
                arena.nodes[pivot.index()].part = if prev_part_is_appendable { prev } else {
                    arena.new_part(pivot, prev, part, arena.gen)
                };
                if pivot_is_first {
                    //println!("{} 3/4", pivot.index());
                    //println!("{:?}", v);
                    // 3:  [.. a]' [x b ..]' -> [.. a]' [x]* [b ..]'
                    // 4:  [.. a]* [x b ..]' -> [.. a    x]* [b ..]'
                    // Update the old pivot part. The next node is the new first element of
                    // the old part, as it wasn't a singleton (cases 1 and 2).
                    arena.parts[part.index()].first = node_next;
                    debug_assert_eq!(arena.nodes[pivot.index()].next, node_next);
                } else {
                    //println!("{} 5/6", pivot.index());
                    // 5:  [.. a]' [b .. x ..]' -> [.. a]' [x]* [b ..]'
                    // 6:  [.. a]* [b .. x ..]' -> [.. a    x]* [b ..]'
                    // Move the pivot in the node list to the correct position.
                    move_before(&mut arena.nodes, pivot, first);
                }
            }

            let curr = arena.nodes[pivot.index()].part;
            debug_assert_eq!(arena.nodes[arena.parts[prev.index()].first.index()].part, prev);
            if arena.parts[part.index()].first.is_valid() {
                debug_assert_eq!(arena.nodes[arena.parts[part.index()].first.index()].part, part);
            }
            debug_assert_eq!(arena.nodes[arena.parts[curr.index()].first.index()].part, curr);

            if part == self.first {
                self.first = arena.nodes[pivot.index()].part;
            }

            debug_assert_eq!(!pivot_part_is_singleton, arena.parts[arena.nodes[pivot.index()].part.index()].gen == arena.gen);
            let b = arena.nodes[pivot.index()].next;
            debug_assert_eq!(arena.parts[arena.nodes[b.index()].part.index()].first, b);
            if !prev_part_is_appendable {
                /*
                if arena.parts[prev.index()].gen == arena.gen {
                    println!("{:?} {:?} [{:?}, ...] [{:?}, ...]", arena.gen, pivot.index(), arena.parts[prev.index()].first.index(), arena.parts[part.index()].first.index());
                    println!("{}: {:?}", prev.index(), arena.parts[prev.index()]);
                    println!("{}: {:?}", arena.nodes[pivot.index()].part.index(), arena.parts[arena.nodes[pivot.index()].part.index()]);
                    println!("{}: {:?}", part.index(), arena.parts[part.index()]);
                } else {
                    let v = self.parts(arena)
                        .map(|part|
                            (arena.parts[part.index()].gen,
                             arena.parts[part.index()].gen == arena.gen,
                             part.elements(arena)
                                 .map(|i| i.index() as u32)
                                 .collect::<Vec<_>>()))
                        .collect::<Vec<_>>();
                    println!("{:?} {:?} [{:?}, ...] [{:?}, ...] {:?}", arena.gen, pivot.index(), arena.parts[prev.index()].first.index(), arena.parts[part.index()].first.index(), v);
                }
                */
                debug_assert_ne!(arena.parts[prev.index()].gen, arena.gen);
            }
        }
    }

    fn refine_backward<I>(&mut self, arena: &mut PartitionArena, pivots: I) where
        I: IntoIterator,
        I::Item: Into<NodeIndex>,
    {
        arena.gen.0 += 1;
        for pivot in pivots {
            let pivot: NodeIndex = pivot.into();
            let Node { next: node_next, part, .. } = arena.nodes[pivot.index()];
            let Part { first, next, .. } = arena.parts[part.index()];

            if arena.parts[part.index()].gen == arena.gen { continue; }

            let next_part_is_appendable = arena.parts[next.index()].gen == arena.gen && self.first != next;
            let pivot_is_first = first == pivot;
            let pivot_part_is_singleton = pivot_is_first && arena.nodes[node_next.index()].part != part;

            if pivot_part_is_singleton {
                // 1:  [.. a] [x]' [b ..]' [c ..] -> [.. a] [x]' [b ..]' [c ..]
                // 2:  [.. a] [x]' [b ..]* [c ..] -> [.. a] [b ..    x]' [c ..]
                if next_part_is_appendable {
                    // Case 2.
                    let c = arena.parts[arena.parts[next.index()].next.index()].first;
                    if c != pivot {
                        move_before(&mut arena.nodes, pivot, c);
                    }
                    arena.nodes[pivot.index()].part = next;
                    arena.remove_part(part);
                    arena.parts[next.index()].gen.0 = arena.gen.0 - 1;
                    if self.first == part {
                        self.first = next;
                    }
                }
            } else {
                arena.nodes[pivot.index()].part = if next_part_is_appendable { next } else {
                    arena.new_part(pivot, part, next, arena.gen)
                };

                if next_part_is_appendable {
                    // 4:  [.. a] [x b    ..]' [c ..]* [d ..] -> [.. a] [b ..]' [c .. x]* [d ..]
                    // 6:  [.. a] [b .. x ..]' [c ..]* [d ..] -> [.. a] [b ..]' [c .. x]* [d ..]
                    let next_next = arena.parts[next.index()].next;
                    let d = arena.parts[next_next.index()].first;
                    if d != pivot {
                        move_before(&mut arena.nodes, pivot, d);
                    }
                } else {
                    // 3:  [.. a] [x b    ..]' [c ..]' -> [.. a] [b ..]' [x]* [c ..]'
                    // 5:  [.. a] [b .. x ..]' [c ..]' -> [.. a] [b ..]' [x]* [c ..]'
                    let c = arena.parts[next.index()].first;
                    if c != pivot {
                        move_before(&mut arena.nodes, pivot, c);
                    }
                }

                if pivot_is_first {
                    // 3:  [.. a] [x b ..]' [c ..]' [d ..] -> [.. a] [b ..]' [x]* [c ..]' [d ..]
                    // 4:  [.. a] [x b ..]' [c ..]* [d ..] -> [.. a] [b ..]' [c ..    x]* [d ..]
                    arena.parts[part.index()].first = node_next;
                } else {
                    // 5:  [.. a] [b .. x ..]' [c ..]' [d ..] -> [.. a] [b ..]' [x]* [c ..]' [d ..]
                    // 6:  [.. a] [b .. x ..]' [c ..]* [d ..] -> [.. a] [b ..]' [c ..    x]* [d ..]
                }
            }

            let curr = arena.nodes[pivot.index()].part;
            debug_assert_eq!(arena.nodes[arena.parts[next.index()].first.index()].part, next);
            if arena.parts[part.index()].first.is_valid() {
                debug_assert_eq!(arena.nodes[arena.parts[part.index()].first.index()].part, part);
            }
            debug_assert_eq!(arena.nodes[arena.parts[curr.index()].first.index()].part, curr);

            debug_assert_eq!(!pivot_part_is_singleton, arena.parts[arena.nodes[pivot.index()].part.index()].gen == arena.gen);

            if !next_part_is_appendable {
                debug_assert_ne!(arena.parts[next.index()].gen, arena.gen);
            }
        }
    }

    pub fn cut(&mut self, arena: &mut PartitionArena, part: PartIndex) -> Option<Partition> {
        if self.first == part {
            self.first = arena.parts[self.first.index()].next;
            if self.first == part { return None; }
        }
        cut(&mut arena.parts, part);
        Some(Partition { first: part })
    }

    pub fn concat(self, other: Self, arena: &mut PartitionArena) -> Self {
        concat(&mut arena.parts, self.head(), other.head());
        self
    }

    pub fn parts<'arena>(&self, arena: &'arena PartitionArena) -> impl Iterator<Item=PartIndex> + 'arena {
        elements(&arena.parts, self)
    }
}

impl PartIndex {
    pub fn move_to_front(&self, a: NodeIndex, arena: &mut PartitionArena) {
        todo!()
    }
    pub fn move_to_back(&self, a: NodeIndex, arena: &mut PartitionArena) {
        let Node { part, next, .. } = arena.nodes[a.index()];
        if arena.parts[part.index()].first == a {
            if arena.nodes[next.index()].part == part {
                arena.parts[part.index()].first = next;
            } else if *self != part {
                arena.remove_part(part);
            }
        }
        let Part { next, .. } = arena.parts[self.index()];
        let b = arena.parts[next.index()].first;
        move_before(&mut arena.nodes, a, b);
    }
    pub fn elements<'arena>(&self, arena: &'arena PartitionArena) -> impl Iterator<Item=NodeIndex> + 'arena {
        let head = arena.parts[self.index()].first;
        let mut next = head;
        let part = *self;
        from_fn(move || {
            if next.is_valid() {
                let current = next;
                let Node { part: p, next: n, .. } = arena.nodes[current.index()];
                next = if n != head { n } else { NodeIndex::invalid() };
                if p == part {
                    return Some(current);
                }
            }
            None
        })
    }
}


fn cut_partition(mut p: Partition, x: PartIndex, arena: &mut PartitionArena) -> (Partition, Partition) {
    let q = p.cut(arena, x).unwrap();
    (q, p)
}

fn crossing_edges(graph: &mut Graph, x: PartIndex, arena: &PartitionArena, set: &mut FastResetBitSet) -> Vec<(NodeIndex, NodeIndex, EdgeIndex)> {
    let mut e = vec![];
    set.clear();
    for u in x.elements(arena) {
        set.set(u);
    }
    for u in x.elements(arena) {
        for (v, i) in graph.incident_edges(u) {
            if !set.get(v) {
                e.push((u, v, i));
            }
        }
    }
    e
}

fn indices(edges: &[(NodeIndex, NodeIndex, EdgeIndex)]) -> impl Iterator<Item=EdgeIndex> + '_ {
    edges.iter().map(|(_, _, e)| *e)
}

fn group_edges<'a, F>(edges: &'a mut Vec<(NodeIndex, NodeIndex, EdgeIndex)>, f: F) -> impl Iterator<Item=impl Iterator<Item=&(NodeIndex, NodeIndex, EdgeIndex)> +'a> +'a
    where
        F: Fn(&(NodeIndex, NodeIndex, EdgeIndex)) -> NodeIndex,
        F: 'a,
{
    edges.sort_unstable_by_key(|e| f(e).index());

    struct GroupIter<'b, F: Fn(&(NodeIndex, NodeIndex, EdgeIndex)) -> NodeIndex> {
        edges: &'b Vec<(NodeIndex, NodeIndex, EdgeIndex)>,
        i: usize,
        f: F,
    }
    impl<'b, F: Fn(&(NodeIndex, NodeIndex, EdgeIndex)) -> NodeIndex> Iterator for GroupIter<'b, F> {
        type Item = Iter<'b, (NodeIndex, NodeIndex, EdgeIndex)>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.i < self.edges.len() {
                let (start, mut end) = (self.i, self.i + 1);
                let key = (self.f)(&self.edges[start]);
                while end < self.edges.len() && (self.f)(&self.edges[end]) == key { end += 1; }
                self.i = end;
                return Some(self.edges[start..end].iter());
            }
            None
        }
    }
    GroupIter { edges, i: 0, f }
}

fn split(graph: &mut Graph, x: PartIndex, p: Partition, arena: &mut PartitionArena, set: &mut FastResetBitSet) -> (Partition, Partition) {

    let mut edges = crossing_edges(graph, x, arena, set);

    assert!(edges.iter().all(|(_, _, i)| i.is_valid()));
    println!("{:?}", edges.iter().map(|(u, v, _)| (u.index(), v.index())).collect::<Vec<_>>());

    graph.remove_edges(indices(&edges));

    let (mut q, mut q_prime) = cut_partition(p, x, arena);

    for pivots in group_edges(&mut edges, |e| { e.0 }) {
        q_prime.refine_forward(arena, pivots.map(|e| e.1));
    }

    for pivots in group_edges(&mut edges, |e| { e.1 }) {
        q.refine_backward(arena, pivots.map(|e| e.0));
    }

    (q, q_prime)
}

fn partition(graph: &mut Graph, p: Partition, arena: &mut PartitionArena, set: &mut FastResetBitSet) -> Partition {
    println!("partition: {:?}", p.parts(arena).map(|part| part.elements(arena).map(|i| i.index() as u32).collect::<Vec<_>>()).collect::<Vec<_>>());
    if p.parts(&arena).count() == 1 {
        p
    } else {
        let x = p.parts(&arena).next().unwrap();
        let (q, q_prime) = split(graph, x, p, arena, set);

        let p1 = partition(graph, q, arena, set);
        let p2 = partition(graph, q_prime, arena, set);
        p1.concat(p2, arena)
    }
}

/*
edges.sort_unstable_by_key(|e| e.0.index());
let mut i = 0;
while i < edges.len() {
    let mut j = i + 1;
    while j < edges.len() && edges[j].0 == edges[i].0 { j += 1; }
    q_prime.refine_forward(arena, heads(&edges[i..j]));
    i = j;
}
edges.sort_unstable_by_key(|e| e.1.index());
let mut i = 0;
while i < edges.len() {
    let mut j = i + 1;
    while j < edges.len() && edges[j].1 == edges[i].1 { j += 1; }
    q.refine_backward(arena, tails(&edges[i..j]));
    i = j;
}
*/

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use petgraph::visit::EdgeRef;
    use rand::{Rng, SeedableRng};
    use rand::seq::{IteratorRandom, SliceRandom};
    use common::instances::ted08_test0;
    use common::set::FastResetBitSet;
    use crate::linked_list::graph::{EdgeIndex, Graph};
    use crate::linked_list::partition;
    use super::*;


    #[test]
    fn new() {
        let (arena, partition) = PartitionArena::new(10);

        assert_eq!(partition.parts(&arena).count(), 1);
        let part = partition.parts(&arena).next().unwrap();
        assert_eq!(part.elements(&arena).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(NodeIndex::new));
    }


    #[test]
    fn from_iter() {
        let elements = [3, 5, 6, 8].map(NodeIndex::new);
        let (arena, partition) = PartitionArena::from_iter(10, elements);

        assert_eq!(partition.parts(&arena).count(), 1);
        let part = partition.parts(&arena).next().unwrap();
        assert_eq!(part.elements(&arena).collect::<Vec<_>>(), elements);
    }

    fn to_vecs(p: &Partition, arena: &PartitionArena) -> Vec<Vec<u32>> {
        p.parts(arena).map(|part| part.elements(arena).map(|i| i.index() as u32).collect()).collect()
    }

    #[test]
    fn refine_forward() {
        let (mut arena, mut partition) = PartitionArena::new(10);

        assert_eq!(partition.parts(&arena).count(), 1);

        assert_eq!(to_vecs(&partition, &arena), [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]);
        partition.refine_forward(&mut arena, [5_u32, 6]);
        assert_eq!(to_vecs(&partition, &arena), [vec![5, 6], vec![0, 1, 2, 3, 4, 7, 8, 9]]);

        partition.refine_forward(&mut arena, [1_u32, 3, 7]);
        assert_eq!(to_vecs(&partition, &arena), [vec![5, 6], vec![1, 3, 7], vec![0, 2, 4, 8, 9]]);

        partition.refine_forward(&mut arena, [5_u32, 6, 1, 3, 0, 4, 9]);
        assert_eq!(to_vecs(&partition, &arena), [vec![5, 6], vec![1, 3], vec![7], vec![0, 4, 9], vec![2, 8]]);

        partition.refine_forward(&mut arena, [9_u32, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(to_vecs(&partition, &arena), [vec![6, 5], vec![3, 1], vec![7], vec![9, 4, 0], vec![8, 2]]);

        partition.refine_forward(&mut arena, [0_u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(to_vecs(&partition, &arena), [vec![5, 6], vec![1, 3], vec![7], vec![0, 4, 9], vec![2, 8]]);
    }

    #[test]
    fn refine_backward() {
        let (mut arena, mut partition) = PartitionArena::new(10);

        assert_eq!(partition.parts(&arena).count(), 1);

        assert_eq!(to_vecs(&partition, &arena), [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]);
        partition.refine_backward(&mut arena, [5_u32, 6]);
        assert_eq!(to_vecs(&partition, &arena), [vec![0, 1, 2, 3, 4, 7, 8, 9], vec![5, 6]]);

        partition.refine_backward(&mut arena, [1_u32, 3, 7]);
        assert_eq!(to_vecs(&partition, &arena), [vec![0, 2, 4, 8, 9], vec![1, 3, 7], vec![5, 6]]);

        partition.refine_backward(&mut arena, [5_u32, 6, 1, 3, 0, 4, 9]);
        assert_eq!(to_vecs(&partition, &arena), [vec![2, 8], vec![0, 4, 9], vec![7], vec![1, 3], vec![5, 6]]);

        partition.refine_backward(&mut arena, [9_u32, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(to_vecs(&partition, &arena), [vec![8, 2], vec![9, 4, 0], vec![7], vec![3, 1], vec![6, 5]]);

        partition.refine_backward(&mut arena, [0_u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(to_vecs(&partition, &arena), [vec![2, 8], vec![0, 4, 9], vec![7], vec![1, 3], vec![5, 6]]);
    }

    #[test]
    fn random_refine_forward() {
        let k = 256;
        let l = 16;
        let m = 1024;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for i in 0..k {
            let size = rng.gen_range(1..m);
            let (mut arena, mut partition) = PartitionArena::new(size);

            let all: Vec<_> = (0..size).collect();
            for j in 0..l {
                let num_elements = rng.gen_range(0..size);

                let mut pivots = all.iter().copied().choose_multiple(&mut rng, num_elements);
                pivots.shuffle(&mut rng);
                partition.refine_forward(&mut arena, pivots);
            }

            let elements: HashSet<_> = partition.parts(&arena).flat_map(|p| p.elements(&arena).map(|i| i.index())).collect();
            assert_eq!(elements, HashSet::from_iter(all));
            // println!("{} {} {:?}", i, size, to_vecs(&partition, &arena));
        }
    }

    #[test]
    fn random_refine_backward() {
        let k = 256;
        let l = 16;
        let m = 1024;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for i in 0..k {
            let size = rng.gen_range(1..m);
            let (mut arena, mut partition) = PartitionArena::new(size);

            let all: Vec<_> = (0..size).collect();
            for j in 0..l {
                let num_elements = rng.gen_range(0..size);

                let mut pivots = all.iter().copied().choose_multiple(&mut rng, num_elements);
                pivots.shuffle(&mut rng);
                partition.refine_backward(&mut arena, pivots);
            }

            let elements: HashSet<_> = partition.parts(&arena).flat_map(|p| p.elements(&arena).map(|i| i.index())).collect();
            assert_eq!(elements, HashSet::from_iter(all));
            // println!("{} {} {:?}", i, size, to_vecs(&partition, &arena));
        }
    }

    #[test]
    fn partition() {
        let mut graph = {
            let test0 = ted08_test0();
            let edges: Vec<_> = test0.edge_references().map(|e| (e.source().index(), e.target().index())).collect();
            Graph::from_edges(test0.node_count(), &edges)
        };

        let (mut arena, mut partition) = PartitionArena::new(graph.node_count());
        partition.refine_forward(&mut arena, [6_u32]);

        let mut set = FastResetBitSet::new(graph.node_count());

        let partition = super::partition(&mut graph, partition, &mut arena, &mut set);

        println!("{:?}", to_vecs(&partition, &arena));
    }
}
