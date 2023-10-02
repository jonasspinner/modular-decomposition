use std::fmt::{Debug, Formatter};
use std::iter::FusedIterator;

struct Node {
    //vertex: u32,
    prev: u32,
    next: u32,
    part: u32,
}

impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} <- {}[p{}] -> {}", self.prev, 0, self.part, self.next)
    }
}

#[derive(Debug, Copy, Clone)]
struct Part {
    first: u32,
    prev: u32,
    next: u32,
    gen: u32,
}

pub struct Partition {
    nodes: Vec<Node>,
    parts: Vec<Part>,
    removed_parts: Vec<u32>,
    first_part: u32,
    gen: u32,
}

impl Partition {
    pub(crate) fn new(size: usize) -> Self {
        assert!(0 < size);
        assert!(size < u32::MAX as _);

        let mut partition = Self { parts: vec![], nodes: Vec::with_capacity(size), removed_parts: vec![], first_part: u32::MAX, gen: 0 };
        partition.nodes.push(Node { prev: u32::MAX, next: 1, part: 0 });
        for u in 1..size {
            let u = u as u32;
            partition.nodes.push(Node { prev: u - 1, next: u + 1, part: 0 });
        }
        partition.nodes[size - 1].next = u32::MAX;
        partition.parts.push(Part { first: 0, prev: u32::MAX, next: u32::MAX, gen: partition.gen });
        partition.first_part = 0;
        assert_ne!(partition.first_part, u32::MAX);
        partition
    }


    fn check_node(nodes: &[Node], idx: u32) -> bool {
        let Node { prev: left, next: right, .. } = nodes[idx as usize];
        if left != u32::MAX && !(nodes[left as usize].next == idx) { return false; }
        if right != u32::MAX && !(nodes[right as usize].prev == idx) { return false; }
        true
    }

    fn insert_before(nodes: &mut [Node], a: u32, b: u32) {
        if a == b { return; }
        // a-1, a, a+1    -->     a-1, a+1
        // b-1, b, b+1            b-1, a, b, b+1
        let Node { prev: a_prev, next: a_next, .. } = nodes[a as usize];
        let Node { prev: b_prev, .. } = nodes[b as usize];

        if a_next == b { return; };

        if a_prev != u32::MAX { nodes[a_prev as usize].next = a_next; }
        if a_next != u32::MAX { nodes[a_next as usize].prev = a_prev; }

        nodes[a as usize].prev = b_prev;
        if b_prev != u32::MAX { nodes[b_prev as usize].next = a; }
        nodes[a as usize].next = b;
        nodes[b as usize].prev = a;

        assert!(Self::check_node(nodes, a));
        assert!(Self::check_node(nodes, b));
    }

    fn new_part(&mut self, first: u32, prev: u32, next: u32, gen: u32) -> u32 {
        let idx = if let Some(idx) = self.removed_parts.pop() {
            let part = &mut self.parts[idx as usize];
            part.first = first;
            part.prev = prev;
            part.next = next;
            part.gen = gen;
            idx
        } else {
            let idx = self.parts.len() as u32;
            self.parts.push(Part { first, prev, next, gen });
            idx
        };
        if prev != u32::MAX { self.parts[prev as usize].next = idx; } else {
            self.first_part = idx;
        }
        if next != u32::MAX { self.parts[next as usize].prev = idx; }
        idx
    }

    fn remove_part(&mut self, idx: u32) {
        let Part { prev, next, .. } = self.parts[idx as usize];
        if prev != u32::MAX { self.parts[prev as usize].next = next; } else {
            self.first_part = next;
        }
        if next != u32::MAX { self.parts[next as usize].prev = prev; }
        self.parts[idx as usize] = Part { first: u32::MAX, prev: u32::MAX, next: u32::MAX, gen: u32::MAX };
        self.removed_parts.push(idx);
    }

    pub(crate) fn refine(&mut self, pivots: impl IntoIterator<Item=u32>) {
        self.gen += 1;
        let is_appendable = |parts: &[Part], gen: u32, idx: u32| {
            idx < u32::MAX && parts[idx as usize].gen == gen
        };

        for pivot in pivots {
            let Node { next: node_next, part, .. } = self.nodes[pivot as usize];
            let Part { first, prev, .. } = self.parts[part as usize];

            debug_assert!(!is_appendable(&self.parts, self.gen, part));

            let prev_part_is_appendable = is_appendable(&self.parts, self.gen, prev);
            let pivot_is_first = first == pivot;
            let pivot_part_is_singleton =
                pivot_is_first &&
                    (node_next == u32::MAX || self.nodes[node_next as usize].part != part);


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
                    // Case 2.
                    // If the current pivot is the last pivot element of its part and the previous
                    // part is appendable, all elements of the original part (before the call to
                    // refine) were pivots and any element in the next part should not be appended
                    // to it. Therefore, mark it as not appendable.
                    self.nodes[pivot as usize].part = prev;
                    self.remove_part(part);
                    self.parts[prev as usize].gen = self.gen - 1;
                }
                debug_assert_eq!(self.nodes[pivot as usize].next, node_next);
            } else {
                // Append to the previous part if it is appendable (cases 4 and 6), otherwise create
                // a new part for the pivot (cases 3 and 5).
                self.nodes[pivot as usize].part = if prev_part_is_appendable { prev } else {
                    self.new_part(pivot, prev, part, self.gen)
                };
                if pivot_is_first {
                    // 3:  [.. a]' [x b ..]' -> [.. a]' [x]* [b ..]'
                    // 4:  [.. a]* [x b ..]' -> [.. a    x]* [b ..]'
                    // Update the old pivot part. The next node is the new first element of
                    // the old part, as it wasn't a singleton (cases 1 and 2).
                    self.parts[part as usize].first = node_next;
                    debug_assert_eq!(self.nodes[pivot as usize].next, node_next);
                } else {
                    // 5:  [.. a]' [b .. x ..]' -> [.. a]' [x]* [b ..]'
                    // 6:  [.. a]* [b .. x ..]' -> [.. a    x]* [b ..]'
                    // Move the pivot in the node list to the correct position.
                    Self::insert_before(&mut self.nodes, pivot, first);
                }
            }

            debug_assert_eq!(!pivot_part_is_singleton, is_appendable(&self.parts, self.gen, self.nodes[pivot as usize].part));
            let b = self.nodes[pivot as usize].next;
            if b != u32::MAX {
                debug_assert_eq!(self.parts[self.nodes[b as usize].part as usize].first, b);
            }
            if !prev_part_is_appendable {
                debug_assert!(!is_appendable(&self.parts, self.gen, prev));
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn to_dot(&self) -> String {
        let mut result = String::new();
        let node_idx = |i| -> u32 { i };
        let part_idx = |i| -> u32 { i + self.nodes.len() as u32 };
        result.push_str(&format!("digraph {{"));
        for (i, node) in self.nodes.iter().enumerate() {
            let i = i as u32;
            result.push_str(&format!("\t{} [ label = \"{}\" ]", node_idx(i), i));
            if node.prev != u32::MAX {
                result.push_str(&format!("\t{} -> {}", node_idx(i), node_idx(node.prev)));
            }
            if node.next != u32::MAX {
                result.push_str(&format!("\t{} -> {}", node_idx(i), node_idx(node.next)));
            }
            result.push_str(&format!("\t{} -> {}", node_idx(i), part_idx(node.part)));
        }
        for (i, part) in self.parts.iter().enumerate() {
            let i = i as u32;
            result.push_str(&format!("\t{} [ label = \"part {}\" ]", part_idx(i), i));
            result.push_str(&format!("\t{} -> {}", part_idx(i), node_idx(part.first)));
            //result.push_str(&format!("\t{} -> {}", part_idx(i), node_idx(part.last)));
        }
        result.push_str(&format!("}}"));
        result
    }
}

pub(crate) struct AllElementsIterator<'a> {
    nodes: &'a [Node],
    next: u32,
    len: u32,
}

impl<'a> AllElementsIterator<'a> {
    fn new(partition: &'a Partition) -> Self {
        Self { nodes: &*partition.nodes, next: partition.parts[partition.first_part as usize].first, len: partition.nodes.len() as _ }
    }
}

impl Iterator for AllElementsIterator<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == u32::MAX { return None; }
        let current = self.next;
        self.next = self.nodes[current as usize].next;
        self.len -= 1;
        Some(current)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len as usize;
        (len, Some(len))
    }
}

impl FusedIterator for AllElementsIterator<'_> {}

impl ExactSizeIterator for AllElementsIterator<'_> { fn len(&self) -> usize { self.len as _ } }

impl Partition {
    pub(crate) fn elements(&self) -> AllElementsIterator { AllElementsIterator::new(self) }
}

pub(crate) struct PartElementsIterator<'a> {
    nodes: &'a [Node],
    next: u32,
    part: u32,
}

impl<'a> PartElementsIterator<'a> {
    fn new(partition: &'a Partition, part: u32) -> Self {
        Self { nodes: &*partition.nodes, next: partition.parts[part as usize].first, part }
    }
}

impl Iterator for PartElementsIterator<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == u32::MAX || self.nodes[self.next as usize].part != self.part { return None; }
        let current = self.next;
        self.next = self.nodes[current as usize].next;
        Some(current)
    }
}

impl FusedIterator for PartElementsIterator<'_> {}

pub(crate) struct PartsIterator<'a> {
    partition: &'a Partition,
    next: u32,
    len: u32,
}

impl<'a> PartsIterator<'a> {
    fn new(partition: &'a Partition) -> Self {
        PartsIterator { partition, next: partition.first_part, len: partition.parts.len() as _ }
    }
}

impl<'a> Iterator for PartsIterator<'a> {
    type Item = PartElementsIterator<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == u32::MAX { return None; }
        let current = self.next;
        self.next = self.partition.parts[current as usize].next;
        self.len -= 1;
        Some(PartElementsIterator::new(self.partition, current))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len as usize;
        (len, Some(len))
    }
}

impl FusedIterator for PartsIterator<'_> {}

impl ExactSizeIterator for PartsIterator<'_> { fn len(&self) -> usize { self.len as _ } }

impl Partition {
    pub(crate) fn parts(&self) -> PartsIterator {
        PartsIterator::new(self)
    }
}

impl Debug for Partition {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:?}", self.nodes)?;
        writeln!(f, "{:?}", self.parts)?;
        for (i, part) in self.parts().enumerate() {
            write!(f, "part {i}: [ ")?;
            for x in part {
                write!(f, "({x}, {}) ", self.nodes[x as usize].part)?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "")
    }
}


#[cfg(test)]
mod test {
    use crate::partition_refinement::Partition;

    fn example_partition() -> Partition {
        let mut partition = Partition::new(8);
        partition.refine([2, 3, 4, 5]);
        partition.refine([4, 5, 6, 7]);
        partition
    }

    fn to_vecs<I>(iter: I) -> Vec<Vec<u32>>
        where I: Iterator,
              <I as Iterator>::Item: Iterator<Item=u32>
    {
        iter.into_iter().map(|elements| elements.collect()).collect()
    }

    #[test]
    fn tiny() {
        let mut partition = Partition::new(4);
        assert_eq!(to_vecs(partition.parts()), [[0, 1, 2, 3]]);

        partition.refine([0, 2]);
        assert_eq!(to_vecs(partition.parts()), [[0, 2], [1, 3]]);

        partition.refine([0, 1, 2, 3]);
        assert_eq!(to_vecs(partition.parts()), [[0, 2], [1, 3]]);
    }

    #[test]
    fn medium() {
        let mut partition = Partition::new(8);
        assert_eq!(to_vecs(partition.parts()), [[0, 1, 2, 3, 4, 5, 6, 7]]);

        partition.refine([0, 1, 2]);
        assert_eq!(to_vecs(partition.parts()), [vec![0, 1, 2], vec![3, 4, 5, 6, 7]]);

        partition.refine([3, 4]);
        assert_eq!(to_vecs(partition.parts()), [vec![0, 1, 2], vec![3, 4], vec![5, 6, 7]]);

        println!("{:?}", partition);
        partition.refine([0, 2, 4, 6]);
        println!("{:?}", partition);
        let parts = to_vecs(partition.parts());
        println!("{:?}", parts);
        assert_eq!(parts, [vec![0, 2], vec![1], vec![4], vec![3], vec![6], vec![5, 7]]);

        partition.refine([0, 1, 2, 3, 4, 5, 6, 7]);
        println!("{:?}", partition);
        let parts = to_vecs(partition.parts());
        println!("{:?}", parts);
        assert_eq!(to_vecs(partition.parts()), [vec![0, 2], vec![1], vec![4], vec![3], vec![6], vec![5, 7]]);
    }

    #[test]
    fn change_ends() {
        let mut partition = Partition::new(8);
        assert_eq!(to_vecs(partition.parts()), [[0, 1, 2, 3, 4, 5, 6, 7]]);

        partition.refine([2, 3, 4, 5]);
        assert_eq!(to_vecs(partition.parts()), [[2, 3, 4, 5], [0, 1, 6, 7]]);

        partition.refine([4, 5, 6, 7]);
        assert_eq!(to_vecs(partition.parts()), [[4, 5], [2, 3], [6, 7], [0, 1]]);

        partition.refine([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(to_vecs(partition.parts()), [vec![4, 5], vec![2, 3], vec![6, 7], vec![0, 1]]);

        partition.refine([5, 3, 6, 0]);
        assert_eq!(to_vecs(partition.parts()), [[5], [4], [3], [2], [6], [7], [0], [1]]);

        partition.refine([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(to_vecs(partition.parts()), [[5], [4], [3], [2], [6], [7], [0], [1]]);
    }
}