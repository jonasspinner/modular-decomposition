use std::fmt::{Debug, Formatter, Write};

struct Node {
    //vertex: u32,
    left: u32,
    right: u32,
    part: u32,
}

impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} <- {}[p{}] -> {}", self.left, 0, self.part, self.right)
    }
}

#[derive(Debug, Copy, Clone)]
struct Part {
    first: u32,
    last: u32,
    left: u32,
    right: u32,
}

pub struct StandardPartitionDataStructure {
    nodes: Vec<Node>,
    parts: Vec<Part>,
    pivots_per_part: Vec<Vec<u32>>,
}

impl StandardPartitionDataStructure {
    pub(crate) fn new(num_vertices: usize) -> Self {
        let mut partition = Self { parts: vec![], nodes: vec![], pivots_per_part: vec![] };
        partition.nodes.reserve(num_vertices);
        for u in (0..num_vertices).map(|i| i as u32) {
            partition.nodes.push(Node { left: u.wrapping_sub(1), right: u + 1, part: 0 });
        }
        if num_vertices > 0 {
            partition.nodes[num_vertices - 1].right = u32::MAX;
        }
        partition.parts.push(Part { first: 0, last: (num_vertices - 1) as u32, left: u32::MAX, right: u32::MAX });
        partition
    }


    fn check_node(nodes: &[Node], idx: u32) -> bool {
        let Node { left, right, .. } = nodes[idx as usize];
        if left != u32::MAX && !(nodes[left as usize].right == idx) { return false; }
        if right != u32::MAX && !(nodes[right as usize].left == idx) { return false; }
        true
    }
    fn insert_after(nodes: &mut [Node], a: u32, b: u32) {
        if a == b { return; }
        // a-1, a, a+1    -->     a-1, a+1
        // b-1, b, b+1            b-1, b, a, b+1
        let Node { left: a_left, right: a_right, .. } = nodes[a as usize];
        let Node { right: b_right, .. } = nodes[b as usize];

        if a_left == b { return; };

        if a_left != u32::MAX { nodes[a_left as usize].right = a_right; }
        if a_right != u32::MAX { nodes[a_right as usize].left = a_left; }

        nodes[a as usize].right = b_right;
        if b_right != u32::MAX { nodes[b_right as usize].left = a; }
        nodes[a as usize].left = b;
        nodes[b as usize].right = a;

        assert!(Self::check_node(nodes, a));
        assert!(Self::check_node(nodes, b));
    }
    fn insert_before(nodes: &mut [Node], a: u32, b: u32) {
        if a == b { return; }
        // a-1, a, a+1    -->     a-1, a+1
        // b-1, b, b+1            b-1, a, b, b+1
        let Node { left: a_left, right: a_right, .. } = nodes[a as usize];
        let Node { left: b_left, .. } = nodes[b as usize];

        if a_right == b { return; };

        if a_left != u32::MAX { nodes[a_left as usize].right = a_right; }
        if a_right != u32::MAX { nodes[a_right as usize].left = a_left; }

        nodes[a as usize].left = b_left;
        if b_left != u32::MAX { nodes[b_left as usize].right = a; }
        nodes[a as usize].right = b;
        nodes[b as usize].left = a;

        assert!(Self::check_node(nodes, a));
        assert!(Self::check_node(nodes, b));
    }

    /// Requires `pivot` to be in ascending order.
    pub(crate) fn refine(&mut self, pivot: impl IntoIterator<Item=u32> + Copy) {
        // Similar to a part, but is allowed to be empty.
        struct IncompletePart {
            first: u32,
            last: u32,
        }
        impl IncompletePart {
            fn new() -> Self { IncompletePart { first: u32::MAX, last: u32::MAX } }
            fn is_empty(&self) -> bool { self.first == u32::MAX && self.last == u32::MAX }
            fn as_part(&self) -> Option<Part> {
                if self.is_empty() { None } else { Some(Part { first: self.first, last: self.last, left: u32::MAX, right: u32::MAX }) }
            }
            fn push(&mut self, nodes: &mut [Node], idx: u32) {
                if self.is_empty() { self.first = idx; } else {
                    StandardPartitionDataStructure::insert_after(nodes, idx, self.last);
                }
                self.last = idx;
            }
        }

        // Utility struct to walk from `first` to `last` of a part. Allows the modification of the
        // node corresponding to the returned index. The sibling node to the right is not allowed
        // to be modified.
        struct PartWalker {
            next: u32,
            last: u32,
        }
        impl PartWalker {
            fn new(part: &Part) -> Self { PartWalker { next: part.first, last: part.last } }
            fn next(&mut self, nodes: &[Node]) -> Option<u32> {
                if self.next == u32::MAX { return None; }
                let current = self.next;
                self.next = if current == self.last { u32::MAX } else { nodes[current as usize].right };
                Some(current)
            }
        }

        // Creates a predicate that returns whether a vertex x belongs to the set of pivots or not.
        // This requires that `pivots` are ordered in ascending value and that successive calls to
        // the predicate are also ordered in ascending value.
        // Note: This does not need to consume the Vec, but this way, we do not need to call `clear`
        //       explicitly.
        let create_pivot_predicate = |pivots: Vec<u32>| {
            let mut iter = pivots.into_iter().peekable();
            let mut predicate = move |vertex: u32| {
                while let Some(&pivot_vertex) = iter.peek() {
                    if pivot_vertex < vertex {
                        iter.next();
                    } else if pivot_vertex == vertex { return true; } else { return false; }
                }
                false
            };
            return predicate;
        };

        // In order to only touch parts which contain an element of the pivot set,
        // we collect all pivot sets grouped by their corresponding part.
        let mut pivots_per_part = std::mem::take(&mut self.pivots_per_part);
        pivots_per_part.resize(self.parts.len(), Vec::new());
        for x in pivot {
            pivots_per_part[self.nodes[x as usize].part as usize].push(x);
        }
        // We visit non-empty groups and work on them. The group is taken when we first visit it
        // and not worked on again a second time after that.
        for s in pivot {
            let i = self.nodes[s as usize].part as usize;

            // part: X, pivot set: S, part1: X ∩ S, part2: X \ S
            let part = self.parts[i];
            let mut part1 = IncompletePart::new();
            let mut part2 = IncompletePart::new();

            let pivots = std::mem::take(&mut pivots_per_part[i]);
            if pivots.is_empty() { continue; }

            let mut is_in_pivot_set = create_pivot_predicate(pivots);

            // For each x ∈ X, put it in part1 if x ∈ S and part2 if x ∉ S
            // Note: This can be further optimized, by stopping the iteration if the pivot set is
            //       empty and completing the parts in O(1) instead of iterating over the remaining
            //       vertices of the part.
            let mut walker = PartWalker::new(&part);
            while let Some(x) = walker.next(&self.nodes) {
                if is_in_pivot_set(x) {
                    part1.push(&mut self.nodes, x);
                } else {
                    part2.push(&mut self.nodes, x);
                }
            }

            // Update the parts

            // X = (X ∩ S) ∪ (X \ S)
            if let Some(part1) = part1.as_part() {
                if let Some(part2) = part2.as_part() {
                    // X ∩ S ≠ ∅ and X \ S ≠ ∅
                    // Add (X \ S) as a part and update the `part` information for the
                    // corresponding nodes
                    self.parts[i] = part1;

                    let j = self.parts.len() as u32;
                    self.parts.push(part2);

                    let mut walker = PartWalker::new(&part2);
                    while let Some(x) = walker.next(&self.nodes) {
                        self.nodes[x as usize].part = j;
                    }

                    debug_assert!(part.first == part1.first || part.first == part2.first);
                    debug_assert!(part.last == part1.last || part.last == part2.last);

                    let j = j as usize;
                    if part.right != u32::MAX { self.parts[part.right as usize].left = j as u32; }
                    self.parts[j].right = part.right;
                    self.parts[j].left = i as u32;
                    self.parts[i].right = j as u32;
                    self.parts[i].left = part.left;
                } else {
                    // X ∩ S ≠ ∅ and X \ S = ∅
                    // No new part
                    debug_assert_eq!(part.first, part1.first);
                    debug_assert_eq!(part.last, part1.last);
                }
            } else {
                // X ∩ S = ∅ and X \ S ≠ ∅
                // No new part
                debug_assert_eq!(part.first, part2.first);
                debug_assert_eq!(part.last, part2.last);
            }
        }

        self.pivots_per_part = std::mem::take(&mut pivots_per_part);
    }

    pub(crate) fn get_parts_ordered(&self) -> Vec<Vec<u32>> {
        let mut parts = Vec::with_capacity(self.parts.len());
        let mut i = 0;
        while i != u32::MAX {
            let mut part = vec![];
            let Part { first, last, right, .. } = self.parts[i as usize];
            let mut current = first;
            loop {
                part.push(current);
                if current == last { break; }
                current = self.nodes[current as usize].right;
            }
            parts.push(part);
            i = right;
        }
        parts
    }

    pub(crate) fn to_dot(&self) -> String {
        let mut result = String::new();
        let node_idx = |i| -> u32 { i };
        let part_idx = |i| -> u32 { i + self.nodes.len() as u32 };
        result.push_str(&format!("digraph {{"));
        for (i, node) in self.nodes.iter().enumerate() {
            let i = i as u32;
            result.push_str(&format!("\t{} [ label = \"{}\" ]", node_idx(i), i));
            if node.left != u32::MAX {
                result.push_str(&format!("\t{} -> {}", node_idx(i), node_idx(node.left)));
            }
            if node.right != u32::MAX {
                result.push_str(&format!("\t{} -> {}", node_idx(i), node_idx(node.right)));
            }
            result.push_str(&format!("\t{} -> {}", node_idx(i), part_idx(node.part)));
        }
        for (i, part) in self.parts.iter().enumerate() {
            let i = i as u32;
            result.push_str(&format!("\t{} [ label = \"part {}\" ]", part_idx(i), i));
            result.push_str(&format!("\t{} -> {}", part_idx(i), node_idx(part.first)));
            result.push_str(&format!("\t{} -> {}", part_idx(i), node_idx(part.last)));
        }
        result.push_str(&format!("}}"));
        result
    }
}

struct OrderedNodesIterator<'a> {
    nodes: &'a [Node],
    next: u32,
    last: u32,
}

impl<'a> OrderedNodesIterator<'a> {
    fn new(nodes : &'a [Node], part : &Part) -> Self {
        Self { nodes, next: part.first, last: part.last }
    }
}

impl Iterator for OrderedNodesIterator<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == u32::MAX { return None; }
        let current = self.next;
        self.next = if current == self.last { u32::MAX } else { self.nodes[current as usize].right };
        Some(current)
    }
}

struct PartRef<'a> {
    partition: &'a StandardPartitionDataStructure,
    idx: u32,
}
impl StandardPartitionDataStructure {
    fn part(&self, idx: u32) -> PartRef {
        PartRef { partition: self, idx }
    }
}
impl<'a> IntoIterator for PartRef<'a> {
    type Item = u32;
    type IntoIter = OrderedNodesIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        OrderedNodesIterator::new(&self.partition.nodes, &self.partition.parts[self.idx as usize])
    }
}

struct OrderedPartsIterator<'a> {
    partition: &'a StandardPartitionDataStructure,
    next: u32,
    last: u32,
}

impl<'a> OrderedPartsIterator<'a> {
    fn new(partition: &'a StandardPartitionDataStructure) -> Self {
        OrderedPartsIterator {partition, next: 0, last: partition.parts.len() as u32 }
    }
}

impl<'a> Iterator for OrderedPartsIterator<'a> {
    type Item = OrderedNodesIterator<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == u32::MAX { return None; }
        let current = self.next;
        let part = &self.partition.parts[current as usize];
        self.next = if current == self.last { u32::MAX } else { part.right };
        Some(OrderedNodesIterator::new(&self.partition.nodes, part))
    }
}

impl StandardPartitionDataStructure {
    fn ordered_parts(&self) -> OrderedPartsIterator {
        OrderedPartsIterator::new(self)
    }
}

impl Debug for StandardPartitionDataStructure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{:?}", self.nodes)?;
        writeln!(f, "{:?}", self.parts)?;
        for (i, part) in self.get_parts_ordered().into_iter().enumerate() {
            write!(f, "part {i}: [ ")?;
            for x in part {
                write!(f, "{x} ")?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "")
    }
}


#[cfg(test)]
mod test {
    use crate::standard_partition_data_structure::StandardPartitionDataStructure;

    fn example_partition() -> StandardPartitionDataStructure {
        let mut partition = StandardPartitionDataStructure::new(8);
        partition.refine([2, 3, 4, 5]);
        partition.refine([4, 5, 6, 7]);
        partition
    }

    #[test]
    fn tiny() {
        let mut partition = StandardPartitionDataStructure::new(4);

        partition.refine([0, 2]);
        assert_eq!(partition.get_parts_ordered(), vec![vec![0, 2], vec![1, 3]]);
    }

    #[test]
    fn medium() {
        let mut partition = StandardPartitionDataStructure::new(8);
        assert_eq!(partition.get_parts_ordered(), vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);

        partition.refine([0, 1, 2]);
        assert_eq!(partition.get_parts_ordered(), vec![vec![0, 1, 2], vec![3, 4, 5, 6, 7]]);

        partition.refine([3, 4]);
        assert_eq!(partition.get_parts_ordered(), vec![vec![0, 1, 2], vec![3, 4], vec![5, 6, 7]]);

        partition.refine([0, 2, 4, 6]);
        assert_eq!(partition.get_parts_ordered(), vec![vec![0, 2], vec![1], vec![4], vec![3], vec![6], vec![5, 7]]);
    }

    #[test]
    fn change_ends() {
        let mut partition = StandardPartitionDataStructure::new(8);
        assert_eq!(partition.get_parts_ordered(), vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);

        partition.refine([2, 3, 4, 5]);
        assert_eq!(partition.get_parts_ordered(), vec![vec![2, 3, 4, 5], vec![0, 1, 6, 7]]);

        partition.refine([4, 5, 6, 7]);
        assert_eq!(partition.get_parts_ordered(), vec![vec![4, 5], vec![2, 3], vec![6, 7], vec![0, 1]]);
    }

    #[test]
    fn part_ref_iter() {
        let partition = example_partition();
        assert_eq!(partition.part(0).into_iter().collect::<Vec<_>>(), vec![4, 5]);
        assert_eq!(partition.part(2).into_iter().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(partition.part(1).into_iter().collect::<Vec<_>>(), vec![6, 7]);
        assert_eq!(partition.part(3).into_iter().collect::<Vec<_>>(), vec![0, 1]);
    }

    #[test]
    fn parts_iter() {
        let partition = example_partition();
        let mut parts = partition.ordered_parts();
        assert_eq!(parts.next().unwrap().collect::<Vec<_>>(), vec![4, 5]);
        assert_eq!(parts.next().unwrap().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(parts.next().unwrap().collect::<Vec<_>>(), vec![6, 7]);
        assert_eq!(parts.next().unwrap().collect::<Vec<_>>(), vec![0, 1]);
        assert!(parts.next().is_none());
    }
}