use std::iter::from_fn;
use std::ops::{AddAssign, SubAssign};
use crate::graph::NodeIndex;
use common::make_index;

make_index!(pub(crate) PartIndex);
make_index!(pub(crate) NodePos);

impl<T: Into<NodePos>> AddAssign<T> for NodePos {
    fn add_assign(&mut self, rhs: T) {
        let rhs = rhs.into();
        self.0 += rhs.0;
    }
}

impl<T: Into<NodePos>> SubAssign<T> for NodePos {
    fn sub_assign(&mut self, rhs: T) {
        let rhs = rhs.into();
        self.0 -= rhs.0;
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct Gen(pub u32);

#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) node: NodeIndex,
    pub(crate) label: u32,
    part: PartIndex,
}

#[derive(Debug, Clone)]
struct PartInner {
    start: NodePos,
    len: u32,
    gen: Gen,
}

pub(crate) struct Partition {
    position: Vec<NodePos>,
    nodes: Vec<Node>,
    parts: Vec<PartInner>,
    removed: Vec<PartIndex>,
    gen: Gen,
}

impl Partition {
    pub(crate) fn new(size: usize) -> Self {
        let position: Vec<_> = (0..size).map(NodePos::new).collect();
        let gen = Gen(0);
        let nodes: Vec<_> = (0..size).map(|i| Node { node: NodeIndex::new(i), label: i as u32, part: PartIndex::new(0) }).collect();
        let parts = vec![PartInner { start: NodePos::new(0), len: size as _, gen }];
        let removed = vec![];
        Self { position, nodes, parts, removed, gen }
    }
    fn new_part(&mut self, start: NodePos, len: u32) -> PartIndex {
        if let Some(part) = self.removed.pop() {
            self.parts[part.index()] = PartInner { start, len, gen: self.gen };
            part
        } else {
            let part = PartIndex::new(self.parts.len());
            self.parts.push(PartInner { start, len, gen: self.gen });
            part
        }
    }
    fn remove_part(&mut self, part: PartIndex) {
        self.removed.push(part);
    }

    pub(crate) fn part_by_index(&self, idx: PartIndex) -> Part {
        self.parts[idx.index()].clone().into()
    }

    pub(crate) fn merge_all_parts(&mut self) -> Part {
        self.removed.clear();
        for i in 0..self.nodes.len() {
            self.nodes[i].label = i as _;
            self.nodes[i].part = PartIndex(0);
        }
        self.gen = Gen(0);
        let part = Part { start: NodePos::new(0), len: self.nodes.len() as _ };
        self.parts.clear();
        self.parts.push(PartInner { start: part.start, len: part.len, gen: Gen(0) });
        part
    }

    pub(crate) fn refine_forward<I>(&mut self, pivots: I) where
        I: IntoIterator,
        I::Item: Into<NodeIndex>,
    {
        self.gen.0 += 1;
        for pivot in pivots {
            let pivot: NodeIndex = pivot.into();

            let pos = self.position[pivot.index()];
            let part = self.nodes[pos.index()].part;
            let start = self.parts[part.index()].start;
            let prev = (start.index() != 0).then(|| {
                self.nodes[start.index() - 1].part
            });
            let new_part = if prev.is_some_and(|prev| self.parts[prev.index()].gen == self.gen) {
                prev.unwrap()
            } else {
                self.new_part(start, 0)
            };

            self.nodes.swap(start.index(), pos.index());
            self.position[pivot.index()] = start;
            self.position[self.nodes[pos.index()].node.index()] = pos;
            self.parts[new_part.index()].len += 1;
            self.parts[part.index()].start += 1_u32;
            self.parts[part.index()].len -= 1;
            self.nodes[start.index()].part = new_part;

            if self.parts[part.index()].len == 0 {
                self.remove_part(part);
                self.parts[new_part.index()].gen.0 = self.gen.0 - 1;
            }

            //println!("i:    {:?}", (0..self.position.len()).collect::<Vec<_>>());
            //println!("pos:  {:?}", self.position.iter().map(|pos| pos.index()).collect::<Vec<_>>());
            //println!("node: {:?}", self.nodes.iter().map(|n| n.node.index()).collect::<Vec<_>>());
            //println!("part: {:?}", self.nodes.iter().map(|n| n.part.index()).collect::<Vec<_>>());
            // println!("{:?}", self.nodes.iter().map(|n| (n.node.index(), n.part.index())).collect::<Vec<_>>());
        }
    }


    pub(crate) fn refine_backward<I>(&mut self, pivots: I) where
        I: IntoIterator,
        I::Item: Into<NodeIndex>,
    {
        self.gen.0 += 1;
        for pivot in pivots {
            let pivot: NodeIndex = pivot.into();
            //println!("pivot {} / gen {}", pivot.index(), self.gen.0);

            let pos = self.position[pivot.index()];
            let part = self.nodes[pos.index()].part;
            let PartInner { start, len, .. } = self.parts[part.index()];
            //println!("{}: ({}, {})", part, start.index(), len);
            let end = NodePos::new(start.index() + ((len - 1) as usize));
            let next = (end.index() + 1 != self.nodes.len()).then(|| {
                self.nodes[end.index() + 1].part
            });
            let new_part = if next.is_some_and(|next| self.parts[next.index()].gen == self.gen) {
                next.unwrap()
            } else {
                self.new_part(NodePos::new(end.0 as usize + 1), 0)
            };

            self.nodes.swap(end.index(), pos.index());
            self.position[pivot.index()] = end;
            self.position[self.nodes[pos.index()].node.index()] = pos;
            self.parts[new_part.index()].start.0 -= 1;
            self.parts[new_part.index()].len += 1;
            self.parts[part.index()].len -= 1;
            self.nodes[end.index()].part = new_part;

            if self.parts[part.index()].len == 0 {
                self.remove_part(part);
                self.parts[new_part.index()].gen.0 = self.gen.0 - 1;
            }

            // println!("i:    {:?}", (0..self.position.len()).collect::<Vec<_>>());
            // println!("pos:  {:?}", self.position.iter().map(|pos| pos.index()).collect::<Vec<_>>());
            // println!("node: {:?}", self.nodes.iter().map(|n| n.node.index()).collect::<Vec<_>>());
            // println!("part: {:?}", self.nodes.iter().map(|n| n.part.index()).collect::<Vec<_>>());
            // println!("{:?}", self.nodes.iter().map(|n| (n.node.index(), n.part.index())).collect::<Vec<_>>());
        }
    }
}


impl Partition {
    pub(crate) fn elements(&self, part: PartIndex) -> impl Iterator<Item=NodeIndex> + '_ {
        let PartInner { start, len, .. } = self.parts[part.index()];
        let start = start.index();
        let end = start + len as usize;
        self.nodes[start..end].iter().map(|n| n.node)
    }

    pub(crate) fn part_by_node(&self, node: NodeIndex) -> PartIndex {
        self.nodes[self.position[node.index()].index()].part
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Dir {
    Forward,
    Backward,
}

impl Dir {
    pub(crate) fn reversed(&self) -> Dir {
        match self {
            Dir::Forward => Dir::Backward,
            Dir::Backward => Dir::Forward,
        }
    }
}

impl Partition {
    pub(crate) fn refine<I>(&mut self, dir: Dir, pivots: I) where
        I: IntoIterator,
        I::Item: Into<NodeIndex>,
    {
        match dir {
            Dir::Forward => self.refine_forward(pivots),
            Dir::Backward => self.refine_backward(pivots),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SubPartition {
    start: NodePos,
    len: u32,
}

impl SubPartition {
    pub(crate) fn new(partition: &Partition) -> Self {
        Self { start: NodePos::new(0), len: partition.nodes.len() as _ }
    }
    pub(crate) fn len(&self) -> usize { self.len as _ }
    pub fn start(&self) -> NodePos { self.start }
    pub fn nodes<'a>(&self, partition: &'a Partition) -> impl Iterator<Item=NodeIndex> + 'a {
        let start = self.start.index();
        let end = start + self.len as usize;
        partition.nodes[start..end].iter().map(|n| n.node)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Part {
    start: NodePos,
    len: u32,
}

impl Part {
    pub(crate) fn new(partition: &Partition) -> Self {
        Self { start: NodePos::new(0), len: partition.nodes.len() as _ }
    }
    pub(crate) fn len(&self) -> usize { self.len as _ }
    pub fn start(&self) -> NodePos { self.start }
    pub fn nodes<'a>(&self, partition: &'a Partition) -> impl Iterator<Item=NodeIndex> + 'a {
        let start = self.start.index();
        let end = start + self.len as usize;
        partition.nodes[start..end].iter().map(|n| n.node)
    }
    pub(crate) fn nodes_raw<'a>(&self, partition: &'a Partition) -> &'a [Node] {
        let start = self.start.index();
        let end = start + self.len as usize;
        &partition.nodes[start..end]
    }
}

impl From<Part> for SubPartition {
    fn from(value: Part) -> Self {
        let Part { start, len } = value;
        Self { start, len }
    }
}

impl From<PartInner> for Part {
    fn from(value: PartInner) -> Self {
        let PartInner { start, len, .. } = value;
        Self { start, len }
    }
}

impl Part {
    pub(crate) fn singleton(self, partition: &mut Partition, u: NodeIndex) -> SubPartition {
        partition.refine_forward([u]);
        self.into()
    }
}

impl Partition {
    pub(crate) fn nodes(&self, p: &SubPartition) -> &[Node] {
        let start = p.start.index();
        let end = start + p.len as usize;
        &self.nodes[start..end]
    }
    pub(crate) fn full_sub_partition(&self) -> SubPartition {
        SubPartition { start: NodePos::new(0), len: self.nodes.len() as _ }
    }
}

impl SubPartition {
    pub(crate) fn first(&self, partition: &Partition) -> PartIndex {
        assert_ne!(self.len, 0);
        let pos = self.start;
        partition.nodes[pos.index()].part
    }
    pub(crate) fn last(&self, partition: &Partition) -> PartIndex {
        assert_ne!(self.len, 0);
        let pos = NodePos::new(self.start.index() + self.len as usize - 1);
        partition.nodes[pos.index()].part
    }
    pub(crate) fn part_indices<'a>(&self, partition: &'a Partition) -> impl Iterator<Item=PartIndex> + 'a {
        let mut pos = self.start;
        let end = NodePos::new(pos.index() + self.len as usize);
        from_fn(move || {
            (pos.0 < end.0).then(|| {
                let part = partition.nodes[pos.index()].part;
                pos.0 += partition.parts[part.index()].len;
                part
            })
        })
    }
    pub(crate) fn parts<'a>(&self, partition: &'a Partition) -> impl Iterator<Item=Part> + 'a {
        self.part_indices(partition).map(|i| {
            let PartInner { start, len, .. } = partition.parts[i.index()];
            Part { start, len }
        })
    }
}


pub(crate) fn divide(p: SubPartition, partition: &Partition) -> (PartIndex, SubPartition, SubPartition, Dir) {
    let first = p.first(&partition);
    let last = p.last(&partition);
    assert_ne!(first, last);
    let PartInner { start: first_start, len: first_len, .. } = partition.parts[first.index()];
    let PartInner { start: last_start, len: last_len, .. } = partition.parts[last.index()];
    if first_len < p.len / 2 {
        (first, SubPartition { start: first_start, len: first_len }, SubPartition { start: NodePos::new(p.start.index() + first_len as usize), len: p.len - first_len }, Dir::Backward)
    } else {
        (last, SubPartition { start: last_start, len: last_len }, SubPartition { start: p.start, len: p.len - last_len }, Dir::Forward)
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn to_vecs(partition: &Partition) -> Vec<Vec<u32>> {
        let mut res = vec![];
        let mut part = PartIndex::invalid();
        for n in &partition.nodes {
            if n.part != part {
                res.push(vec![]);
                part = n.part;
            }
            res.last_mut().unwrap().push(n.node.index() as _);
        }
        res
    }

    #[test]
    fn refine_forward() {
        let mut partition = Partition::new(10);

        assert_eq!(to_vecs(&partition), [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]);
        partition.refine_forward([5_u32, 6]);
        assert_eq!(to_vecs(&partition), [vec![5, 6], vec![2, 3, 4, 0, 1, 7, 8, 9]]);

        partition.refine_forward([1_u32, 3, 7]);
        assert_eq!(to_vecs(&partition), [vec![5, 6], vec![1, 3, 7], vec![0, 2, 4, 8, 9]]);

        partition.refine_forward([5_u32, 6, 1, 3, 0, 4, 9]);
        assert_eq!(to_vecs(&partition), [vec![5, 6], vec![1, 3], vec![7], vec![0, 4, 9], vec![8, 2]]);

        partition.refine_forward([9_u32, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(to_vecs(&partition), [vec![6, 5], vec![3, 1], vec![7], vec![9, 4, 0], vec![8, 2]]);

        partition.refine_forward([0_u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(to_vecs(&partition), [vec![5, 6], vec![1, 3], vec![7], vec![0, 4, 9], vec![2, 8]]);
    }

    #[test]
    fn refine_backward() {
        let mut partition = Partition::new(10);

        assert_eq!(to_vecs(&partition), [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]);
        partition.refine_backward([5_u32, 6]);
        assert_eq!(to_vecs(&partition), [vec![0, 1, 2, 3, 4, 9, 8, 7], vec![6, 5]]);

        partition.refine_backward([1_u32, 3, 7]);
        assert_eq!(to_vecs(&partition), [vec![0, 9, 2, 8, 4], vec![7, 3, 1], vec![6, 5]]);

        partition.refine_backward([5_u32, 6, 1, 3, 0, 4, 9]);
        assert_eq!(to_vecs(&partition), [vec![8, 2], vec![9, 4, 0], vec![7], vec![3, 1], vec![6, 5]]);

        partition.refine_backward([9_u32, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(to_vecs(&partition), [vec![2, 8], vec![0, 4, 9], vec![7], vec![1, 3], vec![5, 6]]);

        partition.refine_backward([0_u32, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(to_vecs(&partition), [vec![8, 2], vec![9, 4, 0], vec![7], vec![3, 1], vec![6, 5]]);
    }
}