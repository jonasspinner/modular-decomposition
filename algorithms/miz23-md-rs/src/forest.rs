use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::iter::FusedIterator;
use std::marker::PhantomData;
use std::ops::Index;

pub(crate) struct Forest<Data> {
    nodes: Vec<Node<Data>>,
    removed: VecDeque<NodeIdx>,
    num_live_nodes: u32,
}

impl<Data> Default for Forest<Data> {
    fn default() -> Self {
        Forest { nodes: vec![], removed: VecDeque::new(), num_live_nodes: 0 }
    }
}

impl<Data> Forest<Data> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Forest { nodes: Vec::with_capacity(capacity), removed: VecDeque::new(), num_live_nodes: 0 }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub(crate) struct NodeIdx(u32);

impl NodeIdx {
    pub fn new(index: usize) -> Self {
        assert!(index < u32::MAX as _);
        Self(index as u32)
    }
    pub fn idx(&self) -> usize {
        self.0 as usize
    }
}

impl From<usize> for NodeIdx {
    fn from(value: usize) -> Self { Self::new(value) }
}

impl PartialEq<u32> for NodeIdx {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}

pub struct Node<Data> {
    pub(crate) data: Data,
    pub(crate) parent: Option<NodeIdx>,
    pub(crate) left: Option<NodeIdx>,
    pub(crate) right: Option<NodeIdx>,
    pub(crate) first_child: Option<NodeIdx>,
    num_children: u32,
    alive: bool,
}

impl<Data> Node<Data> {
    pub fn new(data: Data) -> Self {
        Node { data, parent: None, left: None, right: None, first_child: None, num_children: 0, alive: true }
    }

    pub fn is_alive(&self) -> bool { self.alive }
    pub fn is_root(&self) -> bool { self.parent.is_none() }
    pub fn has_parent(&self) -> bool { self.parent.is_some() }
    pub fn is_first_child(&self) -> bool { self.has_parent() && self.left.is_none() }
    pub fn is_last_child(&self) -> bool { self.has_parent() && self.right.is_none() }
    pub fn is_leaf(&self) -> bool { self.first_child.is_none() }
    pub fn has_child(&self) -> bool { self.first_child.is_some() }
    pub fn has_only_one_child(&self) -> bool { self.num_children == 1 }
    pub fn number_of_children(&self) -> u32 { self.num_children }
}

impl<Data> Index<NodeIdx> for Forest<Data> {
    type Output = Node<Data>;
    fn index(&self, index: NodeIdx) -> &Self::Output { &self.nodes[index.idx()] }
}

impl<Data> std::ops::IndexMut<NodeIdx> for Forest<Data> {
    fn index_mut(&mut self, index: NodeIdx) -> &mut Self::Output { &mut self.nodes[index.idx()] }
}

#[allow(unused)]
pub(crate) struct ChildrenWalker<Data> {
    next: Option<NodeIdx>,
    _data: PhantomData<Data>,
}

#[allow(unused)]
impl<Data> ChildrenWalker<Data> {
    pub(crate) fn new(forest: &Forest<Data>, index: NodeIdx) -> Self {
        let next = forest[index].first_child;
        ChildrenWalker { next, _data: PhantomData }
    }
    fn next(&mut self, forest: &Forest<Data>) -> Option<NodeIdx> {
        if let Some(current) = self.next {
            self.next = forest[current].right;
            Some(current)
        } else {
            None
        }
    }
}

pub(crate) struct ChildrenIter<'a, Data> {
    forest: &'a Forest<Data>,
    current: Option<NodeIdx>,
    num_left: u32,
}

impl<'a, Data> ChildrenIter<'a, Data> {
    fn new(forest: &'a Forest<Data>, index: NodeIdx) -> Self {
        let Node { first_child: current, num_children: num_left, .. } = forest[index];
        ChildrenIter { forest, current, num_left }
    }
}

impl<Data> Iterator for ChildrenIter<'_, Data> {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = self.current {
            self.current = self.forest[current].right;
            self.num_left -= 1;
            Some(current)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.num_left as usize;
        (size, Some(size))
    }
}

impl<Data> ExactSizeIterator for ChildrenIter<'_, Data> {}

impl<Data> FusedIterator for ChildrenIter<'_, Data> {}

impl<Data> Forest<Data> {
    pub fn children(&self, index: NodeIdx) -> ChildrenIter<Data> {
        debug_assert!(self.is_valid(index));
        ChildrenIter::new(self, index)
    }
}


pub(crate) struct AncestorIter<'a, Data> {
    forest: &'a Forest<Data>,
    current: Option<NodeIdx>,
}

impl<'a, Data> AncestorIter<'a, Data> {
    fn new(forest: &'a Forest<Data>, index: NodeIdx) -> Self {
        AncestorIter { forest, current: forest[index].parent }
    }
}

impl<Data> Iterator for AncestorIter<'_, Data> {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current) = self.current {
            self.current = self.forest[current].parent;
            Some(current)
        } else {
            None
        }
    }
}

impl<Data> FusedIterator for AncestorIter<'_, Data> {}

impl<Data> Forest<Data> {
    pub fn ancestors(&self, index: NodeIdx) -> AncestorIter<Data> {
        debug_assert!(self.is_valid(index));

        AncestorIter::new(self, index)
    }
}


pub(crate) struct PreOrderNodeIdxWalker {
    start: NodeIdx,
    next: Option<NodeIdx>,
}

impl PreOrderNodeIdxWalker {
    #[allow(dead_code)]
    fn new(index: NodeIdx) -> Self {
        PreOrderNodeIdxWalker { start: index, next: Some(index) }
    }
}

impl PreOrderNodeIdxWalker {
    fn next<Data>(&mut self, forest: &Forest<Data>) -> Option<NodeIdx> {
        if let Some(next) = self.next {
            let current = next;

            let Node { first_child, right, parent, .. } = forest[current];
            self.next = if let Some(child) = first_child {
                Some(child)
            } else if current == self.start {
                None
            } else if let Some(right) = right {
                Some(right)
            } else {
                let mut ancestor = parent;
                loop {
                    if let Some(index) = ancestor {
                        let Node { parent, right, .. } = forest[index];
                        if index == self.start { break None; } else if right.is_some() { break right; }
                        ancestor = parent;
                    } else { break None; }
                }
            };

            return Some(current);
        }
        None
    }
}

pub(crate) struct PreOrderNodeIdxIter<'a, Data> {
    forest: &'a Forest<Data>,
    walker: PreOrderNodeIdxWalker,
}

impl<'a, Data> PreOrderNodeIdxIter<'a, Data> {
    #[allow(dead_code)]
    fn new(forest: &'a Forest<Data>, index: NodeIdx) -> Self {
        PreOrderNodeIdxIter { forest, walker: PreOrderNodeIdxWalker::new(index) }
    }
}

impl<Data> Iterator for PreOrderNodeIdxIter<'_, Data> {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        self.walker.next(self.forest)
    }
}

impl<Data> FusedIterator for PreOrderNodeIdxIter<'_, Data> {}


impl<Data> Forest<Data> {
    #[allow(dead_code)]
    pub fn pre_order_node_indices(&self, index: NodeIdx) -> PreOrderNodeIdxIter<Data> {
        PreOrderNodeIdxIter::new(self, index)
    }
    pub fn get_bfs_nodes(&self, index: NodeIdx) -> Vec<NodeIdx> {
        let mut ret = vec![];
        let mut q = VecDeque::new();

        q.push_back(index);
        while let Some(x) = q.pop_front() {
            ret.push(x);

            q.extend(self.children(x));
        }
        ret
    }
    pub fn leaves(&self, index: NodeIdx) -> impl Iterator<Item=NodeIdx> + '_ {
        self.pre_order_node_indices(index).filter(|&node| { self[node].is_leaf() })
    }
}

impl<Data> Forest<Data> {
    pub fn subtree_node_mut(&mut self, index: NodeIdx, mut f: impl FnMut(&mut Node<Data>)) {
        let mut walker = PreOrderNodeIdxWalker::new(index);
        while let Some(node) = walker.next(self) {
            f(&mut self[node]);
        }
    }
    pub fn leaves_data_mut(&mut self, index: NodeIdx, mut f: impl FnMut(&mut Data)) {
        self.subtree_node_mut(index, |node| {
            if node.is_leaf() {
                f(&mut node.data);
            }
        });
    }
}

impl<Data> Forest<Data> {
    pub fn get_root(&self, mut index: NodeIdx) -> NodeIdx {
        debug_assert!(self.is_valid(index));

        while let Some(parent_idx) = self[index].parent {
            index = parent_idx;
        }
        index
    }

    pub fn size(&self) -> usize { self.num_live_nodes as usize }
    pub fn capacity(&self) -> usize { self.nodes.len() }
    pub fn is_valid(&self, index: NodeIdx) -> bool { index.idx() < self.capacity() }
    pub fn as_valid(&self, index: Option<NodeIdx>) -> Option<NodeIdx> {
        index.and_then(|index| if self.is_valid(index) { Some(index) } else { None })
    }

    pub fn create_node(&mut self, data: Data) -> NodeIdx {
        self.num_live_nodes += 1;
        if let Some(index) = self.removed.pop_front() {
            debug_assert!(!self.nodes[index.idx()].is_alive());
            self.nodes[index.idx()] = Node::new(data);
            index
        } else {
            let index = NodeIdx::from(self.nodes.len());
            self.nodes.push(Node::new(data));
            index
        }
    }
}


impl<Data> Forest<Data> {
    #[allow(dead_code)]
    pub fn roots(&self) -> impl Iterator<Item=NodeIdx> + '_ {
        self.nodes.iter()
            .enumerate()
            .filter_map(|(i, n)| {
                if n.is_root() && n.is_alive() {
                    Some(NodeIdx(i as u32))
                } else { None }
            })
    }

    pub fn remove(&mut self, index: NodeIdx) {
        debug_assert!(self.is_valid(index));
        self.detach(index);
        debug_assert!(self.nodes[index.idx()].is_leaf());

        self.num_live_nodes -= 1;
        self.nodes[index.idx()].alive = false;
        self.removed.push_back(index);
    }

    pub fn add_child(&mut self, parent: NodeIdx, child: NodeIdx) {
        debug_assert!(self.is_valid(parent));
        debug_assert!(self.is_valid(child));

        if let Some(first) = self[parent].first_child.replace(child) {
            self[first].left = Some(child);
            self[child].right = Some(first);
        }

        self[child].parent = Some(parent);
        self[parent].num_children += 1;
    }

    pub fn detach(&mut self, index: NodeIdx) {
        debug_assert!(self.is_valid(index));

        let Node { parent, left, right, .. } = self[index];
        if let Some(parent) = parent {
            let parent = &mut self[parent];
            parent.num_children -= 1;
            if parent.first_child == Some(index) { parent.first_child = right; }
        }
        if let Some(left) = left { self[left].right = right; }
        if let Some(right) = right { self[right].left = left; }

        let node = &mut self[index];
        node.parent = None;
        node.left = None;
        node.right = None;
    }

    pub fn swap(&mut self, a: NodeIdx, b: NodeIdx) {
        debug_assert!(self.is_valid(a));
        debug_assert!(self.is_valid(b));
        debug_assert_ne!(self.get_root(a), self.get_root(b));

        debug_assert_ne!(a, b);

        let mut update_adj = |x, y| -> (Option<NodeIdx>, Option<NodeIdx>, Option<NodeIdx>) {
            let Node { parent, left, right, .. } = self[x];
            if let Some(parent) = parent {
                if let Some(ref mut first) = self[parent].first_child {
                    if *first == x { *first = y; }
                }
            }
            if let Some(left) = left { self[left].right = Some(y); }
            if let Some(right) = right { self[right].left = Some(y); }
            (parent, left, right)
        };

        let prev_a_adj = update_adj(a, b);
        let prev_b_adj = update_adj(b, a);

        let mut update_self = |x, (parent, left, right)| {
            let x = &mut self[x];
            x.parent = parent;
            x.left = left;
            x.right = right;
        };

        update_self(a, prev_b_adj);
        update_self(b, prev_a_adj);

        /*
        let Node { parent: a_parent, left: a_left, right: a_right, .. } = self[a];
        if let Some(a_parent) = a_parent {
            if let Some(ref mut first) = self[a_parent].first_child {
                if *first == a { *first = b; }
            }
        }
        if let Some(a_left) = a_left { self[a_left].right = Some(b); }
        if let Some(a_right) = a_right { self[a_right].left = Some(b); }


        let Node { parent: b_parent, left: b_left, right: b_right, .. } = self[b];
        if let Some(b_parent) = b_parent {
            if let Some(ref mut first) = self[b_parent].first_child {
                if *first == b { *first = a; }
            }
        }
        if let Some(b_left) = b_left { self[b_left].right = Some(a); }
        if let Some(b_right) = b_right { self[b_right].left = Some(a); }

        // Note: The variables are assigned at the beginning of the function instead of being read again. Unclear if that can lead to bugs.
        let a = &mut self[a];
        a.parent = b_parent;
        a.left = b_left;
        a.right = b_right;
        let b = &mut self[b];
        b.parent = a_parent;
        b.left = a_left;
        b.right = a_right;

         */
    }

    pub fn replace(&mut self, index: NodeIdx, replace_by: NodeIdx) {
        debug_assert!(self.is_valid(index));
        debug_assert!(self.is_valid(replace_by));
        assert_ne!(index, replace_by);
        debug_assert!(!self.ancestors(index).any(|a| a == replace_by));

        self.detach(replace_by);
        self.swap(index, replace_by);
    }

    pub fn move_to(&mut self, index: NodeIdx, new_parent: NodeIdx) {
        debug_assert!(self.is_valid(index));
        debug_assert!(self.is_valid(new_parent));
        assert_ne!(index, new_parent);

        self.detach(index);
        self.add_child(new_parent, index);
    }

    pub fn move_to_before(&mut self, index: NodeIdx, target: NodeIdx) {
        debug_assert!(self.is_valid(index));
        debug_assert!(self.is_valid(target));
        assert!(!self[target].is_root());
        assert_ne!(index, target);
        debug_assert!(!self.ancestors(target).any(|a| a == index));

        self.detach(index);

        let Node { parent: y_parent, left: y_left, .. } = self[target];
        let x = &mut self[index];
        x.parent = y_parent;
        x.left = y_left;
        x.right = Some(target);

        let y_parent = y_parent.expect("has parent");
        self[y_parent].num_children += 1;

        if let Some(y_left) = self[target].left.replace(index) {
            self[y_left].right = Some(index);
        } else {
            self[y_parent].first_child = Some(index);
        }
    }

    pub fn move_to_after(&mut self, index: NodeIdx, target: NodeIdx) {
        debug_assert!(self.is_valid(index));
        debug_assert!(self.is_valid(target));
        assert!(!self[target].is_root());
        debug_assert!(!self.ancestors(target).any(|a| a == index));

        self.detach(index);

        let Node { parent: y_parent, right: y_right, .. } = self[target];
        let x = &mut self[index];
        x.parent = y_parent;
        x.left = Some(target);
        x.right = y_right;

        self[y_parent.unwrap()].num_children += 1;

        if let Some(y_right) = y_right { self[y_right].left = Some(index); }
        self[target].right = Some(index);
    }

    pub fn make_first_child(&mut self, index: NodeIdx) {
        debug_assert!(self.is_valid(index));

        if let Some(parent) = self[index].parent {
            let first_child = self[parent].first_child.expect("parent node must have a child");
            if first_child != index {
                debug_assert!(!self[index].is_root());
                debug_assert!(!self[index].is_first_child());
                debug_assert_eq!(self[first_child].parent, Some(parent));
                self.move_to_before(index, first_child);
            }
        }
    }

    pub fn add_children_from(&mut self, index: NodeIdx, target: NodeIdx) {
        debug_assert!(self.is_valid(index));
        debug_assert!(self.is_valid(target));
        debug_assert!(!self.ancestors(index).any(|a| a == target));

        if index == target { return; }

        let mut child = self[target].first_child;
        while let Some(child_idx) = child {
            self[child_idx].parent = Some(index);
            if self[child_idx].is_last_child() {
                self[child_idx].right = self[index].first_child;
                if let Some(first_child) = self[index].first_child {
                    self[first_child].left = child;
                }
                break;
            }

            child = self.nodes[child_idx.idx()].right;
        }

        if let Some(first) = self[target].first_child.take() {
            self[index].first_child = Some(first);
        }
        self[index].num_children += self[target].num_children;
        self[target].num_children = 0;
    }

    pub fn replace_by_children(&mut self, index: NodeIdx) {
        debug_assert!(self.is_valid(index));
        debug_assert!(!self[index].is_root());

        let mut child = self[index].first_child;
        while let Some(child_idx) = child {
            let next = self[child_idx].right;
            self.move_to_before(child_idx, index);
            child = next;
        }
        self.detach(index);
    }

    pub fn replace_children(&mut self, index: NodeIdx, target: NodeIdx) {
        for c in self.children(index).collect::<Vec<_>>() {
            self.detach(c);
        }
        self.move_to(target, index);
    }

    #[allow(dead_code)]
    fn check_consistency(&self) -> Result<(), String> {
        let mut num_alive = 0;
        for index in (0..self.capacity()).map(|i| NodeIdx(i as u32)) {
            if !self[index].is_alive() { continue; }
            num_alive += 1;

            if let Some(left) = self[index].left {
                if self[left].right != Some(index) {
                    return Err("left->right".into());
                }
            }
            if let Some(right) = self[index].right {
                if self[right].left != Some(index) {
                    return Err("left->right".into());
                }
            }
            if self.children(index).count() != self[index].number_of_children() as usize {
                return Err("number_of_children".into());
            }
            if let Some(parent) = self[index].parent {
                let mut children = self.children(parent);
                if !children.any(|c| c == index) {
                    return Err("parent->child".into());
                }
            }
        }
        if num_alive != self.size() {
            return Err("size".into());
        }
        Ok(())
    }
}

impl<Data: Debug> Forest<Data> {
    #[allow(dead_code)]
    pub(crate) fn to_string(&self, root: Option<NodeIdx>) -> String {
        let mut ss = String::new();

        let hash = |mut x: u64| {
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51afd7ed558ccd);
            x ^= x >> 33;
            x
        };
        let hash_combine = |seed: &mut u64, x: u64| {
            // *seed ^= hash(x) + 0x9e3779b9 + (*seed << 6) + (*seed >> 2);
            let mut result = hash(x);
            result = result.wrapping_add(0x9e3779b9);
            result = result.wrapping_add(*seed << 6);
            result = result.wrapping_add(*seed >> 2);
            *seed ^= result;
        };

        let mut h: u64 = 0;
        for node in &self.nodes {
            if node.alive {
                let p: u64 = node.parent.unwrap_or(NodeIdx::from(0)).idx() as _;
                let l: u64 = node.left.unwrap_or(NodeIdx::from(0)).idx() as _;
                let r: u64 = node.right.unwrap_or(NodeIdx::from(0)).idx() as _;
                let c: u64 = node.first_child.unwrap_or(NodeIdx::from(0)).idx() as _;
                hash_combine(&mut h, p);
                hash_combine(&mut h, l);
                hash_combine(&mut h, r);
                hash_combine(&mut h, c);
            }
        }
        ss.push_str(&format!("[{:#x}]#", h));

        let Some(root) = self.as_valid(root) else {
            ss.push_str(&format!("invalid({:?})", root));
            return ss;
        };
        let mut stack = vec![];
        let mut visited: HashSet<Option<NodeIdx>> = HashSet::new();

        stack.push(None);
        stack.push(Some(root));

        while let Some(p_) = stack.pop() {
            if let Some(p) = p_ {
                if visited.contains(&p_) {
                    return "cycle detected".to_string();
                }
                visited.insert(Some(p));
                ss.push_str(&format!("({:?}", self[p].data));

                let st = self.children(p).collect::<Vec<_>>();
                for &it in st.iter().rev() {
                    stack.push(None);
                    stack.push(Some(it));
                }
            } else {
                ss.push(')');
            }
        }

        ss
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn init_forest() -> Forest<u32> {
        let n = 20;
        let mut forest = Forest::new();
        for i in 0..n {
            forest.create_node(i);
        }
        let relations: Vec<_> = [
            (3, 1), (3, 5), (3, 4), (5, 9), (5, 2), (4, 7), (7, 6), (7, 8),
            (13, 11), (13, 15), (13, 14), (15, 19), (15, 12), (14, 17), (17, 16), (17, 18),
        ].iter().map(|(a, b)| (NodeIdx(*a), NodeIdx(*b))).collect();
        for (new_parent, index) in relations {
            forest.move_to(index, new_parent);
        }
        forest
    }

    fn st(forest: &Forest<u32>, i: u32) -> String {
        let mut ss: String = String::with_capacity(forest.capacity());
        let mut stack = vec![];
        let mut visited: Vec<bool> = (0..forest.capacity()).map(|_| false).collect();

        stack.push(None);
        stack.push(Some(NodeIdx(i)));

        while let Some(p) = stack.pop() {
            if let Some(p) = p {
                if visited[p.idx()] { return "cycle detected".into(); }
                visited[p.idx()] = true;
                ss.push('(');
                ss.push_str(&format!("{:?}", forest[p].data));

                let st = forest.children(p).collect::<Vec<_>>();

                for child in st.iter().rev() {
                    stack.push(None);
                    stack.push(Some(*child));
                }
            } else {
                ss.push(')');
            }
        }

        ss
    }

    #[test]
    fn basic() {
        let forest = init_forest();

        assert_eq!(forest.size(), 20);
        assert_eq!(forest.capacity(), 20);
        assert_eq!(forest.roots().collect::<Vec<_>>(), vec![0, 3, 10, 13]);

        let is_root: Vec<_> = forest.nodes.iter().take(10).map(|node| node.is_root()).collect();
        assert_eq!(is_root, vec![true, false, false, true, false, false, false, false, false, false]);

        let has_parent: Vec<_> = forest.nodes.iter().take(10).map(|node| node.has_parent()).collect();
        assert_eq!(has_parent, vec![false, true, true, false, true, true, true, true, true, true]);

        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn detach() {
        let mut forest = init_forest();

        forest.detach(NodeIdx(3));
        let subtree_3: Vec<_> = forest.pre_order_node_indices(NodeIdx(3)).collect();
        assert_eq!(subtree_3, vec![3, 4, 7, 8, 6, 5, 2, 9, 1]);
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(5(2)(9))(1))");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 3);

        forest.detach(NodeIdx(5));
        let subtree_3: Vec<_> = forest.pre_order_node_indices(NodeIdx(3)).collect();
        let subtree_5: Vec<_> = forest.pre_order_node_indices(NodeIdx(5)).collect();
        assert_eq!(subtree_3, vec![3, 4, 7, 8, 6, 1]);
        assert_eq!(subtree_5, vec![5, 2, 9]);
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 2);

        forest.detach(NodeIdx(4));
        let subtree_3: Vec<_> = forest.pre_order_node_indices(NodeIdx(3)).collect();
        let subtree_4: Vec<_> = forest.pre_order_node_indices(NodeIdx(4)).collect();
        assert_eq!(subtree_3, vec![3, 1]);
        assert_eq!(subtree_4, vec![4, 7, 8, 6]);
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 1);

        forest.detach(NodeIdx(1));
        let subtree_3: Vec<_> = forest.pre_order_node_indices(NodeIdx(3)).collect();
        let subtree_1: Vec<_> = forest.pre_order_node_indices(NodeIdx(1)).collect();
        assert_eq!(subtree_3, vec![3]);
        assert_eq!(subtree_1, vec![1]);
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 0);

        forest.detach(NodeIdx(0));
        let subtree_0: Vec<_> = forest.pre_order_node_indices(NodeIdx(0)).collect();
        assert_eq!(subtree_0, vec![0]);

        assert_eq!(forest.size(), 20);
        assert_eq!(forest.capacity(), 20);
        assert_eq!(forest.roots().collect::<Vec<_>>(), vec![0, 1, 3, 4, 5, 10, 13]);

        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn remove() {
        let mut forest = init_forest();

        forest.remove(NodeIdx(2));
        forest.remove(NodeIdx(9));
        forest.remove(NodeIdx(5));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(1))");
        assert_eq!(forest.size(), 17);
        assert_eq!(forest.capacity(), 20);

        let i99 = forest.create_node(99);
        forest.move_to(i99, NodeIdx(3));
        assert_eq!(st(&forest, 3), "(3(99)(4(7(8)(6)))(1))");
        assert_eq!(forest.size(), 18);
        assert_eq!(forest.capacity(), 20);

        let i98 = forest.create_node(98);
        forest.move_to(i98, i99);
        assert_eq!(st(&forest, 3), "(3(99(98))(4(7(8)(6)))(1))");
        assert_eq!(forest.size(), 19);
        assert_eq!(forest.capacity(), 20);

        let i97 = forest.create_node(97);
        forest.move_to(i97, i98);
        assert_eq!(st(&forest, 3), "(3(99(98(97)))(4(7(8)(6)))(1))");
        assert_eq!(forest.size(), 20);
        assert_eq!(forest.capacity(), 20);

        let i96 = forest.create_node(96);
        forest.move_to(i96, i98);
        assert_eq!(st(&forest, 3), "(3(99(98(96)(97)))(4(7(8)(6)))(1))");
        assert_eq!(forest.size(), 21);
        assert_eq!(forest.capacity(), 21);

        forest.remove(i97);
        assert_eq!(st(&forest, 3), "(3(99(98(96)))(4(7(8)(6)))(1))");
        assert_eq!(forest.size(), 20);
        assert_eq!(forest.capacity(), 21);

        assert_eq!(forest.roots().collect::<Vec<_>>(), vec![0, 3, 10, 13]);
        forest.remove(NodeIdx(0));
        assert_eq!(forest.roots().collect::<Vec<_>>(), vec![3, 10, 13]);

        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn swap() {
        let mut forest = init_forest();

        forest.swap(NodeIdx(5), NodeIdx(15));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(15(12)(19))(1))");
        assert_eq!(st(&forest, 13), "(13(14(17(18)(16)))(5(2)(9))(11))");

        forest.swap(NodeIdx(4), NodeIdx(11));
        assert_eq!(st(&forest, 3), "(3(11)(15(12)(19))(1))");
        assert_eq!(st(&forest, 13), "(13(14(17(18)(16)))(5(2)(9))(4(7(8)(6))))");

        forest.detach(NodeIdx(7));
        forest.swap(NodeIdx(1), NodeIdx(7));
        assert_eq!(st(&forest, 3), "(3(11)(15(12)(19))(7(8)(6)))");

        forest.swap(NodeIdx(7), NodeIdx(1));
        forest.swap(NodeIdx(1), NodeIdx(7));
        assert_eq!(st(&forest, 3), "(3(11)(15(12)(19))(7(8)(6)))");

        // tree.detach(NodeIdx(7));
        // tree.swap(NodeIdx(7), NodeIdx(7));
        // assert_eq!(tree.to_string(7), "(7(8)(6))");
        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn replace() {
        let mut forest = init_forest();

        forest.replace(NodeIdx(3), NodeIdx(5));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(1))");
        assert_eq!(st(&forest, 5), "(5(2)(9))");
        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn move_to_before() {
        let mut forest = init_forest();

        forest.move_to_before(NodeIdx(15), NodeIdx(5));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(15(12)(19))(5(2)(9))(1))");

        forest.move_to_before(NodeIdx(11), NodeIdx(1));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(15(12)(19))(5(2)(9))(11)(1))");

        forest.move_to_before(NodeIdx(14), NodeIdx(4));
        assert_eq!(st(&forest, 3), "(3(14(17(18)(16)))(4(7(8)(6)))(15(12)(19))(5(2)(9))(11)(1))");

        forest.move_to_before(NodeIdx(11), NodeIdx(4));
        assert_eq!(st(&forest, 3), "(3(14(17(18)(16)))(11)(4(7(8)(6)))(15(12)(19))(5(2)(9))(1))");

        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn move_to_after() {
        let mut forest = init_forest();

        forest.move_to_after(NodeIdx(15), NodeIdx(5));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(5(2)(9))(15(12)(19))(1))");

        forest.move_to_after(NodeIdx(11), NodeIdx(1));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(5(2)(9))(15(12)(19))(1)(11))");

        forest.move_to_after(NodeIdx(14), NodeIdx(4));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(14(17(18)(16)))(5(2)(9))(15(12)(19))(1)(11))");

        forest.move_to_after(NodeIdx(11), NodeIdx(4));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(11)(14(17(18)(16)))(5(2)(9))(15(12)(19))(1))");

        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn make_first_child() {
        let mut forest = init_forest();

        forest.make_first_child(NodeIdx(3));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(5(2)(9))(1))");

        forest.make_first_child(NodeIdx(4));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(5(2)(9))(1))");

        forest.make_first_child(NodeIdx(5));
        assert_eq!(st(&forest, 3), "(3(5(2)(9))(4(7(8)(6)))(1))");

        forest.make_first_child(NodeIdx(1));
        assert_eq!(st(&forest, 3), "(3(1)(5(2)(9))(4(7(8)(6))))");

        forest.make_first_child(NodeIdx(2));
        assert_eq!(st(&forest, 3), "(3(1)(5(2)(9))(4(7(8)(6))))");

        forest.make_first_child(NodeIdx(9));
        assert_eq!(st(&forest, 3), "(3(1)(5(9)(2))(4(7(8)(6))))");

        forest.make_first_child(NodeIdx(7));
        assert_eq!(st(&forest, 3), "(3(1)(5(9)(2))(4(7(8)(6))))");
        assert_eq!(forest.check_consistency(), Ok(()));
    }


    #[test]
    fn add_children_from() {
        let mut forest = init_forest();

        forest.add_children_from(NodeIdx(3), NodeIdx(13));
        assert_eq!(st(&forest, 3), "(3(14(17(18)(16)))(15(12)(19))(11)(4(7(8)(6)))(5(2)(9))(1))");
        assert_eq!(st(&forest, 13), "(13)");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 6);
        assert_eq!(forest[NodeIdx(13)].number_of_children(), 0);

        forest.add_children_from(NodeIdx(3), NodeIdx(1));
        assert_eq!(st(&forest, 3), "(3(14(17(18)(16)))(15(12)(19))(11)(4(7(8)(6)))(5(2)(9))(1))");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 6);
        assert_eq!(forest[NodeIdx(1)].number_of_children(), 0);

        forest.add_children_from(NodeIdx(3), NodeIdx(4));
        assert_eq!(st(&forest, 3), "(3(7(8)(6))(14(17(18)(16)))(15(12)(19))(11)(4)(5(2)(9))(1))");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 7);
        assert_eq!(forest[NodeIdx(4)].number_of_children(), 0);

        forest.add_children_from(NodeIdx(4), NodeIdx(5));
        assert_eq!(st(&forest, 3), "(3(7(8)(6))(14(17(18)(16)))(15(12)(19))(11)(4(2)(9))(5)(1))");
        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn replace_by_children() {
        let mut forest = init_forest();

        forest.replace_by_children(NodeIdx(5));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(2)(9)(1))");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 4);

        forest.replace_by_children(NodeIdx(4));
        assert_eq!(st(&forest, 3), "(3(7(8)(6))(2)(9)(1))");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 4);

        forest.replace_by_children(NodeIdx(7));
        assert_eq!(st(&forest, 3), "(3(8)(6)(2)(9)(1))");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 5);

        forest.replace_by_children(NodeIdx(1));
        assert_eq!(st(&forest, 3), "(3(8)(6)(2)(9))");
        assert_eq!(forest[NodeIdx(3)].number_of_children(), 4);
        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn replace_children() {
        let mut forest = init_forest();

        forest.replace_children(NodeIdx(5), NodeIdx(15));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(5(15(12)(19)))(1))");

        forest.replace_children(NodeIdx(1), NodeIdx(0));
        assert_eq!(st(&forest, 3), "(3(4(7(8)(6)))(5(15(12)(19)))(1(0)))");

        forest.replace_children(NodeIdx(3), NodeIdx(1));
        assert_eq!(st(&forest, 3), "(3(1(0)))");

        assert_eq!(forest.check_consistency(), Ok(()));
    }

    #[test]
    fn preorder_node_indices() {
        let mut forest = Forest::new();
        let a = forest.create_node(());
        let b = forest.create_node(());
        let c = forest.create_node(());
        let d = forest.create_node(());
        let e = forest.create_node(());
        let f = forest.create_node(());
        forest.add_child(a, f);
        forest.add_child(a, d);
        forest.add_child(a, b);
        forest.add_child(b, c);
        forest.add_child(d, e);

        //    a
        //   b d f
        //  c   e

        let indices: Vec<_> = forest.pre_order_node_indices(a).collect();
        assert_eq!(indices, vec![a, b, c, d, e, f]);
        let indices: Vec<_> = forest.pre_order_node_indices(b).collect();
        assert_eq!(indices, vec![b, c]);
        let indices: Vec<_> = forest.pre_order_node_indices(c).collect();
        assert_eq!(indices, vec![c]);
        let indices: Vec<_> = forest.pre_order_node_indices(d).collect();
        assert_eq!(indices, vec![d, e]);
        let indices: Vec<_> = forest.pre_order_node_indices(e).collect();
        assert_eq!(indices, vec![e]);
        let indices: Vec<_> = forest.pre_order_node_indices(f).collect();
        assert_eq!(indices, vec![f]);
    }
}