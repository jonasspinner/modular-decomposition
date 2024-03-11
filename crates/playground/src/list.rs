use common::make_index;
use std::convert::{TryFrom, TryInto};
use std::fmt::{Debug, Formatter};
use std::iter::FusedIterator;
use std::ops::{Index, IndexMut};

make_index!(pub(self) InnerHandle);

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub(crate) struct ListHandle(u32);

impl ListHandle {
    pub(crate) fn index(&self) -> usize {
        self.0 as _
    }
}

impl TryFrom<InnerHandle> for ListHandle {
    type Error = ();
    fn try_from(handle: InnerHandle) -> Result<Self, Self::Error> {
        if handle.is_valid() {
            Ok(ListHandle(handle.0))
        } else {
            Err(())
        }
    }
}

impl From<ListHandle> for InnerHandle {
    fn from(handle: ListHandle) -> Self {
        Self(handle.0)
    }
}

#[derive(Debug)]
struct Node<T> {
    value: T,
    prev: InnerHandle,
    next: InnerHandle,
}

impl<T> Node<T> {
    fn new(value: T) -> Self {
        Self { value, prev: InnerHandle::invalid(), next: InnerHandle::invalid() }
    }
}

#[allow(dead_code)]
pub(crate) struct List<T> {
    nodes: Vec<Node<T>>,
    removed: Vec<ListHandle>,
    first: InnerHandle,
    last: InnerHandle,
}

impl<T> Index<ListHandle> for List<T> {
    type Output = T;

    fn index(&self, index: ListHandle) -> &Self::Output {
        &self.nodes[index.index()].value
    }
}

impl<T> IndexMut<ListHandle> for List<T> {
    fn index_mut(&mut self, index: ListHandle) -> &mut Self::Output {
        &mut self.nodes[index.index()].value
    }
}

impl<T> Default for List<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::default(),
            removed: Vec::default(),
            first: InnerHandle::invalid(),
            last: InnerHandle::invalid(),
        }
    }
}

#[allow(dead_code)]
impl<T> List<T> {
    pub(crate) fn new() -> Self {
        Default::default()
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self { nodes: Vec::with_capacity(capacity), ..Default::default() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.first.is_valid()
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len() - self.removed.len()
    }

    fn new_node(&mut self, value: T) -> (ListHandle, &mut Node<T>) {
        if let Some(handle) = self.removed.pop() {
            let node = &mut self.nodes[handle.index()];
            node.value = value;
            (handle, node)
        } else {
            let handle = InnerHandle::new(self.nodes.len()).try_into().unwrap();
            self.nodes.push(Node::new(value));
            (handle, self.nodes.last_mut().unwrap())
        }
    }

    pub(crate) fn next(&self, handle: ListHandle) -> Option<ListHandle> {
        self.nodes[handle.index()].next.try_into().ok()
    }

    pub(crate) fn prev(&self, handle: ListHandle) -> Option<ListHandle> {
        self.nodes[handle.index()].prev.try_into().ok()
    }

    fn get_node_mut(&mut self, handle: InnerHandle) -> Option<&mut Node<T>> {
        handle.try_into().ok().map(|h: ListHandle| &mut self.nodes[h.index()])
    }

    fn get_node(&self, handle: InnerHandle) -> Option<&Node<T>> {
        handle.try_into().ok().map(|h: ListHandle| &self.nodes[h.index()])
    }

    pub(crate) fn handle_front(&self) -> Option<ListHandle> {
        self.first.try_into().ok()
    }
    pub(crate) fn front(&self) -> Option<&T> {
        self.get_node(self.first).map(|f| &f.value)
    }
    pub(crate) fn front_mut(&mut self) -> Option<&mut T> {
        self.get_node_mut(self.first).map(|f| &mut f.value)
    }
    pub(crate) fn handle_back(&self) -> Option<ListHandle> {
        self.last.try_into().ok()
    }
    pub(crate) fn back(&self) -> Option<&T> {
        self.get_node(self.last).map(|f| &f.value)
    }
    pub(crate) fn back_mut(&mut self) -> Option<&mut T> {
        self.get_node_mut(self.last).map(|f| &mut f.value)
    }

    pub(crate) fn push_front(&mut self, value: T) {
        let old_first = self.first;
        let (handle, node) = self.new_node(value);
        node.next = old_first;

        let handle = handle.into();
        if let Some(n) = self.get_node_mut(old_first) {
            n.prev = handle;
        }
        self.first = handle;
        if !self.last.is_valid() {
            self.last = handle;
        }
    }
    pub(crate) fn push_back(&mut self, value: T) {
        let old_last = self.last;
        let (handle, node) = self.new_node(value);
        node.prev = old_last;

        let handle = handle.into();
        if let Some(n) = self.get_node_mut(old_last) {
            n.next = handle;
        }
        self.last = handle;
        if !self.first.is_valid() {
            self.first = handle;
        }
    }
}

#[allow(dead_code)]
impl<T> List<T> {
    pub(crate) fn remove(&mut self, handle: ListHandle) {
        let Node { prev, next, .. } = &mut self.nodes[handle.index()];
        let prev = std::mem::replace(prev, InnerHandle::invalid());
        let next = std::mem::replace(next, InnerHandle::invalid());
        match TryInto::<ListHandle>::try_into(prev) {
            Ok(prev) => self.nodes[prev.index()].next = next,
            Err(()) => self.first = next,
        }
        match TryInto::<ListHandle>::try_into(next) {
            Ok(next) => self.nodes[next.index()].prev = prev,
            Err(()) => self.last = prev,
        }
        self.removed.push(handle);
    }
    pub(crate) fn pop_front(&mut self) {
        if let Ok(first) = TryInto::<ListHandle>::try_into(self.first) {
            self.remove(first)
        }
    }
    pub(crate) fn pop_back(&mut self) {
        if let Ok(last) = TryInto::<ListHandle>::try_into(self.last) {
            self.remove(last)
        }
    }
}

#[derive(Clone)]
pub struct Iter<'a, T> {
    list: &'a List<T>,
    current: Option<ListHandle>,
    len: usize,
}

impl<'a, T> Iter<'a, T> {
    fn new(list: &'a List<T>) -> Self {
        Self { list, current: list.first.try_into().ok(), len: list.len() }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.current.map(|h| {
            self.len -= 1;
            self.current = self.list.next(h);
            &self.list[h]
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<T> List<T> {
    fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }
}

impl<'a, T> IntoIterator for &'a List<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Debug> Debug for List<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut l = f.debug_list();
        for v in self {
            l.entry(v);
        }
        l.finish()
    }
}

#[cfg(test)]
mod test {
    use crate::list::List;
    use std::ops::{Index, IndexMut};

    #[test]
    fn new() {
        let _list = List::<u32>::new();
    }

    #[test]
    fn push_front() {
        let mut dl = List::new();

        dl.push_front(2);
        assert_eq!(dl.front().unwrap(), &2);
        dl.push_front(1);
        assert_eq!(dl.front().unwrap(), &1);
    }

    #[test]
    fn pop_front() {
        let mut d = List::new();
        assert_eq!(d.front(), None);

        d.push_front(1);
        d.push_front(3);
        assert_eq!(d.front(), Some(&3));
        d.pop_front();
        assert_eq!(d.front(), Some(&1));
        d.pop_front();
        assert_eq!(d.front(), None);
    }

    #[test]
    fn ops() {
        let mut l = List::with_capacity(2);

        assert!(l.is_empty());
        assert_eq!(l.len(), 0);

        l.push_front(1);

        let f = l.handle_front().unwrap();
        assert_eq!(l[f], 1);
        assert_eq!(*l.index(f), 1);
        assert_eq!(*l.index_mut(f), 1);

        let b = l.handle_back().unwrap();
        assert_eq!(l[b], 1);
        assert_eq!(*l.index(b), 1);
        assert_eq!(*l.index_mut(b), 1);

        assert_eq!(l.front(), Some(&1));
        assert_eq!(l.back(), Some(&1));
        assert_eq!(l.front_mut(), Some(&mut 1));
        assert_eq!(l.back_mut(), Some(&mut 1));

        l.push_front(2);
        l.push_back(3);

        assert_eq!(l.len(), 3);
        assert_eq!(l.front(), Some(&2));
        assert_eq!(l.back(), Some(&3));

        l.remove(f);

        assert_eq!(l.len(), 2);

        l.push_front(4);
        l.push_back(5);

        assert_eq!(l.len(), 4);

        while !l.is_empty() {
            l.pop_front();
            l.pop_back();
        }

        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
    }
}
