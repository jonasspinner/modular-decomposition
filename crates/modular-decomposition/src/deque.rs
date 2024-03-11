use crate::index::make_index;
use std::ops::{Index, IndexMut};

make_index!(pub(crate) DequeIndex);

pub(crate) struct Deque<T> {
    elements: Vec<T>,
    head: DequeIndex,
    tail: DequeIndex,
}

impl<T> Deque<T> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self { elements: Vec::with_capacity(capacity), head: DequeIndex::new(0), tail: DequeIndex::new(0) }
    }
    pub(crate) fn len(&self) -> usize {
        (self.head.0 - self.tail.0) as usize
    }
    fn from_index_to_pos(idx: DequeIndex, n: usize) -> usize {
        idx.index() % n
    }
}

impl<T: Copy + Default> Deque<T> {
    pub(crate) fn push_back(&mut self, element: T) -> DequeIndex {
        if self.head.0 == self.elements.len() as u32 + self.tail.0 {
            let old_len = self.elements.len();
            let new_len = 1024.max(old_len * 2);
            self.elements.resize(new_len, T::default());
            if self.head != self.tail {
                let h = Self::from_index_to_pos(self.head, old_len);
                let t = Self::from_index_to_pos(self.tail, old_len);
                self.elements.copy_within(0..h, t);
            }
        }
        let idx = self.head;
        let pos = Self::from_index_to_pos(idx, self.elements.len());
        self.elements[pos] = element;
        self.head.0 += 1;
        idx
    }
    pub(crate) fn pop_front(&mut self) -> Option<T> {
        if self.head != self.tail {
            let pos = Self::from_index_to_pos(self.tail, self.elements.len());
            let element = self.elements[pos];
            self.tail.0 += 1;
            if self.head == self.tail {
                self.head.0 = 0;
                self.tail.0 = 0;
            }
            Some(element)
        } else {
            None
        }
    }
}

impl<T> Index<DequeIndex> for Deque<T> {
    type Output = T;

    fn index(&self, index: DequeIndex) -> &Self::Output {
        assert!(self.tail <= index);
        assert!(index < self.head);
        let pos = Self::from_index_to_pos(index, self.elements.len());
        &self.elements[pos]
    }
}

impl<T> IndexMut<DequeIndex> for Deque<T> {
    fn index_mut(&mut self, index: DequeIndex) -> &mut Self::Output {
        assert!(self.tail <= index);
        assert!(index < self.head);
        let pos = Self::from_index_to_pos(index, self.elements.len());
        &mut self.elements[pos]
    }
}
