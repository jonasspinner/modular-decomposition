use std::fmt::Debug;
use std::ops::{Index, IndexMut};

use crate::index::make_index;

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
            let new_len = 4.max(old_len * 2);
            self.elements.resize(new_len, T::default());
            if self.head != self.tail {
                let t0 = Self::from_index_to_pos(self.tail, old_len);
                let h0 = Self::from_index_to_pos(self.head, old_len);
                let t1 = Self::from_index_to_pos(self.tail, new_len);
                let h1 = Self::from_index_to_pos(self.head, new_len);
                // We must make sure that after the resize, the elements start at t1 and end at h1.
                // The elements must either be placed
                // as [..] [t1..old_len] [old_len, h1] [..] or
                // as [0..h1] [..old_len] [old_len..] [t1..]
                if t1 <= h1 {
                    self.elements.copy_within(0..h0, old_len);
                } else {
                    self.elements.copy_within(t0..old_len, t1);
                }
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

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::num::Wrapping;

    use crate::deque::{Deque, DequeIndex};

    #[test]
    fn index() {
        let mut d = Deque::with_capacity(0);
        for i in [2, 3, 5, 7] {
            d.push_back(i);
        }
        assert_eq!(d.pop_front(), Some(2));
        assert_eq!(d[DequeIndex::new(1)], 3);
        assert_eq!(d[DequeIndex::new(2)], 5);
        assert_eq!(d[DequeIndex::new(3)], 7);
        let d = d;
        assert_eq!(d[DequeIndex::new(1)], 3);
        assert_eq!(d[DequeIndex::new(2)], 5);
        assert_eq!(d[DequeIndex::new(3)], 7);
    }

    #[test]
    fn push_back_copy_within() {
        let mut d = Deque::with_capacity(0);
        for i in 0..=1024 {
            d.push_back(i);
        }
        for i in 0..=1024 {
            assert_eq!(d.pop_front(), Some(i));
        }
    }

    #[test]
    fn same_as_vec_deque() {
        let mut d0 = Deque::with_capacity(0);
        let mut d1 = VecDeque::new();
        let mut i = 0;
        let mut j = 0;
        for (k, l) in [(8, 4), (4, 4), (4, 4), (8, 12)] {
            for _ in 0..k {
                d0.push_back(i);
                d1.push_back(i);
                i += 1;
            }
            for _ in 0..l {
                assert_eq!(d0.pop_front(), Some(j));
                assert_eq!(d1.pop_front(), Some(j));
                j += 1;
            }
        }
        assert_eq!(d0.len(), 0);
        assert_eq!(d1.len(), 0);
    }

    #[test]
    fn same_as_vec_deque_pseudorandom() {
        let mut d0 = Deque::with_capacity(0);
        let mut d1 = VecDeque::new();

        let mut seed = Wrapping(0_usize);
        let mut get_rand = || -> usize {
            let value = Wrapping(0_usize);
            seed ^= value + Wrapping(0x9e3779b9) + (seed << 6) + (seed >> 2);
            seed.0
        };

        let mut value = 0;
        let num_repeats = 10000;
        let max_change = 100;

        for _ in 0..num_repeats {
            let n = get_rand() % max_change;
            for _ in 0..n {
                d0.push_back(value);
                d1.push_back(value);
                value += 1;
            }
            assert_eq!(d0.len(), d1.len());
            let n = (get_rand() % max_change).min(d0.len());
            for _ in 0..n {
                assert_eq!(d0.pop_front(), d1.pop_front());
            }
            assert_eq!(d0.len(), d1.len());
        }

        for _ in 0..d0.len() {
            assert_eq!(d0.pop_front(), d1.pop_front());
        }
        assert_eq!(d0.pop_front(), d1.pop_front());
        assert_eq!(d0.len(), 0);
        assert_eq!(d1.len(), 0);
    }
}
