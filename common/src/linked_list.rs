use std::iter::from_fn;
use crate::index::IndexType;

pub trait CyclicLinkedListNode: Sized {
    type Index: IndexType;
    fn prev(&self) -> Self::Index;
    fn prev_mut(&mut self) -> &mut Self::Index;
    fn next(&self) -> Self::Index;
    fn next_mut(&mut self) -> &mut Self::Index;
    fn prev_next(&self) -> (Self::Index, Self::Index);
}

pub trait CyclicLinkedList: Sized {
    type Node: CyclicLinkedListNode;
    fn head(&self) -> <Self::Node as CyclicLinkedListNode>::Index;
    fn head_mut(&mut self) -> &mut <Self::Node as CyclicLinkedListNode>::Index;
}

pub fn move_before<N: CyclicLinkedListNode>(nodes: &mut [N], a: N::Index, b: N::Index) {
    debug_assert!(a.is_valid());
    debug_assert!(b.is_valid());
    debug_assert_ne!(a, b);
    // a-1, a, a+1    -->     a-1, a+1
    // b-1, b, b+1            b-1, a, b, b+1
    let (a_prev, a_next) = nodes[a.index()].prev_next();
    let b_prev = nodes[b.index()].prev();
    if a_next == b { return; };
    *nodes[a_prev.index()].next_mut() = a_next;
    *nodes[a_next.index()].prev_mut() = a_prev;
    *nodes[a.index()].prev_mut() = b_prev;
    *nodes[b_prev.index()].next_mut() = a;
    *nodes[a.index()].next_mut() = b;
    *nodes[b.index()].prev_mut() = a;
}


pub fn move_after<N: CyclicLinkedListNode>(nodes: &mut [N], a: N::Index, b: N::Index) {
    debug_assert!(a.is_valid());
    debug_assert!(b.is_valid());
    debug_assert_ne!(a, b);
    // a-1, a, a+1    -->     a-1, a+1
    // b-1, b, b+1            b-1, b, a, b+1
    let (a_prev, a_next) = nodes[a.index()].prev_next();
    let b_next = nodes[b.index()].next();
    if a_prev == b { return; };

    *nodes[a_prev.index()].next_mut() = a_next;
    *nodes[a_next.index()].prev_mut() = a_prev;

    *nodes[a.index()].prev_mut() = b;
    *nodes[b_next.index()].prev_mut() = a;
    *nodes[a.index()].prev_mut() = b;
    *nodes[b.index()].next_mut() = a;
}

pub fn concat<N: CyclicLinkedListNode>(nodes: &mut [N], a: N::Index, b: N::Index) {
    // a, a+1, ..., a-1, a    -->     b, b+1, ..., b-1, a, a+1, ..., a-1, b
    // b, b+1, ..., b-1, b
    debug_assert!(a.is_valid());
    debug_assert!(b.is_valid());
    debug_assert_ne!(a, b);

    let a_prev = nodes[a.index()].prev();
    let b_prev = nodes[b.index()].prev();

    *nodes[b_prev.index()].next_mut() = a;
    *nodes[a.index()].prev_mut() = b_prev;
    *nodes[a_prev.index()].next_mut() = b;
    *nodes[b.index()].prev_mut() = a_prev;
}

pub fn cut<N: CyclicLinkedListNode>(nodes: &mut [N], a: N::Index) {
    debug_assert!(a.is_valid());

    let (prev, next) = nodes[a.index()].prev_next();
    assert_ne!(prev, a);
    assert_ne!(next, a);
    *nodes[prev.index()].next_mut() = next;
    *nodes[next.index()].prev_mut() = prev;
    *nodes[a.index()].next_mut() = a;
    *nodes[a.index()].prev_mut() = a;
}

pub fn elements<'a, L: CyclicLinkedList>(nodes: &'a [L::Node], list: &L) -> impl Iterator<Item=<<L as CyclicLinkedList>::Node as CyclicLinkedListNode>::Index> + 'a {
    let head = list.head();
    let mut next = head;
    from_fn(move || {
        next.is_valid().then(|| {
            let current = next;
            next = nodes[current.index()].next();
            if next == head { next = <<L as CyclicLinkedList>::Node as CyclicLinkedListNode>::Index::invalid(); }
            current
        })
    })
}


mod list {
    use super::*;

    #[allow(dead_code)]
    pub fn concat<L: CyclicLinkedList>(nodes: &mut [L::Node], a: &L, b: &L) {
        super::concat(nodes, a.head(), b.head());
    }

    #[allow(dead_code)]
    pub fn cut<L: CyclicLinkedList>(nodes: &mut [L::Node], list: &L, a: <<L as CyclicLinkedList>::Node as CyclicLinkedListNode>::Index) -> Option<<<L as CyclicLinkedList>::Node as CyclicLinkedListNode>::Index> {
        let head = list.head();
        let head = if head == a { nodes[head.index()].next() } else { head };
        let head = if head == a { None } else { Some(head) };
        super::cut(nodes, a);
        head
    }
}