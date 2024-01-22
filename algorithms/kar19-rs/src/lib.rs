mod base;
mod shared;
mod improved;

use petgraph::graph::{DiGraph, UnGraph};
use common::modular_decomposition::MDNodeKind;


#[macro_export]
macro_rules! trace {
    ($($x:expr),*) => {
        tracing::trace!($($x),*)
    }
}

pub fn modular_decomposition(graph: &UnGraph<(), ()>) -> DiGraph<MDNodeKind, ()> {
    improved::modular_decomposition(graph)
}