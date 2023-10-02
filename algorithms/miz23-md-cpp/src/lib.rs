use std::ffi::c_int;
use std::iter::zip;
use std::ptr::null_mut;
use miz23_md_cpp_sys::ffi;
use petgraph::{Graph, Undirected};
use petgraph::graph::{DiGraph, UnGraph};
use petgraph::prelude::NodeIndex;
use petgraph::visit::EdgeRef;
use common::modular_decomposition::MDNodeKind;

fn convert_to_node_kind(node: &ffi::Node) -> Result<MDNodeKind, ()> {
    match node.kind {
        ffi::NodeKind::Prime => { Ok(MDNodeKind::Prime) }
        ffi::NodeKind::Series => { Ok(MDNodeKind::Series) }
        ffi::NodeKind::Parallel => { Ok(MDNodeKind::Parallel) }
        ffi::NodeKind::Vertex => { Ok(MDNodeKind::Vertex(node.vertex as _)) }
        ffi::NodeKind::Removed => { Err(()) }
    }
}


pub struct Prepared {
    graph: *mut ffi::Graph,
    num_vertices: usize,
}

impl Drop for Prepared {
    fn drop(&mut self) { unsafe { ffi::miz23_graph_delete(self.graph); }; }
}

pub fn prepare<N, E>(graph: &UnGraph<N, E>) -> Prepared {
    let num_vertices = graph.node_count();
    let mut prepared = Prepared { graph: null_mut(), num_vertices: graph.node_count() };

    let edges: Vec<ffi::Edge> = graph.edge_references().map(|e| {
        let (u, v) = (e.source().index() as _, e.target().index() as _);
        ffi::Edge { u, v }
    }).collect();

    let status = unsafe {
        ffi::miz23_graph_new(
            num_vertices as c_int,
            edges.as_ptr(),
            edges.len() as c_int,
            &mut prepared.graph as *mut *mut _)
    };
    assert_eq!(status, 0);

    prepared
}

pub struct Computed {
    result: *mut ffi::Result,
    num_vertices: usize,
}

impl Drop for Computed {
    fn drop(&mut self) { unsafe { ffi::miz23_result_delete(self.result); }; }
}

impl Prepared {
    pub fn compute(&self) -> Computed {
        let mut computed = Computed { result: null_mut(), num_vertices: self.num_vertices };

        let status = unsafe { ffi::miz23_compute(self.graph, &mut computed.result as *mut *mut _) };
        assert_eq!(status, 0);

        computed
    }
}

impl Computed {
    pub fn finalize(&self) -> DiGraph<MDNodeKind, ()> {
        let num_nodes = unsafe { ffi::miz23_result_size(self.result) } as usize;
        let mut nodes = vec![ffi::Node::default(); num_nodes];
        let mut vertices = vec![-1; self.num_vertices];

        let status = unsafe {
            ffi::miz23_result_copy_nodes(
                self.result,
                nodes.as_mut_ptr(), nodes.len() as _,
                vertices.as_mut_ptr(), vertices.len() as _)
        };
        assert_eq!(status, 0);

        assert!(zip(&nodes, &vertices).all(|(a, b)| a.vertex == *b));
        assert!(nodes.iter().all(|node| node.kind != ffi::NodeKind::Removed));

        let mut md_tree = DiGraph::<_, ()>::with_capacity(nodes.len(), nodes.len().saturating_sub(1));

        for node in &nodes {
            if let Ok(kind) = convert_to_node_kind(node) {
                md_tree.add_node(kind);
            }
        }

        let edges = nodes.iter().enumerate()
            .filter_map(|(idx, node)| {
                if node.kind != ffi::NodeKind::Removed && node.parent >= 0 {
                    Some((NodeIndex::new(node.parent as usize), NodeIndex::new(idx)))
                } else { None }
            });

        for (a, b) in edges {
            md_tree.add_edge(a, b, ());
        }
        md_tree
    }
}


pub fn modular_decomposition<N, E>(graph: &Graph<N, E, Undirected>) -> DiGraph<MDNodeKind, ()>
{
    prepare(graph).compute().finalize()
}


#[cfg(test)]
mod test {
    use petgraph::dot::{Config, Dot};
    use petgraph::visit::IntoNodeReferences;
    use common::instances::ted08_test0;
    use super::*;

    #[test]
    fn ted08_test0_graph() {
        let graph = ted08_test0();

        let md_tree = modular_decomposition(&graph);

        println!("{:?}", Dot::with_config(&md_tree, &[Config::EdgeNoLabel]));
    }

    #[test]
    fn test0() {
        let graph = ted08_test0();

        let md = modular_decomposition(&graph);

        let count_node_kinds = |tree: &DiGraph<MDNodeKind, ()>| -> (usize, usize, usize, usize) {
            tree.node_references()
                .fold((0, 0, 0, 0),
                      |(prime, series, parallel, vertex), (_, k)| {
                          match k {
                              MDNodeKind::Prime => (prime + 1, series, parallel, vertex),
                              MDNodeKind::Series => (prime, series + 1, parallel, vertex),
                              MDNodeKind::Parallel => (prime, series, parallel + 1, vertex),
                              MDNodeKind::Vertex(_) => (prime, series, parallel, vertex + 1)
                          }
                      })
        };

        assert_eq!(md.node_count(), 27);
        assert_eq!(count_node_kinds(&md), (2, 4, 3, 18));
    }
}