mod io;

use std::ffi::{c_int};
use std::fmt::{Debug};
use std::iter::zip;
use std::ptr::null_mut;
use miz23_md_cpp_sys::ffi;
use petgraph::{Graph, Undirected};
use petgraph::adj::{IndexType};
use petgraph::graph::{DiGraph};
use petgraph::prelude::{NodeIndex, UnGraph};
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use crate::MDNodeKind::Prime;

#[derive(Debug)]
enum MDNodeKind {
    Prime,
    Series,
    Parallel,
    Vertex(usize),
}

impl Default for MDNodeKind {
    fn default() -> Self { Prime }
}

impl TryFrom<&ffi::Node> for MDNodeKind {
    type Error = ();
    fn try_from(node: &ffi::Node) -> Result<Self, Self::Error> {
        Ok(match node.kind {
            ffi::NodeKind::Prime => { Prime }
            ffi::NodeKind::Series => { Self::Series }
            ffi::NodeKind::Parallel => { Self::Parallel }
            ffi::NodeKind::Vertex => { Self::Vertex(node.vertex as usize) }
            ffi::NodeKind::Removed => { return Err(()); }
        })
    }
}

fn modular_decomposition<N, E, Ix>(graph: &Graph<N, E, Undirected, Ix>) -> (f32, DiGraph<MDNodeKind, ()>)
    where Ix: IndexType, NodeIndex: From<Ix>
{
    struct MIZ23 {
        inner: *mut ffi::Result,
        num_vertices: usize,
    }
    impl MIZ23 {
        fn modular_decomposition(num_vertices: usize, edges: &Vec<ffi::Edge>) -> Result<Self, ()> {
            let mut result = MIZ23 { inner: null_mut(), num_vertices };
            let status = unsafe { ffi::miz23_modular_decomposition(num_vertices as c_int, edges.as_ptr(), edges.len() as c_int, &mut result.inner as *mut *mut _) };

            if status == 0 {
                Ok(result)
            } else { Err(()) }
        }
    }
    impl Drop for MIZ23 {
        fn drop(&mut self) { unsafe { ffi::miz23_result_delete(self.inner); }; }
    }
    impl MIZ23 {
        fn time(&self) -> f32 { return unsafe { ffi::miz23_result_time(self.inner) } as _; }

        fn get_nodes(&self) -> Result<(Vec<ffi::Node>, Vec<i32>), ()> {
            let num_nodes = unsafe { ffi::miz23_result_size(self.inner) } as usize;
            let mut nodes = vec![ffi::Node::default(); num_nodes];
            let mut vertices = vec![-1; self.num_vertices];

            let status = unsafe {
                ffi::miz23_result_copy_nodes(
                    self.inner,
                    nodes.as_mut_ptr(), nodes.len() as _,
                    vertices.as_mut_ptr(), vertices.len() as _)
            };

            if status == 0 {
                Ok((nodes, vertices))
            } else { Err(()) }
        }
    }

    let edges: Vec<ffi::Edge> = graph.edge_references().map(|e| {
        let (u, v) = (e.source().index() as _, e.target().index() as _);
        ffi::Edge { u, v }
    }).collect();

    let result = MIZ23::modular_decomposition(graph.node_count(), &edges).unwrap();
    let time = result.time();
    let (nodes, vertices) = result.get_nodes().unwrap();

    //println!("{:?}", nodes);
    //println!("{:?}", nodes.iter().map(|node| node.vertex).collect::<Vec<_>>());
    //println!("{:?}", nodes.iter().map(|node| node.parent).collect::<Vec<_>>());
    //println!("{:?}", vertices);
    //println!("{:?}", nodes.iter().map(|node| node.vertices_end - node.vertices_begin).collect::<Vec<_>>());

    assert!(zip(&nodes, &vertices).all(|(a, b)| a.vertex == *b));
    assert!(nodes.iter().all(|node| node.kind != ffi::NodeKind::Removed));

    let mut md_tree = DiGraph::<_, ()>::new();

    for node in &nodes {
        if let Ok(kind) = node.try_into() {
            md_tree.add_node(kind);
        }
    }

    let edges = nodes.iter().enumerate()
        .filter_map(|(idx, node)| {
            if node.kind != ffi::NodeKind::Removed && node.parent > 0 {
                Some((Ix::new(node.parent as usize), Ix::new(idx)))
            } else { None }
        });
    md_tree.extend_with_edges(edges);
    (time, md_tree)
}


fn example_graph() -> UnGraph<(), ()> {
    let mut graph = UnGraph::new_undirected();
    graph.extend_with_edges([
    (0, 4),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
    (5, 14),
    (5, 15),
    (5, 16),
    (5, 17),
    (5, 11),
    (5, 12),
    (5, 10),
    (5, 13),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (8, 9),
    (10, 11),
    (10, 12),
    (10, 13),
    (11, 12),
    (11, 13),
    (12, 13),
    (13, 14),
    (13, 15),
    (13, 16),
    (14, 15),
    (14, 17),
    (15, 17),
    (16, 17),
    ]);
    graph
}

fn main() {
    /*
    let mut paths: Vec<String> = vec!["../extern/miz23-md/data/exact_200.gr".into(), "../../extern/miz23-md/data/exact_050.gr".into(), "../../extern/miz23-md/data/exact_100.gr".into()];

    let num_p = 50;
    for i in 0..num_p {
        let n = 10000;
        let p = ((i as f32) + 1.0) / 50.0;
        let dir_path = "../../extern/miz23-md/data/duplication_divergence_graph-10000";
        let path = format!("{dir_path}/duplication_divergence_graph_{n}_{p:.2}_{i}");
        paths.push(path);
    }

    for path in &paths {
        let graph = io::read_pace2023(path).or_else(|_| io::read_edgelist(path)).expect("could not read file");
        let (time, md_tree) = modular_decomposition(&graph);
        let num_inner_nodes = md_tree.node_references().filter(|(idx, kind)| { !matches!(kind, MDNodeKind::Vertex(_)) }).count();
        println!("{:5} {:9} {time:.6} {num_inner_nodes:5} {:.6}", graph.node_count(), graph.edge_count(), (num_inner_nodes as f32) / (graph.node_count() as f32));
    }
    */

    let graph = example_graph();
    let (time, md_tree) = modular_decomposition(&graph);
    let num_inner_nodes = md_tree.node_references().filter(|(_idx, kind)| { !matches!(kind, MDNodeKind::Vertex(_)) }).count();
    println!("{:5} {:9} {time:.6} {num_inner_nodes:5} {:.6}", graph.node_count(), graph.edge_count(), (num_inner_nodes as f32) / (graph.node_count() as f32));

    // println!("{:?}", Dot::with_config(&md_tree, &[Config::EdgeNoLabel]));
}
