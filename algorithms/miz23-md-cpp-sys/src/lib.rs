pub mod ffi {
    use std::ffi::{c_float, c_int};
    use std::fmt::Debug;

    #[repr(C)]
    #[derive(Copy, Clone)]
    pub struct Edge {
        pub u: c_int,
        pub v: c_int,
    }

    impl<V: Into<c_int>> From<(V, V)> for Edge {
        fn from((u, v): (V, V)) -> Self { Self { u: u.into(), v: v.into() } }
    }

    #[repr(C)]
    #[derive(Debug, Clone, Eq, PartialEq)]
    pub enum NodeKind {
        Prime = 0,
        Series = 1,
        Parallel = 2,
        Vertex = 3,
        Removed = 4,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Eq, PartialEq)]
    pub struct Node {
        pub kind: NodeKind,
        pub vertex: c_int,
        pub parent: c_int,
        pub vertices_begin: c_int,
        pub vertices_end: c_int,
    }

    impl Default for Node {
        fn default() -> Self {
            Node { kind: NodeKind::Removed, vertex: -1, parent: -1, vertices_begin: -1, vertices_end: -1 }
        }
    }

    #[repr(C)]
    pub struct Graph {
        _data: [u8; 0],
        _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
    }

    #[repr(C)]
    pub struct Result {
        _data: [u8; 0],
        _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
    }

    extern "C" {
        pub fn miz23_graph_new(n: c_int, edges: *const Edge, m: c_int, out_graph: *mut *mut Graph) -> c_int;
        pub fn miz23_graph_delete(graph: *mut Graph);
        pub fn miz23_compute(graph: *const Graph, out_computed: *mut *mut Result) -> c_int;
        pub fn miz23_result_delete(result: *mut Result);
        pub fn miz23_result_time(result: *const Result) -> c_float;
        pub fn miz23_result_size(result: *const Result) -> c_int;
        pub fn miz23_result_copy_nodes(result: *const Result, nodes: *mut Node, num_nodes: c_int, vertices: *mut c_int, num_vertices: c_int) -> c_int;
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_int;
    use std::ptr::null_mut;
    use super::ffi::*;

    #[test]
    fn it_works() {
        let num_vertices = 5;
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)].map(Into::into);

        let mut graph = null_mut();
        let status = unsafe { miz23_graph_new(num_vertices as _, edges.as_ptr(), edges.len() as _, &mut graph as *mut *mut _) };
        assert_eq!(status, 0);

        let mut result = null_mut();
        let status = unsafe { miz23_compute(graph, &mut result as *mut *mut _) };
        assert_eq!(status, 0);

        let time = unsafe { miz23_result_time(result) } as f32;
        assert!(time < 1.0);

        let size = unsafe { miz23_result_size(result) } as usize;
        assert_eq!(size, 6);

        let mut nodes = vec![Node::default(); size];
        let mut vertices = vec![-1; num_vertices];

        let status = unsafe { miz23_result_copy_nodes(result, nodes.as_mut_ptr(), nodes.len() as _, vertices.as_mut_ptr(), vertices.len() as _) };
        assert_eq!(status, 0);

        assert_eq!(vertices, (0..num_vertices).rev().map(|i| i as c_int).collect::<Vec<_>>());

        unsafe { miz23_graph_delete(graph) };
        unsafe { miz23_result_delete(result) };
    }
}
