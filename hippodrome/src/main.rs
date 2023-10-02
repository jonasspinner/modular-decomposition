use std::cmp::min;
use std::ffi::OsStr;
use std::fs;
use std::time::Instant;
use petgraph::graph::DiGraph;
use petgraph::{Incoming, Outgoing};
use petgraph::visit::DfsPostOrder;
use common::modular_decomposition::MDNodeKind;


fn canocicalize(md_tree: &DiGraph<MDNodeKind, ()>) -> Vec<(u32, u32, MDNodeKind)> {
    let root = md_tree.externals(Incoming).next().unwrap();
    let mut dfs = DfsPostOrder::new(md_tree, root);

    let mut result = vec![(u32::MAX, 0, MDNodeKind::Vertex(!0)); md_tree.node_count()];

    while let Some(a) = dfs.next(md_tree) {
        let mut min_vertex = u32::MAX;
        let mut num_vertices = 0;
        match md_tree[a] {
            MDNodeKind::Vertex(u) => {
                min_vertex = u as u32;
                num_vertices = 1;
            }
            _ => {}
        }
        for b in md_tree.neighbors_directed(a, Outgoing) {
            let (a_min, a_num, _) = result[b.index()];
            min_vertex = min(min_vertex, a_min);
            num_vertices += a_num;
        }
        result[a.index()] = (min_vertex, num_vertices, md_tree[a]);
    }
    result.sort();
    result
}

fn main() {
    //let graph = common::instances::ted08_test0();

    let mut paths : Vec<_> = fs::read_dir("hippodrome/instances/pace2023").unwrap().map(|p| p.unwrap().path()).collect();
    paths.sort();

    for path in paths {
        let graph = common::io::read_pace2023(&path).unwrap();

        let start = Instant::now();
        let md0 = miz23_md_rs::modular_decomposition(&graph);
        let t0 = start.elapsed();

        let start = Instant::now();
        let md1 = miz23_md_cpp::modular_decomposition(&graph);
        let t1 = start.elapsed();

        assert_eq!(canocicalize(&md0), canocicalize(&md1));
        println!("{}     Rust {:8} μs     C++ {:8} μs", path.file_name().and_then(OsStr::to_str).unwrap(), t0.as_micros(), t1.as_micros());
    }

    let graph = common::io::read_pace2023("hippodrome/instances/pace2023/exact_001.gr").unwrap();

    let md0 = miz23_md_rs::modular_decomposition(&graph);
    let md1 = miz23_md_cpp::modular_decomposition(&graph);

    assert_eq!(canocicalize(&md0), canocicalize(&md1));
}
