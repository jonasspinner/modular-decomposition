use std::cmp::min;
use std::ffi::OsStr;
use std::fs;
use std::time::Instant;
use petgraph::graph::DiGraph;
use petgraph::{Incoming, Outgoing};
use petgraph::visit::DfsPostOrder;
use common::modular_decomposition::MDNodeKind;


fn canonicalize(md_tree: &DiGraph<MDNodeKind, ()>) -> Vec<(u32, u32, MDNodeKind)> {
    let root = md_tree.externals(Incoming).next().unwrap();
    let mut dfs = DfsPostOrder::new(md_tree, root);

    let mut result = vec![(u32::MAX, 0, MDNodeKind::Vertex(!0)); md_tree.node_count()];

    while let Some(a) = dfs.next(md_tree) {
        let mut min_vertex = u32::MAX;
        let mut num_vertices = 0;
        if let MDNodeKind::Vertex(u) = md_tree[a] {
            min_vertex = u as u32;
            num_vertices = 1;
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

        let problem = miz23_md_rs::prepare(&graph);
        let start = Instant::now();
        let result = problem.compute();
        let t0 = start.elapsed();
        let md0 = result.finalize();

        let problem = miz23_md_cpp::prepare(&graph);
        let start = Instant::now();
        let result = problem.compute();
        let t1 = start.elapsed();
        let md1 = result.finalize();


        let start = Instant::now();
        let result = ms00::modular_decomposition(&graph);
        let t2 = start.elapsed();


        assert_eq!(canonicalize(&md0), canonicalize(&md1));

        let fastest_time = min(min(t0.as_nanos(), t1.as_nanos()), t2.as_nanos()) as f64;

        println!("{}     Rust {:8} μs {:3.2}    C++ {:8} μs {:3.2}    MS00 {:8} μs {:3.2}",
                 path.file_name().and_then(OsStr::to_str).unwrap(),
                 t0.as_micros(), (t0.as_nanos() as f64 / fastest_time),
                 t1.as_micros(), (t1.as_nanos() as f64 / fastest_time),
                 t2.as_micros(), (t2.as_nanos() as f64 / fastest_time));
    }

    let graph = common::io::read_pace2023("hippodrome/instances/pace2023/exact_001.gr").unwrap();

    let md0 = miz23_md_rs::modular_decomposition(&graph);
    let md1 = miz23_md_cpp::modular_decomposition(&graph);

    assert_eq!(canonicalize(&md0), canonicalize(&md1));
}
