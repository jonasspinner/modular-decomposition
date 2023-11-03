use std::cmp::min;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::time::Instant;
use petgraph::graph::{DiGraph, UnGraph};
use petgraph::{Incoming, Outgoing};
use petgraph::dot::{Config, Dot};
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
    let mut paths: Vec<_> = fs::read_dir("hippodrome/instances/pace2023").unwrap().map(|p| p.unwrap().path()).collect();
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

        let problem = ms00::prepare(&graph);
        let start = Instant::now();
        let result = problem.compute();
        let t2 = start.elapsed();
        let md2 = result.finalize();


        let md0 = canonicalize(&md0);
        let md1 = canonicalize(&md1);
        let md2 = canonicalize(&md2);
        assert_eq!(md0, md1);
        assert_eq!(md1, md2);

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


#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use petgraph::dot::{Config, Dot};
    use petgraph::graph::{NodeIndex, UnGraph};
    use crate::canonicalize;

    #[test]
    fn exact_053() {
        //let vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71];
        let vertices = [
            4, 5, 6, 7, 8, 9,
            //10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            //20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            //30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            //40, 41, 42, 43, 47, 49,
            //50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
            //60, 61,
            62, 68,
        ];
        let vertices_8 = [
            4, 5, 6, 7, 8, 9, 62, 68,
        ];

        let edges = [
            (1, 2), (2, 3), (2, 4), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
            (13, 14), (14, 15), (14, 16), (17, 18), (17, 19), (17, 20), (17, 21), (17, 22),
            (17, 23), (17, 24), (17, 25), (17, 26), (17, 27), (17, 28), (17, 29), (17, 30),
            (17, 31), (17, 32), (33, 34), (13, 33), (33, 35), (33, 36), (33, 37), (38, 39),
            (38, 40), (38, 41), (38, 42), (38, 43), (38, 44), (23, 25), (19, 23), (1, 45), (1, 4),
            (1, 46), (1, 47), (1, 3), (41, 43), (41, 44), (39, 41), (41, 48), (13, 15), (4, 49),
            (13, 50), (50, 51), (20, 27), (27, 52), (21, 27), (27, 53), (27, 54), (26, 27),
            (22, 27), (27, 28), (27, 30), (27, 32), (37, 55), (35, 37), (34, 37), (36, 37),
            (37, 56), (21, 28), (24, 28), (18, 28), (28, 57), (28, 29), (28, 31), (3, 47), (3, 45),
            (3, 4), (39, 44), (40, 44), (44, 48), (43, 44), (12, 40), (8, 12), (12, 42), (12, 58),
            (55, 56), (36, 55), (20, 22), (22, 26), (22, 30), (22, 32), (19, 25), (58, 59), (7, 59),
            (24, 60), (21, 24), (24, 29), (18, 24), (24, 31), (18, 21), (21, 53), (21, 26),
            (21, 29), (21, 30), (21, 31), (21, 32), (54, 61), (16, 54), (52, 54), (8, 54), (53, 54),
            (58, 62), (40, 58), (42, 58), (43, 58), (6, 9), (7, 9), (9, 63), (9, 11), (9, 10),
            (20, 30), (30, 53), (26, 30), (30, 32), (31, 52), (31, 35), (18, 31), (29, 31),
            (43, 62), (39, 43), (43, 48), (64, 65), (6, 11), (7, 11), (11, 63), (10, 11), (66, 67),
            (46, 68), (45, 47), (46, 47), (4, 47), (10, 69), (16, 69), (18, 29), (39, 40), (8, 52),
            (8, 16), (35, 61), (35, 52), (34, 35), (35, 36), (65, 70), (4, 45), (6, 7), (34, 36),
            (36, 56), (29, 52), (45, 46), (13, 51), (10, 16), (7, 10), (10, 63), (71, 72), (53, 61),
            (52, 61), (7, 63), (52, 53), (26, 53), (32, 53), (40, 42), (20, 26), (20, 32), (26, 32)
        ].map(|(u, v)| (u - 1, v - 1));

        let mut graph = UnGraph::<usize, ()>::new_undirected();
        let mut map = HashMap::<usize, NodeIndex>::new();
        for u in vertices {
            map.insert(u, graph.add_node(u));
        }
        for (u, v) in edges {
            if let (Some(&u), Some(&v)) = (map.get(&u), map.get(&v)) {
                graph.add_edge(u, v, ());
            }
        }

        println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

        let md0 = miz23_md_rs::modular_decomposition(&graph);
        let md1 = ms00::modular_decomposition(&graph);

        println!("{:?}", Dot::with_config(&md0, &[Config::EdgeNoLabel]));
        println!("{:?}", Dot::with_config(&md1, &[Config::EdgeNoLabel]));

        assert_eq!(canonicalize(&md0), canonicalize(&md1));
    }
}