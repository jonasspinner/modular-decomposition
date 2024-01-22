use std::cmp::{min, Reverse};
use std::ffi::OsStr;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::time::Instant;
use petgraph::graph::{DiGraph};
use petgraph::{Incoming, Outgoing};
use petgraph::visit::DfsPostOrder;
use common::modular_decomposition::MDNodeKind;


fn canonicalize(md_tree: &DiGraph<MDNodeKind, ()>) -> Vec<(u32, u32, MDNodeKind)> {
    let Some(root) = md_tree.externals(Incoming).next() else { return vec![] };
    let mut dfs = DfsPostOrder::new(md_tree, root);

    let mut info = vec![(u32::MAX, u32::MAX); md_tree.node_count()];

    while let Some(a) = dfs.next(md_tree) {
        let mut min_vertex = u32::MAX;
        let mut num_vertices = 0;
        if let MDNodeKind::Vertex(u) = md_tree[a] {
            min_vertex = u as u32;
            num_vertices = 1;
        }
        for b in md_tree.neighbors_directed(a, Outgoing) {
            let (a_min, a_num) = info[b.index()];
            min_vertex = min(min_vertex, a_min);
            num_vertices += a_num;
        }
        info[a.index()] = (min_vertex, num_vertices);
    }

    let mut result = vec![];
    let mut stack = vec![root];
    while let Some(a) = stack.pop() {
        let (m, n) = info[a.index()];
        result.push((m, n, md_tree[a]));
        let mut children: Vec<_> = md_tree.neighbors_directed(a, Outgoing).collect();
        children.sort_by_key(|c| Reverse(info[c.index()].0));
        stack.extend(children);
    }
    result
}

fn main() {
    let mut paths: Vec<_> = fs::read_dir("data/02-graphs").unwrap().map(|p| p.unwrap().path()).filter(|p| p.is_file()).collect();
    paths.sort_by_key(|p| p.metadata().unwrap().size());

    for (i, path) in paths.iter().enumerate() {
        let graph = common::io::read_metis(path).unwrap();

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

        let problem = kar19_rs::prepare(&graph);
        let start = Instant::now();
        let result = problem.compute();
        let t3 = start.elapsed();
        let md3 = result.finalize();

        let md0 = canonicalize(&md0);
        let md1 = canonicalize(&md1);
        let md2 = canonicalize(&md2);
        let md3 = canonicalize(&md3);

        assert_eq!(md0, md1);
        assert_eq!(md1, md2);
        assert_eq!(md2, md3);

        let fastest_time = *[t0, t1, t2, t3]
            .map(|t| t.as_nanos())
            .iter().min().unwrap() as f64;

        println!("{i:4.} {:<30.30} miz23-rust {:9} μs {:6.2}  miz23-cpp {:9} μs {:6.2}  ms00 {:9} μs {:6.2}  kar19-rust {:9} μs {:6.2}",
                 path.file_name().and_then(OsStr::to_str).unwrap(),
                 t0.as_micros(), (t0.as_nanos() as f64 / fastest_time),
                 t1.as_micros(), (t1.as_nanos() as f64 / fastest_time),
                 t2.as_micros(), (t2.as_nanos() as f64 / fastest_time),
                 t3.as_micros(), (t3.as_nanos() as f64 / fastest_time),
        );
    }
}


#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use petgraph::dot::{Config, Dot};
    use petgraph::graph::{DiGraph, NodeIndex, UnGraph};
    use petgraph::visit::{EdgeRef, IntoNodeReferences};
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
        #[allow(unused_variables)]
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

    #[test]
    fn pace2023_exact_055() {
        let original = common::io::read_metis("../data/02-graphs/pace2023-exact_055").unwrap();

        let vertices = [0_usize, 135, 154, 159, 145, 20, 26, 83, 84, 107, 2, 160, 158, 157, 119, 9, 98, 87, 73, 142, 85, 127, 108, 153, 151, 17, 23, 69, 1, 12, 48, 126, 130, 139, 141, 44, 100, 50, 54, 152, 155, 146, 156, 143, 58, 163, 129, 91, 133, 13, 81, 89, 97, 62, 24, 113, 101, 128, 47, 162, 79, 41, 111, 148, 105, 27, 144, 52, 45, 71, 161, 150, 140, 14, 104, 65, 95, 76, 147, 82, 75, 43, 102, 3, 5, 6, 7, 8, 10, 11, 15, 16, 18, 19, 21, 22, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 46, 49, 51, 53, 55, 56, 57, 59, 60, 61, 63, 64, 66, 67, 68, 70, 72, 74, 77, 78, 80, 86, 88, 90, 92, 93, 94, 96, 99, 103, 106, 109, 110, 112, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 131, 132, 134, 136, 137, 138, 149, 4];
        let vertices = &vertices[15..48];
        let edges: Vec<_> = original.edge_references().map(|e| (e.source().index(), e.target().index())).collect();

        let mut graph = UnGraph::<(), ()>::new_undirected();
        let mut map = HashMap::<usize, NodeIndex>::new();
        for &u in vertices {
            map.insert(u, graph.add_node(()));
        }
        for (u, v) in edges {
            if let (Some(&u), Some(&v)) = (map.get(&u), map.get(&v)) {
                graph.add_edge(u, v, ());
            }
        }

        let mut graph = UnGraph::new_undirected();
        for _ in 0..33 {
            graph.add_node(());
        }
        let edges = [
            (13_usize, 0_usize), (13, 14), (13, 10), (13, 11), (13, 20),
            (13, 15), (13, 22), (13, 23), (13, 12), (13, 3),
            (13, 5), (13, 2), (13, 1), (13, 21), (13, 7),
            //(13, 16), (13, 6), (13, 31), (13, 17), (13, 18),
            //(13, 19), (13, 4), (13, 28), (13, 26), (13, 9),
            (13, 24), (13, 8), (13, 25), (13, 27), (13, 30),
            (0, 14), (0, 10), (0, 11), (0, 20), (0, 15),
            (0, 22), (0, 23), (0, 29), (0, 12), (0, 3),
            (0, 5), (0, 2), (0, 32), (0, 1), (0, 21),
            (0, 7), (0, 16), (0, 6), (0, 31), (0, 17),
            (0, 18), (0, 19), (0, 4), (0, 28), (0, 26),
            (0, 9), (0, 24), (0, 8), (0, 25), (0, 27),
            (14, 10), (14, 11), (14, 20), (14, 15), (14, 22),
            (14, 23), (14, 12), (14, 3), (14, 5), (14, 2),
            //(14, 1), (14, 21), (14, 7), (14, 16), (14, 6),
            //(14, 31), (14, 17), (14, 18), (14, 19), (14, 4),
            //(14, 28), (14, 26), (14, 9), (14, 24), (14, 8), (14, 25), (14, 27),
            //(10, 11), (10, 20), (10, 15), (10, 22), (10, 23), (10, 29), (10, 12), (10, 3), (10, 5), (10, 2), (10, 1), (10, 21), (10, 7), (10, 16), (10, 6), (10, 31), (10, 17), (10, 18), (10, 19), (10, 4), (10, 28), (10, 26), (10, 9), (10, 24), (10, 8), (10, 25), (10, 27), (10, 30),
            //(11, 20), (11, 15), (11, 22), (11, 23), (11, 29), (11, 12), (11, 3), (11, 5), (11, 2), (11, 1), (11, 21), (11, 7), (11, 16), (11, 6), (11, 31), (11, 17), (11, 18), (11, 19), (11, 4), (11, 28), (11, 26), (11, 9), (11, 24), (11, 8), (11, 25), (11, 27),
            (20, 15), (20, 22), (20, 23), (20, 12), (20, 3),
            (20, 5), (20, 2), (20, 1), (20, 21), (20, 7),
            //(20, 16), (20, 6), (20, 17), (20, 18), (20, 19),
            //(20, 4), (20, 28), (20, 26), (20, 9), (20, 24), (20, 8), (20, 25), (20, 27),
            (15, 22), (15, 23), (15, 29), (15, 12), (15, 3),
            (15, 5), (15, 2), (15, 1), (15, 21), (15, 7),
            //(15, 16), (15, 6), (15, 31), (15, 17), (15, 18),
            //(15, 19), (15, 4), (15, 28), (15, 26), (15, 9),
            //(15, 24), (15, 8), (15, 25), (15, 27),
            (22, 23), (22, 12), (22, 3), (22, 5), (22, 2),
            (22, 32), (22, 1), (22, 21), (22, 7), (22, 16),
            (22, 6), (22, 31), (22, 17), (22, 18), (22, 19),
            (22, 4), (22, 28), (22, 26), (22, 9), (22, 24),
            (22, 8), (22, 25), (22, 27),
            (23, 29), (23, 12), (23, 3), (23, 5), (23, 2),
            (23, 32), (23, 1), (23, 21), (23, 7), (23, 16),
            (23, 6), (23, 31), (23, 17), (23, 18), (23, 19),
            (23, 4), (23, 28), (23, 26), (23, 9), (23, 24),
            (23, 8), (23, 25), (23, 27), (23, 30),
            //(29, 12), (29, 32), (29, 6), (29, 31), (29, 26), (29, 25), (29, 27), (29, 30),
            (12, 3), (12, 5), (12, 2), (12, 1), (12, 21),
            //(12, 7), (12, 16), (12, 6), (12, 31), (12, 17),
            //(12, 18), (12, 19), (12, 4), (12, 28), (12, 26),
            //(12, 9), (12, 24), (12, 8), (12, 25), (12, 27),
            (3, 5), (3, 2), (3, 32), (3, 1), (3, 21), (3, 7),
            (3, 16), (3, 6), (3, 31), (3, 17), (3, 18), (3, 19), (3, 4), (3, 28), (3, 26), (3, 9), (3, 24), (3, 8), (3, 25), (3, 27), (3, 30),
            (5, 2), (5, 32), (5, 1), (5, 21), (5, 7), (5, 16), (5, 6), (5, 31), (5, 17), (5, 18), (5, 19), (5, 4), (5, 28), (5, 26), (5, 9), (5, 24), (5, 8), (5, 25), (5, 27),
            //(2, 32), (2, 1), (2, 21), (2, 7), (2, 16), (2, 6), (2, 31), (2, 17), (2, 18), (2, 19), (2, 4), (2, 28), (2, 26), (2, 9), (2, 24), (2, 8), (2, 25), (2, 27), (2, 30),
            //(32, 1), (32, 7), (32, 16), (32, 6), (32, 31), (32, 17), (32, 18), (32, 4), (32, 28), (32, 9), (32, 8), (32, 25), (32, 30),
            //(1, 21), (1, 7), (1, 16), (1, 6), (1, 31), (1, 17), (1, 18), (1, 19), (1, 4), (1, 28), (1, 26), (1, 9), (1, 24), (1, 8), (1, 25), (1, 27), (1, 30),
            //(21, 7), (21, 16), (21, 6), (21, 17), (21, 18), (21, 19), (21, 4), (21, 28), (21, 26), (21, 9), (21, 24), (21, 8), (21, 25), (21, 27), (21, 30),
            //(7, 16), (7, 6), (7, 31), (7, 17), (7, 18), (7, 19), (7, 4), (7, 28), (7, 26), (7, 9), (7, 24), (7, 8), (7, 25), (7, 27), (7, 30),
            //(16, 6), (16, 17), (16, 18), (16, 19), (16, 4), (16, 28), (16, 26), (16, 9), (16, 24), (16, 8), (16, 25), (16, 27), (16, 30), (6, 31), (6, 17), (6, 18), (6, 19), (6, 4), (6, 28), (6, 26), (6, 9), (6, 24), (6, 8), (6, 25), (6, 27), (6, 30),
            //(31, 4), (31, 26), (31, 9), (31, 25), (31, 30), (17, 18), (17, 19), (17, 4), (17, 28), (17, 26), (17, 9), (17, 24), (17, 8), (17, 25), (17, 27),
            (18, 19), (18, 4), (18, 28), (18, 26), (18, 9), (18, 24), (18, 8), (18, 25), (18, 27), (19, 4),
            //(19, 28), (19, 26), (19, 9), (19, 24), (19, 8), (19, 25), (19, 27), (19, 30),
            //(4, 28), (4, 26), (4, 9), (4, 24), (4, 8), (4, 25), (4, 27),
            //(28, 26), (28, 9), (28, 24), (28, 8), (28, 25), (28, 27), (28, 30),
            //(26, 9), (26, 24), (26, 8), (26, 25), (26, 27), (26, 30),
            (9, 24), (9, 8), (9, 25), (9, 27),
            (24, 8), (24, 25), (24, 27),
            (8, 25), (8, 27), (8, 30),
            (25, 27), (25, 30),
            (27, 30)
        ];
        for (u, v) in edges {
            graph.add_edge(NodeIndex::new(u), NodeIndex::new(v), ());
        }

        println!("{:?}", graph.node_indices().map(|u| u.index()).collect::<Vec<_>>());
        println!("{:?}", graph.edge_references().map(|e| (e.source().index(), e.target().index())).collect::<Vec<_>>());


        //println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));

        let md0 = miz23_md_rs::modular_decomposition(&graph);
        let md1 = kar19_rs::modular_decomposition(&graph);

        let mut md0_edges: Vec<_> = md0.edge_references().map(|e| (e.source(), e.target())).collect();
        md0_edges.sort();
        let mut md0_ = DiGraph::new();
        for (_, w) in md0.node_references() { md0_.add_node(*w); }
        for (a, b) in md0_edges { md0_.add_edge(a, b, ()); }
        let mut md1_edges: Vec<_> = md1.edge_references().map(|e| (e.source(), e.target())).collect();
        md1_edges.sort();
        let mut md1_ = DiGraph::new();
        for (_, w) in md1.node_references() { md1_.add_node(*w); }
        for (a, b) in md1_edges { md1_.add_edge(a, b, ()); }


        println!("{:?}", Dot::with_config(&md0, &[Config::EdgeNoLabel]));
        //println!("{:?}", Dot::with_config(&md1, &[Config::EdgeNoLabel]));

        //println!("{:?}", canonicalize(&md0));
        //println!("{:?}", canonicalize(&md1));

        assert_eq!(canonicalize(&md0), canonicalize(&md1));
    }

    #[test]
    fn pace2023_heuristic_010() {
        let graph = common::io::read_metis("../data/02-graphs/pace2023-heuristic_010").unwrap();

        let md0 = miz23_md_rs::modular_decomposition(&graph);
        let md1 = kar19_rs::modular_decomposition(&graph);

        assert_eq!(canonicalize(&md0), canonicalize(&md1));
    }

    #[test]
    fn pace2023_heuristic_022() {
        let graph = common::io::read_metis("../data/02-graphs/pace2023-heuristic_022").unwrap();

        let md0 = miz23_md_rs::modular_decomposition(&graph);
        let md1 = kar19_rs::modular_decomposition(&graph);


        //println!("{:?}", Dot::with_config(&md0, &[Config::EdgeNoLabel]));
        //println!("{:?}", Dot::with_config(&md1, &[Config::EdgeNoLabel]));

        //println!("{:?}", canonicalize(&md0));
        //println!("{:?}", canonicalize(&md1));

        assert_eq!(canonicalize(&md0), canonicalize(&md1));
    }
}