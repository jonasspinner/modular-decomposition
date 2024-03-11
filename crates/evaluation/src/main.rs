use canonicalize::canonicalize;
use petgraph::graph::NodeIndex;
use petgraph::visit::Bfs;
use std::error::Error;
use std::ffi::OsStr;
use std::os::unix::fs::MetadataExt;
use std::time::Instant;
use std::{env, fs};

pub(crate) mod canonicalize;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let dir = args.get(1).ok_or("usage: evaluation [dir]")?;

    let mut paths: Vec<_> = fs::read_dir(dir)?.map(|p| p.unwrap().path()).filter(|p| p.is_file()).collect();
    paths.sort_by_key(|p| p.metadata().unwrap().size());

    for (i, path) in paths.iter().enumerate() {
        let start = Instant::now();
        let graph = common::io::read_metis(path)?;
        let t_read = start.elapsed();

        let start = Instant::now();
        if graph.node_count() != 0 {
            let mut bfs = Bfs::new(&graph, NodeIndex::new(0));
            while bfs.next(&graph).is_some() {}
        }
        let t_bfs = start.elapsed();

        let problem = linear::prepare(&graph);
        let start = Instant::now();
        let result = problem.compute();
        let t0 = start.elapsed();
        let md0 = result.finalize();

        let problem = linear_ref::prepare(&graph);
        let start = Instant::now();
        let result = problem.compute();
        let t1 = start.elapsed();
        let t1_internal = result.get_internal_time();
        let md1 = result.finalize();

        let problem = skeleton::prepare(&graph);
        let start = Instant::now();
        let result = problem.compute();
        let t2 = start.elapsed();
        let md2 = result.finalize();

        let problem = fracture::prepare(&graph);
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

        let fastest_time = *[t0, t1, t1_internal, t2, t3].map(|t| t.as_nanos()).iter().min().unwrap() as f64;

        println!("{i:4.} {:<30.30}  read {:9} μs {:6.2}  bfs {:9} μs {:6.2}  linear {:9} μs {:6.2}  linear-ref {:9} μs {:6.2} [{:6.4}]  skeleton {:9} μs {:6.2}  fracture {:9} μs {:6.2}",
                 path.file_name().and_then(OsStr::to_str).unwrap(),
                 t_read.as_micros(), (t_read.as_nanos() as f64 / fastest_time),
                 t_bfs.as_micros(), (t_bfs.as_nanos() as f64 / fastest_time),
                 t0.as_micros(), (t0.as_nanos() as f64 / fastest_time),
                 t1.as_micros(), (t1.as_nanos() as f64 / fastest_time),
                 (t1.as_nanos() as f64) / (t1_internal.as_nanos() as f64),
                 t2.as_micros(), (t2.as_nanos() as f64 / fastest_time),
                 t3.as_micros(), (t3.as_nanos() as f64 / fastest_time),
        );
    }
    Ok(())
}
