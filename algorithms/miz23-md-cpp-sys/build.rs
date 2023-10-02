use cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .include("src")
        .file("src/ds/graph/Graph.cpp")
        .file("src/ds/set/FastSet.cpp")
        .file("src/modular/MDTree.cpp")
        .file("src/modular/compute/assembly.cpp")
        .file("src/modular/compute/MDComputeNode.cpp")
        .file("src/modular/compute/MDSolver.cpp")
        .file("src/modular/compute/misc.cpp")
        .file("src/modular/compute/pivot.cpp")
        .file("src/modular/compute/promotion.cpp")
        .file("src/modular/compute/refinement.cpp")
        .file("src/readwrite/edge_list.cpp")
        .file("src/readwrite/pace_2023.cpp")
        .file("src/util/logger.cpp")
        .file("src/util/profiler.cpp")
        .file("src/util/Random.cpp")
        .file("src/util/util.cpp")
        .file("src/ffi.cpp")
        .warnings(false)
        .compile("libmiz23.a");
}