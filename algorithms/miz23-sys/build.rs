use cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .includes(["src", "extern/src/main/cpp"])
        .file("extern/src/main/cpp/ds/set/FastSet.cpp")
        .file("extern/src/main/cpp/modular/MDTree.cpp")
        .file("extern/src/main/cpp/modular/compute/assembly.cpp")
        .file("extern/src/main/cpp/modular/compute/MDComputeNode.cpp")
        .file("extern/src/main/cpp/modular/compute/MDSolver.cpp")
        .file("extern/src/main/cpp/modular/compute/misc.cpp")
        .file("extern/src/main/cpp/modular/compute/pivot.cpp")
        .file("extern/src/main/cpp/modular/compute/promotion.cpp")
        .file("extern/src/main/cpp/modular/compute/refinement.cpp")
        .file("extern/src/main/cpp/readwrite/edge_list.cpp")
        .file("extern/src/main/cpp/util/profiler.cpp")
        .file("extern/src/main/cpp/util/Random.cpp")
        .file("extern/src/main/cpp/util/util.cpp")
        .file("src/ffi.cpp")
        .warnings(false)
        .compile("miz23");
    println!("cargo:rerun-if-changed=src/ffi.cpp");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=build.rs");
}