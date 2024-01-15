from pathlib import Path
import run
import multiprocessing

run.use_cores(multiprocessing.cpu_count() - 1)

#
# build
#

run.group("build")

run.add("build_md", "cargo build --bin md --release", {})
run.add("build_convert", "cargo build --bin convert --release", {})

#
# download
#

run.group("download")

run.add("download_pace2023", "bash scripts/data/download_pace2023.sh", {})
run.add("download_girg_deg_scaling", "bash scripts/data/download_girg_deg_scaling.sh", {})

#
# preprocessing
#

run.group("preprocessing")

pace2023_exact_names = [f"exact_{i:03}" for i in range(1, 201)]
pace2023_heuristic_names = [f"heuristic_{i:03}" for i in range(1, 201)]
run.add("convert_pace2023",
        "cargo run --bin convert --release -- "
        "--input-type pace2023 --output-type metis "
        "--input [[input]] --output [[output]]",
        {
            "name": pace2023_exact_names + pace2023_heuristic_names,
            "input": "data/01-raw/pace2023/[[name]].gr",
            "output": "data/02-graphs/pace2023-[[name]]"
        },
        creates_file="[[output]]")
pace2023_exact_names = [f"pace2023-{name}" for name in pace2023_exact_names]
pace2023_heuristic_names = [f"pace2023-{name}" for name in pace2023_heuristic_names]

girg_deg_scaling_names = [path.name[17:] for path in
                          Path("data/01-raw/girg_deg_scaling/edge_lists_girg_deg_scaling").glob("girg_deg_scaling_*")]
run.add("convert_girg_deg_scaling",
        "cargo run --bin convert --release -- "
        "--input-type edge-list --output-type metis "
        "--input [[input]] --output [[output]]",
        {
            "name": girg_deg_scaling_names,
            "input": "data/01-raw/girg_deg_scaling/edge_lists_girg_deg_scaling/girg_deg_scaling_[[name]]",
            "output": "data/02-graphs/girg_deg_scaling-[[name]]"
        },
        creates_file="[[output]]")

#
# generate
#

run.group("generate")

run.add("generate_gnm",
        "python3 scripts/generate.py gnm [[n]] [[m]] --seed [[seed]] --output [[output]]",
        {
            "n": [10000],
            "m": list(range(5000, 50000 + 1, 5000)),
            "seed": list(range(10)),
            "name": "gnm_n=[[n]]-m=[[m]]-s=[[seed]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_nx-cograph",
        "python3 scripts/generate.py nx-cograph [[n]] --seed [[seed]] --output [[output]]",
        {
            "n": [2 ** 8, 2 ** 10, 2 ** 12],
            "seed": list(range(10)),
            "name": "nx-cograph_n=[[n]]-s=[[seed]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_cograph-uni-deg",
        "python3 scripts/generate.py cograph-uni-deg "
        "[[n]] --a [[a]] --b [[b]] --root-kind=[[root_kind]] --seed [[seed]] --output [[output]]",
        {
            "n": [2 ** 8, 2 ** 10, 2 ** 12],
            "a": 2,
            "b": 8,
            "root_kind": ["series", "parallel"],
            "seed": list(range(10)),
            "name": "cograph-uni-deg_n=[[n]]-a=[[a]]-b=[[b]]-r=[[root_kind]]-s=[[seed]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

#
# plot_graphs
#

run.group("plot_graphs")
run.add("plot_graphs",
        "python3 scripts/plot.py --input [[input]] --output [[output]] --n-max 1000",
        {
            "name": [path.name for path in Path("data/02-graphs").glob("*_*")],
            "input": "data/02-graphs/[[name]]",
            "output": "data/10-graph-plots/[[name]].png"
        },
        creates_file="[[output]]")

#
# graph_stats
#

run.group("graph_stats")

names = [path.name for path in sorted(Path(f"data/02-graphs").glob(f"*_*"), key=lambda path: path.stat().st_size)]
run.add(f"analyze_graphs",
        "python3 scripts/analyze.py graph --input [[input]] --output [[output]]",
        {
            "name": names,
            "input": "data/02-graphs/[[name]]",
            "output": "data/03-graph-stats/[[name]].graph.stats"
        },
        creates_file="[[output]]")

#
# experiments
#

run.group("experiments")

names = [path.name for path in sorted(Path(f"data/02-graphs").glob(f"*_*"), key=lambda path: path.stat().st_size)]
algos = ["miz23-rust", "miz23-cpp", "ms00", "kar19-rust"]
for algo in algos:
    (Path("data/02-graphs") / algo).mkdir(exist_ok=True, parents=True)
    (Path("data/04-algo-runs") / algo).mkdir(exist_ok=True, parents=True)
run.add("md",
        "cargo run --bin md --release -- "
        "--algo [[algo]] --input-type metis "
        "--input [[input]] --output [[output]] --stats [[stats]]",
        {
            "algo": algos,
            "name": names,
            "repetition": list(range(1)),
            "input": "data/02-graphs/[[name]]",
            "output": "data/05-md-trees/[[algo]]/[[name]].md",
            "stats": "data/04-algo-runs/[[algo]]/[[name]]_rep=[[repetition]].runstats",
        },
        creates_file="[[output]]")

#
# md_tree_stats
#

run.group("md_tree_stats")

names = [path.name for path in sorted(Path(f"data/02-graphs").glob(f"*_*"), key=lambda path: path.stat().st_size)]
run.add(f"analyze_md_trees",
        "python3 scripts/analyze.py tree --input [[input]] --output [[output]]",
        {
            "name": names,
            "input": "data/05-md-trees/miz23-rust/[[name]].md",
            "output": "data/06-md-tree-stats/[[name]].md.stats"
        },
        creates_file="[[output]]")

run.run()
