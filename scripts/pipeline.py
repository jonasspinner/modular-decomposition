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
        creates_file="data/02-graphs/pace2023-[[name]]")
pace2023_exact_names = [f"pace2023-{name}" for name in pace2023_exact_names]
pace2023_heuristic_names = [f"pace2023-{name}" for name in pace2023_heuristic_names]

#
# graph_stats
#

run.group("graph_stats")


def graph_stat_group(group_name):
    names = [path.name for path in Path(f"data/02-graphs").glob(f"{group_name}_*")]
    run.add(f"analyze_graph_{group_name}",
            "python3 scripts/analyze.py graph --input [[input]]",
            {
                "name": names,
                "input": "data/02-graphs/[[name]]"
            },
            stdout_file=f"data/03-graph-stats/{group_name}.csv",
            header_command="python3 scripts/analyze.py graph --only-header")


graph_stat_group("pace2023-exact")
graph_stat_group("pace2023-heuristic")

#
# experiments
#

run.group("experiments")

names = list(sorted([path.name for path in Path(f"data/02-graphs").glob("*_*")]))
run.add("md",
        "cargo run --bin md --release -- "
        "--algo [[algo]] --input-type metis "
        "--input [[input]] --output [[output]]",
        {
            "algo": ["miz23-rust"],
            "name": names,
            "input": "data/02-graphs/[[name]]",
            "output": "data/05-md-trees/[[algo]]/[[name]].md"
        },
        creates_file="[[output]]")

#
# md_tree_stats
#

run.group("md_tree_stats")


def md_tree_stat_group(group_name):
    names = list(sorted([path.name for path in Path(f"data/02-graphs").glob(f"{group_name}_*")]))
    run.add(f"analyze_md_tree_{group_name}",
            "python3 scripts/analyze.py tree --input [[input]]",
            {
                "name": names,
                "input": "data/05-md-trees/miz23-rust/[[name]].md"
            },
            stdout_file=f"data/06-md-tree-stats/{group_name}.csv",
            header_command="python3 scripts/analyze.py graph --only-header")


md_tree_stat_group("pace2023-exact")
md_tree_stat_group("pace2023-heuristic")

run.run()
