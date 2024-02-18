from pathlib import Path
import run
import multiprocessing

run.use_cores(min(8, multiprocessing.cpu_count()) - 1)

Path("data/02-graphs").mkdir(exist_ok=True, parents=True)
algos = ["linear-ref", "linear", "skeleton", "fracture"]

#
# build
#

run.group("build")

run.add("build_md", "cargo build --bin md --release", {})
run.add("build_convert", "cargo build --bin convert --release", {})
run.add("build_check_trees", "cargo build --bin check_trees --release", {})

#
# download
#

run.group("download")

run.add("download_pace2023", "bash scripts/data/download_pace2023.sh", {})
run.add("download_girg_deg_scaling", "bash scripts/data/download_girg_deg_scaling.sh", {})
run.add("download_girg", "bash scripts/data/download_girg.sh", {})
run.add("download_real", "bash scripts/data/download_real.sh", {})

#
# convert
#

run.group("convert")

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
            "output": "data/02-graphs/girg-deg-scaling_[[name]]"
        },
        creates_file="[[output]]")

girg_names = [path.name[5:] for path in Path("data/01-raw/girg/edge_lists_girg").glob("girg_*")]
run.add("convert_girg",
        "cargo run --bin convert --release -- "
        "--input-type edge-list --output-type metis "
        "--input [[input]] --output [[output]]",
        {
            "name": girg_names,
            "input": "data/01-raw/girg/edge_lists_girg/girg_[[name]]",
            "output": "data/02-graphs/girg_[[name]]"
        },
        creates_file="[[output]]")

real_names = [path.name for path in Path("data/01-raw/real/edge_lists_real").glob("*")]
run.add("convert_real_names",
        "cargo run --bin convert --release -- "
        "--input-type edge-list --output-type metis "
        "--input [[input]] --output [[output]]",
        {
            "name": real_names,
            "input": "data/01-raw/real/edge_lists_real/[[name]]",
            "output": "data/02-graphs/real_[[name]]"
        },
        creates_file="[[output]]")

#
# generate
#

run.group("generate")

run.add("generate_gnm-log-n=14",
        "python3 scripts/generate.py gnm [[n]] [[m]] --seed [[seed]] --output [[output]]",
        {
            "n": 2 ** 14,
            "m": list(range(0, 2 ** 17 + 1, 2 ** 12)),
            "seed": list(range(10)),
            "name": "gnm_n=[[n]]-m=[[m]]-seed=[[seed]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_gnm-log-n=18",
        "python3 scripts/generate.py gnm [[n]] [[m]] --seed [[seed]] --output [[output]]",
        {
            "n": 2 ** 18,
            "m": list(reversed(range(0, 2 ** 24 + 1, 2 ** 18))),
            "seed": 0,
            "name": "gnm_n=[[n]]-m=[[m]]-seed=[[seed]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_gnm-log-m=20",
        "python3 scripts/generate.py gnm [[n]] [[m]] --seed [[seed]] --output [[output]]",
        {
            "n": list(reversed(range(2 ** 16, 2 ** 22 + 1, 2 ** 16))),
            "m": 2 ** 20,
            "seed": 0,
            "name": "gnm-fixed-m_n=[[n]]-m=[[m]]-seed=[[seed]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_gnm-m=8n",
        "python3 scripts/generate.py gnm [[n]] [[m]] --seed [[seed]] --output [[output]]",
        {
            "n": list(reversed(range(2 ** 14, 2 ** 20 + 1, 2 ** 14))),
            "m": "$((8 * [[n]]))",
            "seed": 0,
            "name": "gnm-m=8n_n=[[n]]-m=[[m]]-seed=[[seed]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")


def add_generate_cograph_uni_deg(k, n, a: int, b: int, num_repeats=10):
    run.add(f"generate_cograph-uni-deg_{k}",
            "python3 scripts/generate.py cograph-uni-deg "
            "[[n]] --a [[a]] --b [[b]] --root-kind=[[root_kind]] --seed [[seed]] --output [[output]]",
            {
                "n": n,
                "a": a,
                "b": b,
                "root_kind": ["series", "parallel"],
                "seed": list(range(num_repeats)),
                "name": "cograph-uni-deg_n=[[n]]-a=[[a]]-b=[[b]]-r=[[root_kind]]-seed=[[seed]]",
                "output": "data/02-graphs/[[name]]"
            },
            creates_file="[[output]]")


add_generate_cograph_uni_deg(1, [
    2 ** 8, 3 * 2 ** 7,
    2 ** 9, 3 * 2 ** 8,
    2 ** 10, 3 * 2 ** 9,
    2 ** 11, 3 * 2 ** 10,
    2 ** 12, 3 * 2 ** 11,
    2 ** 13], 2, 8, num_repeats=10)

add_generate_cograph_uni_deg(2, [2 ** 8, 2 ** 10, 2 ** 12], 256, 256)

add_generate_cograph_uni_deg(3, [2 ** 12], 512, 2048)

add_generate_cograph_uni_deg(4, [2 ** 13], 256, 256)

run.add("generate_path",
        "python3 scripts/generate.py path [[n]] --output [[output]]",
        {
            "n": list(range(0, 2 ** 18 + 1, 2 ** 10)),
            "name": "path_n=[[n]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_cycle",
        "python3 scripts/generate.py cycle [[n]] --output [[output]]",
        {
            "n": list(range(0, 2 ** 18 + 1, 2 ** 10)),
            "name": "cycle_n=[[n]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_empty",
        "python3 scripts/generate.py empty [[n]] --output [[output]]",
        {
            "n": list(range(0, 2 ** 18 + 1, 2 ** 10)),
            "name": "empty_n=[[n]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_complete",
        "python3 scripts/generate.py complete [[n]] --output [[output]]",
        {
            "n": list(range(0, 2 ** 10 + 1, 2 ** 4)),
            "name": "complete_n=[[n]]",
            "output": "data/02-graphs/[[name]]"
        },
        creates_file="[[output]]")

run.add("generate_one-edge",
        "python3 scripts/generate.py gnm [[n]] 1 --seed [[seed]] --output [[output]]",
        {
            "n": list(range(2 ** 4, 2 ** 10 + 1, 2 ** 4)),
            "seed": 0,
            "name": "one-edge_n=[[n]]-seed=[[seed]]",
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
run.add(f"graph_stats",
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

names = [path.name for path in sorted(Path(f"data/02-graphs").glob(f"*_*"), key=lambda path: -path.stat().st_size)]
for algo in algos:
    (Path("data/04-algo-runs") / algo).mkdir(exist_ok=True, parents=True)
    (Path("data/05-md-trees") / algo).mkdir(exist_ok=True, parents=True)
run.add("md",
        "timeout 10m cargo run --bin md --release -- "
        "--algo [[algo]] --input-type metis "
        "--input [[input]] --output [[output]] --stats [[stats]]",
        {
            "algo": algos,
            "name": names,
            "repetition": list(range(3)),
            "input": "data/02-graphs/[[name]]",
            "output": "data/05-md-trees/[[algo]]/[[name]].md",
            "stats": "data/04-algo-runs/[[algo]]/[[name]]_rep=[[repetition]].runstats",
        },
        creates_file="[[stats]]")

#
# md_tree_stats
#

run.group("md_tree_stats")

names = [path.name for path in sorted(Path(f"data/02-graphs").glob(f"*_*"), key=lambda path: path.stat().st_size)]
run.add(f"md_tree_stats",
        "python3 scripts/analyze.py tree --input [[input]] --output [[output]]",
        {
            "name": names,
            "input": f"data/05-md-trees/{algos[0]}/[[name]].md",
            "output": "data/06-md-tree-stats/[[name]].md.stats"
        },
        creates_file="[[output]]")

#
# check
#

run.group("check")

names = [path.name for path in sorted(Path(f"data/02-graphs").glob(f"*_*"), key=lambda path: path.stat().st_size)]
arguments = dict([])
run.add(f"check",
        "cargo run --bin check_trees --release -- [[a]] [[b]] [[c]] [[d]]",
        {
            "name": names,
            **dict([(k, f"data/05-md-trees/{algo}/[[name]].md") for k, algo in zip("abcd", algos)])
        })

#
# compress
#

run.group("compress")


def compress_folder(name):
    prefix = name.split('-')[0]
    run.add(f"compress_{prefix}",
            "cd [[input]] && zip -r ../../[[output]] *",
            {
                "name": name,
                "input": "data/[[name]]",
                "output": "[[name]].zip"
            },
            creates_file="[[output]]")


compress_folder("03-graph-stats")
compress_folder("04-algo-runs")
compress_folder("05-md-trees")
compress_folder("06-md-tree-stats")
compress_folder("10-graph-plots")

run.run()
