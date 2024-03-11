from argparse import ArgumentParser

import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
from typing import List


def combine_csv_lines(header: str, paths: List[Path]) -> pd.DataFrame:
    text = '\n'.join([header] + [path.open().read().strip() for path in paths])
    with tempfile.NamedTemporaryFile() as combined_path:
        with Path(combined_path.name).open("w") as file:
            file.write(text)
        return pd.read_csv(combined_path.name)


def main():
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    graph_stats_path = data_dir / "03-graph-stats"
    md_tree_stats_path = data_dir / "06-md-tree-stats"
    algo_runs_path = data_dir / "04-algo-runs"

    for path in (data_dir, graph_stats_path, md_tree_stats_path, algo_runs_path):
        assert path.exists()
        assert path.is_dir()

    print(f"working on data dir: {data_dir}")

    header = ("name,timeout,n,m,density,deg_min,deg_avg,deg_max,deg_var,deg_coeff_of_var,deg_heterogeneity,num_cc,"
              "max_cc_n,max_cc_m,max_cc_density,max_cc_radius,max_cc_diameter,max_cc_avg_dist,"
              "average_clustering_estimate,num_triangles")
    paths = list(graph_stats_path.glob("*.graph.stats"))
    graph_stats = combine_csv_lines(header, paths)

    header = ("name,root_kind,num_nodes,num_inner,num_leaves,num_prime,num_series,num_parallel,layer_sizes,"
              "num_children,children_sizes_quantiles,children_sizes_min,children_sizes_avg,children_sizes_max,"
              "children_sizes_var,children_sizes_coeff_of_var,children_sizes_heterogeneity,leaf_dist_quantiles,"
              "leaf_dist_min,leaf_dist_avg,leaf_dist_max,leaf_dist_var,leaf_dist_coeff_of_var,"
              "leaf_dist_heterogeneity,inner_out_deg_quantiles,inner_out_deg_min,inner_out_deg_avg,inner_out_deg_max,"
              "inner_out_deg_var,inner_out_deg_coeff_of_var,inner_out_deg_heterogeneity")
    paths = list(md_tree_stats_path.glob("*.md.stats"))
    md_tree_stats = combine_csv_lines(header, paths)

    paths = list(algo_runs_path.glob("*/*.runstats"))
    algo_runs = pd.DataFrame([json.loads(path.open().read().strip()) for path in paths])

    print(graph_stats.shape, md_tree_stats.shape, algo_runs.shape)

    algo_runs = algo_runs.rename(columns={"input": "name"})
    md_tree_stats["name"] = md_tree_stats["name"].str.split("\\.md").str[0]

    df = algo_runs.copy()
    df = pd.merge(df, graph_stats, on="name", how="left")
    df = pd.merge(df, md_tree_stats, on="name", how="left")
    df["dataset"] = df["name"].str.split("_").str[0]
    print(df.shape)
    df.to_csv(data_dir / "experiments.csv", index=False)


if __name__ == '__main__':
    main()
