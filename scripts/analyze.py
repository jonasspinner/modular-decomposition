import re
from collections import Counter
from pathlib import Path
import networkx as nx
import numpy as np
from argparse import ArgumentParser

from typing import Optional

from util import read_metis, read_md_tree_adj, run_with_timeout


def analyze_graph(input_path: Optional[Path], only_header: bool, timeout: int) -> str:
    line = "{name},{timeout}," \
           "{n},{m},{density}," \
           "{deg_min},{deg_avg},{deg_max},{deg_var},{deg_coeff_of_var},{deg_heterogeneity}," \
           "{num_cc},{max_cc_n},{max_cc_m},{max_cc_density},{max_cc_radius},{max_cc_diameter},{max_cc_avg_dist}," \
           "{average_clustering_estimate},{num_triangles}"
    if only_header:
        return line.replace("{", "").replace("}", "")

    assert input_path.exists()

    graph = read_metis(input_path)
    name = input_path.name

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = nx.density(graph)
    d = nx.degree(graph)
    degrees = [d[u] for u in range(n)]
    deg_min = np.min(degrees) if degrees else None
    deg_avg = np.mean(degrees) if degrees else None
    deg_max = np.max(degrees) if degrees else None
    deg_var = np.var(degrees) if degrees else None
    deg_coeff_of_var = np.sqrt(deg_var) / deg_avg if degrees and deg_avg > 0.0 else None
    deg_heterogeneity = None if deg_coeff_of_var is None else np.log10(
        deg_coeff_of_var) if deg_coeff_of_var > 0.0 else -np.inf

    ccs = list(nx.connected_components(graph))
    num_cc = len(ccs)
    max_cc = graph.subgraph(max(ccs, key=len)) if ccs else None
    max_cc_n = max_cc.number_of_nodes() if max_cc else None
    max_cc_m = max_cc.number_of_edges() if max_cc else None
    max_cc_density = nx.density(max_cc) if max_cc else None

    e = run_with_timeout(nx.eccentricity, (max_cc,), timeout) if max_cc else None
    if e is not None:
        e = list(e.values())
        max_cc_radius = min(e)
        max_cc_diameter = max(e)
        max_cc_avg_dist = np.mean(e)
    else:
        max_cc_radius = None
        max_cc_diameter = None
        max_cc_avg_dist = None
    average_clustering_estimate = run_with_timeout(nx.average_clustering, (graph,), timeout) if n else None

    t = run_with_timeout(nx.triangles, (graph,), timeout)
    num_triangles = int(sum(t.values())) // 3 if t is not None else None

    return line.format(**{k: v if v is not None else "" for k, v in locals().items()})


def analyze_tree(input_path: Optional[Path], only_header: bool, timeout: int) -> str:
    line = "{name},{root_kind}," \
           "{num_nodes},{num_inner},{num_leaves}," \
           "{num_prime},{num_series},{num_parallel}," \
           "{layer_sizes},{num_children}," \
           "{children_sizes_quantiles}," \
           "{children_sizes_min},{children_sizes_avg},{children_sizes_max},{children_sizes_var}," \
           "{children_sizes_coeff_of_var},{children_sizes_heterogeneity}," \
           "{leaf_dist_quantiles}," \
           "{leaf_dist_min},{leaf_dist_avg},{leaf_dist_max},{leaf_dist_var}," \
           "{leaf_dist_coeff_of_var},{leaf_dist_heterogeneity}," \
           "{inner_out_deg_quantiles}," \
           "{inner_out_deg_min},{inner_out_deg_avg},{inner_out_deg_max},{inner_out_deg_var}," \
           "{inner_out_deg_coeff_of_var},{inner_out_deg_heterogeneity}"
    if only_header:
        return line.replace("{", "").replace("}", "")

    assert input_path.exists()

    def subtree_size(tree, child) -> int:
        return sum(1 for _ in nx.dfs_preorder_nodes(tree, child))

    with input_path.open() as file:
        tree, _ = read_md_tree_adj(file)
    tree: nx.DiGraph

    name = input_path.name

    if tree.number_of_nodes() == 0:
        return name + re.sub(r"{[^,{}]*}", "", line)

    assert nx.is_arborescence(tree)

    root, = [u for u in tree.nodes() if tree.in_degree(u) == 0]
    leaves = [u for u in tree.nodes() if tree.out_degree(u) == 0]
    inner_nodes = [u for u in tree.nodes() if tree.out_degree(u) != 0]

    root_kind = tree.nodes[root]["label"]
    num_nodes = len(tree.nodes)

    inner_node_kinds = ["Prime", "Series", "Parallel"]
    inner_node_kind_counts = Counter((label for _, label in tree.nodes.data(data="label") if label in inner_node_kinds))
    num_prime = inner_node_kind_counts["Prime"]
    num_series = inner_node_kind_counts["Series"]
    num_parallel = inner_node_kind_counts["Parallel"]
    num_leaves = len(leaves)
    num_inner = num_prime + num_series + num_parallel

    _, root_children = next(nx.bfs_successors(tree, root))
    num_children = len(root_children)

    layer_sizes = "\"{}\"".format([len(layer) for layer in nx.bfs_layers(tree, root)])

    q = np.linspace(0, 1, 51)

    def quantiles(xs):
        return list(np.quantile(xs, q) if len(xs) > len(q) else sorted(xs))

    children_sizes = np.array([subtree_size(tree, child) for child in root_children])
    children_sizes_quantiles = "\"{}\"".format(quantiles(children_sizes))
    children_sizes_min = np.min(children_sizes)
    children_sizes_avg = np.mean(children_sizes)
    children_sizes_max = np.max(children_sizes)
    children_sizes_var = np.var(children_sizes)
    children_sizes_coeff_of_var = np.sqrt(children_sizes_var) / children_sizes_avg
    children_sizes_heterogeneity = np.log10(
        children_sizes_coeff_of_var) if children_sizes_coeff_of_var > 0.0 else -np.inf

    leaf_dist = nx.shortest_path_length(tree, root)
    leaf_dist = np.array([leaf_dist[u] for u in leaves])
    leaf_dist_quantiles = "\"{}\"".format(quantiles(leaf_dist))
    leaf_dist_min = np.min(leaf_dist)
    leaf_dist_avg = np.mean(leaf_dist)
    leaf_dist_max = np.max(leaf_dist)
    leaf_dist_var = np.var(leaf_dist)
    leaf_dist_coeff_of_var = np.sqrt(leaf_dist_var) / leaf_dist_avg
    leaf_dist_heterogeneity = np.log10(leaf_dist_coeff_of_var) if leaf_dist_coeff_of_var > 0.0 else -np.inf

    inner_out_deg = dict((u, len(succ)) for u, succ in nx.bfs_successors(tree, root))
    inner_out_deg = np.array([inner_out_deg[u] if u == root else inner_out_deg[u] - 1 for u in inner_nodes])
    inner_out_deg_quantiles = "\"{}\"".format(quantiles(inner_out_deg))
    inner_out_deg_min = np.min(inner_out_deg)
    inner_out_deg_avg = np.mean(inner_out_deg)
    inner_out_deg_max = np.max(inner_out_deg)
    inner_out_deg_var = np.var(inner_out_deg)
    inner_out_deg_coeff_of_var = np.sqrt(inner_out_deg_var) / inner_out_deg_avg
    inner_out_deg_heterogeneity = np.log10(inner_out_deg_coeff_of_var) if inner_out_deg_coeff_of_var > 0.0 else -np.inf

    return line.format(**{k: v if v is not None else "" for k, v in locals().items()})


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="command", dest="command", required=True)
    graph_parser = subparsers.add_parser("graph")
    tree_parser = subparsers.add_parser("tree")
    for p in (graph_parser, tree_parser):
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--only-header", action="store_true")
        group.add_argument("--input", type=Path)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--timeout", type=int, default=100)
    args = parser.parse_args()

    if args.command == "graph":
        res = analyze_graph(args.input, args.only_header, args.timeout)
    elif args.command == "tree":
        res = analyze_tree(args.input, args.only_header, args.timeout)
    else:
        assert False

    if args.output is not None:
        with args.output.open("w") as output:
            output.write(res)
    else:
        print(res)


if __name__ == "__main__":
    # pace2023_names = [f"exact_{i:03}.gr" for i in range(1, 201)] + [f"heuristic_{i:03}.gr" for i in range(1, 201)]
    # pace2023_dir = Path("../../hippodrome/instances/pace2023")
    # pace2023_paths = [pace2023_dir / name for name in pace2023_names]
    # df = create_csv(Path("pace2023_parameters.csv"), pace2023_paths[:370])
    main()
