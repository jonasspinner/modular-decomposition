from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from util import write_metis


def series_parallel_tree_to_graph(tree: nx.DiGraph) -> nx.Graph:
    assert nx.is_arborescence(tree)
    roots = [u for u in tree.nodes if tree.in_degree(u) == 0]
    assert len(roots) == 1
    root = roots[0]

    graph = nx.empty_graph()
    graphs_for_tree_node = {}
    for u in nx.dfs_postorder_nodes(tree, root):
        children = list(tree.successors(u))

        if tree.nodes[u]["kind"] == "leaf":
            assert len(children) == 0
            graphs_for_tree_node[u] = {u}
            graph.add_node(u)
            continue

        assert len(children) >= 2
        assert u not in graphs_for_tree_node
        graphs_for_tree_node[u] = set.union(*(graphs_for_tree_node[q] for q in children))

        if tree.nodes[u]["kind"] == "series":
            for q, r in combinations(children, 2):
                for a in graphs_for_tree_node[q]:
                    for b in graphs_for_tree_node[r]:
                        graph.add_edge(a, b)
        elif tree.nodes[u]["kind"] == "parallel":
            pass
        else:
            assert False
    return graph


def random_cograph_uni_deg(n: int, a: int, b: int, root_kind: str, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)

    nodes = list(range(n))
    tree = nx.DiGraph()
    for u in nodes:
        tree.add_node(u, kind="leaf")

    degrees = list(range(a, b + 1))
    while len(nodes) != 1:
        d = rng.choice(degrees)
        if 2 + d > len(nodes):
            d = len(nodes)
        children = rng.choice(nodes, d, replace=False)
        nodes = [node for node in nodes if node not in children]
        u = tree.number_of_nodes()
        nodes.append(u)
        tree.add_node(u, kind="inner")
        for v in children:
            tree.add_edge(u, v)

    root = tree.number_of_nodes() - 1
    if root_kind in ["series", "parallel"]:
        kind = root_kind
    elif root_kind == "random":
        kind = rng.choice(["series", "parallel"])
    else:
        assert False

    for layer in nx.bfs_layers(tree, root):
        for u in layer:
            if tree.nodes[u]["kind"] == "inner":
                tree.nodes[u]["kind"] = kind
        kind = "parallel" if kind == "series" else "series"

    graph = series_parallel_tree_to_graph(tree)

    # nx.draw(graph)
    # plt.show()

    # color_d = {"leaf": 0, "series": 1, "parallel": 2}
    # color = [color_d[tree.nodes.data("kind")[u]] for u in range(tree.number_of_nodes())]
    # nx.draw(tree, with_labels=True, node_color=color)
    # plt.show()

    return graph


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="generator", dest="generator", required=True)

    gnm_parser = subparsers.add_parser("gnm")
    gnm_parser.add_argument("n", type=int)
    gnm_parser.add_argument("m", type=int)
    gnm_parser.add_argument("--seed", type=int, required=True)
    gnm_parser.add_argument("--output", type=Path, required=True)

    nx_cograph_parser = subparsers.add_parser("nx-cograph")
    nx_cograph_parser.add_argument("n", type=int)
    nx_cograph_parser.add_argument("--seed", type=int, required=True)
    nx_cograph_parser.add_argument("--output", type=Path, required=True)

    cograph_uni_deg_parser = subparsers.add_parser("cograph-uni-deg")
    cograph_uni_deg_parser.add_argument("n", type=int)
    cograph_uni_deg_parser.add_argument("--a", type=int, default=2)
    cograph_uni_deg_parser.add_argument("--b", type=int, default=2)
    cograph_uni_deg_parser.add_argument("--root-kind", choices=["series", "parallel", "random"], default="random")
    cograph_uni_deg_parser.add_argument("--seed", type=int, required=True)
    cograph_uni_deg_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.generator == "gnm":
        n, m, seed = args.n, args.m, args.seed
        assert 0 <= n
        assert 0 <= m <= (n * (n - 1)) // 2
        assert 0 <= seed < 2 ** 32
        graph = nx.gnm_random_graph(n, m, seed, directed=False)
        with args.output.open("w") as f:
            write_metis(f, graph)
    elif args.generator == "nx-cograph":
        n, seed = args.n, args.seed
        log_n = int(np.log2(n))
        assert 0 <= n
        assert 2 ** log_n == n
        assert 0 <= seed < 2 ** 32
        graph = nx.random_cograph(log_n, seed)
        with args.output.open("w") as f:
            write_metis(f, graph)
    elif args.generator == "cograph-uni-deg":
        n, a, b, root_kind, seed = args.n, args.a, args.b, args.root_kind, args.seed
        assert 0 <= n
        assert 2 <= a <= b
        assert 0 <= seed < 2 ** 32
        graph = random_cograph_uni_deg(n, a, b, root_kind, seed)
        with args.output.open("w") as f:
            write_metis(f, graph)
    else:
        assert False


if __name__ == "__main__":
    main()
