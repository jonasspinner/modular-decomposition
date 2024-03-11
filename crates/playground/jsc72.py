from argparse import ArgumentParser

import networkx as nx
from pathlib import Path
import itertools
from typing import List
from datetime import datetime

import sys

sys.path.insert(1, str(Path(__file__).parent.parent / 'scripts'))

import util


def find_series_parallel_modules(graph: nx.Graph) -> List[List[int]]:
    index = dict()
    modules = []
    for u, v in itertools.combinations(graph.nodes, 2):
        if set(graph[u]) - {v} == set(graph[v]) - {u}:
            if u not in index:
                index[u] = len(modules)
                modules.append({u})
            modules[index[u]].add(v)
            index[v] = index[u]
    return list(map(list, modules))


def find_minimal_strong_module(graph: nx.Graph, u: int, v: int) -> List[int]:
    W = [u, v]
    B = set(graph[u]) - {u, v}
    i = 1
    while i < len(W):
        W_G = set(graph[W[i]]) - set(W)
        W.extend(B.symmetric_difference(W_G))
        B = B.intersection(W_G)
        i += 1
    return W


def find_prime_modules(graph: nx.Graph) -> List[List[int]]:
    H = dict()
    for u, v in graph.edges:
        m = find_minimal_strong_module(graph, u, v)
        if u not in H or len(m) < len(H[u]):
            H[u] = m
        if v not in H or len(m) < len(H[v]):
            H[v] = m

    for u in graph.nodes:
        if u in H:
            h = H[u]
            for v in h:
                if v > u:
                    if v in H:
                        if len(H[v]) >= len(h):
                            del H[v]
                        else:
                            del H[u]
                            break

    return list(H.values())


def condense(graph: nx.Graph, modules: List[List[int]], representative: List[int], md: nx.DiGraph, kind: str):
    if not modules: return False
    for m in modules:
        k = kind if kind != 'series/parallel' else 'series' if graph.has_edge(m[0], m[1]) else 'parallel'
        v = max(md.nodes) + 1
        md.add_node(v, kind=k)
        for u in m:
            if md.nodes[representative[u]]['kind'] == k:
                for _, w in md.out_edges([representative[u]]):
                    md.add_edge(v, w)
                md.remove_node(representative[u])
            else:
                md.add_edge(v, representative[u])
            representative[u] = v
    graph.remove_nodes_from([u for m in modules for u in m[1:]])
    return True


def modular_decomposition(graph: nx.Graph) -> nx.DiGraph:
    assert list(graph.nodes) == list(range(graph.number_of_nodes()))
    md = nx.DiGraph()
    md.add_nodes_from(graph.nodes, kind='vertex')
    representative = [u for u in graph.nodes]

    while graph.number_of_nodes() > 1:
        modules = find_series_parallel_modules(graph)
        if condense(graph, modules, representative, md, 'series/parallel'): continue
        modules = find_prime_modules(graph)
        condense(graph, modules, representative, md, 'prime')
    return nx.convert_node_labels_to_integers(md)


def stats_line(input_name, algo, time, status) -> str:
    if time is None:
        time = "NaN"
    return f"{{\"input\": \"{input_name}\", \"algo\": \"{algo}\", \"time\": {time}, \"status\": \"{status}\"}}"


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--stats", type=Path, required=False)

    args = parser.parse_args()

    graph = util.read_metis(args.input)
    input_name = args.input.name
    algo = "jsc72-py"

    line = stats_line(input_name, algo, None, "unfinished")
    if args.stats is not None:
        with args.stats.open("w") as out:
            out.write(f"{line}\n")

    start = datetime.now()
    md = modular_decomposition(graph)
    end = datetime.now()

    if args.output is not None:
        with args.output.open("w") as out:
            util.write_md_tree_adj(out, md)

    time = (end - start).microseconds
    line = stats_line(input_name, algo, time, "finished")
    if args.stats is not None:
        with args.stats.open("w") as out:
            out.write(f"{line}\n")
    else:
        print(line)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        exit(-1)
