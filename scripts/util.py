from multiprocessing import Pool, TimeoutError
from pathlib import Path
from typing import List, Tuple, TextIO

import networkx as nx


def read_metis(path: Path) -> nx.Graph:
    lines = path.open().readlines()
    lines = [line.strip() for line in lines]
    header, adj = lines[0], lines[1:]
    n, m = header.split()
    n = int(n)
    m = int(m)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for u, neighbors in enumerate(adj):
        edges = [(u, int(v) - 1) for v in neighbors.split()]
        graph.add_edges_from(edges)

    assert graph.number_of_nodes() == n, f"{path.name} {n} != {graph.number_of_nodes()}"
    assert graph.number_of_edges() == m
    return graph


def write_metis(out, graph: nx.Graph):
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    out.write(f"{n} {m}\n")
    for u in range(n):
        for i, v in enumerate(graph.neighbors(u)):
            if i != 0: out.write(" ")
            out.write(f"{v + 1}")
        out.write("\n")


def label_int_to_str(w: int) -> str:
    if w == 0:
        return "Prime"
    elif w == 1:
        return "Series"
    elif w == 2:
        return "Parallel"
    else:
        return f"{w - 3}"


def read_md_tree_adj(file: TextIO) -> Tuple[nx.DiGraph, List[str]]:
    comments = []
    md = nx.DiGraph()
    n, m, fmt = (None, None, None)
    for line in file:
        line = line.strip()
        if line.startswith("%"):
            comments.append(line)
            continue
        n, m, fmt = line.split()
        break

    n, m, fmt = int(n), int(m), int(fmt)
    assert n == 0 or m + 1 == n
    assert fmt == 10

    md.add_nodes_from(range(n))
    u = 0
    for line in file:
        line = line.strip()
        if line.startswith("%"):
            comments.append(line)
            continue
        w, *adj = map(int, line.strip().split())
        md.nodes[u]["label"] = label_int_to_str(w)
        for v in adj:
            md.add_edge(u, v)
        u += 1
    file.seek(0)
    return md, comments


def run_with_timeout(f, args: Tuple, timeout: int):
    with Pool() as p:
        r = p.apply_async(f, args)
        try:
            return r.get(timeout)
        except KeyboardInterrupt:
            return None
        except TimeoutError:
            return None
