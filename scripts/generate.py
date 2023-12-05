from argparse import ArgumentParser
from pathlib import Path

import networkx as nx

from util import write_metis


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="generator", dest="generator", required=True)
    gnm_parser = subparsers.add_parser("gnm")

    gnm_parser.add_argument("n", type=int)
    gnm_parser.add_argument("m", type=int)
    gnm_parser.add_argument("--seed", type=int, required=True)
    gnm_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assert 0 <= args.n
    assert 0 <= args.m <= (args.n * (args.n - 1)) // 2

    if args.generator == "gnm":
        graph = nx.gnm_random_graph(args.n, args.m, args.seed, directed=False)
        with args.output.open("w") as f:
            write_metis(f, graph)
    else:
        assert False


if __name__ == "__main__":
    main()
