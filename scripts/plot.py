from argparse import ArgumentParser
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from util import read_metis


def to_rgb_tuple(color_hex: str):
    return int(color_hex[1:3], 16) / 256, int(color_hex[3:5], 16) / 256, int(color_hex[5:7], 16) / 256


blue = to_rgb_tuple("#5EA6E5")
green = to_rgb_tuple("#6EC620")
yellow = to_rgb_tuple("#EEC200")
red = to_rgb_tuple("#E62711")
violet = to_rgb_tuple("#c6468d")
purple = to_rgb_tuple("#613872")
gray = to_rgb_tuple("#666666")
cmap = LinearSegmentedColormap.from_list(
    "PaperColors",
    [(0.0, blue), (0.333333333333, green), (0.666666666667, yellow), (1.0, red)])


def plot_graph(graph_path: Path, image_path: Path, maximum_number_of_nodes: int):
    assert graph_path.is_file()

    if image_path.is_file():
        return

    graph = read_metis(graph_path)
    degrees = nx.degree(graph)
    graph = graph.subgraph(list(graph.nodes)[:maximum_number_of_nodes])

    pos = nx.spring_layout(graph)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    degrees = dict(nx.degree(graph))
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        alpha=0.5,
        ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        alpha=0.8,
        node_size=50,
        node_color=[degrees[u] for u in graph.nodes],
        cmap=cmap,
        ax=ax)

    ax.axis('off')

    fig.tight_layout()
    fig.patch.set_alpha(0.)
    ax.patch.set_alpha(0.)
    fig.savefig(image_path, bbox_inches=0, transparent=True)
    plt.close(fig)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-max", default=1000, type=int)
    args = parser.parse_args()

    plot_graph(args.input, args.output, args.n_max)


if __name__ == "__main__":
    main()
