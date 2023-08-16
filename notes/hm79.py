from typing import List, TypeVar, Dict, Set, Tuple

import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

V = TypeVar('V')
E = Tuple[V, V]


def edge(a: V, b: V) -> E:
    return min(a, b), max(a, b)


def gamma_classes(G: nx.Graph):
    """"""
    """
    Def.: Gamma-classes
        ab Gamma ac iff b = c or bc notin E(G)
    Def.: A Gamma-class alpha of E(G) covers a vertex x if there exists y in V(G) - {x} such that xy in alpha
    Def.: The subset of V(G) covered by a Gamma-class alpha is externally related in G. Furthermore, if G has at least
       two Gamma-classes, then one of these does not cover V(G) itself.
    """

    """
    Gamma-classes algorithm
    Let G be a non-fragile graph with V(G)= {x_1, ..., x_n}, E(G) = {e_1, ..., e_m}. C is an array of length m, whose
    elements indicate the Gamma-classes (e.g. C(e_i) = alpha iff the edge e_i is in the class alpha). P_alpha is a 'st
    containing all the edges of the Gamma-class alpha. Initially: forall i \in [1, m] C(e_i) <- i; P_i <- e_i:
    For i varying from 1 to n: compute the connected components of G_j, j \in [1, k_i].
    V(G_j) = {x_1^j, ..., x_{L_j}^j}, of the subgraph bar{G}_{N(x_i)} of bar{G}.
    For j varying from 1 to k_i:
        P_{C(x_?^? x_?)} <- U_{h=1}^{h=L_j} P_{C(x_?^? x_?)}
        forall L \in [2, L_j],C(c_L^j, x_i) <- C(x_1^j,x_?)
    """
    Alpha = int

    C: Dict[E, Alpha] = {edge(*e): i for i, e in enumerate(G.edges)}
    P: Dict[Alpha, Set[E]] = {i: {e} for e, i in C.items()}

    G_complement = nx.complement(G)

    for i, x_i in enumerate(G.nodes):
        ccs = list(nx.connected_components(G_complement.subgraph(G.adj[x_i])))
        # print(f"{x_i}\n\t{[G_j for G_j in ccs]}\n\t{list(G.adj[x_i])}\n\t{list(G_complement.subgraph(G.adj[x_i]).edges)}")
        for j, G_j in enumerate(ccs):
            V_j = list(G_j)
            P[C[edge(V_j[0], x_i)]] = set().union(*(P[C[edge(x_h, x_i)]] for x_h in V_j))
            # print(P[C[edge(V_j[0], x_i)]])
            for L in range(1, len(V_j)):
                C[edge(V_j[L], x_i)] = C[edge(V_j[0], x_i)]

    for u, v in G.edges:
        G[u][v]['alpha'] = C[edge(u, v)]

    cmap = plt.get_cmap('Dark2') # plt.cm.Accent
    alpha_colors = {alpha: color for color, alpha in enumerate({alpha for u, v, alpha in G.edges.data('alpha')})}
    edge_color = [alpha_colors[alpha] for u, v, alpha in G.edges.data('alpha')]
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color=edge_color, edge_cmap=cmap, width=2, alpha=1)
    plt.show()

    return [P[c] for c in set(C.values())]


def covered_vertices(alpha_class):
    return list(set().union(*[{u, v} for u, v in alpha_class]))

def decomposition_algorithm(G: nx.Graph):
    """"""
    """
    Def.: A graph G is fragile if G or bar{G} is disconnected.
    """
    """
    Maximal partition procedure
    (a) When G is a fragile graph, the connected components of G and bar{G} give the maximal partition P, end.
    (b) Otherwise, determine the Gamma-classes of G.
    (c) If G has at least two Gamma-classes, then construct the Graph G'.
    (d) Determine the classes of totally equivalent vertices in G': they give the elements of P, end.
    """

    ccs = list(nx.connected_components(G))
    complement_ccs = list(nx.connected_components(nx.complement(G)))
    if len(ccs) > 1:
        print(f"fragile G disconnected")
        return ccs
    if len (complement_ccs) > 1:
        print(f"fragile bar{G} disconnected")
        return complement_ccs

    P_alphas = gamma_classes(G)
    for P_alpha in P_alphas:
        print(P_alpha)
        print(f"   {covered_vertices(P_alpha)}")
        print(f"   {''.join(sorted(covered_vertices(P_alpha)))}")
    # print([P[c] for c in set(C.values())])
    G_prime = None


def main():
    G = nx.Graph({
        'r': "q",
        'n': "mqa",
        'p': "mqa",
        'm': "npqa",
        'q': "mnpra",
        'a': "edcxfghibjklmnpq",
        'b': "a",
        'j': "a",
        'k': "la",
        'l': "ka",
        'h': "fgia",
        'f': "ghia",
        'g': "fhia",
        'i': "fghedca",
        'e': "dxia",
        'd': "exia",
        'c': "xia",
        'x': "edca",
    })

    renamed = nx.quotient_graph(G, [[u] for u in sorted(G.nodes)], relabel=True)
    print(f"p tww {len(G.nodes)} {len(G.edges)}")
    print("\n".join([f"{u+1} {v+1}" for u, v in sorted(renamed.edges)]))

    gamma_classes(G)

    decomposition_algorithm(G)

    nx.draw(G, with_labels=True, node_color='lightblue')
    plt.show()
    nx.draw(nx.complement(G), with_labels=True, node_color='lightblue')
    plt.show()


if __name__ == '__main__':
    main()
