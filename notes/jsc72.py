from collections import defaultdict
from typing import List, Dict, TypeVar, Tuple, Iterable, Any
import networkx as nx
import matplotlib.pyplot as plt

"""
Definitions used by [1].

Def.: condensible subgraph
    A subgraph H is condensible if for all u, v in V(H), N(u) \ V(H) = N(v) \ V(H)
    Note: H is a condensible subgraph <=> H is a module
    Note: When they talk about minimal condensible subraphs, they require |V(H)| >= 2

References
----------
[1] L. O. James, R. G. Stanton, and D. Cowan, “Graph decomposition for undirected graphs,” Utilitas Mathematica, Jan. 1972, [Online]. Available: https://www.researchgate.net/publication/268245143_Graph_decomposition_for_undirected_graphs
"""

V = TypeVar('V')


def find_minimal_modules_from_set(G: nx.Graph, nodes: List[V]) -> List[V]:
    """
    Compute G*(v) to be the smallest condensible subgraph in G which contains the vertices of v.
    """

    """
    Algorithm 2.2
    Step 1:  Set W to v
             set i to 2
             set B to W_G(W, v_1).
    Step 2:  If i > the number of elements in the vector W, then go to Step 4.
    Step 3.  Add to W those vertices that are in one but not both of B and W_G(W, w_i). Let B become the set of vertices
             in both of those sets. Finally, increment i by 1, and go to Step 2.
    """

    """
    W := {v_1, ..., v_n}
    i := 2
    B := N(v_1) \ W
    while i <= |W|
        W_G(W, w_i) := N(w_i) \ W
        W := W u (B △ W_G(W, w_i))
        B :=     (B n W_G(W, w_i))
    return W
    """
    if len(nodes) == 1:
        return list(nodes)
    W = list(nodes)
    i = 1
    B = set(G[W[0]]) - set(W)
    while i < len(W):
        W_G = set(G[W[i]]) - set(W)
        W.extend(list(B.symmetric_difference(W_G)))
        B = B.intersection(W_G)
        i += 1
    return W


def find_prime_modules(G: nx.Graph) -> List[List[V]]:
    """
    """
    """
    Step 1.  For each edge g of G, calculate the minimal condensible subgraph containing that edge. With each vertex i
             of G, associate H_i, the minimal condensible subgraph defined on an edge incident with i. This is the
             minimal condensible subgraph defined on an edge incident with i. This is the minimal condensible subgraph
             containing i. With i, associate M_i = |V(H_i)|.
    Step 2.  For each i go through the vertices h_j \in H_i for j > i. If M_j >= M_i, delete H_j from consideration.
             If M_j < M_i, delete H_i.
    Step 3.  At this point, the H's represent the set of condensible subgraphs.
    """

    """
    H(u) := argmin {|V(H)| : H \in { G*({u, v}) : v \in N(u) }}
    for u in V(G):
        for v in H(u):
            if v > u:
                if |H(v)| >= |H(u)|: del H(v)
                else:                del H(u)
    return H
    """

    assert len(G.edges) > 0

    H: Dict[V, List[V]] = defaultdict(list)
    for u, v in G.edges:
        minimal_edge_module = find_minimal_modules_from_set(G, [u, v])
        if len(H[u]) == 0 or len(minimal_edge_module) < len(H[u]):
            H[u] = minimal_edge_module
        if len(H[v]) == 0 or len(minimal_edge_module) < len(H[v]):
            H[v] = minimal_edge_module

    idx = {u: i for u, i in zip(G.nodes, range(len(G.nodes)))}

    for u in G.nodes:
        if u not in H: continue
        for v in H[u]:
            # if u == v: continue
            if len(H[v]) == 0: continue
            # if not v > u: continue
            if not (idx[v] > idx[u]): continue
            if len(H[v]) >= len(H[u]):
                H[v] = []
            else:
                H[u] = []
                break

    assert all(set(Hu).isdisjoint(set(Hv)) for i, Hu in enumerate(H.values()) for j, Hv in enumerate(H.values()) if i != j)

    return [H[v] for v in G.nodes if len(H[v]) > 0]


def find_non_prime_modules(G: nx.Graph):
    """"""
    """
    Algorithm 3.5
    Step 1.  Set I to 0.
    Step 2.  Increment I. If I equals |V| we are finished. Otherwise, set J to I+1.
    Step 3.  Increment J. If J > |V|, go to Step 2.
    Step 4.  If the Ith and Jth lines in the incidence matrix of G are not identical (except possibly in their Ith and
             Jth positions), then go to Step 3.
             Otherwise, add vertex J to the clique (or independent set) to which I belongs. If I did not previously
             belong to a clique (or independent set) form a new one.
             It is not necessary to record whether it is a clique or independent set.
    """
    index = dict()
    max_sets = []
    for u in G.nodes:
        for v in G.nodes:
            if u >= v: continue
            if set(G[u]) - {v} == set(G[v]) - {u}:
                if u not in index:
                    index[u] = len(max_sets)
                    max_sets.append([u])
                if v not in max_sets[index[u]]:
                    max_sets[index[u]].append(v)
                index[v] = index[u]
    return max_sets


def condense(G, sets: List[Iterable[V]]) -> nx.Graph:
    c = set()
    for X in sets:
        c |= set(X)

    def node_data(B : Iterable[V]) -> Dict[str, Any]:
        labels = G.nodes.data('label')
        return {'label': "".join((labels[x] if labels[x] is not None else str(x)) for x in sorted(B))}

    Q = nx.quotient_graph(G, sets + [[u] for u in G.nodes if u not in c], node_data=node_data, relabel=True)

    return Q

    def f(u):
        return "".join(str(x) for x in sorted(list(u)))

    G_new = nx.Graph()
    G_new.add_nodes_from((f(u), {'original_nodes': u, 'original_graph': G.subgraph(u)}) for u in Q.nodes)
    G_new.add_edges_from([(f(u), f(v)) for u, v in Q.edges])

    print(G.nodes.data(True))

    return G_new


def algorithm_3_4(G: nx.Graph, plot_graphs: bool = False) -> List[List[V]]:
    """
    Algorithm 3.4
    Step 1. Determine the set of maximal condensible cliques and independent sets in G
    Step 2. If we only have the vertex graph, we are finished.
            If Step 1 produced no subgraphs, go to Step 3.
            If Step 1 produced some subgraphs, condense them and go to Step 1.
    Step 3. Determine the set of minimal condensible subgraphs in G
    Step 4. If no condensible subgraphs were obtained, we are finished.
            Otherwise, condense these subgraphs and go to Step 1.
    """

    G = condense(G, [[u] for u in G.nodes])

    modules = []

    def plot(H: nx.Graph):
        if plot_graphs:
            nx.draw(H, with_labels=True, node_color='lightblue', labels=dict(H.nodes.data('label')))
            plt.show()

    plot(G)

    while len(G.nodes) > 1:
        non_prime_modules = find_non_prime_modules(G)
        print("Step 1.", [{G.nodes[u]['label'] for u in X} for X in non_prime_modules])
        if len(non_prime_modules) > 0:
            G = condense(G, non_prime_modules)
            modules.extend(non_prime_modules)
            plot(G)
            continue

        prime_modules = find_prime_modules(G)
        print("Step 3.", [{G.nodes[u]['label'] for u in X} for X in prime_modules])
        if len(prime_modules) > 0:
            G = condense(G, prime_modules)
            modules.extend(prime_modules)
            plot(G)

    return modules


def main():
    G = nx.Graph({
        1: [2, 3, 8],
        2: [1, 3, 8],
        3: [1, 2, 4, 5, 6],
        4: [3, 7, 8],
        5: [3, 7, 8],
        6: [3, 7, 8],
        7: [4, 5, 6, 8],
        8: [1, 2, 4, 6, 7],
    })

    G_ = nx.quotient_graph(G, [[1, 2], [4, 5, 6], [3], [7], [8]])

    G2 = nx.Graph({"A": [3, 8], "B": [3, 7, 8], 3: ["A", "B", 8], 7: ["B", 8], 8: ["A", "B", 3, 7]})

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

    d = {u: i for i, u in enumerate(G.nodes())}
    print("\n".join(f"{d[u]} {d[v]}" for u, v in G.edges()));

    modules = algorithm_3_4(G, plot_graphs=True)
    print(f"modules: {modules}")
    # {1 (2 {(3 (4 5)) (6 7 8) 9 18} 10 (11 12)) (13 (14 15)) 16 17}
    # {a (b {(c (d e)) (f g h) i x} j (k l)) (m (n p)) q r}

    # (P(0)(U(1)(P(U(2)(J(3)(4)))(J(5)(6)(7))(8)(17))(9)(J(10)(11)))(J(12)(U(13)(14)))(15)(16))
    # (P(1)(U(2)(P(U(3)(J(4)(5)))(J(6)(7)(8))(9)(18))(10)(J(11)(12)))(J(13)(U(14)(15)))(16)(17))
    # {  1 (  2 { (  3 (  4  5 ))(  6  7  8 ) 9  18 } 10 (  11  12 ))(  13 (  14  15 )) 16  17 }

    """
    (3) Prime [['q', 'r', 'mnp', 'a', 'bjklcdefghix']]
        'q'
        'r'
        'a'
        (1) Parallel [['bjkl', 'cdefghix']]
            (1) Parallel [['bj', 'kl']]
                (1) Parallel [['b', 'j']]
                (1) Series [['k', 'l']]
            (3) Prime [['i', 'fgh', 'cde', 'x']]
                'i'
                'x'
                (1) Series [['f', 'h', 'g']]
                (1) Parallel [['c', 'de']]
                    'c'
                    (1) Series [['d', 'e']]
        (1) Series [['m', 'np']]
            'm'
            (1) Parallel [['n', 'p']]
    """


if __name__ == '__main__':
    main()
