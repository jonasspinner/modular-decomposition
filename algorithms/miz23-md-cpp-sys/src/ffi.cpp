#include <fstream>
#include <iostream>
#include <numeric>
#include <memory>
#include <cstdint>

#include "modular/MDTree.hpp"

extern "C" {
struct Edge {
    int u;
    int v;
};

enum NodeKind {
    Prime = 0,
    Series = 1,
    Parallel = 2,
    Vertex = 3,
    Removed = 4,
};

struct Node {
    NodeKind kind{NodeKind::Removed};
    int vertex{-1};
    int parent{-1};
    int vertices_begin{-1};
    int vertices_end{-1};

    Node() = default;

    Node(const modular::MDNode &node, int parent_, bool alive)
            : vertex(node.vertex), parent(parent_),
              vertices_begin(node.vertices_begin), vertices_end(node.vertices_end) {
        if (alive) {
            if (node.is_prime_node()) {
                kind = NodeKind::Prime;
            } else if (node.is_join_node()) {
                kind = NodeKind::Series;
            } else if (node.is_union_node()) {
                kind = NodeKind::Parallel;
            } else if (node.is_vertex_node()) {
                kind = NodeKind::Vertex;
            }
        } else {
            kind = NodeKind::Removed;
            vertex = -1;
            parent = -1;
        }
    }
};

struct Graph {
    ds::graph::Graph inner;
    Graph(int n) : inner(n) {}
};

struct Result {
    std::pair<modular::MDTree, float> inner;
};

int miz23_graph_new(int n, const Edge *edges, int m, Graph **out_graph) {
    try {
        auto graph = std::make_unique<Graph>(n);
        for (int i = 0; i < m; ++i) {
            graph->inner.add_edge(edges[i].u, edges[i].v);
        }
        *out_graph = graph.release();
    } catch (...) {
        return -1;
    }
    return 0;
}

void miz23_graph_delete(Graph *graph) {
    if (graph) {
        delete graph;
    }
}

int miz23_compute(const Graph *graph, Result **out_result) {
    try {
        auto result = std::make_unique<Result>();
        result->inner = modular::modular_decomposition_time(graph->inner, false);

        *out_result = result.release();
    } catch (...) {
        return -1;
    }
    return 0;
}

void miz23_result_delete(Result *result) {
    if (result) {
        delete result;
    }
}

float miz23_result_time(const Result *result) {
    return result ? result->inner.second : std::numeric_limits<float>::infinity();
}

int miz23_result_size(const Result *result) {
    return result ? result->inner.first.get_tree().capacity() : 0;
}

int miz23_result_copy_nodes(const Result *result, Node *nodes, int num_nodes,
                            int *vertices, int num_vertices) {
    try {
        const auto &md_tree = result->inner.first;
        const auto &tree = md_tree.get_tree();
        const auto root = md_tree.get_root();

        if (num_nodes != tree.capacity())
            return -1;
        if (num_vertices != tree[root].data.size())
            return -1;

        for (int i = 0; i < num_vertices; ++i) {
            vertices[i] = md_tree.get_vertex(i);
        }

        int next = root;

        while (next != -1) {
            assert(tree.is_valid(next));
            int current = next;

            const auto &md_node = tree[current];
            Node node(md_node.data, md_node.parent, md_node.is_alive());

            nodes[current] = node;

            if (md_node.first_child != -1) {
                next = md_node.first_child;
            } else if (md_node.right != -1) {
                next = md_node.right;
            } else {
                int parent = md_node.parent;
                while (true) {
                    if (parent == -1 || parent == root) {
                        next = -1;
                        break;
                    } else if (tree[parent].right != -1) {
                        next = tree[parent].right;
                        break;
                    }
                    parent = tree[parent].parent;
                }
            }
        }
    } catch (...) {
        return -1;
    }
    return 0;
}
}