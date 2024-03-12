use modular_decomposition::{modular_decomposition, ModuleKind};
use petgraph::dot::Config::EdgeNoLabel;
use petgraph::dot::Dot;
use petgraph::visit::{Dfs, GraphBase, GraphProp, IntoNeighbors, NodeCompactIndexable, NodeCount, NodeIndexable};
use petgraph::{Incoming, Undirected};

struct Graph(Vec<Vec<usize>>);

impl Graph {
    fn from_edges(edges: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut adj = vec![];
        for (u, v) in edges {
            assert_ne!(u, v);
            if u >= adj.len() {
                adj.resize(u + 1, vec![])
            }
            if v >= adj.len() {
                adj.resize(v + 1, vec![])
            }
            adj[u].push(v);
            adj[v].push(u);
        }
        Self(adj)
    }
}

impl GraphBase for Graph {
    type EdgeId = usize;
    type NodeId = usize;
}

impl GraphProp for Graph {
    type EdgeType = Undirected;
}

struct Neighbors<'a>(std::slice::Iter<'a, usize>);

impl<'a> Iterator for Neighbors<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().copied()
    }
}

impl<'a> IntoNeighbors for &'a Graph {
    type Neighbors = Neighbors<'a>;
    fn neighbors(self, a: Self::NodeId) -> Self::Neighbors {
        Neighbors(self.0[a].iter())
    }
}

impl NodeCount for Graph {
    fn node_count(&self) -> usize {
        self.0.len()
    }
}

impl NodeIndexable for Graph {
    fn node_bound(&self) -> usize {
        self.node_count()
    }
    fn to_index(&self, a: Self::NodeId) -> usize {
        a
    }
    fn from_index(&self, i: usize) -> Self::NodeId {
        i
    }
}

impl NodeCompactIndexable for Graph {}

fn main() {
    let graph = Graph::from_edges([(0, 1), (1, 2), (2, 3)]);
    let tree = modular_decomposition(&graph).map(|tree| tree.into_digraph()).unwrap_or_default();
    println!("{:?}", Dot::with_config(&tree, &[EdgeNoLabel]));

    let mut factorizing_permutation = Vec::new();
    let root = tree.externals(Incoming).next().unwrap();
    let mut dfs = Dfs::new(&tree, root);
    while let Some(node) = dfs.next(&tree) {
        if let ModuleKind::Node(u) = tree[node] {
            factorizing_permutation.push(u);
        }
    }
    let factorizing_permutation: Vec<_> = factorizing_permutation.iter().map(|u| u.index()).collect();
    println!("{:?}", factorizing_permutation);
}
