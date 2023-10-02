use petgraph::{Direction};
use petgraph::graph::IndexType;
use petgraph::prelude::{DiGraph, NodeIndex, UnGraph};
use petgraph::visit::Dfs;
use common::modular_decomposition::MDNodeKind;
use miz23_md_cpp;


struct MDTree<Ix: IndexType> {
    md_tree: DiGraph<MDNodeKind, (), Ix>,
}

impl<Ix: IndexType> MDTree<Ix> {
    fn new<N, E>(graph: &UnGraph<N, E, Ix>) -> Self {
        Self { md_tree: miz23_md_cpp::modular_decomposition(&graph) }
    }

    fn ancestors(&self, mut x: NodeIndex<Ix>) -> Vec<NodeIndex<Ix>> {
        let mut result = vec![x];
        while let Some(parent) = self.md_tree.neighbors_directed(x, Direction::Incoming).next() {
            result.push(parent);
            x = parent;
        }
        result.reverse();
        result
    }

    fn last_common_index<T: Eq>(a: &[T], b: &[T]) -> Option<usize> {
        let mut i = 0;
        while i < a.len() && i < b.len() {
            if a[i] == b[i] {
                i += 1;
            } else {
                break;
            }
        }
        if i > 0 {
            Some(i - 1)
        } else {
            None
        }
    }

    fn minimal_strong_module(&self, x: NodeIndex<Ix>, y: NodeIndex<Ix>) -> NodeIndex<Ix> {
        // the minimal strong module containing two vertices x and y is denoted by m(x, y),

        let a = self.md_tree.node_indices().find(|i| self.md_tree[*i] == MDNodeKind::Vertex(x.index())).unwrap();
        let b = self.md_tree.node_indices().find(|i| self.md_tree[*i] == MDNodeKind::Vertex(y.index())).unwrap();

        if a == b { return a; }

        let mut a_ancestors = self.ancestors(a);
        let mut b_ancestors = self.ancestors(b);

        let last = Self::last_common_index(&a_ancestors, &b_ancestors).unwrap();
        assert_eq!(a_ancestors[last], b_ancestors[last]);
        assert!(!matches!(self.md_tree[a_ancestors[last]], MDNodeKind::Vertex(_)));
        a_ancestors[last]
    }

    fn vertices(&self, x: NodeIndex<Ix>) -> Vec<NodeIndex<Ix>> {
        match self.md_tree[x] {
            MDNodeKind::Vertex(u) => { return vec![NodeIndex::new(u)]}
            _ => {}
        }
        let mut dfs = Dfs::new(&self.md_tree, x);
        let mut module = vec![];
        while let Some(c) = dfs.next(&self.md_tree) {
            match self.md_tree[c] {
                MDNodeKind::Vertex(u) => { module.push(NodeIndex::new(u)) }
                _ => {}
            }
        }
        module.sort();
        module
    }

    fn maximal_strong_module(&self, x: NodeIndex<Ix>, y: NodeIndex<Ix>) -> NodeIndex<Ix> {
        // the maximal strong module containing x but not y, for any two different vertices x, y of G, is denoted by M(x, y).

        assert_ne!(x, y);

        let a = self.md_tree.node_indices().find(|i| self.md_tree[*i] == MDNodeKind::Vertex(x.index())).unwrap();
        let b = self.md_tree.node_indices().find(|i| self.md_tree[*i] == MDNodeKind::Vertex(y.index())).unwrap();


        let mut a_ancestors = self.ancestors(a);
        let mut b_ancestors = self.ancestors(b);

        let last = Self::last_common_index(&a_ancestors, &b_ancestors).unwrap();
        assert_eq!(a_ancestors[last], b_ancestors[last]);
        assert!(!matches!(self.md_tree[a_ancestors[last]], MDNodeKind::Vertex(_)));
        a_ancestors[last + 1]
    }

    fn v_modular_partition(v: NodeIndex<Ix>) -> Vec<Vec<NodeIndex<Ix>>> {
        // M(G, v) = {v} u { M | M is a maximal module not containing v }
        todo!()
    }

    fn spine(v: NodeIndex<Ix>) -> DiGraph<MDNodeKind, (), Ix> {
        // spine(G, v) = MD(G/M(G, v))

        //DiGraph::<MDNodeKind, (), Ix>::new()
        todo!()
    }
}


mod test {
    use petgraph::visit::NodeCount;
    use common::instances::ted08_test0;
    use super::*;

    #[test]
    fn print_degrees() {
        let graph = ted08_test0();

        let mut degrees = vec![0; graph.node_count()];
        for u in graph.node_indices() {
            let degree = graph.neighbors(u).count();
            degrees[degree] += 1;
        }

        println!("{degrees:?}");

        for (d, &c) in degrees.iter().enumerate() {
            if c != 0 {
                println!("{d}: {c}");
            }
        }
    }

    #[test]
    fn minimal_strong_module_test() {
        let graph = ted08_test0();
        let md_tree = MDTree::new(&graph);

        let m = |x, y| {
            let x = NodeIndex::new(x);
            let y = NodeIndex::new(y);
            md_tree.vertices(md_tree.minimal_strong_module(x, y)).iter().map(|i| i.index()).collect::<Vec<_>>()
        };

        // { 0 ( 1 { ( 2 ( 3 4 )) ( 5 6 7 ) 8 17 } 9 ( 10 11 ))( 12 ( 13 14 )) 15 16 }
        // (P(0)(U(1)(P(U(2)(J(3)(4)))(J(5)(6)(7))(8)(17))(9)(J(10)(11)))(J(12)(U(13)(14)))(15)(16))
        //  { 0  ( 1  { ( 2  ( 3 4  )) ( 5  6  7 ) 8  17 } 9  ( 10  11 )) ( 12  ( 13  14 )) 15  16 }
        assert_eq!(m(3, 4), [3, 4]);
        assert_eq!(m(2, 3), [2, 3, 4]);
        assert_eq!(m(2, 4), [2, 3, 4]);
        assert_eq!(m(5, 6), [5, 6, 7]);
        assert_eq!(m(5, 7), [5, 6, 7]);
        assert_eq!(m(6, 7), [5, 6, 7]);
        assert_eq!(m(10, 11), [10, 11]);
        assert_eq!(m(12, 13), [12, 13, 14]);
        assert_eq!(m(12, 14), [12, 13, 14]);
        assert_eq!(m(13, 14), [13, 14]);
        assert_eq!(m(2, 5), [2, 3, 4, 5, 6, 7, 8, 17]);
        assert_eq!(m(2, 8), [2, 3, 4, 5, 6, 7, 8, 17]);
        assert_eq!(m(3, 5), [2, 3, 4, 5, 6, 7, 8, 17]);
        assert_eq!(m(3, 8), [2, 3, 4, 5, 6, 7, 8, 17]);
        assert_eq!(m(8, 17), [2, 3, 4, 5, 6, 7, 8, 17]);
    }


    #[test]
    fn maximal_strong_module_test() {
        let graph = ted08_test0();
        let md_tree = MDTree::new(&graph);

        #[allow(non_snake_case)]
        let M = |x, y| {
            let x = NodeIndex::new(x);
            let y = NodeIndex::new(y);
            md_tree.vertices(md_tree.maximal_strong_module(x, y)).iter().map(|i| i.index()).collect::<Vec<_>>()
        };

        // { 0 ( 1 { ( 2 ( 3 4 )) ( 5 6 7 ) 8 17 } 9 ( 10 11 ))( 12 ( 13 14 )) 15 16 }
        // (P(0)(U(1)(P(U(2)(J(3)(4)))(J(5)(6)(7))(8)(17))(9)(J(10)(11)))(J(12)(U(13)(14)))(15)(16))
        //  { 0  ( 1  { ( 2  ( 3 4  )) ( 5  6  7 ) 8  17 } 9  ( 10  11 )) ( 12  ( 13  14 )) 15  16 }
        assert_eq!(M(3, 4), [3]);
        assert_eq!(M(2, 3), [2]);
        assert_eq!(M(2, 4), [2]);
        assert_eq!(M(5, 6), [5]);
        assert_eq!(M(5, 7), [5]);
        assert_eq!(M(6, 7), [6]);
        assert_eq!(M(5, 9), [2, 3, 4, 5, 6, 7, 8, 17]);
        assert_eq!(M(10, 11), [10]);
        assert_eq!(M(12, 13), [12]);
        assert_eq!(M(12, 14), [12]);
        assert_eq!(M(10, 9), [10, 11]);
        assert_eq!(M(11, 9), [10, 11]);
        assert_eq!(M(13, 14), [13]);
        assert_eq!(M(13, 12), [13, 14]);
        assert_eq!(M(14, 12), [13, 14]);
    }
}