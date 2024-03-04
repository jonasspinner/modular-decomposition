use petgraph::graph::{NodeIndex, UnGraph};

pub fn empty_graph(n: usize) -> UnGraph<(), ()> {
    let mut graph = UnGraph::new_undirected();
    for _ in 0..n {
        graph.add_node(());
    }
    graph
}

pub fn complete_graph(n: usize) -> UnGraph<(), ()> {
    let mut graph = empty_graph(n);
    for i in 0..n {
        for j in i + 1..n {
            graph.add_edge(NodeIndex::new(i), NodeIndex::new(j), ());
        }
    }
    graph
}

pub fn path_graph(n: usize) -> UnGraph<(), ()> {
    let mut graph = empty_graph(n);
    for i in 1..n {
        graph.add_edge(NodeIndex::new(i - 1), NodeIndex::new(i), ());
    }
    graph
}


pub fn ted08_test0() -> UnGraph<(), ()> {
    // from test0.txt in ted+08
    //a->e,d,c,x,f,g,h,i,b,j,k,l,m,n,p,q
    //b->a
    //c->x,i,a
    //d->e,x,i,a
    //e->d,x,i,a
    //f->g,h,i,a
    //g->f,h,i,a
    //h->f,g,i,a
    //i->f,g,h,e,d,c,a
    //j->a
    //k->l,a
    //l->k,a
    //m->n,p,q,a
    //n->m,q,a
    //p->m,q,a
    //q->m,n,p,r,a
    //r->q
    //x->e,d,c,a

    // (P(0)(U(1)(P(U(2)(J(3)(4)))(J(5)(6)(7))(8)(17))(9)(J(10)(11)))(J(12)(U(13)(14)))(15)(16))
    //  { 0  ( 1  { ( 2  ( 3 4  )) ( 5  6  7 ) 8  17 } 9  ( 10  11 )) ( 12  ( 13  14 )) 15  16 }
    // [[6], [5, 7], [8], [0], [2, 3, 4], [17], [1, 9, 10, 11], [12, 13, 14], [15], [16]]
    let mut graph = UnGraph::new_undirected();
    graph.extend_with_edges([
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11),
        (0, 12), (0, 13), (0, 14), (0, 15), (0, 17),
        (2, 8), (2, 17),
        (3, 4), (3, 8), (3, 17),
        (4, 8), (4, 17),
        (5, 6), (5, 7), (5, 8),
        (6, 7), (6, 8), (7, 8), (10, 11), (12, 13), (12, 14), (12, 15), (13, 15), (14, 15), (15, 16)]);
    graph
}

pub fn ms00_fig1() -> UnGraph<(), ()> {
    UnGraph::from_edges([(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6), (5, 7), (6, 7), (8, 11), (9, 11), (10, 11)])
}


pub fn graph_001_10_19() -> UnGraph<(), ()> {
    // {(4 3) 2 0 6 1 5 (9 8 7)}
    UnGraph::from_edges([
        (0, 2), (0, 3), (0, 4),
        (1, 2),
        (2, 3), (2, 4), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 5), (4, 6),
        (5, 6),
        (6, 7), (6, 8), (6, 9),
        (7, 8), (7, 9),
        (8, 9)
    ])
}

pub fn pace2023_exact_024() -> UnGraph<(), ()> {
    UnGraph::from_edges([
        (0, 1), (1, 2), (1, 3), (4, 5), (4, 6), (4, 7), (3, 4), (5, 6), (5, 8), (5, 9), (5, 10),
        (5, 11), (5, 12), (3, 5), (5, 13), (6, 8), (3, 6), (6, 14), (6, 15), (6, 16), (8, 10),
        (8, 15), (8, 13), (8, 16), (17, 18), (17, 19), (0, 20), (3, 20), (9, 11), (11, 12), (9, 21),
        (0, 21), (21, 22), (13, 23), (24, 25), (26, 27), (15, 28), (28, 29), (0, 30), (3, 30),
        (9, 31), (10, 31), (26, 31), (12, 31), (32, 33), (13, 33), (34, 35), (14, 35), (9, 10),
        (9, 26), (9, 12), (9, 29), (9, 16), (10, 26), (10, 12), (10, 15), (10, 13), (36, 37),
        (16, 36), (12, 26), (3, 26), (15, 26), (26, 32), (0, 26), (0, 2), (2, 3), (2, 37), (2, 16),
        (12, 29), (12, 15), (12, 32), (0, 12), (12, 16), (0, 3), (18, 19), (16, 37), (32, 35),
        (32, 38), (15, 29), (29, 32), (16, 29), (15, 32), (0, 15), (15, 16), (13, 32), (0, 32),
        (0, 22), (0, 13), (16, 39)
    ])
}

pub fn pace2023_exact_054() -> UnGraph<(), ()> {
    UnGraph::from_edges([
        (0, 1), (0, 2), (1, 2), (2, 3), (4, 5), (6, 7), (6, 8), (0, 9), (9, 10), (0, 6), (0, 8),
        (0, 11), (0, 12), (0, 13), (0, 10), (1, 14), (1, 12), (1, 15), (3, 14), (3, 11), (2, 11),
        (2, 16), (2, 12), (4, 17), (6, 11), (14, 18), (14, 15), (8, 11), (5, 17), (11, 12), (7, 19),
        (7, 20), (7, 10), (13, 19), (10, 20), (10, 19), (12, 16), (1, 16), (1, 3), (3, 16), (3, 12),
        (8, 14), (12, 14), (13, 14), (14, 20), (8, 13), (11, 16), (11, 20), (7, 13), (13, 20),
        (16, 20), (12, 20), (17, 21), (7, 22), (18, 22), (13, 22), (20, 22), (3, 23), (13, 18),
        (18, 20), (24, 25), (8, 16), (5, 26), (17, 26), (21, 27), (3, 22), (22, 28), (22, 29),
        (8, 22), (10, 22), (21, 30), (26, 30), (30, 31), (5, 30), (32, 33), (32, 34), (34, 35),
        (26, 31), (8, 28), (8, 29), (8, 36), (8, 20), (8, 10), (5, 37), (33, 35), (26, 38),
        (23, 36), (11, 23), (3, 20), (20, 29), (10, 28), (11, 36), (8, 23), (20, 28), (39, 40),
        (36, 41), (36, 39), (11, 41), (23, 41), (39, 41), (11, 33), (10, 36), (21, 42), (8, 41),
        (43, 44), (2, 28), (12, 28), (45, 46), (42, 47), (48, 49), (48, 50), (51, 52), (49, 50),
        (53, 54), (55, 56), (57, 58), (8, 59), (23, 39), (47, 60), (58, 61), (58, 62), (43, 63),
        (43, 61), (43, 64), (65, 66), (65, 67), (66, 67), (68, 69), (51, 70), (71, 72), (10, 11),
        (43, 66)
    ])
}
