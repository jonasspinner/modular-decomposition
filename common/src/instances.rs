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

pub fn ted08_test1() -> UnGraph<(), ()> {
    // from test1.txt in ted+08
    // a->e,d,c,x,f,g,h,i,b,j,k,l,m,n,p,q
    // b->a
    // c->x,i,a
    // d->e,x,i,a
    // e->d,x,i,a
    // f->g,i,a
    // g->f,h,a
    // h->g,a
    // i->f,e,d,c,a
    // j->a
    // k->l,a
    // l->k,a
    // m->n,p,q,a
    // n->m,q,a
    // p->m,q,a
    // q->m,n,p,r,a
    // r->q
    // x->e,d,c,a
    todo!()
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