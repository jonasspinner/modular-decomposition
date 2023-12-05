pub(crate) mod kar19 {
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::collections::hash_map::RandomState;
    use petgraph::graph::{NodeIndex, UnGraph};
    use crate::factorizing_permutation::kar19::util::{smaller_larger, splice};
    #[allow(unused_imports)]
    use crate::{d1, d2, trace};

    #[allow(non_snake_case)]
    pub(crate) fn factorizing_permutation(graph: &UnGraph<(), ()>) -> Vec<NodeIndex> {
        let V = graph.node_indices().collect();
        let mut P = vec![V];
        let mut center = NodeIndex::new(0);
        let mut pivots = vec![];
        let mut modules = VecDeque::new();
        let mut first_pivot = HashMap::<Vec<NodeIndex>, NodeIndex>::new();

        partition_refinement(&mut P, &mut center, &mut pivots, &mut modules, &mut first_pivot, &graph);
        P.iter().map(|part| part[0]).collect()
    }

    #[allow(non_snake_case)]
    fn refine(P: &mut Vec<Vec<NodeIndex>>,
              S: HashSet<NodeIndex>,
              x: NodeIndex,
              center: &NodeIndex,
              pivots: &mut Vec<Vec<NodeIndex>>,
              modules: &mut VecDeque<Vec<NodeIndex>>) {
        trace!("refine: {} {:?}", x.index(), S.iter().map(|u| u.index()).collect::<Vec<_>>());
        let mut i = 0_usize.wrapping_sub(1);
        let mut between = false;
        while i.wrapping_add(1) < P.len() {
            i = i.wrapping_add(1);
            let X = &P[i];
            if X.contains(&center) || X.contains(&x) {
                between = !between;
                continue;
            }
            let (X_a, X): (Vec<_>, Vec<_>) = X.iter().partition(|&y| S.contains(y));
            if X_a.is_empty() || X.is_empty() { continue; }
            trace!("refine:P:0: {:?}", d2(&*P));
            P[i] = X.clone();
            P.insert(i + between as usize, X_a.clone());
            trace!("refine:P:1: {:?}", d2(&*P));
            add_pivot(X, X_a, pivots, modules);
            i += 1;
        }
    }

    #[allow(non_snake_case)]
    fn add_pivot(
        X: Vec<NodeIndex>,
        X_a: Vec<NodeIndex>,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>) {
        trace!("add_pivot: {:?} {:?}", d1(&X), d1(&X_a));
        if pivots.contains(&X) {
            pivots.push(X_a);
        } else {
            let i = modules.iter().position(|Y| Y == &X);
            let (S, L) = smaller_larger(X, X_a);
            pivots.push(S);
            if let Some(i) = i {
                modules[i] = L;
            } else {
                modules.push_back(L);
            }
        }
    }

    #[allow(non_snake_case)]
    fn partition_refinement(
        P: &mut Vec<Vec<NodeIndex>>,
        center: &mut NodeIndex,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
        first_pivot: &mut HashMap<Vec<NodeIndex>, NodeIndex>,
        graph: &UnGraph<(), ()>) {
        while init_partition(P, center, pivots, modules, first_pivot, graph) {
            trace!("P: {:?}", d2(&*P));
            trace!("pivots: {:?}", d2(&*pivots));
            trace!("modules: {:?}", d2(&*modules));
            while let Some(E) = pivots.pop() {
                let E_h: HashSet<_, RandomState> = HashSet::from_iter(E.clone());
                for &x in &E {
                    let S = graph.neighbors(x).filter(|v| !E_h.contains(v)).collect();
                    refine(P, S, x, center, pivots, modules);
                }
            }
        }
    }

    #[allow(non_snake_case)]
    fn init_partition(
        P: &mut Vec<Vec<NodeIndex>>,
        center: &mut NodeIndex,
        pivots: &mut Vec<Vec<NodeIndex>>,
        modules: &mut VecDeque<Vec<NodeIndex>>,
        first_pivot: &mut HashMap<Vec<NodeIndex>, NodeIndex>,
        graph: &UnGraph<(), ()>) -> bool {
        if P.iter().all(|p| p.len() <= 1) { return false; }
        if let Some(X) = modules.pop_front() {
            let x = X[0];
            pivots.push(vec![x]);
            first_pivot.insert(X, x);
        } else {
            for (i, X) in P.iter().enumerate() {
                if X.len() <= 1 { continue; }
                let x = first_pivot.get(X).copied().unwrap_or(X[0]);
                let adj: HashSet<_> = graph.neighbors(x).collect();
                let (A, mut N): (Vec<_>, Vec<_>) = X.into_iter().partition(|&y| *y != x && adj.contains(y));
                N.retain(|y| *y != x);
                splice(P, i, A.clone(), x, N.clone());
                let (S, L) = smaller_larger(A, N);
                *center = x;
                pivots.push(S);
                modules.push_back(L);
                break;
            }
        }
        true
    }

    mod util {
        pub(crate) fn smaller_larger<T>(a: Vec<T>, b: Vec<T>) -> (Vec<T>, Vec<T>) {
            if a.len() <= b.len() { (a, b) } else { (b, a) }
        }

        pub(crate) fn splice<T>(vec: &mut Vec<Vec<T>>, i: usize, first: Vec<T>, second: T, third: Vec<T>) {
            match (first.is_empty(), third.is_empty()) {
                (true, true) => { vec[i] = vec![second] }
                (true, false) => {
                    vec[i] = vec![second];
                    vec.insert(i + 1, third)
                }
                (false, true) => {
                    vec[i] = first;
                    vec.insert(i + 1, vec![second]);
                }
                (false, false) => {
                    vec[i] = first;
                    vec.insert(i + 1, vec![second]);
                    vec.insert(i + 2, third)
                }
            }
        }
    }
}