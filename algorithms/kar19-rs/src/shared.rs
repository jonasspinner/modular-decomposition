use tracing::instrument;

#[instrument(skip_all)]
pub(crate) fn create_consecutive_twin_nodes(op: &mut [usize], cl: &mut [usize], lc: &[usize], uc: &[usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);
    let mut l = 0;
    for k in 0..n {
        s.push((k, l));
        l = k;
        s.extend(std::iter::repeat((k, k)).take(op[k]));
        for c in (0..cl[k] + 1).rev() {
            let (j, i) = s.pop().unwrap();

            l = i; // continue twin chain by default
            if i >= j { continue; }
            if i <= lc[j - 1] && lc[j - 1] < uc[j - 1] && uc[j - 1] <= k {
                // this node and prev are twins
                if c > 0 {
                    // not last parens ∴ last twin
                    op[i] += 1;
                    cl[k] += 1;
                    l = k + 1;
                }
            } else {
                // this node and prev aren't twins
                if i < j - 1 {
                    op[i] += 1;
                    cl[j - 1] += 1;
                }
                l = j; // this node starts new chain
            }
        }
    }
}

#[instrument(skip_all)]
pub(crate) fn remove_singleton_dummy_nodes(op: &mut [usize], cl: &mut [usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);

    for j in 0..n {
        s.extend(std::iter::repeat(j).take(op[j]));
        let mut i_ = usize::MAX;
        for _ in 0..cl[j] {
            let i = s.pop().unwrap();
            if i == i_ {
                op[i] -= 1;
                cl[j] -= 1;
            }
            i_ = i;
        }
    }
    op[0] -= 1;
    cl[n - 1] -= 1;
}