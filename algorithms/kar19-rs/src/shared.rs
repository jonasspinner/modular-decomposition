use tracing::instrument;

#[instrument(skip_all)]
pub(crate) fn create_nodes(op: &mut [usize], cl: &mut [usize], lc: &[usize], uc: &[usize]) {
    let n = op.len();
    let mut s = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    let mut l = 0;
    for k in 0..n {
        for _ in 0..op[k] + 1 {
            s.push(k);
            t.push(l);
            l = k;
        }
        for c in (0..cl[k] + 1).rev() {
            let i = t.pop().unwrap();
            let j = s.pop().unwrap();

            l = i;
            if i >= j { continue; }
            if i <= lc[j - 1] && lc[j - 1] < uc[j - 1] && uc[j - 1] <= k {
                if c > 0 {
                    op[i] += 1;
                    cl[k] += 1;
                    l = k + 1;
                }
            } else {
                if i < j - 1 {
                    op[i] += 1;
                    cl[j - 1] += 1;
                }
                l = j;
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