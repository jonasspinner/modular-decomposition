use criterion::{black_box, criterion_group, criterion_main, Criterion};
use common::instances::ted08_test0;


fn criterion_benchmark(c: &mut Criterion) {
    let graph = ted08_test0();

    let g_cpp = miz23_md_cpp::prepare(&graph);
    let g_rs = miz23_md_rs::prepare(&graph);

    c.bench_function("cpp", |b| b.iter(|| black_box(black_box(&g_cpp).compute())));

    c.bench_function("rs", |b| b.iter(|| black_box(black_box(&g_rs).compute())));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);