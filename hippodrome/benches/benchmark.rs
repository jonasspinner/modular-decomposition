use criterion::{black_box, criterion_group, criterion_main, Criterion};
use common::instances::ted08_test0;

use miz23_md_cpp;
use miz23_md_rs;

fn criterion_benchmark(c: &mut Criterion) {
    let graph = ted08_test0();

    c.bench_function("cpp", |b| b.iter(|| miz23_md_cpp::modular_decomposition(black_box(&graph))));

    c.bench_function("rs", |b| b.iter(|| miz23_md_rs::modular_decomposition(black_box(&graph))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);