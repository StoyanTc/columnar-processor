use std::path::Path;

use columnar_processor::processor::{AggregateOp, columnar_processor::ColumnarProcessor};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_aggregate(c: &mut Criterion) {
    let path = Path::new("data/data_10m.csv");

    c.bench_function("aggregate_sum_value", |b| {
        b.iter(|| {
            let mut processor = ColumnarProcessor::new();
            processor.load_csv(path).unwrap();
            processor.aggregate("value", AggregateOp::Sum).unwrap();
        })
    });
}

criterion_group!(benches, bench_aggregate);
criterion_main!(benches);
