use std::path::Path;

use columnar_processor::processor::{AggregateOp, columnar_processor::ColumnarProcessor};
use criterion::{Criterion, criterion_group, criterion_main};
use rayon::ThreadPoolBuilder;

fn bench_scalability(c: &mut Criterion) {
    let sizes = ["10m", "10m"];

    for &rows in &sizes {
        let path = format!("data/data_{}.csv", rows);
        let path = Path::new(&path);

        // test single-threaded
        let id = format!("sum_{}rows_1thread", rows);
        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        c.bench_function(&id, |b| {
            pool.install(|| {
                b.iter(|| {
                    let mut processor = ColumnarProcessor::new();
                    processor.load_csv(&path).unwrap();
                    processor.aggregate("value", AggregateOp::Sum).unwrap();
                })
            })
        });

        // test with 8 threads
        let id = format!("sum_{}rows_8threads", rows);
        let pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        c.bench_function(&id, |b| {
            pool.install(|| {
                b.iter(|| {
                    let mut processor = ColumnarProcessor::new();
                    processor.load_csv(&path).unwrap();
                    processor.aggregate("value", AggregateOp::Sum).unwrap();
                })
            })
        });
    }
}

criterion_group!(benches, bench_scalability);
criterion_main!(benches);
