use columnar_processor::processor::{
    AggregateOp, columnar_processor::ColumnarProcessor, query_builder::QueryCache,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use jemallocator::Jemalloc;
use std::{path::Path, rc::Rc};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn load_and_sum(c: &mut Criterion) {
    let _profiler = dhat::Profiler::new_heap();

    let mut group = c.benchmark_group("ColumnarProcessor");
    group.sample_size(10);

    // Example with synthetic dataset
    let path = Path::new("/home/stoyan/data_10m.csv"); // <-- place your test CSV here

    group.throughput(Throughput::Elements(10_000_000)); // adjust based on dataset size
    group.bench_function("load_csv", |b| {
        b.iter(|| {
            let mut processor = ColumnarProcessor::new();
            processor.load_csv(path).unwrap();
        })
    });

    // Benchmark SUM on the "value" column
    group.bench_function("load_csv_chunked + aggregate_sum_value", |b| {
        b.iter(|| {
            let mut processor = ColumnarProcessor::new();
            processor.load_csv(path).unwrap();
            processor.aggregate("value", AggregateOp::Sum).unwrap();
        })
    });

    group.bench_function("aggregate_only", |b| {
        // Preload once outside the iterator
        let mut processor = ColumnarProcessor::new();
        processor.load_csv(path).unwrap();

        b.iter(|| {
            processor.aggregate("value", AggregateOp::Sum).unwrap();
        });
    });

    // Benchmark SUM on the "value" column cached
    group.bench_function("aggregate_sum_value_cached", |b| {
        let mut processor = ColumnarProcessor::new();
        let mut cache = Rc::new(QueryCache::new());
        processor.load_csv(path).unwrap();
        let processor_rc = Rc::new(processor);

        b.iter(|| {
            let _ = processor_rc
                .query_with_cache(&mut cache)
                .aggregate("value", AggregateOp::Sum)
                .execute();
        })
    });

    // Benchmark GROUP BY average
    group.throughput(Throughput::Elements(10_000_000));
    group.bench_function("group_by_category_avg", |b| {
        b.iter(|| {
            let mut processor = ColumnarProcessor::new();
            processor.load_csv(path).unwrap();
            processor
                .group_by("category", "value", AggregateOp::Avg)
                .unwrap();
        })
    });

    // Benchmark GROUP BY cached average
    group.throughput(Throughput::Elements(10_000_000));
    group.bench_function("group_by_category_avg_cached", |b| {
        let mut processor = ColumnarProcessor::new();
        let cache = Rc::new(QueryCache::new());
        processor.load_csv(path).unwrap();
        let processor_rc = Rc::new(processor);

        b.iter(|| {
            let _ = processor_rc
                .query_with_cache(&cache)
                .aggregate("value", AggregateOp::Avg)
                .group_by("category")
                .execute();
        })
    });

    group.finish();
}

criterion_group!(benches, load_and_sum);
criterion_main!(benches);
