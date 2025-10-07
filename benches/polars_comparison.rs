use columnar_processor::processor::{AggregateOp, columnar_processor::ColumnarProcessor};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use polars::prelude::*;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;

fn generate_test_csv(path: &str, rows: usize) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "id,value,price,category").unwrap();
    for i in 0..rows {
        writeln!(
            file,
            "{},{},{:.2},cat_{}",
            i,
            i * 10,
            (i as f64) * 1.5,
            i % 100
        )
        .unwrap();
    }
}

fn bench_csv_loading(c: &mut Criterion) {
    //let sizes = vec![10_000, 100_000, 1_000_000, 100_000_000];
    let sizes = vec![100_000_000];
    let mut group = c.benchmark_group("csv_loading");
    group.sample_size(10);

    for size in sizes {
        let path = "/home/stoyan/data_10m.csv"; // format!("bench_data_{}.csv", size);
        //generate_test_csv(&path, size);

        group.bench_with_input(BenchmarkId::new("columnar", size), &path, |b, path| {
            b.iter(|| {
                let mut proc = ColumnarProcessor::new();
                proc.load_csv(path.as_ref()).unwrap();
                black_box(proc);
            });
        });

        group.bench_with_input(BenchmarkId::new("polars", size), &path, |b, path| {
            b.iter(|| {
                let df = CsvReadOptions::default()
                    .with_has_header(true)
                    .try_into_reader_with_file_path(Some(PathBuf::from(path)))
                    .unwrap()
                    .finish()
                    .unwrap();

                black_box(df);
            });
        });
    }
    group.finish();
}

fn bench_aggregation(c: &mut Criterion) {
    let path = "/home/stoyan/data_10m.csv"; //"bench_agg.csv";
    //generate_test_csv(path, 1_000_000);

    let mut proc = ColumnarProcessor::new();
    proc.load_csv(path.as_ref()).unwrap();

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(PathBuf::from(path)))
        .unwrap()
        .finish()
        .unwrap();

    let mut group = c.benchmark_group("aggregation");

    // Sum aggregation
    group.bench_function("columnar_sum", |b| {
        b.iter(|| black_box(proc.aggregate("value", AggregateOp::Sum).unwrap()));
    });

    group.bench_function("polars_sum", |b| {
        b.iter(|| black_box(df.column("value").unwrap().sum_reduce().unwrap()));
    });

    // Cached aggregation
    group.bench_function("columnar_sum_cached", |b| {
        proc.aggregate("value", AggregateOp::Sum).unwrap(); // Prime cache
        b.iter(|| black_box(proc.aggregate("value", AggregateOp::Sum).unwrap()));
    });

    group.finish();
}

criterion_group!(benches, bench_csv_loading, bench_aggregation);
criterion_main!(benches);
