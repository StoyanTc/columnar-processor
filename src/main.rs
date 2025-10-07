use std::path::{Path, PathBuf};

use columnar_processor::processor::columnar_processor::ColumnarProcessor;
use jemallocator::Jemalloc;
use polars::{io::SerReader, prelude::CsvReadOptions};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    // The code below is a manual, one-time execution of your benchmark logic.
    // It's not a `criterion` benchmark, but it allows `dhat` to profile it.

    println!("Starting a single run of the 'aggregate_sum_value' benchmark...");

    // Example with synthetic dataset
    let path = Path::new("/home/stoyan/data_10m.csv"); // <-- place your test CSV here

    // Perform the operation you want to profile
    let mut processor = ColumnarProcessor::new();
    processor.load_csv(path).unwrap();
    //let result = processor.aggregate("value", AggregateOp::Sum).unwrap();
    // let _df = CsvReadOptions::default()
    //     .with_has_header(true)
    //     .try_into_reader_with_file_path(Some(PathBuf::from(path)))
    //     .unwrap()
    //     .finish()
    //     .unwrap();

    //println!("Aggregation result: {:?}", result);
}
