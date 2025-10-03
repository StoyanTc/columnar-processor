use std::path::Path;

use columnar_processor::processor::{AggregateOp, columnar_processor::ColumnarProcessor};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    let _profiler = dhat::Profiler::new_heap();

    let path = Path::new("/home/stoyan/data_10m.csv");
    let mut processor = ColumnarProcessor::new();
    processor.load_csv(path).unwrap();

    // Run aggregation
    let _res = processor.aggregate("value", AggregateOp::Sum).unwrap();

    println!("Memory benchmark finished. See dhat-heap.json for details");
}
