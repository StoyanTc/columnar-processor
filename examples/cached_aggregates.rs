use std::rc::Rc;
use std::time::Instant;

use crate::utils::sample_csv_path;
use columnar_processor::processor::AggregateOp;
use columnar_processor::processor::columnar_processor::ColumnarProcessor;
use columnar_processor::processor::query_builder::QueryCache;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = sample_csv_path();
    let mut processor = ColumnarProcessor::new();
    let cache = Rc::new(QueryCache::new());
    processor.load_csv(path.as_path())?;
    let processor_rc = Rc::new(processor);

    // First run (parsing + aggregation)
    let start = Instant::now();
    let result = processor_rc
        .query_with_cache(&cache)
        .aggregate("value", AggregateOp::Sum)
        .execute();
    println!(
        "First run sum: {:?}, elapsed: {:?}",
        result,
        start.elapsed()
    );

    // Second run (should be cached)
    let start = Instant::now();
    let cached_result = processor_rc
        .query_with_cache(&cache)
        .aggregate("value", AggregateOp::Sum)
        .execute();
    println!(
        "Cached run sum: {:?}, elapsed: {:?}",
        cached_result,
        start.elapsed()
    );

    Ok(())
}
