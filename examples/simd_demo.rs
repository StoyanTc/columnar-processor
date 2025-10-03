use std::time::Instant;

use crate::utils::sample_csv_path;
use columnar_processor::processor::AggregateOp;
use columnar_processor::processor::columnar_processor::ColumnarProcessor;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = sample_csv_path();
    let mut processor = ColumnarProcessor::new();
    processor.load_csv(path.as_path())?;

    let start = Instant::now();
    let sum_result = processor.aggregate("value", AggregateOp::Sum)?;
    let elapsed = start.elapsed();
    println!("Sum: {:?}, elapsed: {:?}", sum_result, elapsed);

    // Demonstrates SIMD usage if AVX2 is available
    println!("If CPU supports AVX2, SIMD accelerated aggregation was used");

    Ok(())
}
