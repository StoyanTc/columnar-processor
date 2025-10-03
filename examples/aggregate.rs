use columnar_processor::processor::{AggregateOp, columnar_processor::ColumnarProcessor};

use crate::utils::sample_csv_path;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = sample_csv_path();
    let mut processor = ColumnarProcessor::new();

    // Load CSV
    processor.load_csv(path.as_path())?;

    // Aggregate numeric column
    let sum_result = processor.aggregate("value", AggregateOp::Sum)?;
    println!("Sum of 'value': {:?}", sum_result);

    let avg_result = processor.aggregate("value", AggregateOp::Avg)?;
    println!("Average of 'value': {:?}", avg_result);

    Ok(())
}
