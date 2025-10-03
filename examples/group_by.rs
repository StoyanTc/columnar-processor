use crate::utils::sample_csv_path;
use columnar_processor::processor::{AggregateOp, columnar_processor::ColumnarProcessor};
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = sample_csv_path();

    let mut processor = ColumnarProcessor::new();
    processor.load_csv(path.as_path())?;

    // Group by category and compute average of 'value'
    let grouped = processor.group_by("category", "value", AggregateOp::Avg)?;
    for (k, v) in grouped {
        println!("Category {} => {:?}", k, v);
    }

    Ok(())
}
