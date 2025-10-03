use crate::utils::sample_csv_path;
use columnar_processor::processor::{
    columnar_processor::ColumnarProcessor,
    {FilterPredicate, Value},
};
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = sample_csv_path();

    let mut processor = ColumnarProcessor::new();
    processor.load_csv(path.as_path())?;

    // Filter rows where 'value' > 100
    let filtered_rows =
        processor.filter("value", &FilterPredicate::GreaterThan(Value::Int(100)))?;

    println!("Rows where 'value' > 100: {:?}", filtered_rows);
    Ok(())
}
