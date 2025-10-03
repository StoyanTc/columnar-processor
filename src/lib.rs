//! # ColumnarProcessor
//!
//! `ColumnarProcessor` is a high-performance, memory-efficient, columnar CSV processor
//! written in Rust. It supports:
//!
//! - Memory-mapped CSV loading (zero-copy for large files)
//! - Dynamic schema inference (int, float, string)
//! - Lazy numeric parsing for minimal allocations
//! - SIMD-accelerated aggregation and filtering
//! - Parallel computation with Rayon
//! - Cached aggregation results for low-latency queries
//!
//! # Features
//!
//! - **Columnar storage**: offsets into the CSV buffer
//! - **Aggregations**: sum, count, average, min, max
//! - **Filtering**: equals, greater-than, less-than, between
//! - **Group-by** on string columns
//! - **AVX2 SIMD acceleration** for numeric aggregation (optional, fallback to scalar)
//!
//! # Example
//!
//! ```rust
//! use columnar_processor::{ColumnarProcessor, AggregateOp, FilterPredicate, Value};
//! use std::path::Path;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let path = Path::new("data.csv");
//!     let mut processor = ColumnarProcessor::new();
//!
//!     // Load CSV
//!     processor.load_csv(path)?;
//!
//!     // Aggregate numeric column
//!     let sum_result = processor.aggregate("value", AggregateOp::Sum)?;
//!     println!("Sum of values: {:?}", sum_result);
//!
//!     // Filter rows
//!     let filtered_rows = processor.filter(
//!         "value",
//!         FilterPredicate::GreaterThan(Value::Int(100))
//!     )?;
//!     println!("Rows with value > 100: {:?}", filtered_rows);
//!
//!     // Group by category
//!     let grouped = processor.group_by("category", "value", AggregateOp::Average)?;
//!     for (k, v) in grouped {
//!         println!("Category {} => {:?}", k, v);
//!     }
//!
//!     Ok(())
//! }
//! ```

mod helpers;
pub mod processor;

#[cfg(feature = "python-bindings")]
pub mod python;

#[cfg(feature = "jni-bindings")]
pub mod java;
