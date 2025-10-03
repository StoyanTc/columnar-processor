use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use thiserror::Error;

pub mod columnar_processor;
pub mod query_builder;

/// Error type used across the crate
#[derive(Debug, Error)]
pub enum ProcessorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("Int parse error: {0}")]
    IntParse(#[from] std::num::ParseIntError),

    #[error("Float parse error: {0}")]
    FloatParse(#[from] std::num::ParseFloatError),

    #[error("Schema/parse error: {0}")]
    Parse(String),

    #[error("Missing column: {0}")]
    MissingColumn(String),

    #[error("Mmap not loaded")]
    MmapNotLoaded,
}

#[derive(Debug)]
pub struct ParseSummary {
    pub rows_processed: usize,
    pub errors: Vec<ParseError>,
}

#[derive(Debug)]
pub struct ParseError {
    pub row: usize,
    pub column: String,
    pub value: String,
    pub error: String,
}

/// Value helper for predicates (owned for simplicity)
#[derive(Debug, Clone)]
pub enum Value {
    /// Integer column
    Int(i64),
    /// Float column
    Float(f64),
    /// String column
    Str(String),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::Str(a), Value::Str(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Value::Int(v) => v.hash(state),
            Value::Float(v) => v.to_bits().hash(state),
            Value::Str(v) => v.hash(state),
        }
    }
}

/// Filter predicate
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FilterPredicate {
    Equals(Value),
    GreaterThan(Value),
    LessThan(Value),
    Between(Value, Value),
}

/// Aggregate operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateOp {
    /// Sum of all numeric values
    Sum,
    /// Count of all rows
    Count,
    /// Average of numeric values
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
}

/// Result of an aggregation
#[derive(Debug, Clone, PartialEq)]
pub enum AggregateResult {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone)]
pub enum OperationResult {
    Aggregate(AggregateResult),
    Filter(Vec<usize>),
    GroupBy(HashMap<String, AggregateResult>),
}
