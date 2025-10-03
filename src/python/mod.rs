#![cfg(feature = "python-bindings")]

use pyo3::types::PyModuleMethods;
use pyo3::{Bound, PyResult, Python, pymodule, types::PyModule};

pub mod py_aggregate_op;

pub mod py_columnar_processor;

pub mod py_filter_predicate;

pub mod py_query_builder;

#[pymodule]
fn columnar_processor(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_columnar_processor::PyColumnarProcessor>()?;
    Ok(())
}
