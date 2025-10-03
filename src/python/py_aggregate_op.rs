use pyo3::{pyclass, pymethods};

use crate::processor::AggregateOp;

#[pyclass]
pub struct PyAggregateOp {
    pub inner: AggregateOp,
}

#[pymethods]
impl PyAggregateOp {
    #[staticmethod]
    pub fn sum() -> Self {
        PyAggregateOp {
            inner: AggregateOp::Sum,
        }
    }

    #[staticmethod]
    pub fn avg() -> Self {
        PyAggregateOp {
            inner: AggregateOp::Avg,
        }
    }

    #[staticmethod]
    pub fn count() -> Self {
        PyAggregateOp {
            inner: AggregateOp::Count,
        }
    }

    #[staticmethod]
    pub fn min() -> Self {
        PyAggregateOp {
            inner: AggregateOp::Min,
        }
    }

    #[staticmethod]
    pub fn max() -> Self {
        PyAggregateOp {
            inner: AggregateOp::Max,
        }
    }

    fn __repr__(&self) -> String {
        format!("PyAggregateOp::{:?}", self.inner)
    }
}
