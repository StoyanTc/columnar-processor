use pyo3::{pyclass, pymethods};

use crate::processor::{FilterPredicate, Value};

#[pyclass]
pub struct PyFilterPredicate {
    pub inner: FilterPredicate,
}

#[pymethods]
impl PyFilterPredicate {
    #[staticmethod]
    pub fn equals_int(v: i64) -> Self {
        PyFilterPredicate {
            inner: FilterPredicate::Equals(Value::Int(v)),
        }
    }

    #[staticmethod]
    pub fn equals_float(v: f64) -> Self {
        PyFilterPredicate {
            inner: FilterPredicate::Equals(Value::Float(v)),
        }
    }

    #[staticmethod]
    pub fn equals_str(s: &str) -> Self {
        PyFilterPredicate {
            inner: FilterPredicate::Equals(Value::Str(s.to_owned())),
        }
    }

    #[staticmethod]
    pub fn between_int(a: i64, b: i64) -> Self {
        PyFilterPredicate {
            inner: FilterPredicate::Between(Value::Int(a), Value::Int(b)),
        }
    }

    // helper to get the inner for Rust-side usage
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}
