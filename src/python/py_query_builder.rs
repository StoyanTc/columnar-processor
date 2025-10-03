use pyo3::exceptions;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;

use crate::processor::query_builder::{QueryBuilder, QueryCache, QueryResult};
use crate::processor::{AggregateOp, AggregateResult};
use crate::python::py_filter_predicate::PyFilterPredicate;
use pyo3::IntoPyObject;
use pyo3::types::PyModule;

/// Python bindings for QueryCache
#[pyclass(unsendable)]
pub struct PyQueryCache {
    pub inner: Rc<QueryCache>,
}

#[pymethods]
impl PyQueryCache {
    #[new]
    fn new() -> Self {
        Self {
            inner: Rc::new(QueryCache::new()),
        }
    }
}

/// Python bindings for QueryBuilder
#[pyclass(unsendable)]
pub struct PyQueryBuilder {
    pub inner: QueryBuilder,
}

#[pymethods]
impl PyQueryBuilder {
    /// Add filter
    pub fn filter(&mut self, column: &str, predicate: &PyFilterPredicate) -> PyResult<()> {
        self.inner = self.inner.clone().filter(column, predicate.inner.clone());
        Ok(())
    }

    /// Add group by
    pub fn group_by(&mut self, column: &str) {
        self.inner = self.inner.clone().group_by(column);
    }

    /// Add aggregation
    pub fn aggregate(&mut self, column: &str, op: &str) -> PyResult<()> {
        let op = match op.to_lowercase().as_str() {
            "sum" => AggregateOp::Sum,
            "avg" => AggregateOp::Avg,
            "count" => AggregateOp::Count,
            "min" => AggregateOp::Min,
            "max" => AggregateOp::Max,
            _ => return Err(PyErr::new::<exceptions::PyValueError, _>("Unknown op")),
        };
        self.inner = self.inner.clone().aggregate(column, op);
        Ok(())
    }

    /// Execute query
    pub fn execute(&self) -> PyResult<PyQueryResult> {
        match self.inner.clone().execute() {
            Ok(result) => Ok(PyQueryResult { inner: result }),
            Err(e) => Err(PyErr::new::<exceptions::PyRuntimeError, _>(format!(
                "{:?}",
                e
            ))),
        }
    }
}

/// Python wrapper for QueryResult
#[pyclass]
pub struct PyQueryResult {
    inner: QueryResult,
}

#[pymethods]
impl PyQueryResult {
    fn is_multi_dimensional(&self) -> bool {
        matches!(self.inner, QueryResult::MultiDimensional(_))
    }

    fn dimensions(&self) -> PyResult<Vec<String>> {
        match &self.inner {
            QueryResult::MultiDimensional(res) => Ok(res.dimensions.clone()),
            _ => Err(PyErr::new::<exceptions::PyValueError, _>(
                "Not a MultiDimensional result",
            )),
        }
    }

    fn measures(&self) -> PyResult<Vec<String>> {
        match &self.inner {
            QueryResult::MultiDimensional(res) => Ok(res.measures.clone()),
            _ => Err(PyErr::new::<exceptions::PyValueError, _>(
                "Not a MultiDimensional result",
            )),
        }
    }

    fn data(&self, py: Python<'_>) -> PyResult<Vec<(Vec<String>, HashMap<String, Py<PyAny>>)>> {
        match &self.inner {
            QueryResult::MultiDimensional(res) => {
                let mut converted = Vec::with_capacity(res.data.len());

                for (dims, map) in &res.data {
                    let mut py_map: HashMap<String, Py<PyAny>> = HashMap::new();

                    for (k, v) in map {
                        let obj = match v {
                            AggregateResult::Int(i) => i.into_pyobject(py)?.into(),
                            AggregateResult::Float(f) => f.into_pyobject(py)?.into(),
                        };
                        py_map.insert(k.clone(), obj);
                    }

                    converted.push((dims.clone(), py_map));
                }

                Ok(converted)
            }
            _ => Err(PyErr::new::<exceptions::PyValueError, _>(
                "Not a MultiDimensional result",
            )),
        }
    }
}

#[pyclass]
pub struct PyAggregateResult {
    inner: AggregateResult,
}

#[pymethods]
impl PyAggregateResult {}

/// Module initialization
#[pymodule]
fn pycolumnar_processor(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQueryCache>()?;
    m.add_class::<PyQueryBuilder>()?;
    m.add_class::<PyQueryResult>()?;
    Ok(())
}
