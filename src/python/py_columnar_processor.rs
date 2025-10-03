use arrow2::ffi::export_field_to_c;
use arrow2::ffi::{ArrowArray, ArrowSchema, export_array_to_c};
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyFloat;
use pyo3::types::PyInt;
use pyo3::types::PyList;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use crate::python::py_aggregate_op::PyAggregateOp;
use crate::python::py_filter_predicate::PyFilterPredicate;
// Assume these are defined elsewhere
use crate::processor::{AggregateResult, ProcessorError, columnar_processor::ColumnarProcessor};
use crate::python::py_query_builder::{PyQueryBuilder, PyQueryCache};

/// Convert Rust errors to Python exceptions
impl From<ProcessorError> for PyErr {
    fn from(err: ProcessorError) -> PyErr {
        exceptions::PyRuntimeError::new_err(format!("{:?}", err))
    }
}

#[pyclass(unsendable)]
pub struct PyColumnarProcessor {
    inner: Rc<ColumnarProcessor>,
}

#[pymethods]
impl PyColumnarProcessor {
    #[new]
    pub fn new(path: String) -> PyResult<Self> {
        let mut proc = ColumnarProcessor::new();
        proc.load_csv(Path::new(&path))?; // ✅ mutably while still owned
        Ok(PyColumnarProcessor {
            inner: Rc::new(proc),
        })
    }

    pub fn filter(
        &self,
        _py: Python<'_>,
        column: &str,
        predicate: &PyFilterPredicate,
    ) -> PyResult<Vec<usize>> {
        Ok(self.inner.filter(column, &predicate.inner.clone())?)
    }

    pub fn aggregate(
        &self,
        py: Python<'_>,
        column: &str,
        op: &PyAggregateOp,
    ) -> PyResult<Py<PyAny>> {
        match self.inner.aggregate(column, op.inner)? {
            AggregateResult::Int(v) => Ok(PyInt::new(py, v).into()),
            AggregateResult::Float(v) => Ok(PyFloat::new(py, v).into()),
        }
    }

    /// Expose Arrow data as PyArrow objects
    pub fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (schema, chunk) = self.inner.to_arrow();

        // Export schema fields to C ABI - zero-copy, creates metadata pointers only
        let c_schema_fields: Vec<ArrowSchema> =
            schema.fields.iter().map(export_field_to_c).collect();

        // Export arrays to C ABI - zero-copy, just creates pointers to existing data
        let c_arrays: Vec<ArrowArray> = chunk
            .arrays()
            .iter()
            .map(|array| export_array_to_c(array.to_boxed())) // clone is cheap - reference counting
            .collect();

        // Use PyArrow's C ABI import
        let pyarrow = py.import("pyarrow")?;

        // Create field objects and build schema
        let fields_obj = PyList::empty(py);
        for c_schema_field in &c_schema_fields {
            let field_ptr = c_schema_field as *const ArrowSchema as usize;
            let field_obj = pyarrow
                .getattr("Field")?
                .getattr("_import_from_c")?
                .call1((field_ptr,))?;
            fields_obj.append(field_obj)?;
        }

        // Create schema from fields
        let schema_obj = pyarrow.getattr("schema")?.call1((fields_obj,))?;

        // Import arrays using their corresponding field schemas
        let arrays_obj = PyList::empty(py);
        for (c_array, c_schema_field) in c_arrays.iter().zip(c_schema_fields.iter()) {
            let array_ptr = c_array as *const ArrowArray as usize;
            let schema_ptr = c_schema_field as *const ArrowSchema as usize;
            let arr_obj = pyarrow
                .getattr("Array")?
                .getattr("_import_from_c")?
                .call1((array_ptr, schema_ptr))?;
            arrays_obj.append(arr_obj)?;
        }

        // Create table
        let table = pyarrow.getattr("Table")?.call1((arrays_obj, schema_obj))?;

        Ok(table.into())
    }

    pub fn group_by(
        &self,
        py: Python<'_>,
        group_col: &str,
        agg_col: &str,
        op: &PyAggregateOp,
    ) -> PyResult<Py<PyAny>> {
        let result: HashMap<String, AggregateResult> =
            self.inner.group_by(group_col, agg_col, op.inner)?;

        let py_dict = PyDict::new(py);

        for (key, value) in result {
            let py_value: Py<PyAny> = match value {
                AggregateResult::Int(v) => PyInt::new(py, v).into(),
                AggregateResult::Float(v) => PyFloat::new(py, v).into(),
            };
            py_dict.set_item(key, py_value)?;
        }

        Ok(py_dict.into())
    }

    pub fn query(&self) -> PyQueryBuilder {
        let builder = self.inner.query();
        PyQueryBuilder { inner: builder }
    }

    pub fn query_with_cache(&self, cache: &PyQueryCache) -> PyQueryBuilder {
        let builder = self.inner.query_with_cache(&cache.inner);
        PyQueryBuilder { inner: builder }
    }
}
