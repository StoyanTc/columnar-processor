use arrow2::{
    array::{Array, Float64Array, Int64Array, MutableUtf8Array, Utf8Array},
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
};
use memchr::memchr_iter;
use memmap2::Mmap;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::HashMap,
    fs::File,
    path::Path,
    str,
    sync::{Arc, Mutex},
};

use crate::{
    helpers::simd_helpers::{aggregate_f64_avx2, aggregate_i64_avx2, filter_f64, filter_i64},
    processor::{
        AggregateOp, AggregateResult, FilterPredicate, ParseError, ParseSummary, ProcessorError,
        Value,
    },
};
use atoi::atoi;
use fast_float::parse as fast_float_parse;

/// Supported column
#[derive(Debug)]
pub enum Column {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Str(Vec<(usize, usize)>),
}

/// Main processor for columnar CSV data
///
/// # Examples
///
/// ```rust
/// # use columnar_processor::{ColumnarProcessor, AggregateOp};
/// let mut processor = ColumnarProcessor::new();
/// processor.load_csv("data.csv".as_ref()).unwrap();
/// let sum = processor.aggregate_cached("value", AggregateOp::Sum).unwrap();
/// println!("Sum: {:?}", sum);
/// ```
#[derive(Debug)]
pub struct ColumnarProcessor {
    mmap: Option<Mmap>,   // owns the CSV bytes
    columns: Vec<Column>, // dynamic columns by name
    row_count: usize,
    headers: Vec<String>,
}

impl ColumnarProcessor {
    /// Create an empty processor
    pub fn new() -> Self {
        ColumnarProcessor {
            mmap: None,
            columns: Vec::new(),
            row_count: 0,
            headers: Vec::new(),
        }
    }

    /// Loads a CSV file into memory using memory mapping
    ///
    /// Infers column types from the first data row (Int, Float, Str)
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file
    ///
    /// # Errors
    /// Returns a [`ProcessorError`] if:
    /// - File cannot be opened or mapped
    /// - CSV is malformed (rows mismatch header)
    ///
    /// # Example
    /// ```rust
    /// # use columnar_processor::ColumnarProcessor;
    /// let mut processor = ColumnarProcessor::new();
    /// processor.load_csv("data.csv".as_ref()).unwrap();
    /// ```.
    pub fn load_csv(&mut self, path: &Path) -> Result<ParseSummary, ProcessorError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let buf: &[u8] = &mmap[..];

        // find header end (first newline)
        let header_end_rel = buf
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| ProcessorError::Parse("Missing header line".into()))?;
        let header_line = &buf[..header_end_rel];
        let header_fields: Vec<&[u8]> = header_line.split(|&b| b == b',').collect();
        let headers: Vec<String> = header_fields
            .iter()
            .map(|s| String::from_utf8_lossy(s).to_string())
            .collect();

        // iterate rows after header; get iterator of line slices
        let mut lines = Self::lines(&buf[header_end_rel + 1..]);

        // infer schema from first data line
        let first_line = lines
            .next()
            .ok_or_else(|| ProcessorError::Parse("No data rows after header".into()))?;
        let first_fields: Vec<&[u8]> = first_line.split(|&b| b == b',').collect();
        if first_fields.len() != headers.len() {
            return Err(ProcessorError::Parse(format!(
                "Header length {} != first row fields {}",
                headers.len(),
                first_fields.len()
            )));
        }

        // prepare columns: initialize with first row offsets
        let mut columns: Vec<Column> = Vec::with_capacity(headers.len());
        let base_ptr = buf.as_ptr() as usize;
        // estimate rows for preallocation of the columns' vectors
        let estimated_rows = {
            let data_bytes = buf.len() - header_end_rel - 1;
            let sample_line_bytes = first_line.len() + 1;
            (data_bytes / sample_line_bytes) + 100_000 // 10% buffer
        };
        //let estimated_rows = memchr_iter(b'\n', &buf[header_end_rel + 1..]).count();

        let first_line_start = first_line.as_ptr() as usize - base_ptr;
        for (i, _) in headers.iter().enumerate() {
            let data = if let Some(v) = atoi::<i64>(first_fields[i]) {
                let mut nums = Vec::with_capacity(estimated_rows);
                nums.push(v);
                Column::Int64(nums)
            } else if let Ok(v) = fast_float_parse::<f64, _>(first_fields[i]) {
                let mut nums = Vec::with_capacity(estimated_rows);
                nums.push(v);
                Column::Float64(nums)
            } else {
                let start = first_line_start
                    + (first_fields[i].as_ptr() as usize - first_line.as_ptr() as usize);
                let end = start + first_fields[i].len();
                let mut strs = Vec::with_capacity(estimated_rows);
                strs.push((start, end));
                Column::Str(strs)
            };

            columns.push(data);
        }

        // process the rest of the lines (we already consumed the first data line)
        let mut row_count: usize = 1; // first_line counted
        let mut errors = Vec::new(); // Collect the errors, if any
        let mut fields = Vec::with_capacity(headers.len());
        for line in lines {
            let line_start = line.as_ptr() as usize - base_ptr;
            fields.clear();
            let mut field_start = 0;
            for comma_pos in memchr_iter(b',', line) {
                fields.push(&line[field_start..comma_pos]);
                field_start = comma_pos + 1;
            }
            fields.push(&line[field_start..]);
            if fields.len() != headers.len() {
                return Err(ProcessorError::Parse(format!(
                    "Row {} column count mismatch: expected {}, got {}",
                    row_count + 1,
                    headers.len(),
                    fields.len()
                )));
            }

            for i in 0..headers.len() {
                let col = columns.get_mut(i).unwrap();
                match col {
                    Column::Int64(v) => match atoi::<i64>(fields[i]) {
                        Some(value) => v.push(value),
                        None => errors.push(ParseError {
                            row: row_count + 2, // +2 because of header and 0-indexing
                            column: headers[i].clone(),
                            value: String::from_utf8_lossy(fields[i]).to_string(),
                            error: None,
                        }),
                    },
                    Column::Float64(v) => match fast_float_parse::<f64, _>(fields[i]) {
                        Ok(value) => v.push(value),
                        Err(e) => errors.push(ParseError {
                            row: row_count + 2, // +2 because of header and 0-indexing
                            column: headers[i].clone(),
                            value: String::from_utf8_lossy(fields[i]).to_string(),
                            error: Some(e.to_string()),
                        }),
                    },
                    Column::Str(v) => {
                        let start =
                            line_start + (fields[i].as_ptr() as usize - line.as_ptr() as usize);
                        let end = start + fields[i].len();
                        v.push((start, end))
                    }
                }
            }

            row_count += 1;
        }

        // commit into self
        self.mmap = Some(mmap);
        self.columns = columns;
        self.row_count = row_count;
        self.headers = headers;

        Ok(ParseSummary {
            rows_processed: row_count,
            errors,
        })
    }

    fn lines<'a>(buf: &'a [u8]) -> impl Iterator<Item = &'a [u8]> + 'a {
        let mut start = 0;
        memchr_iter(b'\n', buf)
            .map(move |i| {
                let line = &buf[start..i];
                start = i + 1;
                line
            })
            .filter(|line| line.len() > 0)
    }

    /// Return number of rows
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Helper to slice mmap and return bytes for given offset
    pub fn slice_bytes(&self, start: usize, end: usize) -> Result<&[u8], ProcessorError> {
        let mmap = self.mmap.as_ref().ok_or(ProcessorError::MmapNotLoaded)?;

        if end > mmap.len() || start > end {
            return Err(ProcessorError::Parse("Invalid byte range".into()));
        }

        Ok(&mmap[start..end])
    }

    /// Aggregates a numeric column with optional caching for low-latency
    ///
    /// Uses SIMD acceleration (AVX2) if available.
    ///
    /// # Arguments
    /// * `column` - Column name
    /// * `op` - Aggregate operation
    ///
    /// # Returns
    /// [`AggregateResult`] representing the aggregation result
    ///
    /// # Example
    /// ```rust
    /// # use columnar_processor::{ColumnarProcessor, AggregateOp};
    /// let result = processor.aggregate_cached("value", AggregateOp::Sum).unwrap();
    /// ```
    pub fn filter(
        &self,
        column: &str,
        predicate: &FilterPredicate,
    ) -> Result<Vec<usize>, ProcessorError> {
        let col = self.get_col(column)?;

        match (col, &predicate) {
            (
                Column::Float64(values),
                FilterPredicate::Equals(_)
                | FilterPredicate::GreaterThan(_)
                | FilterPredicate::LessThan(_)
                | FilterPredicate::Between(_, _),
            ) => Ok(filter_f64(&values, predicate.clone())),

            (
                Column::Int64(values),
                FilterPredicate::Equals(_)
                | FilterPredicate::GreaterThan(_)
                | FilterPredicate::LessThan(_)
                | FilterPredicate::Between(_, _),
            ) => Ok(filter_i64(&values, predicate.clone())),

            (Column::Str(offsets), FilterPredicate::Equals(Value::Str(target))) => {
                let mut out = Vec::with_capacity(offsets.len());
                for (i, &(s, e)) in offsets.iter().enumerate() {
                    if self.slice_bytes(s, e)? == target.as_bytes() {
                        out.push(i);
                    }
                }
                Ok(out)
            }

            _ => Err(ProcessorError::Parse(
                "Predicate & column-type combination not supported".into(),
            )),
        }
    }

    /// Aggregates a numeric column with optional caching for low-latency
    ///
    /// Uses SIMD acceleration (AVX2) if available.
    ///
    /// # Arguments
    /// * `column` - Column name
    /// * `op` - Aggregate operation
    ///
    /// # Returns
    /// [`AggregateResult`] representing the aggregation result
    ///
    /// # Example
    /// ```rust
    /// # use columnar_processor::{ColumnarProcessor, AggregateOp};
    /// let result = processor.aggregate("value", AggregateOp::Sum).unwrap();
    /// ```
    pub fn aggregate(
        &self,
        column: &str,
        op: AggregateOp,
    ) -> Result<AggregateResult, ProcessorError> {
        let col = self.get_col(column)?;

        match col {
            Column::Int64(values) => {
                let n = values.len();
                if n == 0 {
                    return Err(ProcessorError::Parse("empty column".into()));
                }

                let result = match op {
                    AggregateOp::Sum
                    | AggregateOp::Min
                    | AggregateOp::Max
                    | AggregateOp::Avg
                    | AggregateOp::Count => {
                        let v = aggregate_i64_avx2(&values, op);
                        match op {
                            AggregateOp::Sum
                            | AggregateOp::Min
                            | AggregateOp::Max
                            | AggregateOp::Count => AggregateResult::Int(v),
                            AggregateOp::Avg => {
                                AggregateResult::Float(v as f64 / values.len() as f64)
                            }
                        }
                    }
                };
                Ok(result)
            }

            Column::Float64(values) => {
                let n = values.len();
                if n == 0 {
                    return Err(ProcessorError::Parse("empty column".into()));
                }

                let result = match op {
                    AggregateOp::Sum
                    | AggregateOp::Min
                    | AggregateOp::Max
                    | AggregateOp::Avg
                    | AggregateOp::Count => {
                        let v = aggregate_f64_avx2(values, op);
                        match op {
                            AggregateOp::Sum | AggregateOp::Min | AggregateOp::Max => {
                                AggregateResult::Float(v)
                            }
                            AggregateOp::Avg => AggregateResult::Float(v),
                            AggregateOp::Count => AggregateResult::Int(values.len() as i64),
                        }
                    }
                };
                Ok(result)
            }

            Column::Str(_) => Err(ProcessorError::Parse(
                "Cannot aggregate string column directly".into(),
            )),
        }
    }

    /// Group-by aggregation on string column
    ///
    /// # Arguments
    /// * `group_col` - Column name for grouping (must be string)
    /// * `agg_col` - Column name for aggregation (numeric)
    /// * `op` - Aggregate operation
    ///
    /// # Returns
    /// HashMap mapping group keys to aggregation results
    ///
    /// # Example
    /// ```rust
    /// # use columnar_processor::{ColumnarProcessor, AggregateOp};
    /// let grouped = processor.group_by("category", "value", AggregateOp::Avg).unwrap();
    /// ```
    pub fn group_by(
        &self,
        group_col: &str,
        agg_col: &str,
        op: AggregateOp,
    ) -> Result<HashMap<String, AggregateResult>, ProcessorError> {
        let gcol = self.get_col(group_col)?;
        let acol = self.get_col(agg_col)?;

        // only support string group keys
        let offsets_keys = match gcol {
            Column::Str(v) => v,
            _ => return Err(ProcessorError::Parse("group_col must be string".into())),
        };

        match acol {
            Column::Int64(values) => {
                let mut map: HashMap<String, (i128, usize, i64, i64)> = HashMap::new();
                // (sum as i128, count, min, max)
                for i in 0..self.row_count {
                    let (ks, ke) = offsets_keys[i];
                    let key_bytes = self.slice_bytes(ks, ke)?;
                    let key = String::from_utf8_lossy(key_bytes).to_string();
                    let v: i64 = values[i];
                    let entry = map.entry(key).or_insert((0i128, 0usize, v, v));
                    entry.0 += v as i128;
                    entry.1 += 1;
                    if v < entry.2 {
                        entry.2 = v;
                    }
                    if v > entry.3 {
                        entry.3 = v;
                    }
                }
                // convert to AggregateResult
                let mut out: HashMap<String, AggregateResult> = HashMap::new();
                for (k, (sum, cnt, min, max)) in map {
                    let res = match op {
                        AggregateOp::Sum => AggregateResult::Int(sum as i64),
                        AggregateOp::Count => AggregateResult::Int(cnt as i64),
                        AggregateOp::Avg => AggregateResult::Float(sum as f64 / cnt as f64),
                        AggregateOp::Min => AggregateResult::Int(min),
                        AggregateOp::Max => AggregateResult::Int(max),
                    };
                    out.insert(k, res);
                }
                Ok(out)
            }

            Column::Float64(values) => {
                let mut map: HashMap<String, (f64, usize, f64, f64)> = HashMap::new();
                for i in 0..self.row_count {
                    let (ks, ke) = offsets_keys[i];
                    let key_bytes = self.slice_bytes(ks, ke)?;
                    let key = String::from_utf8_lossy(key_bytes).to_string();
                    let v: f64 = values[i];
                    let entry = map.entry(key).or_insert((0.0f64, 0usize, v, v));
                    entry.0 += v;
                    entry.1 += 1;
                    if v < entry.2 {
                        entry.2 = v;
                    }
                    if v > entry.3 {
                        entry.3 = v;
                    }
                }
                let mut out: HashMap<String, AggregateResult> = HashMap::new();
                for (k, (sum, cnt, min, max)) in map {
                    let res = match op {
                        AggregateOp::Sum => AggregateResult::Float(sum),
                        AggregateOp::Count => AggregateResult::Int(cnt as i64),
                        AggregateOp::Avg => AggregateResult::Float(sum / cnt as f64),
                        AggregateOp::Min => AggregateResult::Float(min),
                        AggregateOp::Max => AggregateResult::Float(max),
                    };
                    out.insert(k, res);
                }
                Ok(out)
            }

            _ => Err(ProcessorError::Parse("agg_col must be numeric".into())),
        }
    }

    pub fn to_arrow(&self) -> (Schema, Chunk<Arc<dyn Array>>) {
        let mmap = self.mmap.as_ref().expect("mmap must be present");

        let fields: Vec<Field> = self
            .headers
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let dtype = match self.columns.get(i).unwrap() {
                    Column::Int64(_) => DataType::Int64,
                    Column::Float64(_) => DataType::Float64,
                    Column::Str(_) => DataType::Utf8,
                };
                Field::new(h, dtype, true)
            })
            .collect();

        let schema = Schema::from(fields);

        let arrays: Vec<Arc<dyn Array>> = self
            .headers
            .par_iter()
            .enumerate()
            .map(|(i, _)| {
                let col = self.columns.get(i).unwrap();
                match col {
                    Column::Int64(values) => {
                        let arrow_array = Int64Array::from_vec(values.to_vec());
                        Arc::new(arrow_array) as Arc<dyn Array>
                    }
                    Column::Float64(values) => {
                        let arrow_array = Float64Array::from_vec(values.to_vec());
                        Arc::new(arrow_array) as Arc<dyn Array>
                    }
                    Column::Str(offsets) => {
                        let mut arr = MutableUtf8Array::<i32>::with_capacity(offsets.len());
                        for &(start, end) in offsets {
                            let s = std::str::from_utf8(&mmap[start..end]).unwrap();
                            arr.push(Some(s));
                        }

                        let array: Utf8Array<i32> = arr.into();
                        Arc::new(array) as Arc<dyn Array>
                    }
                }
            })
            .collect();

        (schema, Chunk::new(arrays))
    }

    pub fn get_col(&self, col_name: &str) -> Result<&Column, ProcessorError> {
        let col_pos = self
            .headers
            .iter()
            .position(|cn| cn == col_name)
            .ok_or_else(|| ProcessorError::MissingColumn(col_name.to_string()))?;

        let col = self
            .columns
            .get(col_pos)
            .ok_or_else(|| ProcessorError::MissingColumn(col_name.to_string()))?;

        Ok(col)
    }
}

impl Default for ColumnarProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_processor_from_str(csv: &'_ str) -> ColumnarProcessor {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // write CSV to temp file
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "{}", csv).unwrap();
        let path = tmp.path();

        let mut processor = ColumnarProcessor::new();
        processor.load_csv(path).unwrap();
        processor
    }

    #[test]
    fn test_row_count() {
        let csv = "id,value\n1,10\n2,20\n3,30\n";
        let processor = make_processor_from_str(csv);
        assert_eq!(processor.row_count(), 3);
    }

    #[test]
    fn test_aggregate_sum() {
        let csv = "id,value\n1,10\n2,20\n3,30\n";
        let processor = make_processor_from_str(csv);
        let res = processor.aggregate("value", AggregateOp::Sum).unwrap();
        if let AggregateResult::Int(sum) = res {
            assert_eq!(sum, 60);
        } else {
            panic!("Expected integer sum");
        }
    }

    #[test]
    fn test_filter_greater_than() {
        let csv = "id,value\n1,10\n2,20\n3,30\n";
        let processor = make_processor_from_str(csv);
        let rows = processor
            .filter("value", &FilterPredicate::GreaterThan(Value::Int(15)))
            .unwrap();
        assert_eq!(rows, vec![1, 2]);
    }
}
