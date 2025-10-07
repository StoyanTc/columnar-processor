use arrow2::{
    array::{Array, Float64Array, Int64Array, MutableUtf8Array, Utf8Array},
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
};
use memchr::memchr_iter;
use memmap2::Mmap;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{collections::HashMap, fs::File, path::Path, str, sync::Arc};

use crate::{
    helpers::simd_helpers::{aggregate_f64_avx2, aggregate_i64_avx2, filter_f64, filter_i64},
    processor::{
        AggregateOp, AggregateResult, BatchResult, FilterPredicate, ParseError, ParseSummary,
        ProcessorError, Value,
        column::{Column, ColumnType},
    },
};

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
    pub fn load_csv(&mut self, path: &Path) -> Result<ParseSummary, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let buf: &[u8] = &mmap[..];

        // Parse header
        let header_end = buf
            .iter()
            .position(|&b| b == b'\n')
            .ok_or("Missing header line")?;
        let header_line = &buf[..header_end];
        let headers: Vec<String> = header_line
            .split(|&b| b == b',')
            .map(|s| String::from_utf8_lossy(s).to_string())
            .collect();

        let data_start = header_end + 1;
        let data = &buf[data_start..];

        // Infer schema from first line
        let first_line_end = data
            .iter()
            .position(|&b| b == b'\n')
            .ok_or("No data rows")?;
        let first_line = &data[..first_line_end];
        let schema = Self::infer_schema(first_line, &headers)?;

        // Find chunk boundaries (split by newlines)
        let num_threads = rayon::current_num_threads();
        let chunks = Self::find_chunk_boundaries(data, num_threads);

        // Estimate rows per chunk for preallocation
        let estimated_rows_per_chunk = {
            let avg_line_len = first_line.len() + 1;
            (data.len() / num_threads / avg_line_len) + 1000
        };

        // Parse chunks in parallel
        let data_offset = data_start; // Offset from buffer start to data start

        let batch_results: Vec<BatchResult> = chunks
            .par_iter()
            .enumerate()
            .map(|(chunk_idx, (start, end))| {
                Self::parse_chunk(
                    &data[*start..*end],
                    &schema,
                    &headers,
                    estimated_rows_per_chunk,
                    data_offset + start, // Absolute offset in file
                    chunk_idx,
                )
            })
            .collect();

        // Merge batch results into chunked columns
        let mut columns: Vec<Column> = schema
            .iter()
            .map(|col_type| match col_type {
                ColumnType::Int64 => Column::new_int64(),
                ColumnType::Float64 => Column::new_float64(),
                ColumnType::Str => Column::new_str(),
            })
            .collect();

        let mut total_rows = 0;
        let mut all_errors = Vec::new();

        for mut batch in batch_results {
            total_rows += batch.row_count;
            all_errors.extend(batch.errors);

            // Move each column's data
            for col_idx in 0..columns.len() {
                match &mut columns[col_idx] {
                    Column::Int64(chunks) => {
                        // Move the vec out of batch
                        chunks.push(std::mem::take(&mut batch.int64_batches[col_idx]));
                    }
                    Column::Float64(chunks) => {
                        chunks.push(std::mem::take(&mut batch.float64_batches[col_idx]));
                    }
                    Column::Str(chunks) => {
                        chunks.push(std::mem::take(&mut batch.str_batches[col_idx]));
                    }
                }
            }
        }

        self.mmap = Some(mmap);
        self.columns = columns;
        self.headers = headers;
        self.row_count = total_rows;

        Ok(ParseSummary {
            rows_processed: total_rows,
            errors: all_errors,
        })
    }

    fn infer_schema(
        first_line: &[u8],
        headers: &[String],
    ) -> Result<Vec<ColumnType>, Box<dyn std::error::Error>> {
        let fields: Vec<&[u8]> = first_line.split(|&b| b == b',').collect();

        if fields.len() != headers.len() {
            return Err(format!(
                "Header/data mismatch: {} vs {}",
                headers.len(),
                fields.len()
            )
            .into());
        }

        let schema: Vec<ColumnType> = fields
            .iter()
            .map(|field| {
                if atoi_simd::parse::<i64>(field).is_ok() {
                    ColumnType::Int64
                } else if fast_float::parse::<f64, _>(field).is_ok() {
                    ColumnType::Float64
                } else {
                    ColumnType::Str
                }
            })
            .collect();

        Ok(schema)
    }

    fn find_chunk_boundaries(data: &[u8], num_chunks: usize) -> Vec<(usize, usize)> {
        if data.is_empty() {
            return vec![];
        }

        let chunk_size = data.len() / num_chunks;
        let mut boundaries = Vec::with_capacity(num_chunks);
        let mut start = 0;

        for i in 0..num_chunks - 1 {
            let mut end = (i + 1) * chunk_size;

            // Find next newline
            while end < data.len() && data[end] != b'\n' {
                end += 1;
            }

            if end < data.len() {
                end += 1; // Include the newline
            }

            if start < end {
                boundaries.push((start, end));
            }
            start = end;
        }

        // Last chunk gets everything remaining
        if start < data.len() {
            boundaries.push((start, data.len()));
        }

        boundaries
    }

    fn parse_chunk(
        chunk: &[u8],
        schema: &[ColumnType],
        headers: &[String],
        estimated_rows: usize,
        chunk_offset: usize, // Absolute offset of this chunk in the file
        chunk_idx: usize,
    ) -> BatchResult {
        let num_cols = schema.len();

        // Pre-allocate column batches
        let mut int64_cols: Vec<Vec<i64>> = (0..num_cols)
            .map(|i| {
                if matches!(schema[i], ColumnType::Int64) {
                    Vec::with_capacity(estimated_rows)
                } else {
                    Vec::new()
                }
            })
            .collect();

        let mut float64_cols: Vec<Vec<f64>> = (0..num_cols)
            .map(|i| {
                if matches!(schema[i], ColumnType::Float64) {
                    Vec::with_capacity(estimated_rows)
                } else {
                    Vec::new()
                }
            })
            .collect();

        let mut str_cols: Vec<Vec<(usize, usize)>> = (0..num_cols)
            .map(|i| {
                if matches!(schema[i], ColumnType::Str) {
                    Vec::with_capacity(estimated_rows)
                } else {
                    Vec::new()
                }
            })
            .collect();

        let mut errors = Vec::new();
        let mut row_count = 0;
        let mut fields = Vec::with_capacity(num_cols);

        // Iterate lines
        let mut start = 0;
        for newline_pos in memchr_iter(b'\n', chunk) {
            let line = &chunk[start..newline_pos];
            start = newline_pos + 1;

            if line.is_empty() {
                continue;
            }

            // Calculate absolute line position in file
            let line_offset_in_chunk = line.as_ptr() as usize - chunk.as_ptr() as usize;
            let absolute_line_offset = chunk_offset + line_offset_in_chunk;

            // Split line into fields
            fields.clear();
            let mut field_start = 0;
            for comma_pos in memchr_iter(b',', line) {
                fields.push(&line[field_start..comma_pos]);
                field_start = comma_pos + 1;
            }
            fields.push(&line[field_start..]);

            if fields.len() != num_cols {
                errors.push(ParseError {
                    row: chunk_idx * estimated_rows + row_count + 2,
                    column: "".to_string(),
                    value: format!("Expected {} fields, got {}", num_cols, fields.len()),
                    error: None,
                });
                continue;
            }

            // Parse each field according to schema
            for col_idx in 0..num_cols {
                match schema[col_idx] {
                    ColumnType::Int64 => match atoi_simd::parse::<i64>(fields[col_idx]) {
                        Ok(value) => int64_cols[col_idx].push(value),
                        Err(e) => errors.push(ParseError {
                            row: chunk_idx * estimated_rows + row_count + 2,
                            column: headers[col_idx].clone(),
                            value: String::from_utf8_lossy(fields[col_idx]).to_string(),
                            error: Some(e.to_string()),
                        }),
                    },
                    ColumnType::Float64 => match fast_float::parse::<f64, _>(fields[col_idx]) {
                        Ok(value) => float64_cols[col_idx].push(value),
                        Err(e) => errors.push(ParseError {
                            row: chunk_idx * estimated_rows + row_count + 2,
                            column: headers[col_idx].clone(),
                            value: String::from_utf8_lossy(fields[col_idx]).to_string(),
                            error: Some(e.to_string()),
                        }),
                    },
                    ColumnType::Str => {
                        // Store absolute offset into mmap
                        let field_offset_in_line =
                            fields[col_idx].as_ptr() as usize - line.as_ptr() as usize;
                        let absolute_start = absolute_line_offset + field_offset_in_line;
                        let absolute_end = absolute_start + fields[col_idx].len();
                        str_cols[col_idx].push((absolute_start, absolute_end));
                    }
                }
            }

            row_count += 1;
        }

        BatchResult {
            int64_batches: int64_cols,
            float64_batches: float64_cols,
            str_batches: str_cols,
            row_count,
            errors,
        }
    }

    // Helper to get string value from mmap using offsets
    pub fn get_string(&self, start: usize, end: usize) -> &str {
        if let Some(ref mmap) = self.mmap {
            let bytes = &mmap[start..end];
            std::str::from_utf8(bytes).unwrap_or("")
        } else {
            ""
        }
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn headers(&self) -> &[String] {
        &self.headers
    }

    pub fn column(&self, idx: usize) -> Option<&Column> {
        self.columns.get(idx)
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
                Column::Float64(_),
                FilterPredicate::Equals(_)
                | FilterPredicate::GreaterThan(_)
                | FilterPredicate::LessThan(_)
                | FilterPredicate::Between(_, _),
            ) => Ok(filter_f64(
                &col.iter_f64().collect::<Vec<f64>>(),
                predicate.clone(),
            )),

            (
                Column::Int64(_),
                FilterPredicate::Equals(_)
                | FilterPredicate::GreaterThan(_)
                | FilterPredicate::LessThan(_)
                | FilterPredicate::Between(_, _),
            ) => Ok(filter_i64(
                &col.iter_i64().collect::<Vec<i64>>(),
                predicate.clone(),
            )),

            (Column::Str(offsets), FilterPredicate::Equals(Value::Str(target))) => {
                let mut out = Vec::with_capacity(offsets.len());
                for (i, &(s, e)) in col
                    .iter_str()
                    .collect::<Vec<(usize, usize)>>()
                    .iter()
                    .enumerate()
                {
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
                        let v = aggregate_i64_avx2(&col.iter_i64().collect::<Vec<i64>>(), op);
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
                        let v = aggregate_f64_avx2(&col.iter_f64().collect::<Vec<f64>>(), op);
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
            Column::Str(_) => gcol.iter_str().collect::<Vec<(usize, usize)>>(),
            _ => return Err(ProcessorError::Parse("group_col must be string".into())),
        };

        match acol {
            Column::Int64(_) => {
                let mut map: HashMap<String, (i128, usize, i64, i64)> = HashMap::new();
                // (sum as i128, count, min, max)
                for i in 0..self.row_count {
                    let (ks, ke) = offsets_keys[i];
                    let key_bytes = self.slice_bytes(ks, ke)?;
                    let key = String::from_utf8_lossy(key_bytes).to_string();
                    let v: i64 = acol.iter_i64().collect::<Vec<i64>>()[i];
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

            Column::Float64(_) => {
                let mut map: HashMap<String, (f64, usize, f64, f64)> = HashMap::new();
                for i in 0..self.row_count {
                    let (ks, ke) = offsets_keys[i];
                    let key_bytes = self.slice_bytes(ks, ke)?;
                    let key = String::from_utf8_lossy(key_bytes).to_string();
                    let v: f64 = acol.iter_f64().collect::<Vec<f64>>()[i];
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
                    Column::Int64(_) => {
                        let arrow_array =
                            Int64Array::from_vec(col.iter_i64().collect::<Vec<i64>>());
                        Arc::new(arrow_array) as Arc<dyn Array>
                    }
                    Column::Float64(_) => {
                        let arrow_array =
                            Float64Array::from_vec(col.iter_f64().collect::<Vec<f64>>());
                        Arc::new(arrow_array) as Arc<dyn Array>
                    }
                    Column::Str(offsets) => {
                        let mut arr = MutableUtf8Array::<i32>::with_capacity(offsets.len());
                        for (start, end) in col.iter_str().collect::<Vec<(usize, usize)>>() {
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
