use lru::LruCache;

use crate::processor::column::Column;
use crate::processor::columnar_processor::ColumnarProcessor;
use crate::processor::{
    AggregateOp, AggregateResult, FilterPredicate, OperationResult, ProcessorError, Value,
};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq, Clone)]
pub enum QueryKey {
    Aggregate {
        column: String,
        op: AggregateOp,
    },
    GroupBy {
        group_col: String,
        agg_col: String,
        op: AggregateOp,
    }, // or encode predicate
    Filter {
        column: String,
        predicate: FilterPredicate,
    },
}

use std::num::NonZeroUsize;
use std::rc::Rc;

#[derive(Debug)]
pub struct QueryCache {
    cache: RefCell<LruCache<QueryKey, OperationResult>>,
}

impl QueryCache {
    pub fn new() -> Self {
        Self {
            cache: RefCell::new(LruCache::new(NonZeroUsize::new(128).unwrap())),
        }
    }

    pub fn get(&self, key: &QueryKey) -> Option<OperationResult> {
        self.cache.borrow().peek(key).cloned()
    }

    pub fn put(&self, key: QueryKey, value: OperationResult) {
        self.cache.borrow_mut().put(key, value);
    }
}

/// Multi-dimensional query results
#[derive(Debug, Clone, PartialEq)]
pub enum QueryResult {
    /// Single aggregation result
    Aggregate(AggregateResult),
    /// Multiple aggregation results
    MultiAggregate(HashMap<String, AggregateResult>),
    /// Single group-by dimension with single aggregation
    GroupBy(HashMap<String, AggregateResult>),
    /// Single group-by dimension with multiple aggregations  
    GroupByMultiAgg(HashMap<String, HashMap<String, AggregateResult>>),
    /// Multiple group-by dimensions (hierRchical)
    MultiGroupBy(HashMap<String, HashMap<String, AggregateResult>>), // 2-level example
    /// Full multi-dimensional cube
    MultiDimensional(MultiDimensionalResult),
    /// Filtered row data
    Select {
        columns: Vec<String>,
        rows: Vec<usize>,
        data: HashMap<String, Vec<Value>>,
    },
}

/// Represents a multi-dimensional aggregation result
#[derive(Debug, Clone, PartialEq)]
pub struct MultiDimensionalResult {
    /// The dimensions used for grouping (e.g., ["region", "category", "month"])
    pub dimensions: Vec<String>,
    /// The aggregations computed (e.g., ["sum_sales", "avg_price", "count"])
    pub measures: Vec<String>,
    /// The actual data: Vec of (dimension_values, metric_values)
    /// dimension_values[i] corresponds to dimensions[i]
    /// metric_values[j] corresponds to metrics[j]
    pub data: Vec<(Vec<String>, HashMap<String, AggregateResult>)>,
}

/// Advanced query builder supporting multiple dimensions and aggregations
#[derive(Debug)]
pub struct QueryBuilder {
    processor: Rc<ColumnarProcessor>,
    cache: Option<Rc<QueryCache>>,
    filters: Vec<(String, FilterPredicate)>,
    group_by_columns: Vec<String>,
    aggregations: Vec<(String, AggregateOp, Option<String>)>, // (column, op, alias)
    select_columns: Vec<String>,
    limit: Option<usize>,
}

impl Clone for QueryBuilder {
    fn clone(&self) -> Self {
        QueryBuilder {
            processor: Rc::clone(&self.processor), // reuse same processor
            cache: self.cache.as_ref().map(Rc::clone), // reuse same cache if present
            filters: self.filters.clone(),
            group_by_columns: self.group_by_columns.clone(),
            aggregations: self.aggregations.clone(),
            select_columns: self.select_columns.clone(),
            limit: self.limit,
        }
    }
}

impl QueryBuilder {
    pub fn new(processor: Rc<ColumnarProcessor>, cache: Option<Rc<QueryCache>>) -> Self {
        Self {
            processor,
            cache,
            filters: Vec::new(),
            group_by_columns: Vec::new(),
            aggregations: Vec::new(),
            select_columns: Vec::new(),
            limit: None,
        }
    }

    /// Add a filter condition
    pub fn filter(mut self, column: &str, predicate: FilterPredicate) -> Self {
        self.filters.push((column.to_string(), predicate));
        self
    }

    /// Add multiple filter conditions
    pub fn filters(mut self, filters: Vec<(&str, FilterPredicate)>) -> Self {
        for (col, pred) in filters {
            self.filters.push((col.to_string(), pred));
        }
        self
    }

    /// Add a single group-by column
    pub fn group_by(mut self, column: &str) -> Self {
        self.group_by_columns.push(column.to_string());
        self
    }

    /// Add multiple group-by columns for multi-dimensional analysis
    pub fn group_by_multi(mut self, columns: Vec<&str>) -> Self {
        for col in columns {
            self.group_by_columns.push(col.to_string());
        }
        self
    }

    /// Add an aggregation with optional alias
    pub fn aggregate(mut self, column: &str, op: AggregateOp) -> Self {
        self.aggregations.push((column.to_string(), op, None));
        self
    }

    /// Add an aggregation with a custom alias
    pub fn aggregate_as(mut self, column: &str, op: AggregateOp, alias: &str) -> Self {
        self.aggregations
            .push((column.to_string(), op, Some(alias.to_string())));
        self
    }

    /// Add multiple aggregations at once
    pub fn aggregates(mut self, aggs: Vec<(&str, AggregateOp)>) -> Self {
        for (col, op) in aggs {
            self.aggregations.push((col.to_string(), op, None));
        }
        self
    }

    /// Select specific columns to return
    pub fn select(mut self, columns: Vec<&str>) -> Self {
        self.select_columns = columns.into_iter().map(|s| s.to_string()).collect();
        self
    }

    /// Limit number of results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Disable caching
    pub fn no_cache(mut self) -> Self {
        self.cache = None;
        self
    }

    /// Execute the multi-dimensional query
    pub fn execute(self) -> Result<QueryResult, ProcessorError> {
        // Apply filters first
        let filtered_rows = self.apply_filters()?;

        // Determine query type and execute
        match (
            self.group_by_columns.len(),
            self.aggregations.len(),
            self.select_columns.len(),
        ) {
            // No grouping, single aggregation
            (0, 1, 0) => {
                let (col, op, _) = &self.aggregations[0];
                let result = self.execute_simple_aggregation(col, *op, &filtered_rows)?;
                Ok(QueryResult::Aggregate(result))
            }

            // No grouping, multiple aggregations
            (0, agg_count, 0) if agg_count > 1 => {
                let results = self.execute_multi_aggregation(&filtered_rows)?;
                Ok(QueryResult::MultiAggregate(results))
            }

            // Single group-by, single aggregation
            (1, 1, 0) => {
                let group_col = &self.group_by_columns[0];
                let (agg_col, agg_op, _) = &self.aggregations[0];
                let result =
                    self.execute_single_group_by(group_col, agg_col, *agg_op, &filtered_rows)?;
                Ok(QueryResult::GroupBy(result))
            }

            // Single group-by, multiple aggregations
            (1, agg_count, 0) if agg_count > 1 => {
                let group_col = &self.group_by_columns[0];
                let result = self.execute_single_group_by_multi_agg(group_col, &filtered_rows)?;
                Ok(QueryResult::GroupByMultiAgg(result))
            }

            // Multiple group-by columns (multi-dimensional)
            (group_count, agg_count, 0) if group_count > 1 && agg_count >= 1 => {
                let result = self.execute_multi_dimensional(&filtered_rows)?;
                Ok(QueryResult::MultiDimensional(result))
            }

            // Select columns
            (0, 0, _) if !self.select_columns.is_empty() => {
                let rows =
                    filtered_rows.unwrap_or_else(|| (0..self.processor.row_count()).collect());
                let result = self.select_data(rows)?;
                Ok(result)
            }

            // Just filtering
            (0, 0, 0) => {
                let rows =
                    filtered_rows.unwrap_or_else(|| (0..self.processor.row_count()).collect());
                Ok(QueryResult::Select {
                    columns: vec!["row_index".to_string()],
                    rows: rows.clone(),
                    data: {
                        let mut data = HashMap::new();
                        data.insert(
                            "row_index".to_string(),
                            rows.into_iter().map(|i| Value::Int(i as i64)).collect(),
                        );
                        data
                    },
                })
            }

            _ => Err(ProcessorError::Parse("Invalid query combination".into())),
        }
    }

    /// Apply all filters and return filtered row indices
    fn apply_filters(&self) -> Result<Option<Vec<usize>>, ProcessorError> {
        let mut filtered_rows: Option<Vec<usize>> = None;

        for (column, predicate) in &self.filters {
            let current_filter = match &self.cache {
                Some(lru) => {
                    let key = QueryKey::Filter {
                        column: column.to_string(),
                        predicate: predicate.clone(),
                    };
                    if let Some(OperationResult::Filter(result)) = lru.get(&key) {
                        result
                    } else {
                        let result = self.processor.filter(column, predicate)?;
                        lru.put(key, OperationResult::Filter(result.clone()));
                        result
                    }
                }
                None => self.processor.filter(column, predicate)?,
            };

            filtered_rows = Some(match filtered_rows {
                None => current_filter,
                Some(existing) => intersect_sorted_vecs(existing, current_filter),
            });
        }

        Ok(filtered_rows)
    }

    /// Execute simple aggregation on filtered data
    fn execute_simple_aggregation(
        &self,
        column: &str,
        op: AggregateOp,
        filtered_rows: &Option<Vec<usize>>,
    ) -> Result<AggregateResult, ProcessorError> {
        match filtered_rows {
            Some(rows) => self.filtered_aggregate(column, op, rows.clone()),
            None => match &self.cache {
                Some(lru) => {
                    let key = QueryKey::Aggregate {
                        column: column.to_string(),
                        op,
                    };
                    if let Some(OperationResult::Aggregate(result)) = lru.get(&key) {
                        Ok(result)
                    } else {
                        let result = self.processor.aggregate(column, op)?;
                        lru.put(key, OperationResult::Aggregate(result.clone()));
                        Ok(result)
                    }
                }
                None => Ok(self.processor.aggregate(column, op)?),
            },
        }
    }

    /// Execute multiple aggregations
    fn execute_multi_aggregation(
        &self,
        filtered_rows: &Option<Vec<usize>>,
    ) -> Result<HashMap<String, AggregateResult>, ProcessorError> {
        let mut results = HashMap::new();

        for (col, op, alias) in &self.aggregations {
            let result = self.execute_simple_aggregation(col, *op, filtered_rows)?;
            let key = alias
                .clone()
                .unwrap_or_else(|| format!("{}_{:?}", col, op).to_lowercase());
            results.insert(key, result);
        }

        Ok(results)
    }

    /// Execute single group-by with single aggregation
    fn execute_single_group_by(
        &self,
        group_col: &str,
        agg_col: &str,
        agg_op: AggregateOp,
        filtered_rows: &Option<Vec<usize>>,
    ) -> Result<HashMap<String, AggregateResult>, ProcessorError> {
        match filtered_rows {
            Some(rows) => self.filtered_group_by(group_col, agg_col, agg_op, rows.clone()),
            None => match &self.cache {
                Some(lru) => {
                    let key = QueryKey::GroupBy {
                        group_col: group_col.to_string(),
                        agg_col: agg_col.to_string(),
                        op: agg_op,
                    };
                    if let Some(OperationResult::GroupBy(res)) = lru.get(&key) {
                        Ok(res.clone())
                    } else {
                        let result = self.processor.group_by(group_col, agg_col, agg_op)?;
                        lru.put(key, OperationResult::GroupBy(result.clone()));
                        Ok(result)
                    }
                }
                None => Ok(self.processor.group_by(group_col, agg_col, agg_op)?),
            },
        }
    }

    /// Execute single group-by with multiple aggregations
    fn execute_single_group_by_multi_agg(
        &self,
        group_col: &str,
        filtered_rows: &Option<Vec<usize>>,
    ) -> Result<HashMap<String, HashMap<String, AggregateResult>>, ProcessorError> {
        let gcol = self.processor.get_col(group_col)?;
        let offsets_keys = match gcol {
            Column::Str(_) => gcol.iter_str().collect::<Vec<(usize, usize)>>(),
            _ => return Err(ProcessorError::Parse("Group column must be string".into())),
        };

        let rows = filtered_rows
            .as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| (0..self.processor.row_count()).collect::<Vec<_>>());

        // Group data by key first
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

        for row_idx in rows {
            let (ks, ke) = offsets_keys[row_idx];
            let key_bytes = self.processor.slice_bytes(ks, ke)?;
            let key = String::from_utf8_lossy(key_bytes).to_string();
            groups.entry(key).or_default().push(row_idx);
        }

        // Apply all aggregations to each group
        let mut result: HashMap<String, HashMap<String, AggregateResult>> = HashMap::new();

        for (group_key, group_rows) in groups {
            let mut group_aggs = HashMap::new();

            for (agg_col, agg_op, alias) in &self.aggregations {
                let agg_result = self.aggregate_rows(agg_col, *agg_op, &group_rows)?;
                let metric_name = alias
                    .clone()
                    .unwrap_or_else(|| format!("{}_{:?}", agg_col, agg_op).to_lowercase());
                group_aggs.insert(metric_name, agg_result);
            }

            result.insert(group_key, group_aggs);
        }

        Ok(result)
    }

    /// Execute multi-dimensional analysis (multiple group-by columns)
    fn execute_multi_dimensional(
        &self,
        filtered_rows: &Option<Vec<usize>>,
    ) -> Result<MultiDimensionalResult, ProcessorError> {
        let rows = filtered_rows
            .as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| (0..self.processor.row_count()).collect::<Vec<_>>());

        // Get all group-by columns (must be strings)
        let group_cols: Result<Vec<_>, _> = self
            .group_by_columns
            .iter()
            .map(|col_name| {
                let col = self.processor.get_col(col_name)?;
                match col {
                    Column::Str(_) => Ok((
                        col_name.clone(),
                        col.iter_str().collect::<Vec<(usize, usize)>>(),
                    )),
                    _ => Err(ProcessorError::Parse(format!(
                        "Group-by column '{}' must be string",
                        col_name
                    ))),
                }
            })
            .collect();
        let group_cols = group_cols?;

        // Create multi-dimensional groups
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row_idx in rows {
            let mut group_key = Vec::new();

            for (_, offsets) in &group_cols {
                let (ks, ke) = offsets[row_idx];
                let key_bytes = self.processor.slice_bytes(ks, ke)?;
                let key = String::from_utf8_lossy(key_bytes).to_string();
                group_key.push(key);
            }

            groups.entry(group_key).or_default().push(row_idx);
        }

        // Apply aggregations to each group
        let mut data = Vec::new();

        for (group_key, group_rows) in groups {
            let mut measures = HashMap::new();

            for (agg_col, agg_op, alias) in &self.aggregations {
                let result = self.aggregate_rows(agg_col, *agg_op, &group_rows)?;
                let metric_name = alias
                    .clone()
                    .unwrap_or_else(|| format!("{}_{:?}", agg_col, agg_op).to_lowercase());
                measures.insert(metric_name, result);
            }

            data.push((group_key, measures));
        }

        // Create metric names list
        let measure_names: Vec<String> = self
            .aggregations
            .iter()
            .map(|(col, op, alias)| {
                alias
                    .clone()
                    .unwrap_or_else(|| format!("{}_{:?}", col, op).to_lowercase())
            })
            .collect();

        Ok(MultiDimensionalResult {
            dimensions: self.group_by_columns.clone(),
            measures: measure_names,
            data,
        })
    }

    /// Helper to aggregate specific rows for a column
    fn aggregate_rows(
        &self,
        column: &str,
        op: AggregateOp,
        rows: &[usize],
    ) -> Result<AggregateResult, ProcessorError> {
        let col = self.processor.get_col(column)?;

        match col {
            Column::Int64(_) => {
                let filtered_values: Vec<i64> = rows
                    .iter()
                    .map(|&i| col.iter_i64().collect::<Vec<i64>>()[i])
                    .collect();
                self.aggregate_int_values(&filtered_values, op)
            }
            Column::Float64(_) => {
                let filtered_values: Vec<f64> = rows
                    .iter()
                    .map(|&i| col.iter_f64().collect::<Vec<f64>>()[i])
                    .collect();
                self.aggregate_float_values(&filtered_values, op)
            }
            _ => Err(ProcessorError::Parse(
                "Cannot aggregate string column".into(),
            )),
        }
    }

    /// Helper for filtered aggregation
    fn filtered_aggregate(
        &self,
        column: &str,
        op: AggregateOp,
        rows: Vec<usize>,
    ) -> Result<AggregateResult, ProcessorError> {
        let col = self.processor.get_col(column)?;

        match col {
            Column::Int64(_) => {
                let filtered_values: Vec<i64> = rows
                    .iter()
                    .map(|&i| col.iter_i64().collect::<Vec<i64>>()[i])
                    .collect();
                let result = self.aggregate_int_values(&filtered_values, op)?;
                Ok(result)
            }
            Column::Float64(_) => {
                let filtered_values: Vec<f64> = rows
                    .iter()
                    .map(|&i| col.iter_f64().collect::<Vec<f64>>()[i])
                    .collect();
                let result = self.aggregate_float_values(&filtered_values, op)?;
                Ok(result)
            }
            _ => Err(ProcessorError::Parse(
                "Cannot aggregate string column".into(),
            )),
        }
    }

    /// Helper for filtered group by
    fn filtered_group_by(
        &self,
        group_col: &str,
        agg_col: &str,
        op: AggregateOp,
        rows: Vec<usize>,
    ) -> Result<HashMap<String, AggregateResult>, ProcessorError> {
        let gcol = self.processor.get_col(group_col)?;
        let acol = self.processor.get_col(agg_col)?;

        // Only support string group keys
        let offsets_keys = match gcol {
            Column::Str(_) => gcol.iter_str().collect::<Vec<(usize, usize)>>(),
            _ => return Err(ProcessorError::Parse("group_col must be string".into())),
        };

        match acol {
            Column::Int64(_) => {
                let mut map: HashMap<String, Vec<i64>> = HashMap::new();

                for &row_idx in &rows {
                    let (ks, ke) = offsets_keys[row_idx];
                    let key_bytes = self.processor.slice_bytes(ks, ke)?;
                    let key = String::from_utf8_lossy(key_bytes).to_string();
                    let value = acol.iter_i64().collect::<Vec<i64>>()[row_idx];

                    map.entry(key).or_default().push(value);
                }

                let mut result: HashMap<String, AggregateResult> = HashMap::new();
                for (key, group_values) in map {
                    let agg_result = self.aggregate_int_values(&group_values, op)?;
                    result.insert(key, agg_result);
                }

                Ok(result)
            }
            Column::Float64(_) => {
                let mut map: HashMap<String, Vec<f64>> = HashMap::new();

                for &row_idx in &rows {
                    let (ks, ke) = offsets_keys[row_idx];
                    let key_bytes = self.processor.slice_bytes(ks, ke)?;
                    let key = String::from_utf8_lossy(key_bytes).to_string();
                    let value = acol.iter_f64().collect::<Vec<f64>>()[row_idx];

                    map.entry(key).or_default().push(value);
                }

                let mut result: HashMap<String, AggregateResult> = HashMap::new();
                for (key, group_values) in map {
                    let agg_result = self.aggregate_float_values(&group_values, op)?;
                    result.insert(key, agg_result);
                }

                Ok(result)
            }
            _ => Err(ProcessorError::Parse("agg_col must be numeric".into())),
        }
    }

    /// Helper for selecting data
    fn select_data(&self, rows: Vec<usize>) -> Result<QueryResult, ProcessorError> {
        let mut data: HashMap<String, Vec<Value>> = HashMap::new();

        for col_name in &self.select_columns {
            let col = self.processor.get_col(col_name)?;
            let mut column_data = Vec::with_capacity(rows.len());

            for &row_idx in &rows {
                let value = match col {
                    Column::Int64(_) => Value::Int(col.iter_i64().collect::<Vec<i64>>()[row_idx]),
                    Column::Float64(_) => {
                        Value::Float(col.iter_f64().collect::<Vec<f64>>()[row_idx])
                    }
                    Column::Str(_) => {
                        let (start, end) = col.iter_str().collect::<Vec<(usize, usize)>>()[row_idx];
                        let bytes = self.processor.slice_bytes(start, end)?;
                        let s = String::from_utf8_lossy(bytes).to_string();
                        Value::Str(s)
                    }
                };
                column_data.push(value);
            }

            data.insert(col_name.clone(), column_data);
        }

        Ok(QueryResult::Select {
            columns: self.select_columns.clone(),
            rows,
            data,
        })
    }

    /// Helper to aggregate integer values
    fn aggregate_int_values(
        &self,
        values: &[i64],
        op: AggregateOp,
    ) -> Result<AggregateResult, ProcessorError> {
        if values.is_empty() {
            return Err(ProcessorError::Parse(
                "Cannot aggregate empty values".into(),
            ));
        }

        match op {
            AggregateOp::Sum => Ok(AggregateResult::Int(values.iter().sum())),
            AggregateOp::Count => Ok(AggregateResult::Int(values.len() as i64)),
            AggregateOp::Avg => {
                let sum: i64 = values.iter().sum();
                Ok(AggregateResult::Float(sum as f64 / values.len() as f64))
            }
            AggregateOp::Min => Ok(AggregateResult::Int(*values.iter().min().unwrap())),
            AggregateOp::Max => Ok(AggregateResult::Int(*values.iter().max().unwrap())),
        }
    }

    /// Helper to aggregate float values
    fn aggregate_float_values(
        &self,
        values: &[f64],
        op: AggregateOp,
    ) -> Result<AggregateResult, ProcessorError> {
        if values.is_empty() {
            return Err(ProcessorError::Parse(
                "Cannot aggregate empty values".into(),
            ));
        }

        match op {
            AggregateOp::Sum => Ok(AggregateResult::Float(values.iter().sum())),
            AggregateOp::Count => Ok(AggregateResult::Int(values.len() as i64)),
            AggregateOp::Avg => {
                let sum: f64 = values.iter().sum();
                Ok(AggregateResult::Float(sum / values.len() as f64))
            }
            AggregateOp::Min => Ok(AggregateResult::Float(
                values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            )),
            AggregateOp::Max => Ok(AggregateResult::Float(
                values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            )),
        }
    }
}

/// Helper function to intersect sorted vectors
fn intersect_sorted_vecs(mut a: Vec<usize>, mut b: Vec<usize>) -> Vec<usize> {
    a.sort_unstable();
    b.sort_unstable();

    let mut result = Vec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }

    result
}

impl ColumnarProcessor {
    pub fn query(self: &Rc<Self>) -> QueryBuilder {
        QueryBuilder::new(self.clone(), None)
    }

    pub fn query_with_cache(self: &Rc<Self>, cache: &Rc<QueryCache>) -> QueryBuilder {
        QueryBuilder::new(self.clone(), Some(cache.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_dimensional_query() {
        let processor = Rc::new(make_test_processor());

        // Example: Sales analysis by region and product category
        let result = processor
            .query()
            .filter(
                "date",
                FilterPredicate::Between(
                    Value::Str("2024-01-01".to_string()),
                    Value::Str("2024-12-31".to_string()),
                ),
            )
            .group_by_multi(vec!["region", "category"])
            .aggregates(vec![
                ("sales", AggregateOp::Sum),
                ("quantity", AggregateOp::Sum),
                ("price", AggregateOp::Avg),
            ])
            .execute()
            .unwrap();

        match result {
            QueryResult::MultiDimensional(cube) => {
                assert_eq!(cube.dimensions, vec!["region", "category"]);
                assert_eq!(cube.measures.len(), 3);
                assert!(!cube.data.is_empty());

                // Each data point should have values for all dimensions and metrics
                for (dims, metrics) in &cube.data {
                    assert_eq!(dims.len(), 2); // region, category
                    assert_eq!(metrics.len(), 3); // sum_sales, sum_quantity, avg_price
                }
            }
            _ => panic!("Expected MultiDimensional result"),
        }
    }

    fn make_test_processor() -> ColumnarProcessor {
        // Test implementation
        ColumnarProcessor::new()
    }
}
