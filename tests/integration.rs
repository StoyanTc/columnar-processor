use std::rc::Rc;

use columnar_processor::processor::{
    AggregateOp, AggregateResult, columnar_processor::ColumnarProcessor, query_builder::QueryCache,
};

#[test]
fn test_group_by_average() {
    let csv = "category,value\nA,10\nB,20\nA,30\n";
    let mut processor = ColumnarProcessor::new();

    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", csv).unwrap();
    processor.load_csv(tmp.path()).unwrap();

    let res = processor
        .group_by("category", "value", AggregateOp::Avg)
        .unwrap();
    assert_eq!(res["A"], AggregateResult::Float(20.0));
    assert_eq!(res["B"], AggregateResult::Float(20.0));
}

#[test]
fn test_cached_aggregate() {
    let csv = "id,value\n1,10\n2,20\n3,30\n";
    let mut processor = ColumnarProcessor::new();
    let cache = QueryCache::new();

    use std::io::Write;
    use tempfile::NamedTempFile;

    let mut tmp = NamedTempFile::new().unwrap();
    write!(tmp, "{}", csv).unwrap();
    processor.load_csv(tmp.path()).unwrap();
    let processor_rc = Rc::new(processor);
    let cache_rc = Rc::new(cache);

    let sum1 = processor_rc
        .query_with_cache(&cache_rc)
        .aggregate("value", AggregateOp::Sum)
        .execute()
        .unwrap();
    let sum2 = processor_rc
        .query_with_cache(&cache_rc)
        .aggregate("value", AggregateOp::Sum)
        .execute()
        .unwrap(); // from cache
    assert_eq!(sum1, sum2);
}
