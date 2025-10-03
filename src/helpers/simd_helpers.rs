use std::arch::x86_64::{
    __m256i, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _mm256_add_epi64,
    _mm256_add_pd, _mm256_and_pd, _mm256_and_si256, _mm256_castsi256_pd, _mm256_cmp_pd,
    _mm256_cmpeq_epi64, _mm256_cmpgt_epi64, _mm256_loadu_pd, _mm256_loadu_si256, _mm256_max_epi64,
    _mm256_max_pd, _mm256_min_epi64, _mm256_min_pd, _mm256_movemask_pd, _mm256_set1_epi64x,
    _mm256_set1_pd, _mm256_setzero_pd, _mm256_setzero_si256, _mm256_storeu_pd, _mm256_storeu_si256,
};

use crate::processor::{AggregateOp, FilterPredicate, Value};

pub fn aggregate_i64_avx2(values: &[i64], op: AggregateOp) -> i64 {
    if is_x86_feature_detected!("avx2") {
        unsafe { aggregate_i64_avx2_inner(values, op) }
    } else {
        // scalar fallback
        match op {
            AggregateOp::Sum | AggregateOp::Avg => values.iter().copied().sum(),
            AggregateOp::Min => *values.iter().min().unwrap(),
            AggregateOp::Max => *values.iter().max().unwrap(),
            AggregateOp::Count => values.len() as i64,
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn aggregate_i64_avx2_inner(values: &[i64], op: AggregateOp) -> i64 {
    const LANES: usize = 4; // __m256i holds 4 i64s
    let mut sum = _mm256_setzero_si256();
    let mut min = _mm256_set1_epi64x(i64::MAX);
    let mut max = _mm256_set1_epi64x(i64::MIN);

    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let v = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
        sum = _mm256_add_epi64(sum, v);
        min = unsafe { _mm256_min_epi64(min, v) };
        max = unsafe { _mm256_max_epi64(max, v) };
    }

    // horizontal reduction
    let mut sum_arr = [0i64; LANES];
    let mut min_arr = [i64::MAX; LANES];
    let mut max_arr = [i64::MIN; LANES];
    unsafe { _mm256_storeu_si256(sum_arr.as_mut_ptr() as *mut __m256i, sum) };
    unsafe { _mm256_storeu_si256(min_arr.as_mut_ptr() as *mut __m256i, min) };
    unsafe { _mm256_storeu_si256(max_arr.as_mut_ptr() as *mut __m256i, max) };

    let mut total_sum: i64 = sum_arr.iter().sum();
    let mut total_min: i64 = *min_arr.iter().min().unwrap();
    let mut total_max: i64 = *max_arr.iter().max().unwrap();

    for &v in remainder {
        total_sum += v;
        if v < total_min {
            total_min = v;
        }
        if v > total_max {
            total_max = v;
        }
    }

    match op {
        AggregateOp::Sum => total_sum,
        AggregateOp::Min => total_min,
        AggregateOp::Max => total_max,
        AggregateOp::Count => values.len() as i64,
        AggregateOp::Avg => (total_sum as f64 / values.len() as f64) as i64,
    }
}

#[target_feature(enable = "avx2")]
unsafe fn aggregate_f64_avx2_inner(values: &[f64], op: AggregateOp) -> f64 {
    const LANES: usize = 4; // __m256d holds 4 f64s
    let mut sum = _mm256_setzero_pd();
    let mut min = _mm256_set1_pd(f64::INFINITY);
    let mut max = _mm256_set1_pd(f64::NEG_INFINITY);

    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let v = unsafe { _mm256_loadu_pd(chunk.as_ptr()) };
        sum = _mm256_add_pd(sum, v);
        min = _mm256_min_pd(min, v);
        max = _mm256_max_pd(max, v);
    }

    let mut sum_arr = [0f64; LANES];
    let mut min_arr = [f64::INFINITY; LANES];
    let mut max_arr = [f64::NEG_INFINITY; LANES];
    unsafe { _mm256_storeu_pd(sum_arr.as_mut_ptr(), sum) };
    unsafe { _mm256_storeu_pd(min_arr.as_mut_ptr(), min) };
    unsafe { _mm256_storeu_pd(max_arr.as_mut_ptr(), max) };

    let mut total_sum: f64 = sum_arr.iter().sum();
    let mut total_min: f64 = *min_arr
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let mut total_max: f64 = *max_arr
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    for &v in remainder {
        total_sum += v;
        if v < total_min {
            total_min = v;
        }
        if v > total_max {
            total_max = v;
        }
    }

    match op {
        AggregateOp::Sum => total_sum,
        AggregateOp::Min => total_min,
        AggregateOp::Max => total_max,
        AggregateOp::Count => values.len() as f64,
        AggregateOp::Avg => total_sum / values.len() as f64,
    }
}

/// Aggregate over f64 column using AVX2 or scalar fallback
pub fn aggregate_f64_avx2(values: &[f64], op: AggregateOp) -> f64 {
    if is_x86_feature_detected!("avx2") {
        unsafe { aggregate_f64_avx2_inner(values, op) }
    } else {
        match op {
            AggregateOp::Sum | AggregateOp::Avg => values.iter().copied().sum(),
            AggregateOp::Min => *values
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            AggregateOp::Max => *values
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            AggregateOp::Count => values.len() as f64,
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_f64_avx2(values: &[f64], op: FilterPredicate) -> Vec<usize> {
    const LANES: usize = 4; // __m256d holds 4 f64
    let mut out = Vec::with_capacity(values.len());

    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let (v1, v2) = match op {
        FilterPredicate::Equals(Value::Float(t)) => (_mm256_set1_pd(t), _mm256_set1_pd(t)),
        FilterPredicate::GreaterThan(Value::Float(t)) => (_mm256_set1_pd(t), _mm256_set1_pd(0.0)),
        FilterPredicate::LessThan(Value::Float(t)) => (_mm256_set1_pd(t), _mm256_set1_pd(0.0)),
        FilterPredicate::Between(Value::Float(t1), Value::Float(t2)) => {
            (_mm256_set1_pd(t1), _mm256_set1_pd(t2))
        }
        _ => return out, // unsupported
    };

    for (chunk_idx, chunk) in chunks.enumerate() {
        let v = unsafe { _mm256_loadu_pd(chunk.as_ptr()) };
        let mask = match op {
            FilterPredicate::Equals(_) => _mm256_cmp_pd(v, v1, _CMP_EQ_OQ),
            FilterPredicate::GreaterThan(_) => _mm256_cmp_pd(v, v1, _CMP_GT_OQ),
            FilterPredicate::LessThan(_) => _mm256_cmp_pd(v, v1, _CMP_LT_OQ),
            FilterPredicate::Between(_, _) => {
                let gt = _mm256_cmp_pd(v, v1, _CMP_GE_OQ);
                let lt = _mm256_cmp_pd(v, v2, _CMP_LE_OQ);
                _mm256_and_pd(gt, lt)
            }
        };

        let mask_bits = _mm256_movemask_pd(mask);
        for i in 0..LANES {
            if (mask_bits & (1 << i)) != 0 {
                out.push(chunk_idx * LANES + i);
            }
        }
    }

    let base = values.len() - remainder.len();
    for (i, &v) in remainder.iter().enumerate() {
        let keep = match op {
            FilterPredicate::Equals(Value::Float(t)) => v == t,
            FilterPredicate::GreaterThan(Value::Float(t)) => v > t,
            FilterPredicate::LessThan(Value::Float(t)) => v < t,
            FilterPredicate::Between(Value::Float(t1), Value::Float(t2)) => v >= t1 && v <= t2,
            _ => false,
        };
        if keep {
            out.push(base + i);
        }
    }

    out
}

fn filter_f64_scalar(values: &[f64], op: FilterPredicate) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            let keep = match op {
                FilterPredicate::Equals(Value::Float(t)) => v == t,
                FilterPredicate::GreaterThan(Value::Float(t)) => v > t,
                FilterPredicate::LessThan(Value::Float(t)) => v < t,
                FilterPredicate::Between(Value::Float(t1), Value::Float(t2)) => v >= t1 && v <= t2,
                _ => false,
            };
            if keep { Some(i) } else { None }
        })
        .collect()
}

pub fn filter_f64(values: &[f64], op: FilterPredicate) -> Vec<usize> {
    if is_x86_feature_detected!("avx2") {
        unsafe { filter_f64_avx2(values, op) }
    } else {
        filter_f64_scalar(values, op)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn filter_i64_avx2(values: &[i64], op: FilterPredicate) -> Vec<usize> {
    const LANES: usize = 4; // __m256i holds 4 i64
    let mut out = Vec::with_capacity(values.len());

    let chunks = values.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let (v1, v2) = match op {
        FilterPredicate::Equals(Value::Int(t)) => (_mm256_set1_epi64x(t), _mm256_set1_epi64x(t)),
        FilterPredicate::GreaterThan(Value::Int(t)) => {
            (_mm256_set1_epi64x(t), _mm256_setzero_si256())
        }
        FilterPredicate::LessThan(Value::Int(t)) => (_mm256_set1_epi64x(t), _mm256_setzero_si256()),
        FilterPredicate::Between(Value::Int(t1), Value::Int(t2)) => {
            (_mm256_set1_epi64x(t1), _mm256_set1_epi64x(t2))
        }
        _ => return out,
    };

    for (chunk_idx, chunk) in chunks.enumerate() {
        let v = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
        let mask = match op {
            FilterPredicate::Equals(_) => _mm256_cmpeq_epi64(v, v1),
            FilterPredicate::GreaterThan(_) => _mm256_cmpgt_epi64(v, v1),
            FilterPredicate::LessThan(_) => _mm256_cmpgt_epi64(v1, v),
            FilterPredicate::Between(_, _) => {
                let gt = _mm256_cmpgt_epi64(v, v1);
                let lt = _mm256_cmpgt_epi64(v2, v);
                _mm256_and_si256(gt, lt)
            }
        };

        let mask_bits = _mm256_movemask_pd(_mm256_castsi256_pd(mask)); // trick: treat as f64 for movemask
        for i in 0..LANES {
            if (mask_bits & (1 << i)) != 0 {
                out.push(chunk_idx * LANES + i);
            }
        }
    }

    let base = values.len() - remainder.len();
    for (i, &v) in remainder.iter().enumerate() {
        let keep = match op {
            FilterPredicate::Equals(Value::Int(t)) => v == t,
            FilterPredicate::GreaterThan(Value::Int(t)) => v > t,
            FilterPredicate::LessThan(Value::Int(t)) => v < t,
            FilterPredicate::Between(Value::Int(t1), Value::Int(t2)) => v >= t1 && v <= t2,
            _ => false,
        };
        if keep {
            out.push(base + i);
        }
    }

    out
}

fn filter_i64_scalar(values: &[i64], op: FilterPredicate) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            let keep = match op {
                FilterPredicate::Equals(Value::Int(t)) => v == t,
                FilterPredicate::GreaterThan(Value::Int(t)) => v > t,
                FilterPredicate::LessThan(Value::Int(t)) => v < t,
                FilterPredicate::Between(Value::Int(t1), Value::Int(t2)) => v >= t1 && v <= t2,
                _ => false,
            };
            if keep { Some(i) } else { None }
        })
        .collect()
}

pub fn filter_i64(values: &[i64], op: FilterPredicate) -> Vec<usize> {
    if is_x86_feature_detected!("avx2") {
        unsafe { filter_i64_avx2(values, op) }
    } else {
        filter_i64_scalar(values, op)
    }
}
