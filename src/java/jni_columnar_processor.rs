//#![cfg(feature = "jni-bindings")]

use jni::JNIEnv;
use jni::objects::{JClass, JObject, JString};
use jni::sys::{jint, jlong, jobjectArray};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::processor::{AggregateResult, ProcessorError, columnar_processor::ColumnarProcessor};

lazy_static::lazy_static! {
    static ref PROCESSORS: Mutex<HashMap<jlong, ColumnarProcessor>> = Mutex::new(HashMap::new());
}

// Helper to generate IDs for processors
fn next_id() -> jlong {
    use std::sync::atomic::{AtomicI64, Ordering};
    static COUNTER: AtomicI64 = AtomicI64::new(1);
    COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// JNI function to create a new ColumnarProcessor
#[no_mangle]
pub extern "system" fn Java_com_example_columnar_ColumnarProcessor_nativeNew(
    env: JNIEnv,
    _class: JClass,
) -> jlong {
    let cp = ColumnarProcessor::new();
    let id = next_id();
    PROCESSORS.lock().unwrap().insert(id, cp);
    id
}

/// JNI function to load CSV
#[no_mangle]
pub extern "system" fn Java_com_example_columnar_ColumnarProcessor_nativeLoadCsv(
    env: JNIEnv,
    _class: JClass,
    processor_id: jlong,
    jpath: JString,
) -> jint {
    let path: String = match env.get_string(&jpath) {
        Ok(s) => s.into(),
        Err(_) => return -1,
    };

    let mut processors = PROCESSORS.lock().unwrap();
    if let Some(proc) = processors.get_mut(&processor_id) {
        match proc.load_csv(std::path::Path::new(&path)) {
            Ok(_) => 0,
            Err(_) => -2,
        }
    } else {
        -3
    }
}

/// JNI function to get row count
#[no_mangle]
pub extern "system" fn Java_com_example_columnar_ColumnarProcessor_nativeRowCount(
    env: JNIEnv,
    _class: JClass,
    processor_id: jlong,
) -> jint {
    let processors = PROCESSORS.lock().unwrap();
    if let Some(proc) = processors.get(&processor_id) {
        proc.row_count() as jint
    } else {
        -1
    }
}
