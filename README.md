
# Columnar Processor

**Columnar Processor** is a **high-performance CSV loader, parser, and query engine** written in Rust.
It provides efficient data loading, columnar querying, and aggregation capabilities — designed to rival frameworks like Polars for analytical workloads.

## 🚀 Features

* Ultra-fast **CSV parsing and loading** using a custom columnar engine.
* Efficient **querying and aggregation** by column.
* **Benchmarks** comparing performance against Polars (`cargo bench --bench polars_comparison`).
* **Examples** showing practical usage.
* **Bindings** for Python and Java (via JNI).
* **Executable** for standalone performance measurement.

---

## ⚙️ Building the Project

### 🧱 Default Build

```bash
cargo build --release
```

### 🐍 Build with Python Bindings

```bash
cargo build --release --features python-bindings
maturin develop --release
```

This will build and install a Python wheel for local use.

### ☕ Build with JNI Bindings

```bash
cargo build --release --features jni-binding
```

This generates a shared library usable from Java.

---

## 🧪 Running Benchmarks

Benchmarks are located in the `benches` directory.

```bash
cargo bench
```

### 📊 Example: Polars Comparison

```
cargo bench --bench polars_comparison
```

#### Results (100 million rows)

| Engine                 | Time (s)    | Relative Performance |
| ---------------------- | ----------- | -------------------- |
| **Columnar Processor** | 3.40 – 4.10 | ✅ Faster             |
| **Polars**             | 4.20 – 4.33 |                      |

> The Columnar Processor consistently loads and parses large CSV files **10–20% faster than Polars**, depending on dataset size and machine configuration.

---

## ⚡ Quick Start Example

1. Create a small CSV file:

   ```csv
   id,value
   1,10
   2,20
   3,30
   ```

2. Run an example to load and aggregate:

   ```bash
   cargo run --example aggregate_example ./data.csv
   ```

3. Expected output:

   ```
   Aggregating column "value" with SUM: 60
   ```

---

## 🧩 Profiling with `samply`

You can profile the performance of your executable using [`samply`](https://github.com/mstange/samply):

```bash
cargo build --release
samply -- ./target/release/columnar_processor_executable path/to/data.csv
```

Replace `columnar_processor_executable` with your binary name and `path/to/data.csv` with your input file.

---

## License

Licensed under either of

- Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)
at your option.

### Contribution
Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
```
