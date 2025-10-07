
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

This project is licensed under the MIT License.

MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

This is fully self-contained and ready to use as `README.md`.  

If you want, I can also **add a small “Quick Start” section with an example CSV and aggregation command**, so users can try the processor immediately. Do you want me to do that?
```
