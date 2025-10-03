# Columnar Processor Variant

This project is a **Columnar Processor variant** that provides efficient CSV data processing with querying and aggregation capabilities. It also includes benchmarks, examples, and a performance measurement executable.

## Features

- Load data from CSV files.
- Query and aggregate columns efficiently.
- Benchmarks located in the `benches` directory.
- Examples demonstrating usage.
- Executable for measuring performance.

## Building the Project

### Default Build

To build the project with default features:

```bash
cargo build --release
```

### Build with Python Bindings

To enable Python bindings:

```bash
cargo build --release --features python-bindings
```

You can then use `maturin` to build a Python wheel:

```bash
maturin develop --release
```

### Build with JNI Bindings

To enable JNI bindings:

```bash
cargo build --release --features jni-binding
```

This will generate a shared library that can be used in Java applications.

## Running Benchmarks

Benchmarks are located in the `benches` directory. To run them:

```bash
cargo bench
```

## Examples

Examples demonstrating usage of the processor can be found in the `examples` directory. You can run an example with:

```bash
cargo run --example example_name
```

Replace `example_name` with the name of the example file without the `.rs` extension.

## Performance Measurement with `samply`

To profile the performance of your executable using `samply`:

```bash
cargo build --release
samply -- ./target/release/columnar_processor_executable path/to/data.csv
```

Replace `columnar_processor_executable` with the actual name of the built executable and `path/to/data.csv` with your CSV file.

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
