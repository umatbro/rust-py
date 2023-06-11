# Setup

1. Install maturin
    ```bash
    poetry add maturin
    ```
2. Init maturin - https://pyo3.rs/v0.17.1/getting_started#adding-to-an-existing-project\
    ```bash
    maturin new tmp
    ```
3. Copy contents of `pyproject.toml` to root `pyproject.toml`.
4. Copy `src` and `Cargo.toml` to root directory.

# Implementation
1. Show cargo
   * `Cargo.toml`
   * Shot that maturin added pyo3 as dependency
   * Add dependency - `rstest`
* View docs: https://pyo3.rs/v0.19.0/class
* Add `#[pyclass]` and `struct RsMatrix`.

# Build
1. Run `maturin develop` and benchmark.
2. Run `maturin develop --release` and benchmark.