use pyo3::prelude::*;
mod rcf;

/// A Python module implemented in Rust.
#[pymodule]
fn krcf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
