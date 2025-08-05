use pyo3::prelude::*;
mod pyrcf;

#[pymodule]
fn krcf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<pyrcf::RandomCutForest>()?;
    Ok(())
}
