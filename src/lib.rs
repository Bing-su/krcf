use pyo3::prelude::*;
use rcflib::{
    errors::RCFError,
    rcf::{AugmentedRCF, RCFBuilder, RCFOptionsBuilder},
};

struct PyRCFOptions {
    pub dimensions: usize,
    pub shingle_size: usize,
    pub num_trees: Option<usize>,
    pub sample_size: Option<usize>,
    pub output_after: Option<usize>,
    pub random_seed: Option<u64>,
    pub parallel_execution_enabled: Option<bool>,
    pub lambda: Option<f64>,
}

impl Default for PyRCFOptions {
    fn default() -> Self {
        Self {
            dimensions: 1,
            shingle_size: 1,
            num_trees: None,
            sample_size: None,
            output_after: None,
            random_seed: None,
            parallel_execution_enabled: None,
            lambda: None,
        }
    }
}

impl PyRCFOptions {
    pub fn to_rcf_builder(&self) -> RCFBuilder<u64, u64> {
        let mut options = RCFBuilder::<u64, u64>::new(self.dimensions, self.shingle_size);

        macro_rules! set_option {
            ($opt:expr, $method:ident) => {
                if let Some(val) = $opt {
                    options.$method(val);
                }
            };
        }

        set_option!(self.num_trees, number_of_trees);
        set_option!(self.sample_size, tree_capacity);
        set_option!(self.output_after, output_after);
        set_option!(self.random_seed, random_seed);
        set_option!(self.parallel_execution_enabled, parallel_enabled);
        set_option!(self.lambda, time_decay);

        options
            .internal_shingling(true)
            .store_pointsum(true)
            .store_attributes(true)
            .propagate_attribute_vectors(true);
        options
    }

    pub fn to_rcf(
        &self,
    ) -> Result<Box<(dyn AugmentedRCF<u64, u64> + Send + Sync + 'static)>, RCFError> {
        self.to_rcf_builder().build()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn krcf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
