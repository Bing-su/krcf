use bincode::{Decode, Encode};
use rcflib::{
    common::divector::DiVector,
    errors::RCFError,
    rcf::{AugmentedRCF, RCFBuilder, RCFLarge, RCFOptionsBuilder},
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

        options.internal_shingling(true).store_pointsum(true);
        options
    }

    pub fn to_rcf(&self) -> Result<RCFLarge<u64, u64>, RCFError> {
        self.to_rcf_builder().build_large_simple()
    }
}

struct RandomCutForest(RCFLarge<u64, u64>);

impl RandomCutForest {
    pub fn new(options: PyRCFOptions) -> Result<Self, RCFError> {
        let rcf = options.to_rcf()?;
        Ok(Self(rcf))
    }

    pub fn update(&mut self, point: &[f32]) -> Result<(), RCFError> {
        self.0.update(point, 0)
    }

    pub fn score(&self, point: &[f32]) -> Result<f64, RCFError> {
        self.0.score(point)
    }

    pub fn displacement_score(&self, point: &[f32]) -> Result<f64, RCFError> {
        self.0.displacement_score(point)
    }

    pub fn attribution(&self, point: &[f32]) -> Result<DiVector, RCFError> {
        self.0.attribution(point)
    }

    pub fn near_neighbor_list(
        &self,
        point: &[f32],
        percentile: usize,
    ) -> Result<Vec<(f64, Vec<f32>, f64)>, RCFError> {
        self.0.near_neighbor_list(point, percentile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rcfoptions() {
        let opts = PyRCFOptions::default();
        assert_eq!(opts.dimensions, 1);
        assert_eq!(opts.shingle_size, 1);
        assert!(opts.num_trees.is_none());
        assert!(opts.sample_size.is_none());
        assert!(opts.output_after.is_none());
        assert!(opts.random_seed.is_none());
        assert!(opts.parallel_execution_enabled.is_none());
        assert!(opts.lambda.is_none());
    }

    #[test]
    fn test_to_rcf_builder() {
        let opts = PyRCFOptions {
            dimensions: 3,
            shingle_size: 2,
            num_trees: Some(50),
            sample_size: Some(128),
            output_after: Some(10),
            random_seed: Some(42),
            parallel_execution_enabled: Some(true),
            lambda: Some(0.01),
        };
        let builder = opts.to_rcf_builder();

        let rcf = builder.build();
        assert!(rcf.is_ok());

        let rcf = builder.build_large_simple::<u64>();
        assert!(rcf.is_ok());
    }

    #[test]
    fn test_to_rcf_returns_ok() {
        let opts = PyRCFOptions {
            dimensions: 2,
            shingle_size: 1,
            num_trees: Some(10),
            sample_size: Some(32),
            output_after: None,
            random_seed: None,
            parallel_execution_enabled: None,
            lambda: None,
        };
        let rcf = opts.to_rcf();
        assert!(rcf.is_ok());
    }
}
