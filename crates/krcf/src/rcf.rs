use multiversion::multiversion;
use rcflib::{
    common::{
        directionaldensity::InterpolationMeasure, divector::DiVector, rangevector::RangeVector,
    },
    errors::RCFError,
    rcf::{AugmentedRCF, RCFBuilder, RCFLarge, RCFOptionsBuilder},
};

/// Represents the configuration options for creating a Random Cut Forest.
///
/// This struct allows for detailed configuration of the RCF, including its dimensions,
/// capacity, and behavior. It is used with `RandomCutForest::new` to initialize a forest.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomCutForestOptions {
    /// The number of features in each data point. This is a mandatory parameter.
    pub dimensions: usize,
    /// The number of consecutive data points to be combined into a single point in the forest.
    /// This is useful for time-series data and is a mandatory parameter.
    pub shingle_size: usize,
    /// A unique identifier for the forest.
    pub id: Option<u64>,
    /// The number of trees in the forest. A larger number of trees can improve accuracy.
    pub num_trees: Option<usize>,
    /// The maximum number of points stored in each tree. This is also known as the sample size.
    pub sample_size: Option<usize>,
    /// The number of points that must be processed before the forest is ready to provide anomaly scores.
    pub output_after: Option<usize>,
    /// A seed for the random number generator to ensure reproducibility.
    pub random_seed: Option<u64>,
    /// Enables or disables parallel execution for the RCF operations.
    pub parallel_execution_enabled: Option<bool>,
    /// A decay factor for the importance of older points. Helps the model adapt to changing data patterns.
    pub lambda: Option<f64>,
    /// Enables or disables internal rotation of data points to improve performance.
    pub internal_rotation: Option<bool>,
    /// Enables or disables automatic shingling of the input data stream.
    pub internal_shingling: Option<bool>,
    /// Determines whether to propagate attribute vectors through the forest.
    pub propagate_attribute_vectors: Option<bool>,
    /// Determines whether to store the sum of points in the trees.
    pub store_pointsum: Option<bool>,
    /// Determines whether to store attributes associated with the points.
    pub store_attributes: Option<bool>,
    /// The fraction of initial points to accept unconditionally. (0, 1]
    pub initial_accept_fraction: Option<f64>,
    /// The fraction of the bounding box cache to use. [0, 1]
    pub bounding_box_cache_fraction: Option<f64>,
}

impl Default for RandomCutForestOptions {
    fn default() -> Self {
        Self {
            dimensions: 1,
            shingle_size: 1,
            id: None,
            num_trees: None,
            sample_size: None,
            output_after: None,
            random_seed: None,
            parallel_execution_enabled: None,
            lambda: None,
            internal_rotation: None,
            internal_shingling: None,
            propagate_attribute_vectors: None,
            store_pointsum: None,
            store_attributes: None,
            initial_accept_fraction: None,
            bounding_box_cache_fraction: None,
        }
    }
}

impl Into<RCFBuilder> for RandomCutForestOptions {
    fn into(self) -> RCFBuilder {
        self.to_rcf_builder()
    }
}

impl RandomCutForestOptions {
    /// Converts the `RandomCutForestOptions` into an `RCFBuilder`.
    ///
    /// This method facilitates the construction of a Random Cut Forest by applying the specified
    /// options to an `RCFBuilder` from the underlying `rcflib`.
    pub fn to_rcf_builder(&self) -> RCFBuilder {
        let mut options = RCFBuilder::new(self.dimensions, self.shingle_size);

        macro_rules! set_option {
            ($opt:expr, $method:ident) => {
                if let Some(val) = $opt {
                    options.$method(val);
                }
            };
        }

        set_option!(self.id, id);
        set_option!(self.num_trees, number_of_trees);
        set_option!(self.sample_size, tree_capacity);
        set_option!(self.output_after, output_after);
        set_option!(self.random_seed, random_seed);
        set_option!(self.parallel_execution_enabled, parallel_enabled);
        set_option!(self.lambda, time_decay);

        set_option!(self.internal_rotation, internal_rotation);
        set_option!(self.internal_shingling, internal_shingling);
        set_option!(
            self.propagate_attribute_vectors,
            propagate_attribute_vectors
        );
        set_option!(self.store_pointsum, store_pointsum);
        set_option!(self.store_attributes, store_attributes);
        set_option!(self.initial_accept_fraction, initial_accept_fraction);
        set_option!(
            self.bounding_box_cache_fraction,
            bounding_box_cache_fraction
        );

        options
    }

    /// Builds a `RCFLarge` instance from the options.
    ///
    /// This is a convenience method that uses `to_rcf_builder` to construct and then build
    /// a simple, large-scale Random Cut Forest.
    pub fn to_rcf(&self) -> Result<RCFLarge<u64, u64>, RCFError> {
        self.to_rcf_builder().build_large_simple()
    }
}

/// A wrapper around `rcflib::rcf::RCFLarge` that provides a high-level API for the Random Cut Forest algorithm.
///
/// This struct is the main entry point for interacting with the RCF. It supports serialization and deserialization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomCutForest(RCFLarge<u64, u64>);

#[multiversion(targets = "simd")]
fn update(rcf: &mut RandomCutForest, point: &[f32]) -> Result<(), RCFError> {
    rcf.0.update(point, 0)
}

#[multiversion(targets = "simd")]
fn score(rcf: &RandomCutForest, point: &[f32]) -> Result<f64, RCFError> {
    rcf.0.score(point)
}

#[multiversion(targets = "simd")]
fn displacement_score(rcf: &RandomCutForest, point: &[f32]) -> Result<f64, RCFError> {
    rcf.0.displacement_score(point)
}

#[multiversion(targets = "simd")]
fn attribution(rcf: &RandomCutForest, point: &[f32]) -> Result<DiVector, RCFError> {
    rcf.0.attribution(point)
}

#[multiversion(targets = "simd")]
fn near_neighbor_list(
    rcf: &RandomCutForest,
    point: &[f32],
    percentile: usize,
) -> Result<Vec<(f64, Vec<f32>, f64)>, RCFError> {
    rcf.0.near_neighbor_list(point, percentile)
}

#[multiversion(targets = "simd")]
fn density(rcf: &RandomCutForest, point: &[f32]) -> Result<f64, RCFError> {
    rcf.0.density(point)
}

#[multiversion(targets = "simd")]
fn directional_density(rcf: &RandomCutForest, point: &[f32]) -> Result<DiVector, RCFError> {
    rcf.0.directional_density(point)
}

#[multiversion(targets = "simd")]
fn density_interpolant(
    rcf: &RandomCutForest,
    point: &[f32],
) -> Result<InterpolationMeasure, RCFError> {
    rcf.0.density_interpolant(point)
}

#[multiversion(targets = "simd")]
pub fn extrapolate(rcf: &RandomCutForest, look_ahead: usize) -> Result<RangeVector<f32>, RCFError> {
    rcf.0.extrapolate(look_ahead)
}

impl RandomCutForest {
    /// Creates a new `RandomCutForest` with the specified options.
    ///
    /// # Arguments
    ///
    /// * `options` - A `RandomCutForestOptions` struct that defines the configuration of the forest.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `RandomCutForest` instance or an `RCFError` if creation fails.
    pub fn new(options: RandomCutForestOptions) -> Result<Self, RCFError> {
        let rcf = options.to_rcf()?;
        Ok(Self(rcf))
    }

    /// Transforms a single-dimensional point into a shingled point.
    ///
    /// # Arguments
    ///
    /// * `point` - A slice of f32 representing the data point.
    ///
    /// # Returns
    ///
    /// A `Result` containing the shingled point as a `Vec<f32>` or an `RCFError`.
    pub fn shingled_point(&self, point: &[f32]) -> Result<Vec<f32>, RCFError> {
        self.0.shingled_point(point)
    }

    /// Updates the forest with a new data point.
    ///
    /// # Arguments
    ///
    /// * `point` - A slice of f32 representing the data point to add to the forest.
    pub fn update(&mut self, point: &[f32]) -> Result<(), RCFError> {
        update(self, point)
    }

    /// Computes the anomaly score for a given data point.
    ///
    /// A higher score indicates a higher likelihood of the point being an anomaly.
    ///
    /// # Arguments
    ///
    /// * `point` - The data point to score.
    pub fn score(&self, point: &[f32]) -> Result<f64, RCFError> {
        score(self, point)
    }

    /// Computes the displacement score for a given data point.
    ///
    /// This score measures how much the model's structure would change if the point were added.
    ///
    /// # Arguments
    ///
    /// * `point` - The data point to score.
    pub fn displacement_score(&self, point: &[f32]) -> Result<f64, RCFError> {
        displacement_score(self, point)
    }

    /// Computes the attribution of the anomaly score to each dimension of the data point.
    ///
    /// # Arguments
    ///
    /// * `point` - The data point for which to compute attribution.
    pub fn attribution(&self, point: &[f32]) -> Result<DiVector, RCFError> {
        attribution(self, point)
    }

    /// Finds a list of near neighbors for a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - The point to find neighbors for.
    /// * `percentile` - The percentile of neighbors to return.
    pub fn near_neighbor_list(
        &self,
        point: &[f32],
        percentile: usize,
    ) -> Result<Vec<(f64, Vec<f32>, f64)>, RCFError> {
        near_neighbor_list(self, point, percentile)
    }

    /// Estimates the density of the data at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - The point at which to estimate density.
    pub fn density(&self, point: &[f32]) -> Result<f64, RCFError> {
        density(self, point)
    }

    /// Computes the directional density of the data at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - The point at which to compute directional density.
    pub fn directional_density(&self, point: &[f32]) -> Result<DiVector, RCFError> {
        directional_density(self, point)
    }

    /// Computes an interpolation measure for the density at a given point.
    ///
    /// # Arguments
    ///
    /// * `point` - The point for which to compute the density interpolant.
    pub fn density_interpolant(&self, point: &[f32]) -> Result<InterpolationMeasure, RCFError> {
        density_interpolant(self, point)
    }

    /// Extrapolates future data points based on the current state of the model.
    ///
    /// # Arguments
    ///
    /// * `look_ahead` - The number of future time steps to extrapolate.
    pub fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector<f32>, RCFError> {
        extrapolate(self, look_ahead)
    }

    /// Returns the number of dimensions of the data points in the forest.
    pub fn dimensions(&self) -> usize {
        self.0.dimensions()
    }

    /// Returns the shingle size used by the forest.
    pub fn shingle_size(&self) -> usize {
        self.0.shingle_size()
    }

    /// Checks if internal shingling is enabled.
    pub fn is_internal_shingling_enabled(&self) -> bool {
        self.0.is_internal_shingling_enabled()
    }

    /// Checks if the forest is ready to produce anomaly scores.
    pub fn is_output_ready(&self) -> bool {
        self.0.is_output_ready()
    }

    /// Returns the total number of data points seen by the forest.
    pub fn entries_seen(&self) -> u64 {
        self.0.entries_seen()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rcfoptions() {
        let opts = RandomCutForestOptions::default();
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
        let opts = RandomCutForestOptions {
            dimensions: 3,
            shingle_size: 2,
            num_trees: Some(50),
            sample_size: Some(128),
            output_after: Some(10),
            random_seed: Some(42),
            parallel_execution_enabled: Some(true),
            lambda: Some(0.01),
            ..Default::default()
        };
        let builder = opts.to_rcf_builder();

        let rcf = builder.build::<u64, u64>();
        assert!(rcf.is_ok());

        let rcf = builder.build_large_simple::<u64>();
        assert!(rcf.is_ok());
    }

    #[test]
    fn test_to_rcf_returns_ok() {
        let opts = RandomCutForestOptions {
            dimensions: 2,
            shingle_size: 1,
            num_trees: Some(10),
            sample_size: Some(32),
            ..Default::default()
        };
        let rcf = opts.to_rcf();
        assert!(rcf.is_ok());
    }

    #[test]
    fn test_random_cut_forest_creation() {
        let opts = RandomCutForestOptions {
            dimensions: 2,
            shingle_size: 2,
            random_seed: Some(42),
            ..Default::default()
        };
        let rcf = RandomCutForest::new(opts).unwrap();
        assert_eq!(rcf.dimensions(), 4);
        assert_eq!(rcf.shingle_size(), 2);
        assert!(rcf.is_internal_shingling_enabled());
        assert!(!rcf.is_output_ready());
        assert_eq!(rcf.entries_seen(), 0);
    }

    #[test]
    fn test_random_cut_forest_update_and_score() {
        let opts = RandomCutForestOptions {
            dimensions: 2,
            shingle_size: 1,
            num_trees: Some(10),
            sample_size: Some(32),
            ..Default::default()
        };
        let mut rcf = RandomCutForest::new(opts).unwrap();
        let point = vec![1.0, 2.0];
        rcf.update(&point).unwrap();

        let score = rcf.score(&point).unwrap();
        assert!(score.is_finite());
        assert!(score < 2.0);

        let displacement_score = rcf.displacement_score(&point).unwrap();
        assert!(displacement_score.is_finite());

        let attribution = rcf.attribution(&point).unwrap();
        assert_eq!(attribution.high.len(), 2);
        assert_eq!(attribution.low.len(), 2);

        let neighbors = rcf.near_neighbor_list(&point, 10).unwrap();
        assert!(!neighbors.is_empty());

        for (dist, neighbor, _) in neighbors {
            assert!(dist.is_finite());
            assert_eq!(neighbor.len(), 2);
            assert!(neighbor.iter().all(|&x| x.is_finite()));
        }
    }

    #[test]
    fn test_rcf_serde_json() {
        let opts = RandomCutForestOptions {
            dimensions: 3,
            random_seed: Some(42),
            ..Default::default()
        };
        let mut rcf = RandomCutForest::new(opts).unwrap();
        let serialized = serde_json::to_string(&rcf).unwrap();
        let mut deserialized: RandomCutForest = serde_json::from_str(&serialized).unwrap();

        let point = vec![1.0, 2.0, 3.0];
        rcf.update(&point).unwrap();
        let score1 = rcf.score(&point).unwrap();
        deserialized.update(&point).unwrap();
        let score2 = deserialized.score(&point).unwrap();

        assert_eq!(score1, score2);
    }

    #[test]
    fn test_rcf_update_invalid_dimension() {
        let opts = RandomCutForestOptions {
            dimensions: 2,
            ..Default::default()
        };
        let mut rcf = RandomCutForest::new(opts).unwrap();
        // point의 차원이 다르면 에러가 발생해야 함
        let point = vec![1.0, 2.0, 3.0];
        let result = rcf.update(&point);
        assert!(result.is_err());
    }

    #[test]
    fn test_rcf_score_before_update() {
        let opts = RandomCutForestOptions {
            dimensions: 2,
            ..Default::default()
        };
        let rcf = RandomCutForest::new(opts).unwrap();

        let point = vec![0.0, 0.0];
        let score = rcf.score(&point).unwrap();
        assert!(score.is_finite());
    }

    #[test]
    fn test_rcf_entries_seen_and_output_ready() {
        let opts = RandomCutForestOptions {
            dimensions: 2,
            shingle_size: 1,
            num_trees: Some(5),
            sample_size: Some(10),
            output_after: Some(2),
            ..Default::default()
        };
        let mut rcf = RandomCutForest::new(opts).unwrap();
        let point = vec![1.0, 2.0];
        assert_eq!(rcf.entries_seen(), 0);
        assert!(!rcf.is_output_ready());
        rcf.update(&point).unwrap();
        assert_eq!(rcf.entries_seen(), 1);
        rcf.update(&point).unwrap();
        assert_eq!(rcf.entries_seen(), 2);
        // output_after=2이므로 이제 output_ready가 true여야 함
        rcf.update(&point).unwrap();
        assert!(rcf.is_output_ready());
    }

    #[test]
    fn test_rcf_shingled_point_and_internal_shingling() {
        let dim = 2;
        let shingle_size = 2;
        let opts = RandomCutForestOptions {
            dimensions: dim,
            shingle_size: shingle_size,
            random_seed: Some(42),
            ..Default::default()
        };
        let rcf = RandomCutForest::new(opts).unwrap();
        assert_eq!(rcf.shingle_size(), shingle_size);
        assert!(rcf.is_internal_shingling_enabled());
        let point = vec![1.0, 2.0];
        let shingled = rcf.shingled_point(&point);
        assert!(shingled.is_ok());
        let shingled_vec = shingled.unwrap();
        assert_eq!(shingled_vec.len(), dim * shingle_size);
    }

    #[test]
    fn test_rcf_density_and_directional_density() {
        let dim = 3;
        let shingle_size = 2;
        let opts = RandomCutForestOptions {
            dimensions: dim,
            shingle_size: shingle_size,
            random_seed: Some(42),
            ..Default::default()
        };
        let mut rcf = RandomCutForest::new(opts).unwrap();
        let point = vec![1.0; dim];
        rcf.update(&point).unwrap();
        rcf.update(&point).unwrap();
        rcf.update(&point).unwrap();
        let density = rcf.density(&point).unwrap();
        assert!(density.is_finite());
        let dir_density = rcf.directional_density(&point).unwrap();
        assert_eq!(dir_density.high.len(), dim * shingle_size);
        assert_eq!(dir_density.low.len(), dim * shingle_size);
    }

    #[test]
    fn test_rcf_density_interpolant_and_extrapolate() {
        let opts = RandomCutForestOptions {
            dimensions: 2,
            shingle_size: 2,
            random_seed: Some(42),
            ..Default::default()
        };
        let mut rcf = RandomCutForest::new(opts).unwrap();
        let point = vec![1.0, 2.0];
        rcf.update(&point).unwrap();
        rcf.update(&point).unwrap();
        rcf.update(&point).unwrap();
        let interpolant = rcf.density_interpolant(&point).unwrap();
        // InterpolationMeasure 타입의 필드가 있는지 확인 (예: measure, distance, probability_mass, sample_size)
        let _ = interpolant.measure;
        let _ = interpolant.distance;
        let _ = interpolant.probability_mass;
        let _ = interpolant.sample_size;
        let extrapolated = rcf.extrapolate(1).unwrap();
        assert_eq!(extrapolated.values.len(), 2);
    }
}
