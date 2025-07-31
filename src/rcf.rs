use rcflib::{
    common::{
        directionaldensity::InterpolationMeasure, divector::DiVector, rangevector::RangeVector,
    },
    errors::RCFError,
    rcf::{AugmentedRCF, RCFBuilder, RCFLarge, RCFOptionsBuilder},
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RandomCutForestOptions {
    pub dimensions: usize,
    pub shingle_size: usize,
    pub num_trees: Option<usize>,
    pub sample_size: Option<usize>,
    pub output_after: Option<usize>,
    pub random_seed: Option<u64>,
    pub parallel_execution_enabled: Option<bool>,
    pub lambda: Option<f64>,
}

impl Default for RandomCutForestOptions {
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

impl RandomCutForestOptions {
    pub fn to_rcf_builder(&self) -> RCFBuilder {
        let mut options = RCFBuilder::new(self.dimensions, self.shingle_size);

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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct RandomCutForest(RCFLarge<u64, u64>);

impl RandomCutForest {
    pub fn new(options: RandomCutForestOptions) -> Result<Self, RCFError> {
        let rcf = options.to_rcf()?;
        Ok(Self(rcf))
    }

    pub fn shingled_point(&self, point: &[f32]) -> Result<Vec<f32>, RCFError> {
        self.0.shingled_point(point)
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

    pub fn density(&self, point: &[f32]) -> Result<f64, RCFError> {
        self.0.density(point)
    }

    pub fn directional_density(&self, point: &[f32]) -> Result<DiVector, RCFError> {
        self.0.directional_density(point)
    }

    pub fn density_interpolant(&self, point: &[f32]) -> Result<InterpolationMeasure, RCFError> {
        self.0.density_interpolant(point)
    }

    pub fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector<f32>, RCFError> {
        self.0.extrapolate(look_ahead)
    }

    pub fn dimensions(&self) -> usize {
        self.0.dimensions()
    }

    pub fn shingle_size(&self) -> usize {
        self.0.shingle_size()
    }

    pub fn is_internal_shingling_enabled(&self) -> bool {
        self.0.is_internal_shingling_enabled()
    }

    pub fn is_output_ready(&self) -> bool {
        self.0.is_output_ready()
    }

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
            output_after: None,
            random_seed: None,
            parallel_execution_enabled: None,
            lambda: None,
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
            output_after: None,
            random_seed: None,
            parallel_execution_enabled: None,
            lambda: None,
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
