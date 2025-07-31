use crate::rcf;
use anyhow::Result;
use pyo3::prelude::*;
use rcflib::common::{directionaldensity, divector, rangevector};

#[derive(IntoPyObject)]
pub struct DiVector {
    pub high: Vec<f64>,
    pub low: Vec<f64>,
}

impl From<divector::DiVector> for DiVector {
    fn from(rcf_divector: divector::DiVector) -> Self {
        Self {
            high: rcf_divector.high,
            low: rcf_divector.low,
        }
    }
}

#[derive(IntoPyObject)]
pub struct RangeVector {
    pub values: Vec<f64>,
    pub upper: Vec<f64>,
    pub lower: Vec<f64>,
}

fn convert_to_f64(v: Vec<f32>) -> Vec<f64> {
    v.into_iter().map(|x| x as f64).collect()
}

impl From<rangevector::RangeVector<f32>> for RangeVector {
    fn from(rcf_rangevector: rangevector::RangeVector<f32>) -> Self {
        Self {
            values: convert_to_f64(rcf_rangevector.values),
            upper: convert_to_f64(rcf_rangevector.upper),
            lower: convert_to_f64(rcf_rangevector.lower),
        }
    }
}

impl From<rangevector::RangeVector<f64>> for RangeVector {
    fn from(rcf_rangevector: rangevector::RangeVector<f64>) -> Self {
        Self {
            values: rcf_rangevector.values,
            upper: rcf_rangevector.upper,
            lower: rcf_rangevector.lower,
        }
    }
}

#[derive(IntoPyObject)]
pub struct InterpolationMeasure {
    pub measure: DiVector,
    pub distance: DiVector,
    pub probability_mass: DiVector,
    pub sample_size: f32,
}

impl From<directionaldensity::InterpolationMeasure> for InterpolationMeasure {
    fn from(rcf_interpolation_measure: directionaldensity::InterpolationMeasure) -> Self {
        Self {
            measure: rcf_interpolation_measure.measure.into(),
            distance: rcf_interpolation_measure.distance.into(),
            probability_mass: rcf_interpolation_measure.probability_mass.into(),
            sample_size: rcf_interpolation_measure.sample_size,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, FromPyObject)]
pub struct RandomCutForestOptions {
    pub dimensions: usize,
    pub shingle_size: usize,
    pub num_trees: Option<usize>,
    pub sample_size: Option<usize>,
    pub output_after: Option<usize>,
    pub random_seed: Option<u64>,
    pub parallel_execution_enabled: Option<bool>,
    pub lambda: Option<f64>,
}

impl RandomCutForestOptions {
    fn to_rcf_options(&self) -> rcf::RandomCutForestOptions {
        rcf::RandomCutForestOptions {
            dimensions: self.dimensions,
            shingle_size: self.shingle_size,
            num_trees: self.num_trees,
            sample_size: self.sample_size,
            output_after: self.output_after,
            random_seed: self.random_seed,
            parallel_execution_enabled: self.parallel_execution_enabled,
            lambda: self.lambda,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomCutForest {
    pub rcf: rcf::RandomCutForest,
    options: RandomCutForestOptions,
}

impl RandomCutForest {
    pub fn new(options: RandomCutForestOptions) -> Result<Self> {
        let rcf = rcf::RandomCutForest::new(options.to_rcf_options())?;
        Ok(Self { rcf, options })
    }

    pub fn shingled_point(&self, point: &[f32]) -> Result<Vec<f32>> {
        Ok(self.rcf.shingled_point(point)?)
    }

    pub fn update(&mut self, point: &[f32]) -> Result<()> {
        Ok(self.rcf.update(point)?)
    }

    pub fn score(&self, point: &[f32]) -> Result<f64> {
        Ok(self.rcf.score(point)?)
    }

    pub fn displacement_score(&self, point: &[f32]) -> Result<f64> {
        Ok(self.rcf.displacement_score(point)?)
    }

    pub fn attribution(&self, point: &[f32]) -> Result<DiVector> {
        Ok(self.rcf.attribution(point)?.into())
    }

    pub fn near_neighbor_list(
        &self,
        point: &[f32],
        percentile: usize,
    ) -> Result<Vec<(f64, Vec<f32>, f64)>> {
        Ok(self.rcf.near_neighbor_list(point, percentile)?)
    }

    pub fn density(&self, point: &[f32]) -> Result<f64> {
        Ok(self.rcf.density(point)?)
    }

    pub fn directional_density(&self, point: &[f32]) -> Result<DiVector> {
        Ok(self.rcf.directional_density(point)?.into())
    }

    pub fn density_interpolant(&self, point: &[f32]) -> Result<InterpolationMeasure> {
        Ok(self.rcf.density_interpolant(point)?.into())
    }

    pub fn extrapolate(&self, look_ahead: usize) -> Result<RangeVector> {
        Ok(self.rcf.extrapolate(look_ahead)?.into())
    }

    pub fn dimensions(&self) -> usize {
        self.rcf.dimensions()
    }

    pub fn shingle_size(&self) -> usize {
        self.rcf.shingle_size()
    }

    pub fn is_internal_shingling_enabled(&self) -> bool {
        self.rcf.is_internal_shingling_enabled()
    }

    pub fn is_output_ready(&self) -> bool {
        self.rcf.is_output_ready()
    }

    pub fn entries_seen(&self) -> u64 {
        self.rcf.entries_seen()
    }

    // ------------ Python Magic Methods ------------

    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    pub fn __copy__(&self) -> Self {
        self.clone()
    }

    pub fn __deepcopy__(&self, _memo: Option<&PyAny>) -> Self {
        self.clone()
    }

    pub fn __getnewargs__(&self) -> (RandomCutForestOptions,) {
        (self.options.clone(),)
    }
}
