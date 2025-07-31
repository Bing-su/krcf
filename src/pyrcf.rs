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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, FromPyObject, IntoPyObject)]
#[pyo3(from_item_all)]
pub struct RandomCutForestOptions {
    pub dimensions: usize,
    pub shingle_size: usize,
    #[pyo3(default)]
    pub num_trees: Option<usize>,
    #[pyo3(default)]
    pub sample_size: Option<usize>,
    #[pyo3(default)]
    pub output_after: Option<usize>,
    #[pyo3(default)]
    pub random_seed: Option<u64>,
    #[pyo3(default)]
    pub parallel_execution_enabled: Option<bool>,
    #[pyo3(default)]
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

#[pyclass(module = "krcf")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomCutForest {
    pub rcf: rcf::RandomCutForest,
    pub options: RandomCutForestOptions,
}

#[pymethods]
impl RandomCutForest {
    #[new]
    pub fn new(options: RandomCutForestOptions) -> Result<Self> {
        let rcf = rcf::RandomCutForest::new(options.to_rcf_options())?;
        Ok(Self { rcf, options })
    }

    pub fn shingled_point(&self, point: Vec<f32>) -> Result<Vec<f32>> {
        Ok(self.rcf.shingled_point(&point)?)
    }

    pub fn update(&mut self, point: Vec<f32>) -> Result<()> {
        Ok(self.rcf.update(&point)?)
    }

    pub fn score(&self, point: Vec<f32>) -> Result<f64> {
        Ok(self.rcf.score(&point)?)
    }

    pub fn displacement_score(&self, point: Vec<f32>) -> Result<f64> {
        Ok(self.rcf.displacement_score(&point)?)
    }

    pub fn attribution(&self, point: Vec<f32>) -> Result<DiVector> {
        Ok(self.rcf.attribution(&point)?.into())
    }

    pub fn near_neighbor_list(
        &self,
        point: Vec<f32>,
        percentile: usize,
    ) -> Result<Vec<(f64, Vec<f32>, f64)>> {
        Ok(self.rcf.near_neighbor_list(&point, percentile)?)
    }

    pub fn density(&self, point: Vec<f32>) -> Result<f64> {
        Ok(self.rcf.density(&point)?)
    }

    pub fn directional_density(&self, point: Vec<f32>) -> Result<DiVector> {
        Ok(self.rcf.directional_density(&point)?.into())
    }

    pub fn density_interpolant(&self, point: Vec<f32>) -> Result<InterpolationMeasure> {
        Ok(self.rcf.density_interpolant(&point)?.into())
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
    // ------------ Serialization Methods ------------

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    #[classmethod]
    pub fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, string: String) -> Result<Self> {
        Ok(serde_json::from_str(&string)?)
    }

    // ------------ Python Magic Methods ------------

    fn __repr__(&self) -> String {
        format!(
            "RandomCutForest(dimensions={}, shingle_size={}, num_trees={:?}, sample_size={:?}, output_after={:?}, random_seed={:?}, parallel_execution_enabled={:?}, lambda={:?}, is_output_ready={}, entries_seen={})",
            self.options.dimensions,
            self.options.shingle_size,
            self.options.num_trees,
            self.options.sample_size,
            self.options.output_after,
            self.options.random_seed,
            self.options.parallel_execution_enabled,
            self.options.lambda,
            self.rcf.is_output_ready(),
            self.rcf.entries_seen(),
        )
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __getnewargs__(&self) -> (RandomCutForestOptions,) {
        (self.options.clone(),)
    }

    fn __getstate__(&self) -> Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    fn __setstate__(&mut self, state: String) -> Result<()> {
        let deserialized: Self = serde_json::from_str(&state)?;
        *self = deserialized;
        Ok(())
    }
}
