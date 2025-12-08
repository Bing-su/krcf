use anyhow::Result;
use krcf;
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

#[derive(IntoPyObject)]
pub struct NearNeighbor {
    pub score: f64,
    pub point: Vec<f32>,
    pub distance: f64,
}

impl From<(f64, Vec<f32>, f64)> for NearNeighbor {
    fn from((score, point, distance): (f64, Vec<f32>, f64)) -> Self {
        Self {
            score,
            point,
            distance,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, FromPyObject, IntoPyObject)]
#[pyo3(from_item_all)]
pub struct RandomCutForestOptions {
    pub dimensions: usize,
    pub shingle_size: usize,
    #[pyo3(default)]
    pub id: Option<u64>,
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
    #[pyo3(default)]
    pub internal_rotation: Option<bool>,
    #[pyo3(default)]
    pub internal_shingling: Option<bool>,
    #[pyo3(default)]
    pub propagate_attribute_vectors: Option<bool>,
    #[pyo3(default)]
    pub store_pointsum: Option<bool>,
    #[pyo3(default)]
    pub store_attributes: Option<bool>,
    #[pyo3(default)]
    pub initial_accept_fraction: Option<f64>,
    #[pyo3(default)]
    pub bounding_box_cache_fraction: Option<f64>,
}

impl Into<krcf::RandomCutForestOptions> for RandomCutForestOptions {
    fn into(self) -> krcf::RandomCutForestOptions {
        krcf::RandomCutForestOptions {
            dimensions: self.dimensions,
            shingle_size: self.shingle_size,
            id: self.id,
            num_trees: self.num_trees,
            sample_size: self.sample_size,
            output_after: self.output_after,
            random_seed: self.random_seed,
            parallel_execution_enabled: self.parallel_execution_enabled,
            lambda: self.lambda,
            internal_rotation: self.internal_rotation,
            internal_shingling: self.internal_shingling,
            propagate_attribute_vectors: self.propagate_attribute_vectors,
            store_pointsum: self.store_pointsum,
            store_attributes: self.store_attributes,
            initial_accept_fraction: self.initial_accept_fraction,
            bounding_box_cache_fraction: self.bounding_box_cache_fraction,
        }
    }
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

#[pyclass(module = "krcf.krcf", str)]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RandomCutForest {
    pub rcf: krcf::RandomCutForest,
    pub options: RandomCutForestOptions,
}

#[pymethods]
impl RandomCutForest {
    #[new]
    pub fn new(options: RandomCutForestOptions) -> Result<Self> {
        let rcf = krcf::RandomCutForest::new(options.clone().into())?;
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
    ) -> Result<Vec<NearNeighbor>> {
        let list = self.rcf.near_neighbor_list(&point, percentile)?;
        Ok(list.into_iter().map(NearNeighbor::from).collect())
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

    pub fn options(&self) -> RandomCutForestOptions {
        self.options.clone()
    }

    #[pyo3(name = "clone")]
    pub fn clone_py(&self) -> Self {
        self.clone()
    }

    // ------------ Serialization Methods ------------

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    #[classmethod]
    pub fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, value: String) -> Result<Self> {
        Ok(serde_json::from_str(&value)?)
    }

    pub fn to_msgpack(&self) -> Result<Vec<u8>> {
        Ok(rmp_serde::to_vec_named(self)?)
    }

    #[classmethod]
    pub fn from_msgpack(_cls: &Bound<'_, pyo3::types::PyType>, value: Vec<u8>) -> Result<Self> {
        Ok(rmp_serde::from_slice(&value)?)
    }

    // ------------ Python Magic Methods ------------

    fn __repr__(&self) -> String {
        self.to_string()
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

    fn __getstate__(&self) -> Result<Vec<u8>> {
        Ok(rmp_serde::to_vec_named(&self)?)
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> Result<()> {
        *self = rmp_serde::from_slice(&state)?;
        Ok(())
    }
}

impl std::fmt::Display for RandomCutForest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
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
}
