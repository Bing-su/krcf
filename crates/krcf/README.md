# krcf

A Rust crate that provides a high-level API for the Random Cut Forest (RCF) algorithm, built on top of the `krcflib` library. This crate is designed to be used as a library in other Rust projects and serves as the core for the Python bindings.

## Features

- **Simple API:** A straightforward interface for creating, updating, and scoring with Random Cut Forests.
- **Configurable:** Easily configure the RCF with a wide range of options.
- **Serialization/Deserialization:** Support for serializing and deserializing RCF models using `serde`.
- **Comprehensive Functionality:** Exposes key RCF capabilities including anomaly scoring, displacement scores, attribution, density estimation, and forecasting.

## Usage

Add `krcf` to your `Cargo.toml`:

```toml
[dependencies]
krcf = "0.2.0" # Replace with the desired version
```

### Basic Example

Here's how you can use `krcf` to perform anomaly detection:

```rust
use krcf::{RandomCutForest, RandomCutForestOptions};

fn main() {
    // 1. Configure the Random Cut Forest
    let options = RandomCutForestOptions {
        dimensions: 2,
        shingle_size: 1,
        num_trees: Some(30),
        sample_size: Some(256),
        random_seed: Some(42),
        ..Default::default()
    };

    // 2. Create a new RCF instance
    let mut forest = RandomCutForest::new(options).expect("Failed to create RCF");

    // 3. Update the forest with data points
    forest.update(&[1.0, 1.5]).unwrap();
    forest.update(&[1.2, 1.8]).unwrap();
    forest.update(&[0.9, 1.6]).unwrap();

    // 4. Score a new data point to detect anomalies
    let point_to_score = &[10.0, -5.0];
    let score = forest.score(point_to_score).unwrap();

    println!("Anomaly score for {:?}: {}", point_to_score, score);

    // A higher score suggests a higher likelihood of being an anomaly.
    // For this example, the score will be high because the point is far from the initial cluster.
}
```

### API Reference

#### `RandomCutForestOptions`

A struct to configure the `RandomCutForest`.

- `dimensions` (usize): The number of features in each data point.
- `shingle_size` (usize): The size of the shingle to use for time-series data.
- `num_trees` (Option<usize>): The number of trees in the forest.
- `sample_size` (Option<usize>): The number of points to sample for each tree.
- `output_after` (Option<usize>): The number of points to process before the model is ready for scoring.
- `random_seed` (Option<u64>): A seed for the random number generator for reproducibility.
- ... and other advanced options.

#### `RandomCutForest`

The main struct representing the RCF model.

- `new(options: RandomCutForestOptions) -> Result<Self, RCFError>`: Creates a new `RandomCutForest`.
- `update(&mut self, point: &[f32]) -> Result<(), RCFError>`: Updates the forest with a new data point.
- `score(&self, point: &[f32]) -> Result<f64, RCFError>`: Computes the anomaly score for a point.
- `displacement_score(&self, point: &[f32]) -> Result<f64, RCFError>`: Computes the displacement score.
- `attribution(&self, point: &[f32]) -> Result<DiVector, RCFError>`: Computes the attribution of the anomaly score to each dimension.
- `density(&self, point: &[f32]) -> Result<f64, RCFError>`: Estimates the density at a given point.
- `extrapolate(&self, look_ahead: usize) -> Result<RangeVector<f32>, RCFError>`: Forecasts future values.

For more details on the available methods and options, please refer to the source code documentation.
