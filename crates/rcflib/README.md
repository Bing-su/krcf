# krcflib

Original: [aws/random-cut-forest-by-aws](https://github.com/aws/random-cut-forest-by-aws)

A Rust implementation of the Random Cut Forest (RCF) algorithm for anomaly detection, forecasting, and data analysis on streaming data. Random Cut Forests are a form of unsupervised machine learning that can detect anomalous data points within a dataset.

This library provides a flexible and efficient implementation of RCF, suitable for multi-dimensional data streams.

## Features

*   **Anomaly Detection:** Assigns an anomaly score to each data point. Higher scores indicate a higher likelihood of being an anomaly.
*   **Streaming Data:** Designed to work with continuous data streams, updating the model as new data arrives.
*   **Multi-dimensional Data:** Handles data with multiple features.
*   **Imputation/Extrapolation:** Can be used to predict missing values or forecast future values.
*   **Configurable:** Allows tuning of various parameters like the number of trees, tree capacity (shingle size), and time decay to suit different data characteristics.

## Usage

Here is a basic example of how to use `krcflib` to detect anomalies in a multi-dimensional dataset.

First, add `krcflib` to your `Cargo.toml`:

```toml
[dependencies]
krcflib = "4.0.0" # Replace with the desired version
rand = "0.9"
rand_chacha = "0.9"
```

Then, you can use it in your code like this:

(note: package name is `krcflib`, but you should import it as `rcflib`)

```rust
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::common::multidimdatawithkey;
use rcflib::rcf::RCFBuilder;

fn main() {
    // Define the parameters for the forest
    let shingle_size = 8;
    let base_dimension = 5;
    let data_size = 100000;
    let number_of_trees = 30;
    let capacity = 256;
    let time_decay = 0.1 / capacity as f64;

    // Build the Random Cut Forest
    let mut forest = RCFBuilder::new(base_dimension, shingle_size)
        .tree_capacity(capacity)
        .number_of_trees(number_of_trees)
        .random_seed(17)
        .parallel_enabled(false)
        .internal_shingling(true)
        .time_decay(time_decay)
        .build_default()
        .unwrap();

    // Generate some sample multi-dimensional data
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut amplitude = Vec::new();
    for _i in 0..base_dimension {
        amplitude.push((1.0 + 0.2 * rng.gen::<f32>()) * 60.0);
    }

    let data_with_key = multidimdatawithkey::MultiDimDataWithKey::multi_cosine(
        data_size,
        &vec![60; base_dimension],
        &amplitude,
        5.0, // noise
        0,
        base_dimension,
    )
    .unwrap();

    // Process the data through the forest
    for i in 0..data_with_key.data.len() {
        // Get the anomaly score for the current point
        let score = forest.score(&data_with_key.data[i]).unwrap();

        // Update the forest with the new point
        forest.update(&data_with_key.data[i], 0).unwrap();
    }

    println!("Successfully processed {} data points.", forest.entries_seen());
    println!("Final PointStore size: {}", forest.point_store_size());
    println!("Total size of the forest: {} bytes (approx)", forest.size());
}
```

### Explanation

1.  **`RCFBuilder`**: This is the main entry point for creating a `RandomCutForest`. It allows you to configure the forest's parameters.
    *   `base_dimension`: The number of features in your data.
    *   `shingle_size`: The number of recent data points to consider together as a single point in the forest. This is useful for detecting anomalies in time-series data. If you are not using time-series data, you can set this to 1.
    *   `number_of_trees`: The number of trees in the forest. More trees can lead to more accurate results but will increase memory usage and processing time.
    *   `tree_capacity`: The maximum number of points stored in each tree.
    *   `time_decay`: A parameter that determines how much weight is given to older points. A non-zero value helps the model adapt to changing data patterns.

2.  **`score()`**: This method takes a data point and returns an anomaly score. The score is a measure of how much the new point deviates from the patterns learned by the forest.

3.  **`update()`**: This method adds a new data point to the forest, updating the trees. This allows the model to learn from the new data and adapt over time.
