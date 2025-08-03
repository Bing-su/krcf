# krcf

## Project Description

Python bindings for the [aws/random-cut-forest-by-aws](https://github.com/aws/random-cut-forest-by-aws) library, providing a fast and efficient implementation of Random Cut Forests for anomaly detection.

Key features:

- Fast anomaly detection using Random Cut Forests
- Attribution and scoring for detected anomalies
- Support for shingling and advanced configuration

## API Usage

### Basic Example

```python
from krcf import RandomCutForest, RandomCutForestOptions

# Define forest options
options: RandomCutForestOptions = {
    "dimensions": 3,  # Number of features in each data point
    "shingle_size": 2,  # Shingle size for time series
    "output_after": 1,  # Number of points before output is ready
}
forest = RandomCutForest(options)

# Update the forest with new data points
forest.update([1.0, 2.0, 3.0])
forest.update([2.0, 3.0, 4.0])

# Compute anomaly score
score = forest.score([2.0, 3.0, 4.0])
print("Anomaly score:", score)
# >>> Anomaly score: 1.0

# Get attribution vector
attr = forest.attribution([2.0, 3.0, 4.0])
print("Attribution:", attr)
# >>> Attribution: {'high': [0.3333333333333335, 0.3333333333333335, 0.3333333333333335, 0.0, 0.0, 0.0], 'low': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
```

### API Reference

#### RandomCutForest(options)

Create a new Random Cut Forest instance.

- `options`: A dictionary or `RandomCutForestOptions` specifying configuration (see below).

#### Methods

- `update(point: Sequence[float]) -> None`: Update the forest with a new data point.
- `score(point: Sequence[float]) -> float`: Compute the anomaly score for a data point.
- `displacement_score(point: Sequence[float]) -> float`: Compute the displacement score.
- `attribution(point: Sequence[float]) -> dict`: Get the attribution vector for a data point.
- `density(point: Sequence[float]) -> float`: Compute the density estimate for a data point.
- `near_neighbor(point: Sequence[float], percentile: int) -> list`: Find near neighbors for a data point.
- `is_output_ready() -> bool`: Check if the forest is ready to output scores.

#### RandomCutForestOptions

Options for configuring the forest (all except `dimensions` and `shingle_size` are optional):

- `dimensions` (int): Number of features in each data point (required)
- `shingle_size` (int): Shingle size for time series (required)
- `num_trees` (int): Number of trees in the forest
- `sample_size` (int): Sample size for each tree
- `output_after` (int): Number of points before output is ready
- `random_seed` (int): Random seed for reproducibility
- ...and more advanced options

For more details, see the docstrings in the code or the API reference.
