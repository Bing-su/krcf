from __future__ import annotations

try:
    from typing import Required, TypedDict
except ImportError:
    from typing_extensions import Required, TypedDict


class RandomCutForestOptions(TypedDict, total=False):
    dimensions: Required[int]
    "Number of features in each data point"
    shingle_size: Required[int]
    "Shingle size for time series"
    id: int | None
    "Unique identifier for the forest"
    num_trees: int | None
    "Number of trees in the forest"
    sample_size: int | None
    "Sample size for each tree"
    output_after: int | None
    "Number of points before producing output"
    random_seed: int | None
    "Random seed for reproducibility"
    parallel_execution_enabled: bool | None
    "Enable parallel execution"
    lambda: float | None  # pyright: ignore[reportGeneralTypeIssues]
    "The decay factor used by stream samplers in this forest."
    internal_rotation: bool | None
    "Enable internal rotation"
    internal_shingling: bool | None
    "Enable internal shingling"
    propagate_attribute_vectors: bool | None
    "Propagate attribute vectors"
    store_pointsum: bool | None
    "Store point sum for each tree"
    store_attributes: bool | None
    "Store attributes in the forest"
    initial_accept_fraction: float | None
    "Initial fraction of points accepted. (0,1]"
    bounding_box_cache_fraction: float | None
    "Fraction of bounding box cache to use. [0,1]"
