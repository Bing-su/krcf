from __future__ import annotations

try:
    from typing import Required, TypedDict
except ImportError:
    from typing_extensions import Required, TypedDict


class RandomCutForestOptions(TypedDict, total=False):
    dimensions: Required[int]
    shingle_size: Required[int]
    id: int | None
    num_trees: int | None
    sample_size: int | None
    output_after: int | None
    random_seed: int | None
    parallel_execution_enabled: bool | None
    lambda: float | None  # pyright: ignore[reportGeneralTypeIssues]
    internal_rotation: bool | None
    internal_shingling: bool | None
    propagate_attribute_vectors: bool | None
    store_pointsum: bool | None
    store_attributes: bool | None
    initial_accept_fraction: float | None
    bounding_box_cache_fraction: float | None
