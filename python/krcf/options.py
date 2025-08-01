from __future__ import annotations

try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict


class RandomCutForestOptions(TypedDict):
    dimensions: int
    shingle_size: int
    id: NotRequired[int | None]
    num_trees: NotRequired[int | None]
    sample_size: NotRequired[int | None]
    output_after: NotRequired[int | None]
    random_seed: NotRequired[int | None]
    parallel_execution_enabled: NotRequired[bool | None]
    lambda: NotRequired[float | None]  # pyright: ignore[reportGeneralTypeIssues]
    internal_rotation: NotRequired[bool | None]
    internal_shingling: NotRequired[bool | None]
    propagate_attribute_vectors: NotRequired[bool | None]
    store_pointsum: NotRequired[bool | None]
    store_attributes: NotRequired[bool | None]
    initial_accept_fraction: NotRequired[float | None]
    bounding_box_cache_fraction: NotRequired[float | None]
