from __future__ import annotations

from typing import NotRequired, TypedDict


class RandomCutForestOptions(TypedDict):
    dimensions: int
    shingle_size: int
    num_trees: NotRequired[int | None]
    sample_size: NotRequired[int | None]
    output_after: NotRequired[int | None]
    random_seed: NotRequired[int | None]
    parallel_execution_enabled: NotRequired[bool | None]
    lambda: NotRequired[float | None]  # pyright: ignore[reportGeneralTypeIssues]
