from __future__ import annotations

import copy
import json
import pickle
import platform
from collections import UserList
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable

import jsonpickle
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from krcf import RandomCutForest, RandomCutForestOptions


@st.composite
def points(draw: st.DrawFn, *, dim: int | None = None) -> np.ndarray:
    if dim is None:
        dim = draw(st.integers(min_value=1, max_value=20))
    num_points = draw(st.integers(min_value=1, max_value=100))
    dtype = draw(st.sampled_from([np.float32, np.float64]))
    return draw(
        arrays(
            dtype,
            shape=(num_points, dim),
            elements=st.floats(
                min_value=-1e9,
                max_value=1e9,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )


@given(points=points())
def test_score_with_random_point(points: np.ndarray):
    dim = points.shape[1]
    opts: RandomCutForestOptions = {
        "dimensions": dim,
        "shingle_size": 2,
        "output_after": 1,
    }
    forest = RandomCutForest(opts)
    for point in points:
        forest.update(point)
    score = forest.score(points[-1])
    assert isinstance(score, float)
    assert score >= 0
    if forest.is_output_ready():
        anomaly_score = forest.score([10**18] * dim)
        assert isinstance(anomaly_score, float)
        assert anomaly_score >= 1.5


@given(points=points())
def test_attribution_shape(points: np.ndarray):
    dim = points.shape[1]
    shingle_size = 2
    opts: RandomCutForestOptions = {
        "dimensions": dim,
        "shingle_size": shingle_size,
        "output_after": 1,
    }
    forest = RandomCutForest(opts)
    for point in points:
        forest.update(point)
    attr = forest.attribution(points[-1])
    assert sorted(attr) == ["high", "low"]
    assert len(attr["high"]) == dim * shingle_size
    assert len(attr["low"]) == dim * shingle_size


@given(points=points())
def test_density_is_float(points: np.ndarray):
    dim = points.shape[1]
    opts: RandomCutForestOptions = {
        "dimensions": dim,
        "shingle_size": 2,
        "output_after": 1,
    }
    forest = RandomCutForest(opts)
    for point in points:
        forest.update(point)
    density = forest.density(points[-1])
    assert isinstance(density, float)
    assert density >= 0


@given(points=points())
def test_near_neighbor_list(points: np.ndarray):
    dim = points.shape[1]
    opts: RandomCutForestOptions = {
        "dimensions": dim,
        "shingle_size": 2,
        "output_after": 1,
    }
    forest = RandomCutForest(opts)
    for point in points:
        forest.update(point)

    try:
        neighbors = forest.near_neighbor_list([1.0] * dim, percentile=1)
    except RuntimeError:
        return
    assert isinstance(neighbors, list)
    assert len(neighbors) > 0
    assert sorted(neighbors[0]) == ["distance", "point", "score"]


def test_options():
    opts: RandomCutForestOptions = {
        "dimensions": 5,
        "shingle_size": 3,
        "output_after": 256,
    }
    forest = RandomCutForest(opts)
    options = forest.options()
    for k, v in options.items():
        if k in opts:
            assert v == opts[k]
        else:
            assert v is None, f"Unexpected option {k} with value {v}"


def test_to_json():
    opts: RandomCutForestOptions = {
        "dimensions": 5,
        "shingle_size": 3,
        "output_after": 256,
        "id": 1000000007,
    }
    forest = RandomCutForest(opts)
    json_str = forest.to_json()
    assert isinstance(json_str, str)

    json_data = json.loads(json_str)
    assert isinstance(json_data, dict)
    assert "options" in json_data
    assert isinstance(json_data["options"], dict)
    assert json_data["options"]["dimensions"] == opts["dimensions"]
    assert json_data["options"]["shingle_size"] == opts["shingle_size"]
    assert json_data["options"]["output_after"] == opts["output_after"]
    assert json_data["options"]["id"] == opts["id"]

    assert "rcf" in json_data
    assert isinstance(json_data["rcf"], dict)

    forest2 = RandomCutForest.from_json(json_str)
    assert isinstance(forest2, RandomCutForest)
    assert forest2.options().get("id") == opts["id"]


@given(
    options=st.fixed_dictionaries(
        {
            "dimensions": st.integers(min_value=1, max_value=20),
            "shingle_size": st.integers(min_value=1, max_value=10),
            "output_after": st.integers(min_value=1, max_value=10),
            "id": st.integers(min_value=0, max_value=10**18) | st.none(),
            "random_seed": st.integers(min_value=0, max_value=10**18) | st.none(),
            "num_trees": st.integers(min_value=1, max_value=100) | st.none(),
            "sample_size": st.integers(min_value=10, max_value=512) | st.none(),
            "parallel_execution_enabled": st.booleans() | st.none(),
            "lambda": st.floats(min_value=0.0, max_value=1.0) | st.none(),
            "internal_rotation": st.booleans() | st.none(),
            "internal_shingling": st.just(True) | st.none(),  # noqa: FBT003
            "propagate_attribute_vectors": st.booleans() | st.none(),
            "store_pointsum": st.booleans() | st.none(),
            "store_attributes": st.booleans() | st.none(),
            "initial_accept_fraction": st.floats(min_value=0.01, max_value=1.0)
            | st.none(),
            "bounding_box_cache_fraction": st.floats(min_value=0.0, max_value=1.0)
            | st.none(),
        }
    ),
    data=st.data(),
)
@settings(deadline=None)
def test_rcf_with_options(options: RandomCutForestOptions, data: st.DataObject):
    propagate_attribute_vectors = options.get("propagate_attribute_vectors")
    store_attributes = options.get("store_attributes")
    assume(
        not propagate_attribute_vectors
        or (propagate_attribute_vectors and store_attributes)
    )

    forest = RandomCutForest(options)
    dim = options["dimensions"]
    p = data.draw(points(dim=dim))
    scores = []

    for point in p:
        scores.append(forest.score(point))
        forest.update(point)

    assert all(isinstance(score, float) for score in scores)
    assert all(score >= 0 for score in scores)


@given(
    options=st.fixed_dictionaries(
        {
            "dimensions": st.integers(min_value=1, max_value=20),
            "shingle_size": st.integers(min_value=1, max_value=10),
            "output_after": st.integers(min_value=1, max_value=10),
            "id": st.integers(min_value=0, max_value=10**18) | st.none(),
            "random_seed": st.integers(min_value=0, max_value=10**18),
            "num_trees": st.integers(min_value=1, max_value=50) | st.none(),
            "sample_size": st.integers(min_value=10, max_value=256) | st.none(),
            "parallel_execution_enabled": st.booleans() | st.none(),
            "lambda": st.floats(min_value=0.0, max_value=1.0) | st.none(),
            "internal_rotation": st.booleans() | st.none(),
            "internal_shingling": st.just(True) | st.none(),  # noqa: FBT003
            "store_pointsum": st.booleans() | st.none(),
            "store_attributes": st.booleans() | st.none(),
            "initial_accept_fraction": st.floats(min_value=0.01, max_value=1.0)
            | st.none(),
            "bounding_box_cache_fraction": st.floats(min_value=0.0, max_value=1.0)
            | st.none(),
        }
    ),
)
@settings(deadline=None, max_examples=30)
@pytest.mark.parametrize("module", [pickle, jsonpickle])
@pytest.mark.xfail(
    platform.python_implementation() == "PyPy",
    reason="pypy + jsonpickle error",
)
def test_rcf_pickle(options: RandomCutForestOptions, module: Any):
    forest = RandomCutForest(options)
    dim = options["dimensions"]
    rng = np.random.default_rng(options.get("random_seed", 0))
    p = rng.random(((options.get("output_after") or 1) * 2, dim))

    for point in p:
        forest.update(point)

    serialized = module.dumps(forest)
    deserialized = module.loads(serialized)

    score1 = forest.score(p[0])
    score2 = deserialized.score(p[0])

    assert score1 == score2
    assert forest.is_output_ready() == deserialized.is_output_ready()
    assert forest.entries_seen() == deserialized.entries_seen()

    for point in p:
        deserialized.update(point)


@given(
    options=st.fixed_dictionaries(
        {
            "dimensions": st.integers(min_value=1, max_value=20),
            "shingle_size": st.integers(min_value=1, max_value=10),
            "output_after": st.integers(min_value=1, max_value=10),
            "id": st.integers(min_value=0, max_value=10**18) | st.none(),
            "random_seed": st.integers(min_value=0, max_value=10**18),
            "num_trees": st.integers(min_value=1, max_value=50) | st.none(),
            "sample_size": st.integers(min_value=10, max_value=256) | st.none(),
            "parallel_execution_enabled": st.booleans() | st.none(),
            "lambda": st.floats(min_value=0.0, max_value=1.0) | st.none(),
            "internal_rotation": st.booleans() | st.none(),
            "internal_shingling": st.just(True) | st.none(),  # noqa: FBT003
            "store_pointsum": st.booleans() | st.none(),
            "store_attributes": st.booleans() | st.none(),
            "initial_accept_fraction": st.floats(min_value=0.01, max_value=1.0)
            | st.none(),
            "bounding_box_cache_fraction": st.floats(min_value=0.0, max_value=1.0)
            | st.none(),
        }
    ),
)
@settings(deadline=None, max_examples=30)
@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy, RandomCutForest.clone])
def test_rcf_copy(options: RandomCutForestOptions, copier: Callable[[Any], Any]):
    forest = RandomCutForest(options)
    dim = options["dimensions"]
    rng = np.random.default_rng(options.get("random_seed", 0))
    p = rng.random(((options.get("output_after") or 1) * 2, dim))

    for point in p:
        forest.update(point)

    copied = copier(forest)
    score1 = forest.score(p[0])
    score2 = copied.score(p[0])

    assert score1 == score2
    assert forest.is_output_ready() == copied.is_output_ready()
    assert forest.entries_seen() == copied.entries_seen()

    for point in p:
        copied.update(point)


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.xfail(
    platform.python_implementation() == "PyPy",
    reason="pypy threading issues",
)
def test_rcf_thread_safety(parallel: bool):  # noqa: FBT001
    dim = 10
    shingle_size = 4
    options: RandomCutForestOptions = {
        "dimensions": dim,
        "shingle_size": shingle_size,
        "output_after": 1,
        "parallel_execution_enabled": parallel,
    }

    forest = RandomCutForest(options)

    pp = np.random.random((200, dim))
    scores = []
    updates = []
    with ThreadPoolExecutor() as executor:
        for point in pp:
            fut1 = executor.submit(forest.score, point)
            scores.append(fut1)
            fut2 = executor.submit(forest.update, point)
            updates.append(fut2)

    scores = [fut.result() for fut in scores]
    _ = [fut.result() for fut in updates]

    assert all(isinstance(score, float) for score in scores)
    assert all(score >= 0 for score in scores)

    anomaly = np.random.random(dim) * 10**9
    score = forest.score(anomaly.tolist())
    assert isinstance(score, float)
    assert score >= 2


def test_rcf_types():
    dim = 10
    opts: RandomCutForestOptions = {
        "dimensions": dim,
        "shingle_size": 1,
        "output_after": 1,
    }

    forest = RandomCutForest(opts)

    input1 = np.full(dim, True, dtype=np.bool)  # noqa: FBT003
    forest.update(input1)

    input2 = np.full(dim, 1, dtype=np.int32)
    forest.update(input2)

    input3 = np.full(dim, 1.0, dtype=np.float16)
    forest.update(input3)

    input4 = np.full(dim, 1.0, dtype=np.float64)
    forest.update(input4)

    input5 = [Decimal("1.0")] * dim
    forest.update(input5)

    input6 = UserList([1] * dim)
    forest.update(input6)

    input7 = [Fraction(1, 1)] * dim
    forest.update(input7)

    input8 = [True] * dim
    forest.update(input8)

    class FloatLike:
        def __float__(self) -> float:
            return 1.0

    input9 = [FloatLike() for _ in range(dim)]
    forest.update(input9)
