import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from krcf import RandomCutForest, RandomCutForestOptions


def update_and_score(rcf: RandomCutForest, dim: int) -> list[float]:
    rng = np.random.default_rng(42)

    points = rng.normal(size=(5000, dim))
    idx = rng.choice(points.shape[0], size=100, replace=False)
    points[idx] *= 10000
    scores = []
    for point in points:
        scores.append(rcf.score(point))
        rcf.update(point)
    return scores


def test_update_and_score(benchmark: BenchmarkFixture):
    dim = 5
    options = RandomCutForestOptions(
        dimensions=dim,
        shingle_size=4,
        num_trees=100,
        sample_size=256,
        output_after=1,
        random_seed=42,
        parallel_execution_enabled=True,
    )
    rcf = RandomCutForest(options)

    scores = benchmark(lambda: update_and_score(rcf, dim))
    assert any(score > 1 for score in scores)


def test_serialization(benchmark: BenchmarkFixture):
    dim = 5
    options = RandomCutForestOptions(
        dimensions=dim,
        shingle_size=4,
        num_trees=100,
        sample_size=256,
        parallel_execution_enabled=True,
    )
    rcf = RandomCutForest(options)

    update_and_score(rcf, dim)
    benchmark(lambda: rcf.from_msgpack(rcf.to_msgpack()))
