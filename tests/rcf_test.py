from hypothesis import given
from hypothesis import strategies as st
from krcf import RandomCutForest, RandomCutForestOptions


def list_of_list_floats(size: int) -> st.SearchStrategy[list[list[float]]]:
    return st.lists(
        st.lists(
            st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
            min_size=size,
            max_size=size,
        ),
        min_size=1,
    )


@given(points=list_of_list_floats(10))
def test_score_with_random_point(points: list[list[float]]):
    opts: RandomCutForestOptions = {
        "dimensions": 10,
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
        anomaly_score = forest.score([10**9] * 10)
        assert isinstance(anomaly_score, float)
        assert anomaly_score >= 1.5


@given(points=list_of_list_floats(5))
def test_attribution_shape(points: list[list[float]]):
    dim = 5
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


@given(points=list_of_list_floats(2))
def test_density_is_float(points: list[list[float]]):
    opts: RandomCutForestOptions = {
        "dimensions": 2,
        "shingle_size": 2,
        "output_after": 1,
    }
    forest = RandomCutForest(opts)
    for point in points:
        forest.update(point)
    density = forest.density(points[-1])
    assert isinstance(density, float)
    assert density >= 0


@given(points=list_of_list_floats(5))
def test_near_neighbor_list(points: list[list[float]]):
    opts: RandomCutForestOptions = {
        "dimensions": 5,
        "shingle_size": 2,
        "output_after": 1,
    }
    forest = RandomCutForest(opts)
    for point in points:
        forest.update(point)
    try:
        neighbors = forest.near_neighbor_list([1.0] * 5, percentile=1)
    except RuntimeError:
        return

    assert isinstance(neighbors, list)
    assert len(neighbors) > 0
    assert sorted(neighbors[0]) == ["distance", "point", "score"]
