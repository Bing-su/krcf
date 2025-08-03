from typing import TypedDict


class DiVector(TypedDict):
    high: list[float]
    "Value vector in the upper direction"
    low: list[float]
    "Value vector in the lower direction"


class RangeVector(TypedDict):
    values: list[float]
    "Central or predicted value vector"
    upper: list[float]
    "Upper bound vector"
    lower: list[float]
    "Lower bound vector"


class InterpolationMeasure(TypedDict):
    measure: DiVector
    "Interpolated measure (upper/lower)"
    distance: DiVector
    "Distance information (upper/lower)"
    probability_mass: DiVector
    "Probability mass (upper/lower)"
    sample_size: float
    "Sample size"


class NearNeighbor(TypedDict):
    score: float
    "Score of the neighbor"
    point: list[float]
    "Coordinates of the neighbor"
    distance: float
    "Distance value"
