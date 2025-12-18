from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import SupportsFloat, TypeAlias

import numpy as np

from .options import RandomCutForestOptions
from .types import DiVector, InterpolationMeasure, NearNeighbor, RangeVector

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__version__: str

_1DArray: TypeAlias = (
    Sequence[SupportsFloat]
    | np.ndarray[tuple[int], np.dtype[np.floating | np.integer | np.bool]]
)

class RandomCutForest:
    """
    Random Cut Forest (RCF) for anomaly detection and time series analysis.

    Parameters
    ----------
    options : RandomCutForestOptions
        Configuration options for the forest.
    """

    def __init__(self, options: RandomCutForestOptions): ...
    def shingled_point(self, point: _1DArray) -> list[float]:
        """
        Returns the shingled version of the input point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        list[float]
            Shingled data point.
        """
    def update(self, point: _1DArray) -> None:
        """
        Updates the forest with a new data point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        None
        """
    def score(self, point: _1DArray) -> float:
        """
        Computes the anomaly score for the given point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        float
            Anomaly score.
        """
    def displacement_score(self, point: _1DArray) -> float:
        """
        Computes the displacement score for the given point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        float
            Displacement score.
        """
    def attribution(self, point: _1DArray) -> DiVector:
        """
        Returns the attribution vector for the input point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        DiVector
            Attribution vector.
        """
    def near_neighbor_list(
        self, point: _1DArray, percentile: int
    ) -> list[NearNeighbor]:
        """
        Returns a list of near neighbors for the input point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.
        percentile : int
            Percentile for neighbor selection.

        Returns
        -------
        list of NearNeighbor
            List of near neighbors containing score, neighbor point, and distance.
        """
    def density(self, point: _1DArray) -> float:
        """
        Computes the density estimate for the input point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        float
            Density estimate.
        """
    def directional_density(self, point: _1DArray) -> DiVector:
        """
        Computes the directional density for the input point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        DiVector
            Directional density vector.
        """
    def density_interpolant(self, point: _1DArray) -> InterpolationMeasure:
        """
        Computes the density interpolant for the input point.

        Parameters
        ----------
        point : Sequence[float]
            Input data point.

        Returns
        -------
        InterpolationMeasure
            Interpolated density measure.
        """
    def extrapolate(self, look_ahead: int) -> RangeVector:
        """
        Extrapolates future values based on the current forest state.

        Parameters
        ----------
        look_ahead : int
            Number of steps to look ahead.

        Returns
        -------
        RangeVector
            Extrapolated value ranges.
        """
    def dimensions(self) -> int:
        """
        Returns the number of dimensions in the input data.

        Returns
        -------
        int
            Number of dimensions.
        """
    def shingle_size(self) -> int:
        """
        Returns the shingle size used by the forest.

        Returns
        -------
        int
            Shingle size.
        """
    def is_internal_shingling_enabled(self) -> bool:
        """
        Indicates if internal shingling is enabled.

        Returns
        -------
        bool
            True if enabled, False otherwise.
        """
    def is_output_ready(self) -> bool:
        """
        Indicates if the forest is ready to produce output.

        Returns
        -------
        bool
            True if ready, False otherwise.
        """
    def entries_seen(self) -> int:
        """
        Returns the number of entries processed by the forest.

        Returns
        -------
        int
            Number of entries seen.
        """
    def to_json(self) -> str:
        """
        Serializes the forest to a JSON string.

        Returns
        -------
        str
            JSON representation of the forest.
        """
    @classmethod
    def from_json(cls, value: str) -> Self:
        """
        Deserializes a forest from a JSON string.

        Parameters
        ----------
        value : str
            JSON string.

        Returns
        -------
        Self
            Deserialized forest instance.
        """
    def to_msgpack(self) -> bytes:
        """
        Serializes the forest to a MessagePack byte string.

        Returns
        -------
        bytes
            MessagePack representation of the forest.
        """
    @classmethod
    def from_msgpack(cls, value: bytes) -> Self:
        """
        Deserializes a forest from a MessagePack byte string.

        Parameters
        ----------
        value : bytes
            MessagePack byte string.

        Returns
        -------
        Self
            Deserialized forest instance.
        """
    def options(self) -> RandomCutForestOptions:
        """
        Returns the options used to create the forest.

        Returns
        -------
        RandomCutForestOptions
            Dictionary containing the Random Cut Forest options.
        """
    def clone(self) -> Self:
        """
        Returns a clone of the current forest instance.

        Returns
        -------
        Self
            Cloned forest instance.
        """
