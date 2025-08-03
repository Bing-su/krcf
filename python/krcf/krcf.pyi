import sys
from collections.abc import Sequence

from .options import RandomCutForestOptions
from .types import DiVector, InterpolationMeasure, NearNeighbor, RangeVector

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__version__: str

class RandomCutForest:
    """
    Random Cut Forest (RCF) for anomaly detection and time series analysis.

    Parameters
    ----------
    options : RandomCutForestOptions
        Configuration options for the forest.
    """

    def __init__(self, options: RandomCutForestOptions): ...
    def shingled_point(self, point: Sequence[float]) -> list[float]:
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
    def update(self, point: Sequence[float]) -> None:
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
    def score(self, point: Sequence[float]) -> float:
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
    def displacement_score(self, point: Sequence[float]) -> float:
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
    def attribution(self, point: Sequence[float]) -> DiVector:
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
        self, point: Sequence[float], percentile: int
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
    def density(self, point: Sequence[float]) -> float:
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
    def directional_density(self, point: Sequence[float]) -> DiVector:
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
    def density_interpolant(self, point: Sequence[float]) -> InterpolationMeasure:
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
