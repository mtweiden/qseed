import abc

from typing import Any

from bqskit import Circuit
from bqskit.ir import Operation
from bqskit.qis.graph import CouplingGraph


class Recommender(abc.ABC):
    """
    An abstract class that encodes operations and recommends seed circuits.
    """
    def __init__(self, coupling_graph: CouplingGraph | None = None) -> None:
        """
        The coupling_graph describes seed topology. If none, it is all-to-all.
        """
        if coupling_graph and not isinstance(coupling_graph, CouplingGraph):
            raise ValueError(
                'coupling_graph must be of type CouplingGraph, got '
                f'{type(coupling_graph)}.'
            )
        self.coupling_graph = coupling_graph

    @abc.abstractmethod
    def encode(self, operation: Operation) -> Any:
        """
        Encode an operation for seed circuit recommendation.
        """

    @abc.abstractmethod
    def recommend(
        self,
        encoding: Any,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        """
        Recommend a list of seeds given an encoding of an operation.
        """
