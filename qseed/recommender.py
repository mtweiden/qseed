import abc

from typing import Any

from bqskit import Circuit
from bqskit.ir import Operation


class Recommender(abc.ABC):
    """
    An abstract class that encodes operations and recommends seed circuits.
    """
    @abc.abstractmethod
    def encode(self, operation: Operation) -> Any:
        """
        Encode an operation for seed circuit recommendation.
        """

    @abc.abstractmethod
    def _recommend(
        self,
        encoding: Any,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        """
        Recommend a list of seeds given an encoding of an operation.
        """

    @abc.abstractmethod
    def recommend(
        self,
        operation: Operation,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        """
        Recommend a list of seeds given an encoding of an operation.
        """
