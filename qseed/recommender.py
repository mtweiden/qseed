import abc

from typing import Any

from bqskit import Circuit


class Recommender(abc.ABC):
    """
    An abstract class that encodes circuits and recommends seed circuits.
    """
    @abc.abstractmethod
    def encode(self, circuit: Circuit) -> Any:
        """
        Encode an circuit for seed recommendation.
        """

    @abc.abstractmethod
    def _recommend(
        self,
        encoding: Any,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        """
        Recommend a list of seeds given an encoding of a circuit.
        """

    @abc.abstractmethod
    def recommend(
        self,
        circuit: Circuit,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        """
        Recommend a list of seeds given an encoding of a circuit.
        """
