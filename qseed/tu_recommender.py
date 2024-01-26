import torch.nn as nn
import torch
import numpy as np
import logging

from typing import Any, Sequence

from bqskit import Circuit
from bqskit.qis.graph import CouplingGraph
from bqskit.ir import Circuit
from torch import Tensor, topk

from qseed.recommender import Recommender

_logger = logging.getLogger(__name__)


class TorchUnitaryRecommender(Recommender):
    """
    A recommender that uses a PyTorch nn.Module to analyze unitaries.
    """

    def __init__(
        self,
        recommender_model: nn.Module,
        seed_circuits: Sequence[Circuit],
        coupling_graph: CouplingGraph | None = None,
        recommender_model_state: dict[str, Tensor] | None = None,
    ) -> None:
        """
        Constructor for the TorchUnitaryRecommender.

        Note:
            It cannot be determined by TorchUnitaryRecommender if the provided
            `recommender_model` and `recommender_state` correspond to the
            given `seed_circuits` and `coupling_graph` other than by checking
            dimensions. This must be verified by the user.

        args:
            recommender_model (nn.Module): A specification of the recommender
                model's architecture.

            seed_circuits (Sequence[Circuit]): Possible circuits that the
                `recommender_model` can propose.

            coupling_graph (CouplingGraph | None): If provided, seed circuits
                must conform to this CouplingGraph. Otherwise, any qubit
                iteractions are allowed. (Default: None)

            recommender_model_state (dict[str, Tensor] | None): The optional
                `state_dict` for the `recommender_model`. If not provided,
                then it is assumed that the state has already been loaded.
                (Default: None)

        raises:
            ValueError: If the number of seed_circuits is not the size of
                the `recommender_model` output.

            ValueError: If a coupling_graph is provided but one or more of
                the `seed_circuits` does not conform.
        """
        # Check that each seed circuit matches coupling_graph
        if coupling_graph and not all(
            edge in coupling_graph for s in seed_circuits
            for edge in s.coupling_graph
        ):
            raise ValueError(
                f'Provided coupling_graph ({coupling_graph}) does not match '
                'seed coupling_graph.'
            )
        # Check output of recommender model matches seed_circuits size
        try:
            if recommender_model.num_templates != len(seed_circuits):
                raise ValueError(
                    'The size of the recommender_model output is '
                    f'{recommender_model.num_templates}, but '
                    f'{len(seed_circuits)} seeds were provided.'
                )
        except AttributeError:
            _logger.warning(
                '`recommender_model` has no `num_templates` attribute, '
                'cannot ensure its compatibility with `seed_circuits`.'
            )

        self.coupling_graph = coupling_graph
        if self.coupling_graph is None:
            num_q = max(s.num_qudits for s in seed_circuits)
            self.coupling_graph = CouplingGraph.all_to_all(num_q)

        self.seed_circuits = seed_circuits
        self.recommender_model = recommender_model
        # Load recommender model state
        if recommender_model_state is not None:
            if isinstance(recommender_model_state, str):
                recommender_model_state = torch.load(
                    recommender_model_state, map_location='cpu'
                )
            self.recommender_model.load_state_dict(
                recommender_model_state, strict=True
            )
        self.recommender_model.eval()

    def encode(self, circuit: Circuit) -> Any:
        """
        Encode a circuit's unitary as a PyTorch Tensor.

        args:
            circuit (Circuit): The circuit to be encoded.

        returns:
            (Tensor): A flattened Tensor view of the unitary. Real components
                are stacked on top of imaginary ones.
        """
        unitary = circuit.get_unitary().numpy
        real_x = np.real(unitary).flatten()
        imag_x = np.imag(unitary).flatten()
        x = np.hstack([real_x, imag_x])
        x = Tensor(x)
        return x

    def _recommend(
        self,
        encoding: Tensor,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        """
        Recommend seed circuits based off Tensor encoding of a circuit.

        args:
            encoding (Tensor): A Tensor encoding of a circuit.

            seeds_per_rec (int): The number of seeds to recommend per call to
                `self.recommend`.

        returns:
            (list[Circuit]): A list of seed circuits taken from
                `self.seed_circuits`.
        """
        encoding = encoding.float()
        logits = self.recommender_model(encoding)
        _, indices = topk(logits, seeds_per_rec, dim=-1)
        return [self.seed_circuits[i] for i in indices]

    def batched_recommend(
        self,
        batched_circuits: Sequence[Circuit],
        seeds_per_rec: int = 1,
    ) -> Sequence[Sequence[Circuit]]:
        batch_size = len(batched_circuits)
        batched_encodings = torch.stack(
            [self.encode(op) for op in batched_circuits]
        )
        batched_logits = self.recommender_model(batched_encodings)
        _, indices = topk(batched_logits, seeds_per_rec, dim=-1)
        indices = indices.tolist()
        return [self.seed_circuits[i] for i in indices]
        # batched_seeds = []
        # for b in range(batch_size):
        #    seeds = [self.seed_circuits[i] for i in indices[b]]
        #    batched_seeds.append(seeds)
        # return batched_seeds

    def recommend(
        self,
        circuit: Circuit,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        """
        Recommend seed circuits based off Tensor encoding of a circuit.

        args:
            circuit (Circuit): The circuit for which to predict seeds.

            seeds_per_rec (int): The number of seeds to recommend per call to
                `self.recommend`.

        returns:
            (list[Circuit]): A list of seed circuits taken from
                `self.seed_circuits`.
        """
        encoding = self.encode(circuit)
        seeds = self._recommend(encoding, seeds_per_rec)
        return seeds
