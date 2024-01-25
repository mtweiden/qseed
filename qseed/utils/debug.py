import torch.nn as nn
from torch import Tensor
from bqskit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.gates import U3Gate, CNOTGate
from bqskit.qis.graph import CouplingGraph
from random import choice, randint
from typing import Any

from qseed.recommender import Recommender


class DebugRecommender(Recommender):
    def __init__(
        self,
        seeds: list[Circuit],
        coupling: CouplingGraph | None = None
    ) -> None:
        super().__init__(coupling)
        self.seeds = seeds

    def encode(self, operation: Operation) -> Any:
        return choice(range(len(self.seeds)))

    def recommend(
        self,
        encoding: Any,
        seeds_per_rec: int = 1
    ) -> list[Circuit]:
        return [self.seeds[encoding] for _ in range(seeds_per_rec)]


class DebugModel(nn.Module):
    def __init__(self, num_qudits: int, num_seeds: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * 4 ** num_qudits, num_seeds)
        self.num_templates = num_seeds

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class DebugCircuitGenerator:
    def __init__(
        self,
        num_qudits: int,
        num_circuits: int,
        coupling_graph: CouplingGraph | None = None,
        max_gates: int = 4
    ) -> None:
        self.circuits = []
        if coupling_graph is None:
            coupling_graph = CouplingGraph.all_to_all(num_qudits)
        for _ in range(num_circuits):
            seed = Circuit(num_qudits)
            for q in range(num_qudits):
                seed.append_gate(U3Gate(), (q))
            for _ in range(randint(1, max_gates)):
                u, v = choice(list(coupling_graph._edges))
                seed.append_gate(CNOTGate(), (u, v))
                seed.append_gate(U3Gate(), (u))
                seed.append_gate(U3Gate(), (v))
            self.circuits.append(seed)
