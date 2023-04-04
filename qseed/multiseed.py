"""This module implements the MultiSeedLayerGenerator class."""
from __future__ import annotations

import logging
from typing import Any
from typing import Sequence

from bqskit.compiler.passdata import PassData
from bqskit.compiler.machine import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

_logger = logging.getLogger(__name__)


class MultiSeedLayerGenerator(LayerGenerator):
    """
    The MultiSeedLayerGenerator class.

    Acts like a SimpleLayerGenerator, but adds support for having multiple
    starting heads of a solution tree.
    """

    def __init__(
        self,
        seed_circuits: Sequence[Circuit],
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
        back_step_size: int = 1,
    ) -> None:
        """
        Construct a MultiSeedLayerGenerator.

        Args:
            seed_circuits (Sequence[Circuit]): The initial layer of circuits
                to use as seeds in the synthesis flow.

            forward_generator (LayerGenerator): The mechanism for adding new
                gates in the synthesis flow.

            back_step_size (int): Number of nodes in the search tree to
                remove when going upwards towards the root. (Default: 1)

        Raises:
            TypeError: If `seed_circuits` is not a Sequence.

            TypeError: If each element in `seed_circuits` is not a Circuit.

            TypeError: If the number of qudits in each circuit in
                `seed_circuits` are not all the same.

            TypeError: If `forward_generator` is not a LayerGenerator.
        """
        if not isinstance(seed_circuits, Sequence):
            raise TypeError(
                f'Expected Sequence for `seed_circuits`, '
                f'got {type(seed_circuits)}.',
            )

        if not all([isinstance(c, Circuit) for c in seed_circuits]):
            raise TypeError('All objects in `seed_circuits` must be Circuits.')

        if not all([
            c.num_qudits == seed_circuits[0].num_qudits
            and c.radixes[0] == seed_circuits[0].radixes[0]
            for c in seed_circuits
        ]):
            raise TypeError(
                'All `seed_circuits` must be the same width and radix.',
            )

        if not isinstance(forward_generator, LayerGenerator):
            raise TypeError('forward_generator must be a LayerGenerator.')

        self.seed_circuits = seed_circuits
        self.forward_generator = forward_generator
        self.back_step_size = back_step_size

    def gen_initial_layer(
        self,
        target: UnitaryMatrix | StateVector,
        data: dict[str, Any],
    ) -> Circuit:
        """
        Generate the initial layer, see LayerGenerator for more.

        Raises:
            TypeError: If `target` is not a UnitaryMatrix or StateVector.

            ValueError: If `target` has a radix mismatch with
                any of the `self.seed_circuits`.

            ValueError: If 'target has a dimension mismatch with any of the
                `self.seed_circuits`.
        """

        if not isinstance(target, (UnitaryMatrix, StateVector)):
            raise TypeError(f'Expected unitary or state, got {type(target)}')

        for radix in target.radixes:
            if radix != self.seed_circuits[0].radixes[0]:
                raise ValueError(
                    'Radix mismatch between `target` and `seed_circuits`.',
                )

        if not all([target.dim == s.dim for s in self.seed_circuits]):
            raise ValueError('Seed dimensions do not match with target.')

        empty_circuit = Circuit(target.num_qudits, target.radixes)

        data['seed_seen_before'] = {self.hash_structure(empty_circuit)}

        assert len(empty_circuit) == 0

        return empty_circuit

    def gen_successors(
        self,
        circuit: Circuit,
        data: PassData,
    ) -> list[Circuit]:
        """
        Generate the successors of a circuit node.

        Raises:
            TypeError: If circuit is not a Circuit.

            ValueError: If circuit is a single-qudit circuit.

            ValueError: If a coupling map is provided, but it does not conform
                to the connectivity of the seeds.
        """
        if 'seeds' in data:
            seeds = data['seeds']
        else:
            seeds = self.seed_circuits

        if 'machine_model' in data:
            model: MachineModel = data.model
            filtered_seeds = []
            for seed in seeds:
                seed_graph = seed.coupling_graph
                model_graph = model.coupling_graph
                if seed_graph.is_embedded_in(model_graph):
                    filtered_seeds.append(seed)
            if (seed_diff := len(filtered_seeds) - len(seeds)) != 0:
                _logger.warn(
                    f'{seed_diff} seeds filtered out due to topology mismatch.',
                )
            seeds = filtered_seeds

        if circuit.num_operations == 0:
            return seeds

        if not isinstance(circuit, Circuit):
            raise TypeError(
                f'Expected a Sequence of Circuits, got {type(circuit)}.',
            )

        if circuit.num_qudits < 2:
            raise ValueError('Cannot expand a single-qudit circuit.')

        # Generate successors
        successors: list[Circuit] = self.forward_generator.gen_successors(
            circuit, data,
        )

        # Search reverse direction
        ancestor_circuits = self.remove_atomic_units(circuit)
        successors = ancestor_circuits + successors

        filtered_successors = []
        for s in successors:
            h = self.hash_structure(s)
            if h not in data['seed_seen_before']:
                data['seed_seen_before'].add(h)
                filtered_successors.append(s)
        return filtered_successors

    def remove_atomic_units(self, circuit: Circuit) -> list[Circuit]:
        """
        Search for the last `back_step_size` number of atomic units:

            -- two_qudit_gate -- single_qudit_gate_1 --
                    |
            -- two_qudit_gate -- single_qudit_gate_2 --

        and remove them.
        """
        num_removed = 0
        ancestor_circuits = []

        circuit_copy = circuit.copy()
        for cycle, op in circuit.operations_with_cycles(reverse=True):

            if num_removed >= self.back_step_size:
                break
            if op.num_qudits == 1:
                continue

            for place in op.location:
                point = (cycle + 1, place)
                if not circuit_copy.is_point_idle(point):
                    circuit_copy.pop(point)

            circuit_copy.pop((cycle, op.location[0]))

            ancestor_circuits.append(circuit_copy)
            circuit_copy = circuit_copy.copy()
            num_removed += 1

        return ancestor_circuits

    @staticmethod
    def hash_structure(circuit: Circuit) -> int:
        hashes = []
        for cycle, op in circuit.operations_with_cycles():
            hashes.append(hash((cycle, str(op))))
            if len(hashes) > 100:
                hashes = [sum(hashes)]
        return sum(hashes)
