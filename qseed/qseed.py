"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from typing import Any
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions.residuals.hilbertschmidt import (
    HilbertSchmidtResidualsGenerator,
)
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.multiseed import MultiSeedLayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics.astar import AStarHeuristic
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class QSeedSynthesisPass(BasePass):
    """
    QSeedSynthesisPass class.

    QSeed calls QSearch but can begin synthesis from circuits that are already
    populated with gates.
    """

    def __init__(
        self,
        seed_circuits: Circuit | Sequence[Circuit],
        forward_generator: LayerGenerator = SimpleLayerGenerator(),
        back_step_size: int = 1,
        heuristic_function: HeuristicFunction = AStarHeuristic(),
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = None,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        The constructor for the QSeedSynthesisPass.

        Args:
            seed_circuits (Circuit | Sequence[Circuit]): A number of circuits
                from which synthesis will be started.

            forward_generator (LayerGenerator): Defines how synthesis will
                proceed when adding new gates. (Default: SimpleLayerGenerator)

            back_step_size (int): The number of atomic synthesis gate units to
                remove when going towards the root in the QSearch synthesis
                tree. (Default: 1)

            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-10)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
        """
        # Seeded synthesis parameters
        self.seed_circuits = seed_circuits if isinstance(
            seed_circuits, Sequence,
        ) else [seed_circuits]
        self.back_step_size = back_step_size
        self.layer_generator = MultiSeedLayerGenerator(
            self.seed_circuits,
            forward_generator=forward_generator,
            back_step_size=back_step_size,
        )

        # Regular QSearch parameters
        self.heuristic_function = heuristic_function
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth
        self.instantiate_options = instantiate_options

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:
        """
        Synthesis abstract method to synthesize a UnitaryMatrix into a Circuit.

        Args:
            utry (UnitaryMatrix): The unitary to synthesize.

            data (Dict[str, Any]): Associated data for the pass.
                Can be used to provide auxillary information from
                previous passes. This function should never error based
                on what is in this dictionary.

        Note:
            This function should be self-contained and have no side effects.
        """
        seeded_synth = QSearchSynthesisPass(
            heuristic_function=self.heuristic_function,
            layer_generator=self.layer_generator,
            success_threshold=self.success_threshold,
            cost=self.cost,
            max_layer=self.max_layer,
            store_partial_solutions=self.store_partial_solutions,
            partials_per_depth=self.partials_per_depth,
            instantiate_options=self.instantiate_options,
        )
        return seeded_synth.synthesize(utry, data)

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        if len(data) == 0:
            data = dict()

        target_utry = self.get_target(circuit, data)
        circuit.become(self.synthesize(target_utry, data))
