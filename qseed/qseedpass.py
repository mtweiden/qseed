"""This module implements the SynthesisPass abstract class."""
from __future__ import annotations

from typing import Any
from typing import Sequence
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.opt.cost.functions.residuals.hilbertschmidt import (
    HilbertSchmidtResidualsGenerator,
)
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.generator import LayerGenerator
from qseed.multiseed import MultiSeedLayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics.astar import AStarHeuristic
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

_logger = logging.getLogger(__name__)

class QSeedSynthesisPass(BasePass):
    """
    QSeedSynthesisPass class.

    QSeed calls QSearch but can begin synthesis from circuits that are already
    populated with gates.
    """

    def __init__(
        self,
        seed_circuits: Circuit | Sequence[Circuit] | None = None, 
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
            seed_circuits (Circuit | Sequence[Circuit] | None): A number of 
                circuits from which synthesis will be started. (Default: None)

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
        if seed_circuits is None:
            self.seed_circuits = None
        else:
            self.seed_circuits = seed_circuits if isinstance(
                seed_circuits, Sequence,
            ) else [seed_circuits]

        self.back_step_size = back_step_size
        self.forward_generator = forward_generator
        self.back_step_size = back_step_size

        # Regular QSearch parameters
        self.heuristic_function = heuristic_function
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth
        self.instantiate_options = instantiate_options

    def _check_size(self, utry: UnitaryMatrix, seeds : list[Circuit]) -> bool:
        return utry.num_qudits == seeds[0].num_qudits

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
        # Seeds from recommender pass
        if 'recommended_seeds' in data:
            if self._check_size(utry, data['recommended_seeds'][0]):
                seeds = data['recommended_seeds'].pop(0)
            else:
                raise RuntimeError(
                    'Recommended seeds are a different size than the target.'
                )
        # Manually entered seeds
        elif 'seeds' in data:
            if self._check_size(utry, data['seeds']):
                seeds = data['seeds'] if isinstance(data['seeds'], Sequence) \
                    else [data['seeds']]
            else:
                raise RuntimeError(
                    'Manually entered seeds are a different size than the '
                    'target.'
                )
        # Default seeds set at construction time
        else:
            if self._check_size(utry, self.seed_circuits):
                seeds = self.seed_circuits
            else:
                raise RuntimeError(
                    'Default seeds are a different size than the target.'
                )
        
        if seeds is None:
            raise RuntimeError(
                'No seeds at contructor time or in `data[seeds]`.'
            )

        layer_generator = MultiSeedLayerGenerator(
            seed_circuits=seeds,
            forward_generator=self.forward_generator,
            back_step_size=self.back_step_size,
        )

        seeded_synth = QSearchSynthesisPass(
            layer_generator=layer_generator,
            heuristic_function=self.heuristic_function,
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
