from typing import Callable
from typing import Sequence

import logging

import torch
from torch import Tensor

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.qis.graph import CouplingGraph
from bqskit.compiler.passdata import PassData
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_sequence

from math import ceil

from qseed.recommender import Recommender
from qseed.tu_recommender import TorchUnitaryRecommender
from qseed.utils.loading import load_seed_circuits
from qseed.models.unitary_learner import UnitaryLearner

from timeit import default_timer

_logger = logging.getLogger(__name__)


class QSeedRecommenderPass(BasePass):

    # pass_down_block_specific_key_prefix = (
    #     'ForEachBlockPass_specific_pass_down_seed_circuits'
    # )
    pass_down_block_specific_key_prefix = 'seed_circuits'
    """Key for injecting a map from block number to seed circuits."""
    recommender_state_key = 'recommender_model_state'

    def __init__(
        self,
        load_function: Callable,
        seeds_per_rec: int = 3,
        batch_size: int = 64,
    ) -> None:
        """
        Construct a QSeedRecommenderPass.

        This pass optimizes Circuits partitioned into 3 qubit blocks.

        args:
            seeds_per_rec (int): The number of seeds to recommend per circuit
                Operation. (Default: 3)

            batch_size (int): Max number of operations to pass to a recommender
                at a time. (Default: 64)
        """
        if not is_integer(seeds_per_rec):
            raise TypeError(
                f'seeds_per_rec must be an integer, got {type(seeds_per_rec)}.'
            )
        if seeds_per_rec <= 0:
            raise ValueError('seeds_per_rec must be positive nonzero.')

        self.seeds_per_rec = seeds_per_rec
        self.batch_size = batch_size
        self.seeds = load_seed_circuits()

        self.load_function = load_function

    def check_recommenders(
        self,
        recommender_models: Sequence[Recommender],
    ) -> None:
        if not is_sequence(recommender_models):
            raise TypeError(
                '`recommender_models` must be a Sequence, got '
                f'{type(recommender_models)}.'
            )
        if not all(isinstance(r, Recommender) for r in recommender_models):
            raise TypeError(
                'recommenders must be of type Recommender'
            )
        if len(recommender_models) != 3:
            raise ValueError(
                '`recommender_models` must be a Sequence of length'
                f' 3, got length {len(recommender_models)}.'
            )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Run QSeed recommendation on the given `circuit`.
        """
        start_time = default_timer()
        if self.pass_down_block_specific_key_prefix in data:
            msg = (
                f'Key "{self.pass_down_block_specific_key_prefix}" already '
                'found in PassData. It will be overwritten.'
            )
            _logger.warning(msg)

        # Load models directly from worker's local cache
        worker_cache = get_runtime().get_cache()
        if 'recommenders' not in worker_cache:
            recommenders = self.load_function()
            worker_cache['recommenders'] = recommenders
            # m = (
            #     'The `recommenders` must be loaded into the worker\'s '
            #     'local cache before this pass can be run.'
            # )
            # raise RuntimeError(m)
        else:
            recommenders = worker_cache['recommenders']
        self.check_recommenders(recommenders)

        coupling_recommender_map = {
            rec.coupling_graph: i for (i, rec) in enumerate(recommenders)
        }

        # TODO: Batch into `self.batch_size` batches
        topology = data.connectivity
        if circuit.num_qudits != 3:
            return
        rec_index = self.assign_recommender(
            circuit, coupling_recommender_map, topology
        )
        if rec_index < 0:
            return

        seeds = recommenders[rec_index].recommend(circuit, self.seeds_per_rec)
        data[self.pass_down_block_specific_key_prefix] = seeds
        duration = default_timer() - start_time
        print(f'Finished recommending: {duration:>0.3f}s')

    def assign_recommender(
        self,
        circuit: Circuit,
        topology_map: dict[CouplingGraph, int],
        topology: CouplingGraph
    ) -> int:
        """Assign a recommender based off topology of subcircuit."""
        # topology aware case
        location = [_ for _ in range(circuit.num_qudits)]
        graph = topology.get_subgraph(location)
        if graph not in topology_map:
            return -1
        else:
            return topology_map[graph]
