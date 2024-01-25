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

    pass_down_block_specific_key_prefix = (
        'ForEachBlockPass_specific_pass_down_seed_circuits'
    )
    """Key for injecting a map from block number to seed circuits."""
    recommender_state_key = 'recommender_model_state'

    def __init__(
        self,
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
            m = (
                'The `recommenders` must be loaded into the worker\'s '
                'local cache before this pass can be run.'
            )
            raise RuntimeError(m)
        recommenders = worker_cache['recommenders']
        self.check_recommenders(recommenders)

        coupling_recommender_map = {
            rec.coupling_graph: i for (i, rec) in enumerate(recommenders)
        }

        # TODO: Batch into `self.batch_size` batches
        seed_map = {}
        topology = data.connectivity
        op_by_recommender = [[] for _ in range(len(recommenders))]
        block_num_by_recommender = [[] for _ in range(len(recommenders))]
        for block_num, block in enumerate(circuit):
            if not isinstance(block.gate, CircuitGate) or block.num_qudits != 3:
                continue
            rec_index = self.assign_recommender(
                block, coupling_recommender_map, topology
            )
            if rec_index >= 0:
                op_by_recommender[rec_index].append(block)
                block_num_by_recommender[rec_index].append(block_num)

        for rec_num in range(len(recommenders)):
            operations = op_by_recommender[rec_num]
            num_batches = int(ceil(len(operations) / self.batch_size))
            batched_seeds = []
            for batch in range(num_batches):
                start = batch * self.batch_size
                stop = (batch + 1) * self.batch_size
                batched_operations = operations[start:stop]
                batched_seeds.extend(
                    recommenders[rec_num].batched_recommend(
                        batched_operations, self.seeds_per_rec
                    )
                )
            for block_num, indices in zip(
                block_num_by_recommender[rec_num], batched_seeds
            ):
                seed_map[block_num] = [
                    self.seeds[rec_num][i] for i in indices
                ]
        data[self.pass_down_block_specific_key_prefix] = seed_map
        duration = default_timer() - start_time
        print(f'Finished recommending: {duration}s')

    def assign_recommender(
        self,
        block: Operation,
        topology_map: dict[CouplingGraph, int],
        topology: CouplingGraph
    ) -> int:
        """Assign a recommender based off topology of subcircuit."""
        # topology aware case
        subnum = {
            block.location[i]: i for i in range(len(block.location))
        }
        subgraph = topology.get_subgraph(block.location, subnum)
        if subgraph not in topology_map:
            return -1
        else:
            return topology_map[subgraph]
