import logging

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitGate
from bqskit.ir.operation import Operation
from bqskit.qis.graph import CouplingGraph
from bqskit.compiler.passdata import PassData
from bqskit.utils.typing import is_sequence, is_integer

from qseed.recommender import Recommender
from qseed.tu_recommender import TorchUnitaryRecommender
from models.unitary_learner import UnitaryLearner

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

        raises:
        """
        if not is_integer(seeds_per_rec):
            raise TypeError(
                f'seeds_per_rec must be an integer, got {type(seeds_per_rec)}.'
            )
        if seeds_per_rec <= 0:
            raise ValueError('seeds_per_rec must be positive nonzero.')

        self.seeds_per_rec = seeds_per_rec
        self.batch_size = batch_size

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Run QSeed recommendation on the given `circuit`.
        """
        if self.pass_down_block_specific_key_prefix in data:
            msg = (
                f'Key "{self.pass_down_block_specific_key_prefix}" already '
                'found in PassData. It will be overwritten.'
            )
            _logger.warning(msg)

        # Load Recommenders from PassData
        if self.recommender_state_key not in data:
            raise RuntimeError(
                'A recommender states must be set in PassData using '
                f'the key {self.recommender_state_key}.'
            )
        states = data[self.recommender_state_key]
        recommenders = [
            TorchUnitaryRecommender(
                UnitaryLearner(), seeds, graph, state,
            ) for state, seeds, graph in states
        ]
        coupling_recommender_map = {
            rec.coupling_graph: i for (i, rec) in enumerate(recommenders)
        }

        seed_map = {}
        # rec_info: (index, encoding)
        rec_info = [[] for _ in range(len(recommenders))]
        # Bin blocks by topology
        topology = data.connectivity
        for block_num, block in enumerate(circuit):
            if not isinstance(block.gate, CircuitGate) or block.num_qudits != 3:
                continue
            rec_index = self.assign_recommender(
                block, coupling_recommender_map, topology
            )
            if rec_index >= 0:
                encoding = recommenders[rec_index].encode(block)
                rec_info[rec_index].append((block_num, encoding))

        # TODO: Batch into `self.batch_size` batches
        # Call recommenders on batches, fill seed_map
        for rec_index, rec in enumerate(recommenders):
            for block_num, encoding in rec_info[rec_index]:
                seeds = rec.recommend(encoding, self.seeds_per_rec)
                seed_map[block_num] = seeds
        data[self.pass_down_block_specific_key_prefix] = seed_map

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
