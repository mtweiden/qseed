"""HandlerPass packages makes synthesis of partitioned circuits easier."""
from __future__ import annotations

import logging
import pickle
from typing import Any, Sequence
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit import Circuit
from bqskit import MachineModel
from bqskit.ir.region import CircuitRegion
from bqskit.ir.location import CircuitLocation
from bqskit.ir.point import CircuitPoint
from bqskit.ir.operation import Operation
from qseed.qseedpass import QSeedSynthesisPass
from timeit import default_timer as time
from bqskit.ir.gates import CNOTGate, SwapGate

_logger = logging.getLogger(__name__)

# WidthPredicate to filter out 2 qubit blocks

# NOTES
# Inputs:
#   - quantum circuit (if mapped, then to mesh topology)
#
# Outputs:
#   - optimized circuit (if mapped before, still mapped)
#   - statistics log file about depth, gate counts, compilation time
#
# Assumptions:
#   - All blocks are of blocksize 3. Smaller blocks are filtered out.
#   - 4 QSeed agents, each to handle one topology (3 linear, 1 complete).
#   - Circuits will all look like QFTs, so use static recommendation
#     of things that work for QFTs given that topology.
#
# TODO: 
#       
# Completed:
#   - Record time, depth, and gate counts at start and end of compilation.
#       - This should not be a new pass, just part of what happens when
#         run is called on the handler.
#   - Function that takes in a CircuitGate and returns an integer which
#     corresponds to the topology that block uses.

class HandlerPass(BasePass):
    def __init__(
        self,
    ) -> None:
        """
        The constructure for the HandlerPass. This function sets up what the
        handler will perform before synthesis is called on blocks.
        """
        self.num_qubits = 3
        self.topologies = {
            0: ((0,1),(1,2)),
            1: ((0,1),(0,2)),
            2: ((0,2),(1,2)),
            3: ((0,1),(1,2),(0,2)),
        }

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        start_stats = time(), circuit.depth, circuit.gate_counts

        for cycle, op in circuit.operations_with_cycles():
            if op.num_qudits != self.num_qubits:
                continue
            # Determine seeds
            block = circuit.from_operation(op)
            block.unfold_all()
            original_cnots = self._count_cnots(block)
            seeds = self.seed_recommender(block, circuit)

            # Set up seeds
            qseed = QSeedSynthesisPass(seeds)
            sub_data = {}
            if 'machine_model' in data:
                model = data['machine_model']
                sub_data['machine_model'] = self._sub_machine(model, op.location)
            
            # Run seeded synthesis
            qseed.run(block, sub_data)
            opt_cnots = self._count_cnots(block)

            # Replace if optimized block is smaller
            if opt_cnots < original_cnots:
                circuit.replace((cycle, op.location[0]), block)

        stop_stats = time(), circuit.depth, circuit.gate_counts
        self.record_stats(start_stats, stop_stats)
    
    def seed_recommender(
        self, 
        block: Circuit, 
        circuit: Circuit | None = None,
    ) -> list[Circuit]:
        """
        Recommend a circuit seed to use for seeded synthesis.

        NOTE: Currently only a static recommender strategy is supported.
        TODO: Integrate ML recommender strategy.

        Args:
            block (Circuit): Subcircuit to be optimized.
            
            circuit (Circuit): Complete partitioned circuit that is being 
                optimized.
        
        Returns:
            seeds (list[Circuit]): A list of candidate seed circuits.
        """
        topology_code = self._extract_subtopology(block)
        if topology_code not in [0,1,2,3]:
            return [Circuit(self.num_qubits)]
        
        # NOTE: ONLY QFT IS CURRENTLY SUPPORTED
        if topology_code == 0:
            template_pickle_file = 'qft_templates_a.pickle'
        elif topology_code == 1:
            template_pickle_file = 'qft_templates_b.pickle'
        elif topology_code == 2:
            template_pickle_file = 'qft_templates_c.pickle'
        elif topology_code == 3:
            template_pickle_file = 'qft_templates_d.pickle'
        
        with open(f'static_seeds/{template_pickle_file}','rb') as f:
            return pickle.load(f)
    
    def _count_cnots(self, circuit : Circuit) -> int:
        counts = circuit.gate_counts
        cnots = counts[CNOTGate()] if CNOTGate() in counts else 0
        swaps = counts[SwapGate()] if SwapGate() in counts else 0
        return cnots + swaps
    
    def _sub_machine(
        self, 
        model : MachineModel,
        location : CircuitLocation,
    ) -> MachineModel:
        graph = model.coupling_graph
        subgraph = graph.get_subgraph(location)
        return MachineModel(len(location), subgraph)

    def _extract_subtopology(self, block: Circuit) -> int:
        """
        Return the integer code associated with the subtopology of the block
        located at `region` in `circuit`.

        Args:
            circuit (Circuit): Partitioned quantum circuit.

            point (CircuitPoint): Location of the desired block.

        Returns:
            topology_id (int): Integer id code for block's subtopology.
        
        Raises:
            RuntimeError: If operation at `point` is not a circuit block.
        """
        block.unfold_all()
        block_edges = [e for e in block.coupling_graph]

        for top_id,topology in self.topologies.items():
            if len(topology) != len(block_edges):
                continue
            if all([e in block_edges for e in topology]):
                return top_id
        return -1 # TOPOLOGY NOT FOUND

    def record_stats(self, start_stats: tuple, stop_stats: tuple) -> None:
        """Log statistics about the run."""
        start_time, start_depth, start_counts = start_stats
        opt_time, opt_depth, opt_counts = stop_stats
        duration = opt_time - start_time
        start_multi = sum(
            [start_counts[g] for g in start_counts if g.num_qudits > 1]
        )
        opt_multi = sum(
            [opt_counts[g] for g in opt_counts if g.num_qudits > 1]
        )
        time_str = f'Optimization time: {duration:>0.3f}s'
        depth_str = f'Optimized depth: {start_depth} -> {opt_depth}'
        count_str = f'Optimized >1q counts: {start_multi} -> {opt_multi}'
        _logger.info(time_str)
        _logger.info(depth_str)
        _logger.info(count_str)