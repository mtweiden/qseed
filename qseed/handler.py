"""HandlerPass packages makes synthesis of partitioned circuits easier."""
from __future__ import annotations

import logging
import pickle
import torch
from typing import Any, Sequence
from timeit import default_timer as time
import numpy as np

from bqskit.compiler.passdata import PassData
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.compiler import CompilationTask, Compiler
from bqskit.passes import QuickPartitioner
from bqskit import Circuit
from bqskit.ir.gates import CNOTGate, SwapGate, U3Gate
from bqskit.passes.util import RecordStatsPass
from bqskit.ir import Operation
from bqskit.ir.circuit import CircuitGate
from bqskit.passes import UnfoldPass

from qseed.models.pauli_learner import PauliLearner
from qseed.recommender import TopologyAwareRecommenderPass
from qseed.qseedpass import QSeedSynthesisPass


_logger = logging.getLogger(__name__)


class Handler:
    def __init__(
        self,
    ) -> None:
        """
        The constructure for the HandlerPass. This function sets up what the
        handler will perform before synthesis is called on blocks.
        """
        self.num_qubits = 3
        self.topologies = ['a','b','c']
        models = []
        states = []
        templates = []
        for topology in self.topologies:
            models.append(PauliLearner())
            states.append(
                torch.load(
                    f'qseed/models/learner_{topology}.model',
                    map_location='cuda',
                )
            )
            with open(f'templates/circuits_{topology}.pickle','rb') as f:
                templates.append(pickle.load(f))
        self.recommender = TopologyAwareRecommenderPass(
            models, states, templates,
        )
        self.recorder = RecordStatsPass()
    
    def _filter(self, circuit : Circuit | Operation | CircuitGate) -> bool:
        return circuit.num_qudits == 3

    def handle(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        start_time = time()
        start_total_cnots, opt_total_cnots = 0,0
        start_total_u3s, opt_total_u3s = 0,0
        
        _logger.info('QSeed compilation')

        start_total_cnots, start_total_u3s = self._count_gates(circuit)

        block_passes = [
            self.recommender, 
            QSeedSynthesisPass(),
            self.recorder,
        ]
        task = CompilationTask(
            circuit, 
            [
                QuickPartitioner(block_size=3),
                ForEachBlockPass(
                    block_passes,
                    collection_filter=self._filter,
                ),
                UnfoldPass(),
            ]
        )
        with Compiler() as compiler:
            new_circuit = compiler.compile(task)
        opt_total_cnots, opt_total_u3s = self._count_gates(new_circuit)

        start_stats = start_time, start_total_cnots, start_total_u3s
        stop_stats = time(), opt_total_cnots, opt_total_u3s
        self.record_stats(start_stats, stop_stats)
    
    def _count_gates(self, circuit : Circuit) -> int:
        counts = circuit.gate_counts
        cnots = counts[CNOTGate()] if CNOTGate() in counts else 0
        swaps = counts[SwapGate()] if SwapGate() in counts else 0
        u3s   = counts[U3Gate()] if U3Gate() in counts else 0
        return cnots + 3*swaps, u3s

    def record_stats(self, start_stats: tuple, stop_stats: tuple) -> None:
        """Log statistics about the run."""
        start_time, start_cnots, start_u3s = start_stats
        #opt_time, opt_cnots, opt_u3s, calls = stop_stats
        opt_time, opt_cnots, opt_u3s = stop_stats
        duration = opt_time - start_time
        #mean_calls = np.mean(calls)
        #std_calls  = np.std(calls)
        time_str = f'Optimization time: {duration:>0.3f}s'
        depth_str = f'Optimized u3 gates: {start_u3s} -> {opt_u3s}'
        count_str = f'Optimized cx gates: {start_cnots} -> {opt_cnots}'
        #calls_str = f'Calls: mean - {mean_calls}  std - {std_calls}'
        _logger.info(time_str)
        _logger.info(depth_str)
        _logger.info(count_str)
        #_logger.info(calls_str)
