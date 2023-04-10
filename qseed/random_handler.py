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
from bqskit.passes import ScanningGateRemovalPass

from qseed.models.unitary_learner import UnitaryLearner
from qseed.recommender import TopologyAwareRecommenderPass
from qseed.qseedpass import QSeedSynthesisPass
from qseed.randrecforeach import RandomRecForEachBlockPass


_logger = logging.getLogger(__name__)


class RandomHandler:
	def __init__(
		self,
	) -> None:
		"""
		The constructor for the HandlerPass. This function sets up what the
		handler will perform before synthesis is called on blocks.
		"""
		self.num_qubits = 3
		self.topologies = ['a','b','c']
		self.models, self.states, self.templates = [], [], []
		for topology in self.topologies:
			with open(f'templates/circuits_{topology}.pickle','rb') as f:
				self.templates.append(pickle.load(f))
		self.recorder = RecordStatsPass()
	
	def _filter(self, circuit : Circuit | Operation | CircuitGate) -> bool:
		return circuit.num_qudits == self.num_qubits

	def handle(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
		start_time = time()
		start_total_cnots, opt_total_cnots = 0,0
		start_total_u3s, opt_total_u3s = 0,0
		
		_logger.info('Random compilation')

		start_total_cnots, start_total_u3s = self._count_gates(circuit)

		block_passes = [
			QSeedSynthesisPass(),
			ScanningGateRemovalPass(),
			self.recorder,
		]
		task = CompilationTask(
			circuit, 
			[
				QuickPartitioner(block_size=3),
				RandomRecForEachBlockPass(
					block_passes,
					self.templates,
					collection_filter=self._filter,
				),
				UnfoldPass(),
			]
		)
		with Compiler(num_workers=64) as compiler:
			new_circuit = compiler.compile(task)
		stop_time = time()
		opt_total_cnots, opt_total_u3s = self._count_gates(new_circuit)
		inst_calls = self._get_calls(data)

		start_stats = start_time, start_total_cnots, start_total_u3s
		stop_stats = stop_time, opt_total_cnots, opt_total_u3s, inst_calls
		self.record_stats(start_stats, stop_stats)
	
	def _count_gates(self, circuit : Circuit) -> int:
		counts = circuit.gate_counts
		cnots = counts[CNOTGate()] if CNOTGate() in counts else 0
		swaps = counts[SwapGate()] if SwapGate() in counts else 0
		u3s   = counts[U3Gate()] if U3Gate() in counts else 0
		return cnots + 3*swaps, u3s
	
	def _get_calls(self, data : PassData) -> int:
		if 'ForEachBlockPass_data' not in data:
			return []
		calls = [] 
		if not 'ForEachBlockPass_data' in data:
			return calls
		for fakesubdata in data['ForEachBlockPass_data']:
			for subdata in fakesubdata:
				if 'instantiation_calls' in subdata:
					# -1 because we don't count empty circuit instantiation
					calls.append(subdata['instantiation_calls'] - 1)
		return calls


	def record_stats(self, start_stats: tuple, stop_stats: tuple) -> None:
		"""Log statistics about the run."""
		start_time, start_cnots, start_u3s = start_stats
		opt_time, opt_cnots, opt_u3s, calls = stop_stats

		duration = opt_time - start_time
		mean_calls = np.mean(calls) if len(calls) > 0 else 0
		std_calls  = np.std(calls)  if len(calls) > 0 else 0

		time_str = f'Optimization time: {duration:>0.3f}s'
		depth_str = f'Optimized u3 gates: {start_u3s} -> {opt_u3s}'
		count_str = f'Optimized cx gates: {start_cnots} -> {opt_cnots}'
		calls_str = f'Calls: mean - {mean_calls}  std - {std_calls}'

		_logger.info(time_str)
		_logger.info(depth_str)
		_logger.info(count_str)
		_logger.info(calls_str)