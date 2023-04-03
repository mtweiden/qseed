from __future__ import annotations
import torch
import numpy as np
import pickle
import logging

from bqskit import Circuit
from bqskit.passes import QuickPartitioner
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import ForEachBlockPass
from bqskit.compiler import Compiler, CompilationTask
from bqskit.ir.gates import CNOTGate, U3Gate, SwapGate
from bqskit.ir.operation import Operation
from timeit import default_timer

from qseed import QSeedSynthesisPass
from qseed.models import PauliLearner
from qseed import PauliRecommenderPass

def count_gates(circuit : Circuit | Operation) -> int:
    if isinstance(circuit, Operation):
        circuit = Circuit.from_operation(circuit)
    counts = circuit.gate_counts
    cnots = counts[CNOTGate()] if CNOTGate() in counts else 0
    swaps = counts[SwapGate()] if SwapGate() in counts else 0
    u3s   = counts[U3Gate()] if U3Gate() in counts else 0
    return cnots + 3*swaps, u3s

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    state_path = 'qseed/models/pauli_encoder.model'
    temp_path = 'static_seeds/templates.pickle'

    model = PauliLearner()
    state = torch.load(state_path,map_location='cpu')
    with open(temp_path,'rb') as f:
        templates = pickle.load(f)

    partitioner = QuickPartitioner()
    recommender = PauliRecommenderPass(model, state, templates)
    qseed = QSeedSynthesisPass()
    #remover = ScanningGateRemovalPass()
    #tasks = [
    #    partitioner, ForEachBlockPass(
    #        [recommender, qseed],
    #        collection_filter=lambda x: x.num_qudits == 3,
    #        replace_filter=lambda x,y: count_gates(x)[0] < count_gates(y)[0]
    #    )
    #]

    circuit = Circuit.from_file('examples/qasm/qft_20.qasm')
    #circuit = Circuit.from_file('examples/qasm/tfim_20.qasm')
    new_circuit = Circuit(circuit.num_qudits)

    original_cnots, original_u3s = count_gates(circuit)
    optimized_u3s, optimized_cnots = 0, 0

    data = {}
    partitioner.run(circuit, data)

    start = default_timer()
    #with Compiler() as compiler:
    #    new_circuit = compiler.compile(CompilationTask(circuit, tasks))
    #    optimized_cnots, optimized_u3s = count_gates(new_circuit)
    
    for op in circuit:
        if op.num_qudits != 3:
            continue
        block = Circuit.from_operation(op)
        block.unfold_all()

        original_block = block.copy()
        old_cnots, old_u3s = count_gates(block)
        recommender.run(block, data)
        qseed.run(block, data)
        #remover.run(block, data)
        new_cnots, new_u3s = count_gates(block)

        if old_cnots < new_cnots:
            new_cnots, new_u3s = old_cnots, old_u3s
            block = original_block
        
        new_circuit.append_circuit(block, op.location)

        original_cnots += old_cnots
        original_u3s   += old_u3s 
        optimized_cnots += new_cnots
        optimized_u3s   += new_u3s 

    avg_insts = np.mean(data['num_instantiation_calls'])
    inst_str = f'Average instantiation calls: {avg_insts}'
    time_str = f'Optimization time: {default_timer() - start:>0.3f}s'
    u3_str   = f'Optimized u3 gates: {original_u3s} -> {optimized_u3s}'
    cx_str   = f'Optimized cx gates: {original_cnots} -> {optimized_cnots}'
    logging.info(inst_str)
    logging.info(time_str)
    logging.info(u3_str)
    logging.info(cx_str)