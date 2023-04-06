import torch
import pickle
import logging
import numpy as np

from bqskit import Circuit
from bqskit.compiler import Compiler, CompilationTask
from bqskit.ir.gates import CNOTGate, SwapGate
from bqskit.passes import QuickPartitioner, UnfoldPass, ForEachBlockPass
from qseed import QSeedSynthesisPass
from qseed.recommender import TopologyAwareRecommenderPass
from qseed.models import UnitaryLearner

from examples.util import size_limit

def num_cnots(circuit : Circuit) -> int:
    gates = circuit.gate_counts
    num_gates = 0
    num_gates +=   gates[CNOTGate()] if CNOTGate() in gates else 0
    num_gates += 3*gates[SwapGate()] if SwapGate() in gates else 0
    return num_gates

if __name__ == '__main__':
    file_name = 'mapped-qft_8'
    logging.basicConfig(level=logging.INFO)

    np.random.seed(1234)

    circuit = Circuit.from_file(f'qasm/qft/{file_name}.qasm')

    models, states, templates = [], [], []
    for topology in ['a','b','c']:
        models.append(UnitaryLearner())
        path= f'qseed/models/unitary_learner_{topology}.model'
        states.append(torch.load(path,map_location='cpu'))
        with open(f'templates/circuits_{topology}.pickle','rb') as f:
            templates.append(pickle.load(f))

    partitioner = QuickPartitioner()
    recommender = TopologyAwareRecommenderPass(models, states, templates)
    qseed = QSeedSynthesisPass()
    unfold = UnfoldPass()

    # Partition
    with Compiler(num_workers=64) as compiler:
        task1 = CompilationTask(circuit, partitioner)
        partitioned_circuit = compiler.compile(task1)

    print('finished partitioning')

    # Recommend
    with Compiler(num_workers=64) as compiler:
        recom = ForEachBlockPass(recommender, collection_filter=size_limit)
        task = CompilationTask(partitioned_circuit, recom)
        _, data = compiler.compile(task, request_data=True)
    
    
    for fakesubdata in data['ForEachBlockPass_data']:
        for subdata in fakesubdata:
            assert 'recommended_seeds' in subdata
    
    # QSeed
    with Compiler(num_workers=64) as compiler:
        task = CompilationTask(partitioned_circuit, qseed)
        optimized_circuit = compiler.compile(task)
    
    # Unfold
    with Compiler(num_workers=64) as compiler:
        #task = CompilationTask(optimized_circuit, unfold)
        task = CompilationTask(partitioned_circuit, unfold)
        final_circuit = compiler.compile(task)

    print(f'Original : {num_cnots(circuit)}')
    print(f'Optimized: {num_cnots(final_circuit)}')
