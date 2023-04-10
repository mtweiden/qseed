import pickle
import torch

from timeit import default_timer as time

from bqskit.compiler import Compiler, CompilationTask
from bqskit import Circuit
from bqskit.passes import QuickPartitioner, UnfoldPass
from bqskit.ir import Operation

from qseed.qseedpass import QSeedSynthesisPass
from qseed.recforeach import RecForEachBlockPass 
from qseed.models import UnitaryLearner

def filter(circuit : Operation) -> bool:
    return circuit.num_qudits == 3

if __name__ == '__main__':

    file = '../qasm/qft/mapped-qft_5.pickle'

    circuit = Circuit.from_file(file)

    topologies = ['a','b','c']
    models, states, templates = [], [], []
    for topology in topologies:
        models.append(UnitaryLearner())
        path = f'qseed/models/unitary_learner_{topology}.model'
        states.append(torch.load(path,map_location='cpu'))
        with open(f'templates/circuits_{topology}.pickle','rb') as f:
            templates.append(pickle.load(f))

    block_passes = [
        QSeedSynthesisPass(),
    ]
    task = CompilationTask(
        circuit, 
        [
            QuickPartitioner(block_size=3),
            RecForEachBlockPass(
                block_passes,
                models,
                states,
                templates,
                collection_filter=filter,
            ),
            UnfoldPass(),
        ]
    )
    with Compiler(num_workers=64) as compiler:
        start_time = time()
        new_circuit = compiler.compile(task)
        stop_time = time()

    print(f'Compilation time: {stop_time - start_time:>0.3f}s')