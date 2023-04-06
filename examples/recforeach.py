from bqskit import Circuit
from bqskit.compiler import CompilationTask, Compiler
import numpy as np
from qseed import QSeedSynthesisPass
from qseed.recforeach import RecForEachBlockPass
from qseed.models import UnitaryLearner
from qseed.recommender import TopologyAwareRecommenderPass
from bqskit.passes import QuickPartitioner, UnfoldPass
import torch
import pickle
from examples.util import size_limit, num_cnots

import logging


if __name__ == '__main__':
    file_name = 'mapped-qft_16'
    logging.basicConfig(level=logging.INFO, filename=file_name)

    np.random.seed(1234)

    circuit = Circuit.from_file(f'qasm/qft/{file_name}.qasm')

    models, states, templates = [], [], []
    for topology in ['a','b','c']:
        models.append(UnitaryLearner())
        path = f'qseed/models/unitary_learner_{topology}.model'
        states.append(torch.load(path,map_location='cpu'))
        with open(f'templates/circuits_{topology}.pickle','rb') as f:
            templates.append(pickle.load(f))
    recommender = TopologyAwareRecommenderPass(models, states, templates)

    part = QuickPartitioner()
    qseed = QSeedSynthesisPass()
    foreach = RecForEachBlockPass(
        [qseed],
        models,
        states,
        templates,
        collection_filter=size_limit,
    )
    unfold = UnfoldPass()

    with Compiler(num_workers=64) as compiler:
        task = CompilationTask(circuit, [part, foreach, unfold])
        final_circuit = compiler.compile(task)

    print(f'Original : {num_cnots(circuit)}')
    print(f'Optimized: {num_cnots(final_circuit)}')