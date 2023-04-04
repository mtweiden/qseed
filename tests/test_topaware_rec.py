from __future__ import annotations

from qseed.recommender import TopologyAwareRecommenderPass
import pickle
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from qseed.models import PauliLearner
from qseed import QSeedSynthesisPass
import torch
import numpy as np
from bqskit.compiler import Compiler 
from bqskit.compiler import CompilationTask
from bqskit.passes import QuickPartitioner

def get_unit() -> Circuit:
    unit = Circuit(2)
    unit.append_gate(CNOTGate(), [0, 1])
    unit.append_gate(U3Gate(), [0])
    unit.append_gate(U3Gate(), [1])
    return unit

def init_circuit(num_qudits: int) -> Circuit:
    circuit = Circuit(num_qudits)
    for q in range(num_qudits):
        circuit.append_gate(U3Gate(), [q])
    return circuit

class TestQSeed:
    def test_recommender_constructor(self) -> None:
        models = []
        states = []
        templates = []
        for t in ['a','b','c']:
            models.append(PauliLearner())
            states.append(
                torch.load(f'qseed/models/learner_{t}.model',map_location='cpu')
            )
            with open(f'templates/templates_{t}.pickle','rb') as f:
                templates.append(pickle.load(f))

        recommender = TopologyAwareRecommenderPass(models, states, templates)
    
    def test_recommender_run(self) -> None:
        models = []
        states = []
        templates = []
        for t in ['a','b','c','d']:
            models.append(PauliLearner())
            states.append(
                torch.load(f'qseed/models/learner_{t}.model',map_location='cpu')
            )
            with open(f'templates/templates_{t}.pickle','rb') as f:
                templates.append(pickle.load(f))

        k = 3
        recommender = TopologyAwareRecommenderPass(
            recommender_models=models,
            model_states=states,
            template_lists=templates,
            seeds_per_inference=k
        )

        circuits = [init_circuit(3) for _ in range(4)]

        circuits[0].append_circuit(get_unit(), [0,1])
        circuits[0].append_circuit(get_unit(), [1,2])
        circuits[0].append_circuit(get_unit(), [0,1])

        circuits[1].append_circuit(get_unit(), [0,1])
        circuits[1].append_circuit(get_unit(), [0,2])
        circuits[1].append_circuit(get_unit(), [0,1])

        circuits[2].append_circuit(get_unit(), [1,2])
        circuits[2].append_circuit(get_unit(), [0,2])
        circuits[2].append_circuit(get_unit(), [1,2])

        circuits[3].append_circuit(get_unit(), [0,1])
        circuits[3].append_circuit(get_unit(), [1,2])
        circuits[3].append_circuit(get_unit(), [0,2])

        np.random.seed(1234)
        for circuit in circuits:
            circuit.set_params(np.random.randn(circuit.num_params))

        for i,circuit in enumerate(circuits):
            assert recommender._detect_connectivity(circuit) == i

        data = {}
        for block in circuits:
            recommender.run(block, data)

        assert 'recommended_seeds' in data

        assert len(data['recommended_seeds']) == len(circuits)
        for recommendations in data['recommended_seeds']:
            assert len(recommendations) == k

