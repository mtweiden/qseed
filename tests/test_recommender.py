from __future__ import annotations

from qseed.pauli_recommender import PauliRecommenderPass
import pickle
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from qseed.models import PauliLearner
import torch
import numpy as np

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
        state_path = 'qseed/models/pauli_encoder.model'
        temp_path = 'static_seeds/templates.pickle'

        model = PauliLearner()
        state = torch.load(state_path,map_location='cpu')
        with open(temp_path,'rb') as f:
            templates = pickle.load(f)

        recommender = PauliRecommenderPass(model, state, templates)
    
    def test_recommender_run(self) -> None:
        state_path = 'qseed/models/pauli_encoder.model'
        temp_path = 'static_seeds/templates.pickle'

        model = PauliLearner()
        state = torch.load(state_path,map_location='cpu')
        with open(temp_path,'rb') as f:
            templates = pickle.load(f)

        recommender = PauliRecommenderPass(model, state, templates)

        circuit = init_circuit(3)
        circuit.append_circuit(get_unit(),[0,1])
        circuit.append_circuit(get_unit(),[1,2])
        circuit.append_circuit(get_unit(),[0,1])
        circuit.append_circuit(get_unit(),[1,2])

        np.random.seed(1234)
        circuit.set_params(np.random.randn(circuit.num_params))

        data = {}
        recommender.run(circuit, data)

        assert 'recommended_seeds' in data
        print(data['recommended_seeds'])