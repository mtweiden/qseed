from __future__ import annotations

from qseed.pauli_recommender import PauliRecommenderPass
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
from bqskit.passes import ForEachBlockPass

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
    
    #def test_recommender_qseed(self) -> None:
    #    state_path = 'qseed/models/pauli_encoder.model'
    #    temp_path = 'static_seeds/templates.pickle'

    #    model = PauliLearner()
    #    state = torch.load(state_path,map_location='cpu')
    #    with open(temp_path,'rb') as f:
    #        templates = pickle.load(f)

    #    recommender = PauliRecommenderPass(model, state, templates)
    #    qseed = QSeedSynthesisPass()

    #    circuit = init_circuit(3)
    #    for i in range(3):
    #        circuit.append_circuit(get_unit(),[0,1])
    #        circuit.append_circuit(get_unit(),[1,2])

    #    np.random.seed(1234)
    #    circuit.set_params(np.random.randn(circuit.num_params))

    #    data = {}
    #    recommender.run(circuit, data)
    #    qseed.run(circuit, data)
    
    #def test_recommender_partitions(self) -> None:
    #    state_path = 'qseed/models/pauli_encoder.model'
    #    temp_path = 'static_seeds/templates.pickle'

    #    model = PauliLearner()
    #    state = torch.load(state_path,map_location='cpu')
    #    with open(temp_path,'rb') as f:
    #        templates = pickle.load(f)

    #    partitioner = QuickPartitioner()
    #    recommender = PauliRecommenderPass(model, state, templates)
    #    qseed = QSeedSynthesisPass()

    #    circuit = init_circuit(5)
    #    for i in range(3):
    #        circuit.append_circuit(get_unit(),[0,1])
    #        circuit.append_circuit(get_unit(),[1,2])
    #        circuit.append_circuit(get_unit(),[2,3])
    #        circuit.append_circuit(get_unit(),[3,4])

    #    np.random.seed(1234)
    #    circuit.set_params(np.random.randn(circuit.num_params))

    #    data = {}
    #    partitioner.run(circuit, data)
    #    for op in circuit:
    #        if op.num_qudits != 3:
    #            continue
    #        block = Circuit.from_operation(op)
    #        block.unfold_all()
    #        recommender.run(block, data)
    #        qseed.run(block, data)

    def test_with_compiler(self) -> None:
        state_path = 'qseed/models/pauli_encoder.model'
        temp_path = 'static_seeds/templates.pickle'

        model = PauliLearner()
        state = torch.load(state_path,map_location='cpu')
        with open(temp_path,'rb') as f:
            templates = pickle.load(f)

        partitioner = QuickPartitioner(3)
        recommender = PauliRecommenderPass(model, state, templates)
        qseed = QSeedSynthesisPass()
        tasks = [
            partitioner, ForEachBlockPass([recommender, qseed],
            collection_filter=lambda x: x.num_qudits == 3)
        ]

        circuit = init_circuit(5)
        for i in range(3):
            circuit.append_circuit(get_unit(),[0,1])
            circuit.append_circuit(get_unit(),[1,2])
            circuit.append_circuit(get_unit(),[2,3])
            circuit.append_circuit(get_unit(),[3,4])

        np.random.seed(1234)
        circuit.set_params(np.random.randn(circuit.num_params))

        with Compiler() as compiler:
            compilation_task = CompilationTask(circuit, tasks)
            out = compiler.compile(compilation_task)
