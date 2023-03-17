from __future__ import annotations

from bqskit import MachineModel
import pickle
from bqskit.ir.circuit import Circuit
from bqskit.ir.location import CircuitLocation
from bqskit.ir.operation import Operation
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.qis.graph import CouplingGraph
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.partitioning.quick import QuickPartitioner
from qseed import QSeedSynthesisPass
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from qseed.handler import Handler
from timeit import default_timer as time

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

def toffoli_unitary() -> UnitaryMatrix:
    return UnitaryMatrix([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])

def swap_unitaries() -> tuple[UnitaryMatrix, UnitaryMatrix]:
    swap1 = UnitaryMatrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ])
    swap2 = UnitaryMatrix([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ])
    return (swap1, swap2)


class TestQSeed:
    def test_random_2_qubit(self) -> None:
        utry = UnitaryMatrix.random(2)
        target = Circuit.from_unitary(utry)
        unit = get_unit()
        template = init_circuit(2)
        template.append_circuit(unit, [0, 1])

        qseed = QSeedSynthesisPass(template)
        qseed.run(target)
        dist = target.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_simple_swap(self) -> None:
        unit = get_unit()

        template = Circuit(2)
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [0, 1])

        qseed = QSeedSynthesisPass(template)

        print()
        for utry in swap_unitaries():
            target = Circuit.from_unitary(utry)
            start_time = time()
            qseed.run(target)
            print(f'{time() - start_time}')
            dist = target.get_unitary().get_distance_from(utry)
            assert dist <= 1e-5

    def test_option_swap(self) -> None:
        print('\noptional seeds')
        unit = get_unit()
        template = init_circuit(2)
        qseed = QSeedSynthesisPass(template)

        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [1, 0])
        template.append_circuit(unit, [0, 1])

        data = {'seeds': [template]}
        utry, _ = swap_unitaries()

        target = Circuit.from_unitary(utry)
        start_time = time()
        qseed.run(target, data)
        print(f'{time() - start_time}')
        dist = target.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_over_expressive_swap(self) -> None:
        print('\nover expressive swap')
        unit = get_unit()

        template = Circuit(2)
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [0, 1])

        qseed = QSeedSynthesisPass(template)

        for utry in swap_unitaries():
            target = Circuit.from_unitary(utry)
            start_time = time()
            qseed.run(target)
            print(f'{time() - start_time}')
            dist = target.get_unitary().get_distance_from(utry)
            assert dist <= 1e-5

    def test_under_expressive_swap(self) -> None:
        print('\nunder expressive swap')
        unit = get_unit()

        template = Circuit(2)
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [0, 1])

        qseed = QSeedSynthesisPass(template)

        for utry in swap_unitaries():
            target = Circuit.from_unitary(utry)
            start_time = time()
            qseed.run(target)
            print(f'{time() - start_time}')
            dist = target.get_unitary().get_distance_from(utry)
            assert dist <= 1e-5

    def test_qseed_toffoli(self) -> None:
        print('\nqseed')
        unit = get_unit()
        template = init_circuit(3)
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [0, 2])
        template.append_circuit(unit, [1, 2])

        toffoli = toffoli_unitary()
        qseed = QSeedSynthesisPass(template)
        target = Circuit.from_unitary(toffoli)
        remover = ScanningGateRemovalPass()
        start_time = time()
        qseed.run(target)
        remover.run(target)
        print(f'{time()-start_time}')
        print(target.gate_counts)
        dist = target.get_unitary().get_distance_from(toffoli)
        assert dist <= 1e-5

    def test_over_expressive_template(self) -> None:
        print('\nover expressive')
        unit = get_unit()

        template = init_circuit(3)
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [0, 1])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [0, 2])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [1, 2])
        template.append_circuit(unit, [0, 2])
        template.append_circuit(unit, [1, 2])

        toffoli = toffoli_unitary()

        qseed = QSeedSynthesisPass(template)
        target = Circuit.from_unitary(toffoli)
        start_time = time()
        qseed.run(target)
        print(f'{time()-start_time}')
        print(target.gate_counts)
        dist = target.get_unitary().get_distance_from(toffoli)
        assert dist <= 1e-5

    def test_multiple_randoms(self) -> None:
        print('\nmulti-seeded')
        unit = get_unit()
        template1 = init_circuit(2)
        template2 = init_circuit(2)
        template1.append_circuit(unit, [0, 1])
        template2.append_circuit(unit, [0, 1])
        template2.append_circuit(unit, [0, 1])
        template2.append_circuit(unit, [0, 1])

        templates = [template1, template2]

        qseed = QSeedSynthesisPass(templates)

        utry = UnitaryMatrix.random(2)
        target = Circuit.from_unitary(utry)

        qseed.run(target)
        print(target.gate_counts)
        dist = target.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5

    def test_gate_count(self) -> None:
        circuit = init_circuit(3)
        circuit.append_circuit(get_unit(), [0,1])
        circuit.append_circuit(get_unit(), [0,1])
        circuit.append_circuit(get_unit(), [0,2])
        circuit.append_circuit(get_unit(), [0,2])
        circuit.append_circuit(get_unit(), [1,2])
        circuit.append_circuit(get_unit(), [1,2])

        handler = Handler()
        assert handler._count_gates(circuit)[0] == 6

    def test_sub_machine(self) -> None:
        circuit = init_circuit(3)
        circuit.append_circuit(get_unit(), [0,1])
        circuit.append_circuit(get_unit(), [1,2])

        model = MachineModel(10)
        complete = model.coupling_graph.all_to_all(10)
        linear   = model.coupling_graph.linear(10)
        complete_model = MachineModel(10, complete)
        linear_model = MachineModel(10, linear)

        location = CircuitLocation([2,3,4])

        handler = Handler()
        complete_submachine = handler._sub_machine(complete_model, location)
        linear_submachine = handler._sub_machine(linear_model, location)
        complete_sub = CouplingGraph([(0,1),(1,2),(0,2)], 3)
        linear_sub   = CouplingGraph([(0,1),(1,2),], 3)

        for edge in complete_submachine.coupling_graph:
            assert edge in complete_sub 
        for edge in complete_sub:
            assert edge in complete_submachine.coupling_graph

        for edge in linear_submachine.coupling_graph:
            assert edge in linear_sub 
        for edge in linear_sub:
            assert edge in linear_submachine.coupling_graph

    def test_extract_subtopology(self) -> None:
        handler = Handler()
        circuit = init_circuit(3)
        assert handler._extract_subtopology(circuit) == -1

        circuit.append_circuit(get_unit(), [0,2])
        circuit.append_circuit(get_unit(), [0,1])

        part = QuickPartitioner(3)
        part.run(circuit)

        new_circuit = Circuit(10)
        new_circuit.append_circuit(circuit, [1,3,4])
        new_circuit.append_circuit(circuit, [2,6,8])

        for op in new_circuit:
            block = Circuit.from_operation(op)
            assert handler._extract_subtopology(block) == 1
    
    def test_static_recommender(self) -> None:
        handler = Handler()
        circ_a = init_circuit(3)
        circ_a.append_circuit(get_unit(), [0,1])
        circ_a.append_circuit(get_unit(), [1,2])
        circ_b = init_circuit(3)
        circ_b.append_circuit(get_unit(), [0,1])
        circ_b.append_circuit(get_unit(), [0,2])
        circ_c = init_circuit(3)
        circ_c.append_circuit(get_unit(), [0,2])
        circ_c.append_circuit(get_unit(), [1,2])
        circ_d = init_circuit(3)
        circ_d.append_circuit(get_unit(), [0,1])
        circ_d.append_circuit(get_unit(), [1,2])
        circ_d.append_circuit(get_unit(), [0,2])

        op_a = Operation(CircuitGate(circ_a), [0,1,2])
        op_b = Operation(CircuitGate(circ_b), [2,3,4])
        op_c = Operation(CircuitGate(circ_c), [0,1,2])
        op_d = Operation(CircuitGate(circ_d), [2,3,4])

        circuit = Circuit(5)
        circuit.append(op_a)
        circuit.append(op_b)
        circuit.append(op_c)
        circuit.append(op_d)

        with open('static_seeds/qft_templates_a.pickle','rb') as f:
            templates_a = pickle.load(f)
        with open('static_seeds/qft_templates_b.pickle','rb') as f:
            templates_b = pickle.load(f)
        with open('static_seeds/qft_templates_c.pickle','rb') as f:
            templates_c = pickle.load(f)
        with open('static_seeds/qft_templates_d.pickle','rb') as f:
            templates_d = pickle.load(f)

        for i,op in enumerate(circuit):
            block = Circuit.from_operation(op)
            seeds = handler.seed_recommender(block)

            if i == 0:
                assert seeds == templates_a
            if i == 1:
                assert seeds == templates_b
            if i == 2:
                assert seeds == templates_c
            if i == 3:
                assert seeds == templates_d

