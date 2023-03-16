from __future__ import annotations

from random import randint
from random import seed
from timeit import default_timer as time

from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from bqskit.passes.processing.scan import ScanningGateRemovalPass
from bqskit.passes.synthesis.qsearch import QSearchSynthesisPass
from bqskit.passes.synthesis.qseed import QSeedSynthesisPass
from bqskit.qis.unitary import UnitaryMatrix


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

    # def test_multiple_seeds(self) -> None:
    #    print('\nback tracking')
    #    unit = get_unit()

    #    seed(12345)
    #    target = init_circuit(3)
    #    num_units = 5
    #    locations = [[0,1] if i % 2 == 0 else [1,2] for i in range(num_units)]
    #    for location in locations:
    #        params = 2*np.pi*np.random.random(U3Gate().num_params*2)
    #        unit.set_params(params)
    #        target.append_circuit(unit, location)
    #
    #    utry = target.get_unitary()
    #
    #    template1 : Circuit = init_circuit(3)
    #    template2 : Circuit = init_circuit(3)
    #
    #    template1.append_circuit(unit, [0,1])
    #    template1.append_circuit(unit, [0,1])
    #    template1.append_circuit(unit, [0,1])

    #    template2.append_circuit(unit, [1,2])
    #    template2.append_circuit(unit, [1,2])
    #    template2.append_circuit(unit, [1,2])

    #    qseed = QSeedSynthesisPass([template1, template2])
    #
    #    start_time = time()
    #    qseed.run(target)
    #    print(f'{time()-start_time}')
    #    dist = target.get_unitary().get_distance_from(utry)
    #    assert dist <= 1e-5

    # def test_under_expressive_template(self) -> None:
    #    print("\nunder expressive")
    #    unit = get_unit()

    #    template = init_circuit(3)
    #    template.append_circuit(unit, [0,1])
    #    template.append_circuit(unit, [1,2])
    #    template.append_circuit(unit, [1,2])
    #    template.append_circuit(unit, [0,1])

    #    toffoli = toffoli_unitary()

    #    qseed = QSeedSynthesisPass(template)
    #    target = Circuit.from_unitary(toffoli)
    #    start_time = time()
    #    qseed.run(target)
    #    print(f'{time()-start_time}')
    #    dist = target.get_unitary().get_distance_from(toffoli)
    #    assert dist <= 1e-5

    def test_random_init_toffoli(self) -> None:
        print('\nrandom seed')
        seed(12345)
        unit = get_unit()
        template = init_circuit(3)
        num_random_edges = 5
        graph_edges = [(0, 1), (1, 2), (0, 2)]
        units_to_insert = [randint(0, 2) for _ in range(num_random_edges)]
        for edge_num in units_to_insert:
            template.append_circuit(unit, graph_edges[edge_num])

        qseed = QSeedSynthesisPass(template)

        toffoli = toffoli_unitary()

        target = Circuit.from_unitary(toffoli)
        start_time = time()
        qseed.run(target)
        print(f'{time()-start_time}')
        print(target.gate_counts)
        dist = target.get_unitary().get_distance_from(toffoli)
        assert dist <= 1e-5

    def test_qsearch_toffoli(self, compiler: Compiler) -> None:
        print('\nqsearch')
        toffoli = toffoli_unitary()
        qsearch = QSearchSynthesisPass()

        target = Circuit.from_unitary(toffoli)
        task = CompilationTask(target, [qsearch])
        start_time = time()
        target = compiler.compile(task)
        print(f'{time()-start_time}')
        print(target.gate_counts)
        dist = target.get_unitary().get_distance_from(toffoli)
        assert dist <= 1e-5
