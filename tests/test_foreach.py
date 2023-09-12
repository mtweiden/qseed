from bqskit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.compiler import Compiler
from bqskit.compiler import Workflow
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import QuickPartitioner
from bqskit.passes import UnfoldPass
from bqskit.passes import UpdateDataPass

from qseed.foreach import ForEachBlockPass


class TestForEach:
    def test_tautology(self) -> None:
        assert True

    def test_constructor(self) -> None:
        qsearch = QSearchSynthesisPass()
        partitioner = QuickPartitioner()
        workload = [partitioner, qsearch]
        foreach = ForEachBlockPass(workload)
        assert foreach

    def test_specific(self) -> None:
        circuit = Circuit(4)
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (1, 2))
        circuit.append_gate(CNOTGate(), (0, 1))
        circuit.append_gate(CNOTGate(), (0, 3))
        circuit.append_gate(CNOTGate(), (1, 3))

        seed = Circuit(3)
        seed.append_gate(CNOTGate(), (0, 1))
        seed.append_gate(CNOTGate(), (1, 2))
        seeds = {1: [seed]}

        key = 'ForEachBlockPass_specific_pass_down_seed_circuits'

        partitioner = QuickPartitioner()
        updater = UpdateDataPass(key, seeds)
        qsearch = QSearchSynthesisPass()
        foreach = ForEachBlockPass(qsearch)
        unfolder = UnfoldPass()
        workflow = Workflow([partitioner, updater, foreach, unfolder])
        with Compiler() as compiler:
            compiled, data = compiler.compile(
                circuit, workflow, request_data=True
            )
        dist = compiled.get_unitary().get_distance_from(circuit.get_unitary())
        assert dist <= 1e-5
        assert key in data
