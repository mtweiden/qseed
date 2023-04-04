from __future__ import annotations

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import U3Gate
from qseed.handler import Handler
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.compiler import Compiler, CompilationTask

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

class TestHandler:
    def test_handler_constructor(self) -> None:
        handler = Handler()
    
    def test_handler_run(self) -> None:
        np.random.seed(1234)
        circuit = Circuit.from_unitary(UnitaryMatrix.random(3))

        handler = Handler()

        data = {}
        handler.run(circuit, data)
        
        print(handler.recorder.stats_lists)
        