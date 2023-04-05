from bqskit import Circuit
from bqskit.qis import UnitaryMatrix
from qseed import Handler
import numpy as np

from bqskit.ir.gates import CNOTGate, U3Gate

import logging


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.random.seed(1234)

    circuit = Circuit(3)
    circuit.append_gate(CNOTGate(), [0,1])
    circuit.append_gate(CNOTGate(), [0,1])
    circuit.append_gate(CNOTGate(), [1,2])
    circuit.append_gate(CNOTGate(), [0,2])

    handler = Handler()

    data = {}
    handler.handle(circuit, data)