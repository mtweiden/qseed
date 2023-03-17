from bqskit import Circuit
from qseed import Handler
from qseed import QSearchHandler
from bqskit.passes import QuickPartitioner
from bqskit.compiler import MachineModel
from bqskit.qis import CouplingGraph
import numpy as np

import logging

if __name__ == '__main__':

    logging.basicConfig(filename='examples/logs/qft_20.log',level=logging.INFO)

    # Input a mapped quantum circuit
    circuit = Circuit.from_file('examples/qasm/qft_20.qasm')
    x = int(np.sqrt(circuit.num_qudits))
    machine_model = MachineModel(circuit.num_qudits, CouplingGraph.grid(x,x))
    data = {'machine_model': machine_model}

    qseed_circuit = circuit.copy()
    qsearch_circuit = circuit.copy()
    partitioner = QuickPartitioner()
    partitioner.run(qseed_circuit, data)
    partitioner.run(qsearch_circuit, data)

    # Run QSeed
    handler = Handler()
    handler.run(qseed_circuit, data)

    # Run QSearch
    #circuit.save('examples/qasm/opt-qft_20.qasm')
    handler = QSearchHandler()
    handler.run(qsearch_circuit, data)
