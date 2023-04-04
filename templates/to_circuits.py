from os import listdir
from bqskit import Circuit
import pickle
import numpy as np
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.parameterized.u3 import U3Gate

def make_template(coupling_map : list[tuple[int]], num_q : int = 3) -> Circuit:
    circuit = Circuit(num_q)
    for i in range(num_q):
        circuit.append_gate(U3Gate(), [i])
    for u,v in coupling_map:
        circuit.append_gate(CNOTGate(), (u,v))
        circuit.append_gate(U3Gate(), [u])
        circuit.append_gate(U3Gate(), [v])
    return circuit


if __name__ == '__main__':
    topologies = ['a','b','c','d']

    for top in topologies:
        with open(f'templates_{top}.pickle','rb') as f:
            templates = [make_template(t) for t in pickle.load(f)]
        with open(f'circuits_{top}.pickle','wb') as f:
            pickle.dump(templates, f)