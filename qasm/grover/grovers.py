from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import ZGate
import numpy as np
from qiskit.compiler.transpiler import transpile
from qiskit.compiler.transpiler import CouplingMap



if __name__ == '__main__':
    num_qubits = 10

    reg = QuantumRegister(num_qubits)
    circ = QuantumCircuit(reg)

    cz_gate = ZGate().control(num_qubits-1)
    reverse_order = [_ for _ in range(num_qubits-1,-1,-1)]

    # Setup
    for i in range(num_qubits):
        circ.h(i)

    # Grover iterations
    for i in range(int(np.sqrt(num_qubits))):
        # Oracle
        circ.append(cz_gate, reverse_order)

        # Diffusion
        for i in range(num_qubits):
            circ.h(i)
            circ.x(i)
        circ.append(cz_gate, reverse_order)
        for i in range(num_qubits):
            circ.h(i)
            circ.x(i)

    # Unmapped
    circ = transpile(circ, basis_gates=['u3','cx'])
    with open(f'grover_{num_qubits}.qasm', 'w') as f:
        f.write(circ.qasm())

    x = int(np.ceil(np.sqrt(num_qubits)))
    mesh = CouplingMap.from_grid(x, x)

    # Mapped
    circ = transpile(circ, basis_gates=['u3','cx','swap'], coupling_map=mesh)
    with open(f'mapped-grover_{num_qubits}.qasm', 'w') as f:
        f.write(circ.qasm())

 