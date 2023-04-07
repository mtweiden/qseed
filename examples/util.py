from bqskit.ir.operation import Operation
from bqskit.ir.gates import CNOTGate, SwapGate
from bqskit import Circuit

def size_limit(op: Operation) -> bool:
    return op.num_qudits == 3

def num_cnots(circuit : Circuit) -> int:
    gates = circuit.gate_counts
    num_gates = 0
    num_gates +=   gates[CNOTGate()] if CNOTGate() in gates else 0
    num_gates += 3*gates[SwapGate()] if SwapGate() in gates else 0
    return num_gates