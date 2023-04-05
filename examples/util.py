from bqskit.ir.operation import Operation

def size_limit(op: Operation) -> bool:
    return op.num_qudits == 3