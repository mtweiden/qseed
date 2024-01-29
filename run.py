from argparse import ArgumentParser
import pickle
import numpy as np

from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.compiler import Workflow
from bqskit.passes import SetModelPass
from bqskit.passes import QuickPartitioner
from bqskit.passes.control.foreach import gen_less_than_multi
from qseed.qsearch import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.qis.graph import CouplingGraph
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import SwapGate

from qseed.qseedpass import QSeedRecommenderPass
from qseed.foreach import ForEachBlockPass
from qseed.utils.loading import load_recommenders
from qseed.deletefromdata import DeleteSeedsPass
from qseed.unfold import UnfoldPass
from qseed.timepass import TimePass
from qseed.printpass import PrintPass


from timeit import default_timer

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('qasm_file', type=str)
    parser.add_argument('--qsearch', action='store_true')
    args = parser.parse_args()

    circuit = Circuit.from_file(args.qasm_file)

    n = int(np.ceil(np.sqrt(circuit.num_qudits)))
    topology = [_ for _ in CouplingGraph.grid(n, n)]
    machine = MachineModel(circuit.num_qudits, topology)
    machine_setter = SetModelPass(machine)
    partitioner = QuickPartitioner()
    qsearch = QSearchSynthesisPass()
    scanning = ScanningGateRemovalPass()
    unfold = UnfoldPass()
    tp = TimePass()

    if args.qsearch:
        foreach = ForEachBlockPass(
            [qsearch, scanning],
            replace_filter=gen_less_than_multi(machine)
        )
    else:
        qseed = QSeedRecommenderPass(
            load_recommenders,
            seeds_per_rec=3,
            batch_size=32,
        )
        foreach = ForEachBlockPass(
            [qseed, qsearch, scanning],
            replace_filter=gen_less_than_multi(machine)
        )
    workflow = Workflow(
        [
            # tp, 
            # PrintPass('Machine Setter'),
            machine_setter,
            # tp, 
            # PrintPass('Partitioner'),
            partitioner,
            # tp, 
            # PrintPass('ForEach'),
            foreach,
            # tp, 
            # PrintPass('Unfold'),
            unfold,
            # tp, 
        ]
    )

    algo = 'QSearch' if args.qsearch else 'QSeed'
    title = f'Compiling with {algo}...'
    print(title)
    start_time_1 = default_timer()
    compiler = Compiler(num_workers=8)
    stop_time_1 = default_timer()
    duration_1 = stop_time_1 - start_time_1
    print(f'{duration_1=:>0.3f}')
    start_time_2 = default_timer()
    compiled, data = compiler.compile(circuit, workflow, request_data=True)
    # compiled = compiler.compile(circuit, workflow)
    stop_time_2 = default_timer()
    duration_2 = stop_time_2 - start_time_2
    print(f'{duration_2=:>0.3f}')
    start_time_3 = default_timer()
    compiler.close()
    stop_time_3 = default_timer()
    duration_3 = stop_time_3 - start_time_3
    print(f'{duration_3=:>0.3f}')

    inst_calls = [
        d[i]['instantiation_calls']
        if 'instantiation_calls' in d[i] else 0
        for d in data['ForEachBlockPass_data']
        for i in range(len(d))
    ]
    counts = compiled.gate_counts
    print(f'Mean inst calls: {np.mean(inst_calls):>0.3f}')
    print(f'Gate counts: {counts}')
    cx_count = counts[CNOTGate()]
    if SwapGate() in counts:
        cx_count += 3 * counts[SwapGate()]
    print(f'CXGates: {cx_count}')
    print(f'Total Duration: {stop_time_3 - start_time_1:>0.3f}s\n')

    if '/' in args.qasm_file:
        input_name = args.qasm_file.split('/')[-1]
    else:
        input_name = args.qasm_file
    if args.qsearch:
        input_name = 'qsearch-' + input_name
    input_name = input_name.replace('qasm', 'pkl')
    with open(f'compiled_circuits/{input_name}', 'wb') as f:
        pickle.dump(compiled, f)


if __name__ == "__main__":
    main()
