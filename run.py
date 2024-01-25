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
# from bqskit.passes import QSearchSynthesisPass
from qseed.qsearch import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.qis.graph import CouplingGraph
from bqskit.runtime import get_runtime

from qseed.qseedpass import QSeedRecommenderPass
from qseed.foreach import ForEachBlockPass
from qseed.cacheloaderpass import CacheLoaderPass
from qseed.utils.loading import load_recommenders
from qseed.timepass import TimePass
from qseed.deletefromdata import DeleteFromDataPass

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
    foreach = ForEachBlockPass(
        [qsearch, scanning],
        replace_filter=gen_less_than_multi(machine)
    )
    unfold = UnfoldPass()
    delete = DeleteFromDataPass()

    if args.qsearch:
        workflow = Workflow([machine_setter, partitioner, foreach, delete, unfold])
    else:
        cacheloader = CacheLoaderPass('recommenders', load_recommenders)
        qseed = QSeedRecommenderPass(seeds_per_rec=3, batch_size=32)

        workflow = Workflow(
            [
                machine_setter,
                partitioner,
                cacheloader,
                qseed,
                foreach,
                delete,
                unfold
            ]
        )

    start_time = default_timer()
    print('Compiling...')
    with Compiler(num_workers=8) as compiler:
        compiled, data = compiler.compile(circuit, workflow, request_data=True)
    stop_time = default_timer()
    duration = stop_time - start_time
    print(f'Duration: {duration:>0.3f}s')

    inst_calls = [
        d[i]['instantiation_calls']
        if 'instantiation_calls' in d[i] else 0
        for d in data['ForEachBlockPass_data']
        for i in range(len(d))
    ]
    print(f'Mean inst calls: {np.mean(inst_calls)}')

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
