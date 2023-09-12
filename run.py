from argparse import ArgumentParser
import torch
import torch.nn as nn
import pickle
import numpy as np

from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.compiler import Workflow
from bqskit.passes import SetModelPass
from bqskit.passes import QuickPartitioner
from bqskit.passes.control.foreach import gen_less_than_multi
from bqskit.passes import QSearchSynthesisPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.passes import UpdateDataPass
from bqskit.qis.graph import CouplingGraph

from qseed.qseedpass import QSeedRecommenderPass
from qseed.tu_recommender import TorchUnitaryRecommender
from qseed.foreach import ForEachBlockPass

from models.unitary_learner import UnitaryLearner

from utils.couplings import get_linear_couplings

# Debug
from utils.debug import DebugRecommender


def load_recommender_models() -> list[nn.Module]:
    models = []
    for topology_code in ['a', 'b', 'c']:
        path = f'models/unitary_learner_{topology_code}.model'
        learner = UnitaryLearner()
        learner.load_state_dict(torch.load(path))
        models.append(learner)
    return models


def load_recommender_states() -> list[dict]:
    states = []
    for topology_code in ['a', 'b', 'c']:
        path = f'models/unitary_learner_{topology_code}.model'
        states.append(path)
    return states


def load_seed_circuits() -> list[list[Circuit]]:
    seed_circuits = []
    for topology_code in ['a', 'b', 'c']:
        path = f'seeds/seed_circuits_{topology_code}.pkl'
        with open(path, 'rb') as f:
            seeds = pickle.load(f)
        seed_circuits.append(seeds)
    return seed_circuits


def load_coupling_graphs() -> list[CouplingGraph]:
    couplings = get_linear_couplings()
    return [CouplingGraph(coupling) for coupling in couplings]


def load_recommenders(
    recommender_models: list[nn.Module],
    seed_circuits: list[Circuit],
    coupling_graphs: list[CouplingGraph],
) -> list[TorchUnitaryRecommender]:
    recommenders = []
    for model, seeds, coupling in zip(
        recommender_models, seed_circuits, coupling_graphs
    ):
        #recommender = DebugRecommender(seeds, coupling)
        recommender = TorchUnitaryRecommender(
            model, seeds, coupling
        )
        recommenders.append(recommender)
    return recommenders


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('qasm_file', type=str)
    parser.add_argument('--qsearch', action='store_true')
    args = parser.parse_args()

    circuit = Circuit.from_file(args.qasm_file)

    n = int(np.ceil(np.sqrt(circuit.num_qudits)))
    topology = CouplingGraph.grid(n, n)
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

    if args.qsearch:
        workflow = Workflow([machine_setter, partitioner, foreach, unfold])
    else:
        recommender_states = load_recommender_states()
        seed_circuits = load_seed_circuits()
        coupling_graphs = load_coupling_graphs()

        assert len(recommender_states) == len(seed_circuits)
        assert len(seed_circuits) == len(coupling_graphs)

        rec_states = [
            (s, c, g) for (s, c, g) in zip(
                recommender_states, seed_circuits, coupling_graphs
            )
        ]

        rec_setter = UpdateDataPass(
            QSeedRecommenderPass.recommender_state_key,
            rec_states,
        )

        qseed = QSeedRecommenderPass(
            seeds_per_rec=3,
            batch_size=64
        )
        #import pdb; pdb.set_trace()

        workflow = Workflow(
            [machine_setter, partitioner, rec_setter, qseed, foreach, unfold]
        )

    print('Compiling...')
    #import pdb; pdb.set_trace()
    with Compiler() as compiler:
        compiled = compiler.compile(circuit, workflow)

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
