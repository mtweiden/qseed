import torch
import torch.nn as nn
import pickle

from bqskit import Circuit
from bqskit.qis.graph import CouplingGraph

from qseed.models.unitary_learner import UnitaryLearner
from qseed.tu_recommender import TorchUnitaryRecommender
from qseed.utils.couplings import get_linear_couplings


def load_models() -> list[nn.Module]:
    models = []
    dev = torch.device('cpu')
    for topology_code in ['a', 'b', 'c']:
        path = f'qseed/models/unitary_learner_{topology_code}.model'
        learner = UnitaryLearner()
        learner.load_state_dict(torch.load(path, map_location=dev))
        models.append(learner)
    return models


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


def load_recommenders() -> list[TorchUnitaryRecommender]:
    models = load_models()
    seed_circuits = load_seed_circuits()
    coupling_graphs = load_coupling_graphs()
    recommenders = []
    for model, seeds, coupling in zip(
        models, seed_circuits, coupling_graphs
    ):
        recommender = TorchUnitaryRecommender(
            model, seeds, coupling
        )
        recommenders.append(recommender)
    print('Loaded recommenders.')
    return recommenders
