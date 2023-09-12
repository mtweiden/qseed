import torch
import numpy as np
import pickle
from bqskit import Circuit
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.qis.graph import CouplingGraph

from qseed.tu_recommender import TorchUnitaryRecommender
from models.unitary_learner import UnitaryLearner

from utils.debug import DebugModel
from utils.debug import DebugCircuitGenerator


class TestTorchUnitaryRecommender:
    def test_tautology(self) -> None:
        assert True

    def test_constructor(self) -> None:
        num_qudits, num_seeds = 3, 4
        model = DebugModel(num_qudits, num_seeds)
        graph = CouplingGraph([(0, 1), (1, 2)])
        seed_gen = DebugCircuitGenerator(num_qudits, num_seeds, graph, 4)
        seeds = seed_gen.circuits
        recommender = TorchUnitaryRecommender(
            recommender_model=model,
            seed_circuits=seeds,
            coupling_graph=graph,
        )
        assert recommender

    def test_encode(self) -> None:
        num_qudits, num_seeds = 2, 4
        model = DebugModel(num_qudits, num_seeds)
        seed_gen = DebugCircuitGenerator(num_qudits, num_seeds)
        seeds = seed_gen.circuits
        recommender = TorchUnitaryRecommender(
            recommender_model=model,
            seed_circuits=seeds,
        )
        unitary = UnitaryMatrix.random(2)
        circuit = Circuit.from_unitary(unitary)
        op = circuit.pop((0, 0))
        encoding = recommender.encode(op)
        assert encoding.shape == (2 * 4 ** num_qudits,)

        real_x = np.real(unitary).flatten()
        imag_x = np.imag(unitary).flatten()
        stacked_unitary = np.hstack([real_x, imag_x])
        assert all(encoding == stacked_unitary)

    def test_recommend(self) -> None:
        num_qudits, num_seeds = 2, 4
        model = DebugModel(num_qudits, num_seeds)
        seed_gen = DebugCircuitGenerator(num_qudits, num_seeds)
        seeds = seed_gen.circuits
        recommender = TorchUnitaryRecommender(
            recommender_model=model,
            seed_circuits=seeds,
        )
        encoding = torch.randn((2 * 4 ** num_qudits, ))

        recommendations = recommender.recommend(encoding, seeds_per_rec=2)
        for seed in recommendations:
            assert seed in seeds

    def test_with_torch_model(self) -> None:
        model_path = 'models/unitary_learner_a.model'
        learner = UnitaryLearner()
        learner.load_state_dict(torch.load(model_path))
        seeds_path = 'seeds/seed_circuits_a.pkl'
        with open(seeds_path, 'rb') as f:
            seeds = pickle.load(f)
        coupling = CouplingGraph([(0, 1), (1, 2)])
        recommender = TorchUnitaryRecommender(learner, seeds, coupling)

        encoding = torch.randn((2 * 4 ** 3, ))

        recommendations = recommender.recommend(encoding, seeds_per_rec=2)
        for seed in recommendations:
            assert seed in seeds
