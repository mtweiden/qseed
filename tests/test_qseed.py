from random import choice
from bqskit import Circuit
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.compiler import Compiler
from bqskit.compiler import Workflow
from bqskit.compiler import MachineModel
from bqskit.passes import QuickPartitioner
from bqskit.passes import SetModelPass

from qseed.qseedpass import QSeedRecommenderPass
from qseed.tu_recommender import TorchUnitaryRecommender
from utils.debug import DebugCircuitGenerator
from utils.debug import DebugRecommender
from utils.debug import DebugModel
from utils.couplings import get_linear_couplings


class TestQSeedRecommenderPass:
    #    def test_tautology(self) -> None:
    #        assert True
    #
    #    def test_constructor(self) -> None:
    #        num_qudits, num_seeds = 3, 5
    #        seed_gen = DebugCircuitGenerator(num_qudits, num_seeds)
    #        seeds = seed_gen.circuits
    #        recommender = DebugRecommender(seeds)
    #        qseed = QSeedRecommenderPass(recommender)
    #        assert qseed
    #
    #    def test_single_recommender(self) -> None:
    #        num_qudits, num_seeds = 3, 5
    #        seed_gen = DebugCircuitGenerator(num_qudits, num_seeds)
    #        seeds = seed_gen.circuits
    #        recommender = DebugRecommender(seeds)
    #        qseed = QSeedRecommenderPass(recommender, seeds_per_rec=2)
    #        partitioner = QuickPartitioner()
    #
    #        circuit = Circuit.from_unitary(UnitaryMatrix.random(num_qudits))
    #
    #        workflow = Workflow([partitioner, qseed])
    #
    #        with Compiler() as c:
    #            compiled, data = c.compile(circuit, workflow, request_data=True)
    #
    #        key = qseed.pass_down_block_specific_key_prefix
    #        assert key in data
    #        seed_map = data[key]
    #        assert len(seed_map) == 1
    #        assert 0 in seed_map.keys()
    #        assert isinstance(seed_map[0], list)
    #        assert len(seed_map[0]) == qseed.seeds_per_rec
    #        assert all(isinstance(s, Circuit) for s in seed_map[0])
    #        assert all(s in seeds for s in seed_map[0])
    #
    #    def test_assign_recommender(self) -> None:
    #        num_qudits, num_seeds = 3, 25
    #        recommenders = []
    #        all_seeds, couplings = [], []
    #        for graph in get_linear_couplings():
    #            coupling = CouplingGraph(graph)
    #            seed_gen = DebugCircuitGenerator(
    #                num_qudits, num_seeds, coupling, max_gates=8
    #            )
    #            seeds = seed_gen.circuits
    #            recommender = DebugRecommender(seeds, coupling)
    #            recommenders.append(recommender)
    #            all_seeds.append(seeds)
    #            couplings.append(coupling)
    #
    #        qseed = QSeedRecommenderPass(recommenders)
    #
    #        for coupling_index, seeds in enumerate(all_seeds):
    #            topology = couplings[coupling_index]
    #            for circuit in seeds:
    #                region = {
    #                    i: (0, circuit.depth - 1)
    #                    for i in range(circuit.num_qudits)
    #                }
    #                circuit.fold(region)
    #                for block in circuit:
    #                    assignment = qseed.assign_recommender(block, topology)
    #                    assert assignment == coupling_index
    #
    #    def test_triple_recommender(self) -> None:
    #        num_qudits, num_seeds = 3, 10
    #        recommenders = []
    #        for graph in get_linear_couplings():
    #            coupling = CouplingGraph(graph)
    #            seed_gen = DebugCircuitGenerator(
    #                num_qudits, num_seeds, coupling, max_gates=8
    #            )
    #            seeds = seed_gen.circuits
    #            recommender = DebugRecommender(seeds, coupling)
    #            recommenders.append(recommender)
    #
    #        qseed = QSeedRecommenderPass(recommenders, seeds_per_rec=3)
    #        partitioner = QuickPartitioner()
    #
    #        topology = CouplingGraph(choice(get_linear_couplings()))
    #        circuit_gen = DebugCircuitGenerator(num_qudits, 10, topology, 12)
    #        mach_model = MachineModel(num_qudits, topology)
    #        setter = SetModelPass(mach_model)
    #
    #        workflow = Workflow([setter, partitioner, qseed])
    #
    #        for circuit in circuit_gen.circuits:
    #            with Compiler() as c:
    #                compiled, data = c.compile(
    #                    circuit, workflow, request_data=True
    #                )
    #            key = qseed.pass_down_block_specific_key_prefix
    #            assert key in data
    #            seed_map = data[key]
    #            # Partitioner formed odd artifact
    #            if len(seed_map) == 0:
    #                continue
    #            assert len(seed_map) == 1
    #            assert 0 in seed_map.keys()
    #            assert isinstance(seed_map[0], list)
    #            assert len(seed_map[0]) == qseed.seeds_per_rec
    #            assert all(isinstance(s, Circuit) for s in seed_map[0])

    def test_triple_recommender_torch(self) -> None:
        num_qudits, num_seeds = 3, 10
        recommenders = []
        for graph in get_linear_couplings():
            coupling = CouplingGraph(graph)
            seed_gen = DebugCircuitGenerator(
                num_qudits, num_seeds, coupling, max_gates=8
            )
            seeds = seed_gen.circuits
            model = DebugModel(3, len(seeds))
            recommender = TorchUnitaryRecommender(model, seeds, coupling)
            recommenders.append(recommender)

        qseed = QSeedRecommenderPass(recommenders, seeds_per_rec=3)
        partitioner = QuickPartitioner()

        topology = CouplingGraph(choice(get_linear_couplings()))
        circuit_gen = DebugCircuitGenerator(num_qudits, 10, topology, 12)
        mach_model = MachineModel(num_qudits, topology)
        setter = SetModelPass(mach_model)

        workflow = Workflow([setter, partitioner, qseed])

        for circuit in circuit_gen.circuits:
            with Compiler() as c:
                compiled, data = c.compile(
                    circuit, workflow, request_data=True
                )
            key = qseed.pass_down_block_specific_key_prefix
            assert key in data
            seed_map = data[key]
            # Partitioner formed odd artifact
            if len(seed_map) == 0:
                continue
            assert len(seed_map) == 1
            assert 0 in seed_map.keys()
            assert isinstance(seed_map[0], list)
            assert len(seed_map[0]) == qseed.seeds_per_rec
            assert all(isinstance(s, Circuit) for s in seed_map[0])
