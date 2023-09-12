### Architecture
================================
QSeed
    args:
        num_qudits (int): Number of qudits in seeds.

        recommender (SeedRecommenderPass): A SeedRecommenderPass analyzes a 
            partitioned circuit and propose seed circuits. 
--------------------------------
  SeedRecommenderPass
  ForEachBlockPass
    SeedSetterPass
    QSearch
================================

## NOTES
Recommender(abc.ABC)
    """
    An abstract class that encodes operations and recommends seed circuits.
    """
    @abc.abstractmethod
    def encode(self, operation: Operation) -> Any:
        """
        Encode an operation for seed circuit recommendation.
        """

    @abc.abstractmethod
    def recommend(self, encoding: Any, seeds_per_rec: int = 1) -> list[Circuit]:
        """
        Recommend a list of seeds given an encoding of an operation.
        """

TorchUnitaryRecommender(Recommender):
    """
    A recommender that uses a PyTorch nn.Module to analyze unitaries.
    """
    def __init__(
        self,
        recommender_model: nn.Module,
        recommender_state: dict[str, Tensor],
        seed_circuits: Sequence[Circuit],
        coupling_graph: CouplingGraph | None = None, 
    ) -> None:
        """
        Constructor for the TorchUnitaryRecommender.

        Note:
            It cannot be determined by TorchUnitaryRecommender if the provided
            `recommender_model` and `recommender_state` correspond to the
            given `seed_circuits` and `coupling_graph` other than by checking
            dimensions. This must be verified by the user.

        args:
            recommender_model (nn.Module): A specification of the recommender
                model's architecture.

            recommender_state (dict[str, Tensor]): The `state_dict` for the
                `recommender_model`.

            seed_circuits (Sequence[Circuit]): Possible circuits that the
                `recommender_model` can propose.

            coupling_graph (CouplingGraph | None): If provided, seed circuits
                must conform to this CouplingGraph. Otherwise, any qubit
                iteractions are allowed. (Default: None)

        raises:
            ValueError: If the number of seed_circuits is not the size of
                the `recommender_model` output.

            ValueError: If a coupling_graph is provided but one or more of
                the `seed_circuits` does not conform.
            
            ValueError: If `seed_circuits` have different numbers of qudits,
                or if the number of qudits in the `seed_circuits` differs
                from the number of qudits in the `coupling_graph`.
        """

    def encode(self, operation: Operation) -> Tensor:
        """
        Encode an operation's unitary as a PyTorch Tensor.

        args:
            operation (Operation): The operation to be encoded.

        returns:
            (Tensor): A flattened Tensor view of the unitary. Real components
                are stacked on top of imaginary ones.
        """

    def recommend(self, encoding: Any, seeds_per_rec: int = 1) -> list[Circuit]:
        """
        Recommend seed circuits based off Tensor encoding of an operation.

        args:
            encoding (Any): A Tensor encoding of an operation.

            seeds_per_rec (int): The number of seeds to recommend per call to
                `self.recommend`.

        returns:
            (list[Circuit]): A list of seed circuits taken from 
                `self.seed_circuits`.
        """


QSeedSynthesisPass(BasePass):

This pass accepts a partitioned quantum circuit. Having a seed recommender
for each individual block does not work because linux will kill the process
(too many open files). Therefore, there needs to be a check that the input
circuit is already partitioned.

The partitioned circuit is scanned. Operations are filtered by topology.
Filtered Operations are then batched and sent to recommenders. Circuit
recommendations are gathered then sorted. Seeded synthesis is run on each
block. The results are incorporated into the new circuit if the
filter_function criteria is met.

Take inspiration from the ForEachBlockPass.

    def __init__(
        self,
        recommender: Recommender | Sequence[Recommender],
        seeds_per_rec: int = 3,
        batch_size: int = 64,
        replace_filter: ReplaceFilterFn | str = 'always', 
    ) -> None:
        """
        Construct a QSeedSynthesisPass.

        This pass optimizes Circuits partitioned into 3 qubit blocks.

        args:
            recommender (Recommender | Sequence[Recommender]): Takes as input
                an Operation and produces as output a list[Circuit]. If a
                Sequence is provided, each Recommender is assumed have a 
                specified CouplingGraph. Each Recommender should already have
                its state loaded.

            seeds_per_rec (int): The number of seeds to recommend per circuit
                Operation. (Default: 3)

            batch_size (int): Max number of operations to pass to a recommender
                at a time. (Default: 64)

            replace_filter (ReplaceFilterFn | str): Determines if a block will
                be replaced in the original circuit. (Default: 'always')

        raises:
            ValueError: If a Sequence is provided for `recommender`, but the
                not all three 3-qudit topologies have respective recommenders.
                If non-topology aware synthesis is desired, only provide a 
                single recommender.
        """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Run QSeed optimization on the given `circuit`.

        raises:
            ValueError: If `circuit` is not partitioned into 3 qubit blocks.
        """
        # Error checking
        # Enumerate blocks and determine their CouplingGraph
        # Bin blocks by topology and batch into `self.batch_size` batches
        # Call recommenders on each batch
        # Sort `seed_circuits` by block enumeration
        # ForEach loop must
        # 1) set seeds in each PassData, ignore 1 and 2 qudit blocks
        # 2) call seeded QSearch
