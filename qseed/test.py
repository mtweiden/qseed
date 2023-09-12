from bqskit import Circuit
from bqskit.compiler.passdata import PassData
from bqskit.qis.graph import CouplingGraph
from qseed.recommender import Recommender


if __name__ == "__main__":

    seeds = {i: Circuit(3) for i in range(3)}
    data = PassData(Circuit(3))
    data['seeds'] = seeds

    graph_1 = CouplingGraph([(0, 1), (1, 2)])
    graph_2 = CouplingGraph([(0, 2), (1, 2)])
    graph_3 = CouplingGraph([(0, 1), (0, 2)])

    recs = {
        graph_1: 0,
        graph_2: 1,
        graph_3: 2,
    }

    print(recs)
    print(recs[graph_1])
