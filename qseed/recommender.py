from typing import Any, Sequence
from bqskit import Circuit
from bqskit.compiler import BasePass
from torch.nn import Module
from torch import tensor
from torch import topk
from qseed.encoding import pauli_encoding

# TODO:
# - add cuda support

class TopologyAwareRecommenderPass(BasePass):

    def __init__(
        self,
        recommender_models  : Sequence[Module],
        model_states        : Sequence[dict[str,Any]],
        template_lists      : Sequence[Sequence[Circuit]],
        seeds_per_inference : int = 3
    ) -> None:
        """
        Use the `recommender_models` to make predictions about which seeds
        should be used to synthesize circuits. This recommender uses the 
        pauli coefficients vector to encode unitaries, and assumes the 
        connectivity of the input must be maintained in the output circuit.

        Args:
            recommender_models (Sequence[torch.nn.Module]): Model used to make 
                seed recommendations.
            
            model_states (Sequence[dict[str,Any]]): Weights and biases for the
                input `recommender_models`.
            
            template_lists (Sequence[list[Circuit]]): The outputs of 
                `recommender_model` must correspond to template circuits in 
                this list.
            
            seeds_per_inference (int): The number of seeds to recommend
                per circuit.
        
        Note:
            A torch.nn.Module typically does not have a way to verify output
            dimension. Having a mismatch between the template_list and the
            recommender model output will cause errors.
        """
        assert len(recommender_models) == len(model_states)

        self.models = [m.float() for m in recommender_models]
        for model,state in zip(self.models, model_states):
            model.load_state_dict(state)
        self.template_lists = template_lists
        self.seeds_per_inference = seeds_per_inference
    
    def _encode(self, circuit : Circuit) -> tensor:
        """
        Function that encodes a circuit into some format that the recommender
        model can take as input.

        Args:
            circuit (Circuit): The circuit to be encoded.
        
        Returns:
            encoded_circuit (torch.tensor): The encoded circuit which can be
                fed to the recommender model as an input.
        """
        return pauli_encoding(circuit)
    
    def _decode(self, model_output : tensor, topology : str) -> list[Circuit]:
        """
        Function that takes an encoded recommender model output, and transforms
        it into a Circuit.

        Args:
            model_output (torch.tensor): The encoded output of a recommender 
                model.
        
        Returns:
            recommendations (list[Circuit]): A list of recommendation seed
                circuits.
        """
        top = ['a','b','c','d'].index(topology)
        _,indices = topk(model_output, self.seeds_per_inference, dim=-1)
        return [self.template_lists[top][int(i)] for i in indices]
    
    def _detect_connectivity(self, circuit: Circuit) -> str:
        """
        The input `circuit` is assumed to have 3 qubits, and be one of 4
        possible connectivities.

        'a' - linear   - [(0,1),(1,2)]
        'b' - linear   - [(0,1),(0,2)]
        'c' - linear   - [(0,2),(1,2)]
        'd' - complete - [(0,1),(1,2),(0,2)]
        """
        if circuit.num_qudits != 3:
            raise RuntimeError(
                'Recommender currently only supports blocksize 3 circuits.'
            )
        a,b,c = circuit.coupling_graph.get_qudit_degrees()
        if a == 1 and b == 2 and c == 1:
            return 'a'
        elif a == 2 and b == 1 and c == 1:
            return 'b'
        elif a == 1 and b == 1 and c == 2:
            return 'c'
        elif a == 2 and b == 2 and c == 2:
            return 'd'
        else: # no or very little connectivity case
            return 'a'
    
    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """
        Calls the recommender model on the given `circuit`, storing the 
        recommended seed circuits in the `data` dictionary.
        """
        if 'recommended_seeds' not in data:
            data['recommended_seeds'] = []

        encoded_circuit = self._encode(circuit)
        connectivity_code = self._detect_connectivity(circuit)
        model_output = self.model(encoded_circuit)
        recommendations = self._decode(model_output, connectivity_code)

        data['recommended_seeds'].append(recommendations)