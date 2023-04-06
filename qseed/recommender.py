from typing import Any, Sequence
from bqskit import Circuit
from bqskit.compiler import BasePass
from torch.nn import Module
from torch import tensor
from torch import topk
from qseed.encoding import pauli_encoding
from bqskit.compiler.passdata import PassData
from timeit import default_timer

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
    
    def encode(self, circuit : Circuit) -> tensor:
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
    
    def decode(self, model_output : tensor, topology : int) -> list[Circuit]:
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
        _,indices = topk(model_output, self.seeds_per_inference, dim=-1)
        return [self.template_lists[topology][int(i)] for i in indices]
    
    def detect_connectivity(self, circuit: Circuit) -> str:
        """
        The input `circuit` is assumed to have 3 qubits, and be one of 4
        possible connectivities.

            0 - linear   - [(0,1),(1,2)]
            1 - linear   - [(0,1),(0,2)]
            2 - linear   - [(0,2),(1,2)]
            3 - complete - [(0,1),(1,2),(0,2)]
        """
        if circuit.num_qudits != 3:
            raise RuntimeError(
                f'Recommender currently only supports blocksize 3 circuits. '
                f'Provided circuit has size {circuit.num_qudits}.'
            )
        a,b,c = circuit.coupling_graph.get_qudit_degrees()
        if a == 1 and b == 2 and c == 1:
            return 0
        elif a == 2 and b == 1 and c == 1:
            return 1
        elif a == 1 and b == 1 and c == 2:
            return 2
        elif a == 2 and b == 2 and c == 2:
            if len(self.models) < 4: # no complete graph recommender
                return 0
            return 3
        else: # no or very little connectivity case
            return 0
    
    async def run(self, circuit: Circuit, data: PassData) -> None:
        """
        Calls the recommender model on the given `circuit`, storing the 
        recommended seed circuits in the `data` dictionary.
        """
        if 'recommended_seeds' not in data:
            data['recommended_seeds'] = []

        connectivity_code = self._detect_connectivity(circuit)

        enc_start = default_timer()
        encoded_circuit = tensor(self._encode(circuit)).float()
        enc_end = default_timer()

        inf_start = default_timer()
        model_output = self.models[connectivity_code](encoded_circuit)
        inf_end = default_timer()

        rec_start = default_timer()
        recommendations = self._decode(model_output, connectivity_code)
        rec_end = default_timer()

        data['recommended_seeds'].append(recommendations)

        #print(f'Encoding : {enc_end - enc_start}')
        #print(f'Inference: {inf_end - inf_start}')
        #print(f'Recommend: {rec_end - rec_start}')
