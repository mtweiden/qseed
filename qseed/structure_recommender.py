from typing import Any, Sequence
from bqskit import Circuit
from bqskit.compiler import BasePass
from torch.nn import Module
from torch import tensor
from torch import topk
from encoding import structural_encoding

# TODO:
# - add cuda support

class StructureRecommenderPass(BasePass):

    def __init__(
        self,
        recommender_model : Module,
        template_list : Sequence[Circuit],
        seeds_per_inference : int = 3
    ) -> None:
        """
        Use the `recommender_model` to make predictions about which seeds
        should be used to synthesize circuits.

        Args:
            recommender_model (torch.nn.Module): Model used to make seed
                recommendations.
            
            template_list (list[Circuit]): The outputs of `recommender_model`
                must correspond to template circuits in this list.
            
            seeds_per_inference (int): The number of seeds to recommend
                per circuit.
        
        Note:
            A torch.nn.Module typically does not have a way to verify output
            dimension. Having a mismatch between the template_list and the
            recommender model output will cause errors.
        """
        self.model = recommender_model
        self.template_list = template_list
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
        return structural_encoding(circuit)
    
    def _decode(self, model_output : tensor) -> list[Circuit]:
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
        return [self.template_list[int(i)] for i in indices]
    
    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """
        Calls the recommender model on the given `circuit`, storing the 
        recommended seed circuits in the `data` dictionary.
        """
        if 'recommended_seeds' not in data:
            data['recommended_seeds'] = []
        
        encoded_circuit = self._encode(circuit)
        model_output = self.model(encoded_circuit)
        recommendations = self._decode(model_output)

        data['recommended_seeds'].append(recommendations)
