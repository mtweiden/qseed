"""Learn circuit to template mapping given Pauli vectors."""
from torch import tensor, randn
import torch.nn as nn
from learning.pauli_encoder import PauliEncoder
from learning.pauli_decoder import PauliDecoder

class AddGaussianNoise(object):
	def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
		self.mean = mean
		self.std = std
	
	def __call__(self, tensor: tensor) -> tensor:
		return tensor + self.std * randn(tensor.size()) + self.mean
	
	def __repr__(self) -> str:
		return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class PauliAutoencoder(nn.Module):
	def __init__(self, num_qubits : int = 3):
		if num_qubits != 3:
			raise ValueError(
				f'Only widths of 3 qubits are supported ({num_qubits} != 3).'
			)
		super().__init__()
		self.num_qubits = num_qubits
		self.pauli_len = 4 ** num_qubits

		self.encoder = PauliEncoder(self.num_qubits)
		self.decoder = PauliDecoder(self.num_qubits)
	
	def forward(self, x):
		return self.decoder(self.encoder(x))