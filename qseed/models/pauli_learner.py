"""Learn circuit to template mapping given Pauli vectors."""
import torch.nn as nn
from learning.pauli_encoder import PauliEncoder

class PauliLearner(nn.Module):
	def __init__(self, num_qubits : int = 3):
		if num_qubits != 3:
			raise ValueError(
				f'Only widths of 3 qubits are supported ({num_qubits} != 3).'
			)
		super().__init__()
		self.num_qubits = num_qubits
		self.pauli_len = 4 ** num_qubits
		self.dropout_p = 0.4

		self.num_templates = 1100

		# Encoder
		self.encoder = PauliEncoder(self.num_qubits)

		# Network for learning mapping to templates
		widths = [16, 32]
		# Set up residual blocks	
		self.resblocks = nn.ModuleList([
			nn.Sequential(
				nn.Linear(width, width),
				nn.LayerNorm(width),
				nn.GELU(),
		 ) for width in widths
		])
		# Set up blocks between residual blocks
		inter_widths = [(widths[i],widths[i+1]) for i in range(len(widths)-1)]
		inter_widths.append((widths[-1], self.num_templates))
		self.interblocks = nn.ModuleList([
			nn.Sequential(
				nn.Linear(in_width, out_width),
				nn.LayerNorm(out_width),
				nn.GELU(),
				nn.Dropout(p=self.dropout_p)
			) for in_width, out_width in inter_widths
		])

		self.output_layer = nn.Sequential(
			nn.Linear(self.num_templates, self.num_templates)
		)
	
	def forward(self, x):
		x = self.encoder(x)
		for res_block, inter_block in zip(self.resblocks, self.interblocks):
			x = res_block(x) + x
			x = inter_block(x)
		return self.output_layer(x)
