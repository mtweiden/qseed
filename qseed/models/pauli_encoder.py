"""Learn circuit to template mapping given Pauli vectors."""
import torch
import torch.nn as nn

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

		# Number of output templates
		if num_qubits == 3:
			self.num_templates = 12 # all is 353
		else:
			self.num_templates = 1000

		#self.hidden_depth = 128
		widths = [100, 128, 64, 50, 32, 16] # shallow
		self.layer_widths = [(widths[i],widths[i+1]) for i in range(len(widths)-1)]
		self.num_hidden_layers = len(widths)

		self.input_layer  = nn.Sequential(
			nn.Linear(self.pauli_len, widths[0]),
			nn.LayerNorm(widths[0]),
			nn.GELU(),
			nn.Dropout(p=self.dropout_p)
		)
		self.hidden_layers = nn.ModuleList([
			nn.Sequential(
				nn.Linear(in_width, out_width),
				nn.LayerNorm(out_width),
				nn.GELU(),
				nn.Dropout(p=self.dropout_p)	
			) for (in_width, out_width) in self.layer_widths
		])
		self.output_layer = nn.Sequential(
			nn.Linear(widths[-1], self.num_templates),
		)
	
	def forward(self, x):
		x = self.input_layer(x)
		for hidden_layer in self.hidden_layers:
			x = hidden_layer(x)
		return self.output_layer(x)
