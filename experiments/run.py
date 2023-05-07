from argparse import ArgumentParser
import numpy as np
import logging
import pickle

from bqskit import Circuit
from bqskit.compiler import MachineModel
from bqskit.qis.graph import CouplingGraph

from qseed import Handler
from qseed import QSearchHandler
from qseed import RandomHandler

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('file_path')
	parser.add_argument('--qsearch', action='store_true')
	parser.add_argument('--random', action='store_true')
	args = parser.parse_args()

	path = args.file_path
	name = path.split('/')[-1].split('.')[0]
	print(name)

	np.random.seed(1234)

	if args.qsearch:
		handler_type = 'qsearch'
		handler = QSearchHandler()
	elif args.random:
		handler_type = 'random'
		handler = RandomHandler()
	else:
		handler_type = 'qseed'
		handler = Handler()

	print(handler_type)

	circuit = Circuit.from_file(path)
	n = int(np.ceil(np.sqrt(circuit.num_qudits)))
	graph   = CouplingGraph.grid(n, n)
	machine = MachineModel(n*n, graph)
	data = {'machine_model': machine, 'circuit_name': f'{name}.pickle'}

	circuit = Circuit.from_file(path)
	log_name = f'experiments/logs/{handler_type}/{name}.log'
	logging.basicConfig(level=logging.INFO, filename=log_name)
	new_circuit = handler.handle(circuit, data)

	with open(f'experiments/circuits/{handler_type}/{name}.pickle','wb') as f:
		pickle.dump(new_circuit, f)
