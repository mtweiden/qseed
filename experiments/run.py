from argparse import ArgumentParser
import numpy as np
import logging

from bqskit import Circuit
from qseed import Handler
from qseed import QSearchHandler

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file_path')
    parser.add_argument('--qsearch', action='store_true')
    args = parser.parse_args()

    path = args.file_path
    name = path.split('/')[-1].split('.')[0]
    print(name)

    np.random.seed(1234)

    if not args.qsearch:
        print('qseed')
        logging.basicConfig(level=logging.INFO, filename=f'experiments/logs/qseed/{name}.log')
        qseed_circuit = Circuit.from_file(path)
        qseed_handler = Handler()
        qseed_data = {}
        qseed_handler.handle(qseed_circuit, qseed_data)

    else:
        print('qsearch')
        logging.basicConfig(level=logging.INFO, filename=f'experiments/logs/qsearch/{name}.log')
        qsearch_circuit = Circuit.from_file(path)
        qsearch_handler = QSearchHandler()
        qsearch_data = {}
        qsearch_handler.handle(qsearch_circuit, qsearch_data)
