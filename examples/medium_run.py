from bqskit import Circuit
from qseed import Handler
import numpy as np
from qseed import QSearchHandler


import logging


if __name__ == '__main__':
    file_name = 'mapped-qft_16'
    logging.basicConfig(level=logging.INFO, filename=file_name)

    np.random.seed(1234)

    qseed_circuit = Circuit.from_file(f'qasm/qft/{file_name}.qasm')
    #qsearch_circuit = Circuit.from_file(f'qasm/qft/{file_name}.qasm')

    qseed_handler = Handler()
    #qsearch_handler = QSearchHandler()

    qseed_data = {}
    qseed_handler.handle(qseed_circuit, qseed_data)
    #print('='*80)
    #qsearch_data = {}
    #qsearch_handler.handle(qsearch_circuit, qsearch_data)
