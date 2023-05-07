# Quantum Seeded Synthesis BQSKit Pass

A Python package that implements seeded unitary synthesis.

## Installation

This is available for Python 3.8+ on Linux, macOS, and Windows.

```sh
git clone git@github.com:mtweiden/qseed.git
cd qseed
pip install -e .
```

## Basic Usage
The easiest way to run QSeed is to call the `experiments/run.py` script. It requires a qasm file as input. Adding the `--qsearch` or `--random` flags will perform synthesis with QSearch or with a random seed recommendation scheme.
```
python experiments/run.py qasm/<qasm_family>/mapped-<qasm_file> [--qsearch | --random]
```
Outputs are stored in the corresponding `experiments/circuits` directory.

## How to Cite

To cite qseed, `arXiv link coming soon!`

To cite BQSKit, you can use the [software disclosure](https://www.osti.gov/biblio/1785933).

## License
...

## Copyright
...
