# Quantum Seeded Synthesis BQSKit Pass

A Python package that implements machine learning powered seeded quantum circuit synthesis.

## Installation

This is available for Python 3.8+ on Linux, macOS, and Windows.

```sh
git clone git@github.com:mtweiden/qseed.git
cd qseed
pip install -e .
```

## Basic Usage
The easiest way to run QSeed is to call the `run.py` script. It requires a qasm file as input. Adding the `--qsearch` flag will perform synthesis with QSearch.
```sh
python run.py qasm/<qasm_family>/mapped-<qasm_file> [--qsearch]
```
Outputs are stored in the `compiled_circuits` directory.

## How to Cite

To cite qseed:
```
@misc{
    weiden2023improving,
    title={Improving Quantum Circuit Synthesis with Machine Learning}, 
    author={Mathias Weiden and Ed Younis and Justin Kalloor and John Kubiatowicz and Costin Iancu},
    year={2023},
    eprint={2306.05622},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```

To cite BQSKit, you can use the [software disclosure](https://www.osti.gov/biblio/1785933).
