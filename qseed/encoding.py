import numpy as np
from torch import tensor
from scipy.linalg import expm
from bqskit.utils.math import unitary_log_no_i, pauli_expansion
from bqskit.utils.math import dot_product, PauliMatrices
from bqskit import Circuit
from bqskit.qis.unitary import UnitaryMatrix


def to_canonical_unitary(unitary : np.array) -> np.array:
    """
    Convert a unitary to a special unitary, eliminating the global phase factor.
    Ensure that unitaries that differ in global phase by a root of unitary are
    mapped to the same matrix.

    Arguments:
        unitary (np.array): A unitary matrix.
    
    Returns:
        (np.array): A canonical unitary matrix in the speical unitary group.
    """
    determinant = np.linalg.det(unitary)
    dimension = len(unitary)
    global_phase = np.angle(determinant) / dimension
    global_phase = global_phase % (2 * np.pi / dimension)
    global_phase_factor = np.exp(-1j * global_phase)
    special_unitary = global_phase_factor * unitary
    # Standardize speical unitary to account for exp(-i2pi/N) differences
    std_phase = np.angle(special_unitary[0,0])
    correction_phase = 0 - std_phase
    std_correction = np.exp(1j * correction_phase)
    return std_correction * special_unitary

def pauli_encoding(circuit : Circuit) -> tensor:
    """
    Get the canonical pauli coefficient vector of the unitary associated with
    `circuit`.

    Args:
        circuit (Circuit): Circuit with an associated unitary.
    
    Returns:
        pauli_vector (np.array): Pauli coefficient vector encoding of the
            unitary associated with `circuit`.
    """
    unitary = circuit.get_unitary().numpy
    canonical_unitary = to_canonical_unitary(unitary)
    H = unitary_log_no_i(canonical_unitary)
    return tensor(pauli_expansion(H)).float()

def structural_encoding(circuit : Circuit) -> tensor:
    """
    Encode the location of the first 

    Args:
        circuit (Circuit): Circuit with an associated unitary.
    
    Returns:
        pauli_vector (np.array): Pauli coefficient vector encoding of the
            unitary associated with `circuit`.
    """
    edge_to_int = {
        (0,1) : 1, (1,2) : 2, (0,2) : 3,
        (1,0) : 1, (2,1) : 2, (2,0) : 3,
    }
    encoded_edges = []
    for op in circuit:
        if op.num_qudits == 2:
            encoded_edges.append(edge_to_int(tuple(op.location)))
    return tensor(encoded_edges[:16]).float()

