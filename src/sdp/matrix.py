# src/sdp/matrix.py

"""
This module contains functions to generate matrices 
"""

import numpy as np
import numpy.typing as npt

def omega(d: int) -> np.complex128:
    """
    Generate omega
    Omega is a complex number that represents the d-th root of unity.
    It is defined as exp(2πi/d), where i is the imaginary unit.
    This function returns the value of omega for a given dimension d.
    """
    return np.exp(2j * np.pi / d)

def genShiftMatrix(d: int) -> npt.NDArray[np.complex128]:
    """
    Generate shift matrix
    es:
    X = [[0, 0, 0, 1],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0]]
    Is a unitary matrix that shifts the elements of a vector to the right by one position,
    with the last element wrapping around to the first position.
    """
    X = np.eye(d, k=-1)
    X[0, d-1] = 1.0
    return np.array(X)

def genPhaseMatrix(d: int,
                   omega_value: complex | None = None
                   ) -> npt.NDArray[np.complex128]:
    """
    Generate phase matrix
    es:
    Z = [[1, 0, 0, 0],
         [0, 1j, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, -1j]]
    Each value of the diagonal is an omega fraction of movement from -1 to 1 in the complex plane.
    """
    if omega_value is None:
        omega_value = omega(d)
    Z = np.diag([omega_value**j for j in range(d)])
    return np.array(Z)

def genWeylHeisenbergOperators(d: int,
                               Xa_d: npt.NDArray[np.complex128] | None = None,
                               Zb_d: npt.NDArray[np.complex128] | None = None
                               ) -> dict[tuple[int, int], npt.NDArray[np.complex128]]:
    """
    Generate Weyl Heisenberg Operators:
    es:
    Uab_d = Xa_d*Zb_d for each a, b in {0, 1, ..., d-1}
    Note that the Weyl-Heisenberg operators are a set of unitary matrices that form a basis for the space of d x d complex matrices.
    Fundamental property: Zb_d*Xa_d = omega^(-ab)*Xa_d*Zb_d
    """
    if Xa_d is None:
        Xa_d = genShiftMatrix(d)
    
    if Zb_d is None:
        Zb_d = genPhaseMatrix(d)
    
    U: dict[tuple[int, int], npt.NDArray[np.complex128]] = {}
    for a in range(d):
        Xa = np.linalg.matrix_power(Xa_d, a)
        for b in range(d):
            Zb = np.linalg.matrix_power(Zb_d, b)
            U[(a, b)] = Xa @ Zb
    return U

def genBipartiteWeylHeisenbergOperators(d: int,
                               U1_d: dict[tuple[int, int], npt.NDArray[np.complex128]] | None = None,
                               U2_d: dict[tuple[int, int], npt.NDArray[np.complex128]] | None = None
                               ) -> dict[tuple[int, int, int, int], npt.NDArray[np.complex128]]:
    """
    Combine both Weyl Heisenberg operators using Kronecker product to create the d^4 bipartite operators
    """
    U_bipartite = {}
    for (a1, b1), U1 in U1_d.items():
        for (a2, b2), U2 in U2_d.items():
            U_bipartite[(a1, b1, a2, b2)] = np.kron(U1, U2)
    return U_bipartite

def genBipartiteWeylHeisenbergOperators(d: int,
                               Xa1_d: npt.NDArray[np.complex128] | None = None,
                               Zb1_d: npt.NDArray[np.complex128] | None = None,
                               Xa2_d: npt.NDArray[np.complex128] | None = None,
                               Zb2_d: npt.NDArray[np.complex128] | None = None
                               ) -> dict[tuple[int, int, int, int], npt.NDArray[np.complex128]]:
    """
    Override the previous function to generate bipartite Weyl Heisenberg operators directly from the shift and phase matrices.
    """
    U1_d = genWeylHeisenbergOperators(d, Xa1_d, Zb1_d)
    U2_d = genWeylHeisenbergOperators(d, Xa2_d, Zb2_d)
    return genBipartiteWeylHeisenbergOperators(d, U1_d, U2_d)
