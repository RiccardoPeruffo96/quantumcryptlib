# src/sdp/matrix.py

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