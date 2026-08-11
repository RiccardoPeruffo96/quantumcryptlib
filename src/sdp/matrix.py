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

    Args:
        d: The dimension of the qudit (d >= 2).

    Returns:
        np.complex128: The d-th root of unity, omega = exp(2πi/d).
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

    Args:
        d: The dimension of the qudit (d >= 2).

    Returns:
        npt.NDArray[np.complex128]: The d x d shift matrix.
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

    Args:
        d: The dimension of the qudit (d >= 2).
        omega_value: Optional; the value of omega to use. If None, it will be computed.

    Returns:
        npt.NDArray[np.complex128]: The d x d phase matrix.
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

    Args:
        d: The dimension of the qudit (d >= 2).
        Xa_d: Optional; the shift matrix. If None, it will be generated.
        Zb_d: Optional; the phase matrix. If None, it will be generated.

    Returns:
        dict[tuple[int, int], npt.NDArray[np.complex128]]: A dictionary mapping (a, b) pairs to their corresponding Weyl-Heisenberg operator matrices Uab_d.
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
                               U1_d: dict[tuple[int, int], npt.NDArray[np.complex128]],
                               U2_d: dict[tuple[int, int], npt.NDArray[np.complex128]]
                               ) -> dict[tuple[int, int, int, int], npt.NDArray[np.complex128]]:
    """
    Combine both Weyl Heisenberg operators using Kronecker product to create the d^4 bipartite operators

    Args:
        d: The dimension of the qudit (d >= 2).
        U1_d: A dictionary mapping (a1, b1) pairs to their corresponding Weyl-Heisenberg operator matrices for the first qudit.
        U2_d: A dictionary mapping (a2, b2) pairs to their corresponding Weyl-Heisenberg operator matrices for the second qudit.

    Returns:
        dict[tuple[int, int, int, int], npt.NDArray[np.complex128]]: A dictionary mapping (a1, b1, a2, b2) tuples to their corresponding bipartite Weyl-Heisenberg operator matrices.
    """
    U_bipartite: dict[tuple[int, int, int, int], npt.NDArray[np.complex128]] = {}
    for (a1, b1), U1 in U1_d.items():
        for (a2, b2), U2 in U2_d.items():
            U_bipartite[(a1, b1, a2, b2)] = np.kron(U1, U2)
    return U_bipartite

# TODO: This is a draft version not tested
def gen_phase_error_cost_matrix(d: int) -> npt.NDArray[np.complex128]:
    """
    Generates the cost matrix C representing the Phase Error operator.
    
    In QKD SDP formulations, minimizing Tr(C * rho) finds the maximum possible 
    fidelity or minimum phase error under Eve's attack.
    
    Args:
        d: Dimension of single qudit (dim of bipartite space is d^2 x d^2).
        
    Returns:
        npt.NDArray[np.complex128]: Hermitian matrix C of size (d^2, d^2).
    """
    dim = d * d

    # 1. Define the ideal maximally entangled state |Phi+> = (1/sqrt(d)) * sum(|i,i>)
    phi_plus = np.zeros((dim, 1), dtype=np.complex128)
    for i in range(d):
        # Index in bipartite basis corresponding to |i, i> -> i*d + i
        phi_plus[i * d + i] = 1.0 / np.sqrt(d)

    # 2. Build the projector onto the ideal state: P_ideal = |Phi+><Phi+|
    P_ideal = phi_plus @ phi_plus.conj().T

    # 3. Cost matrix C = I - P_ideal (Phase error operator)
    # Minimizing Tr(C @ rho) is equivalent to maximizing Fidelity Tr(P_ideal @ rho)
    C = np.eye(dim, dtype=np.complex128) - P_ideal

    return C

def gen_W_QBER_Z(d: int) -> npt.NDArray[np.complex128]:
    """
    Generate W_QBER_Z operator.
    Total dimension of the matrix: (d^2) x (d^2)

    Args:
        d: Dimension of single qudit (dim of bipartite space is d^2 x d^2).

    Returns:
        npt.NDArray[np.complex128]: Hermitian matrix W_QBER_Z of size (d^2, d^2).
    """
    # This operator projects onto the subspace spanned by the states |i,i> for i in {0, 1, ..., d-1}.
    proj_correct = np.zeros((d**2, d**2), dtype=complex)
    for i in range(d):
        # State |i>
        ket_i = np.zeros((d, 1), dtype=complex)
        ket_i[i] = 1.0
        proj_i = ket_i @ ket_i.T.conj() # |i><i|
        # Tensor product of proj_i with itself to get the bipartite projector
        proj_correct += np.kron(proj_i, proj_i)
    
    # W_QBER_Z = I_total - proj_correct (for each i != j)
    W_qber_z = np.eye(d**2, dtype=complex) - proj_correct
    return W_qber_z

def gen_W_visibility_X(d: int,
                      Xa_d: npt.NDArray[np.complex128] | None = None) -> npt.NDArray[np.complex128]:
    """
    Generate the Visibility operator along the X basis for two qudits.
    
    Args:
        d: Dimension of single qudit (dim of bipartite space is d^2 x d^2).
        Xa_d: Optional; the shift matrix. If None, it will be generated.
        
    Returns:
        npt.NDArray[np.complex128]: Hermitian matrix W_VisibilityX of size (d^2, d^2).    
    """

    X_d = Xa_d
    if(Xa_d is None):
        X_d = genShiftMatrix(d)

    W_vis_x = np.zeros((d**2, d**2), dtype=complex)
    
    for k in range(1, d):
        X_k = np.linalg.matrix_power(X_d, k)
        W_vis_x += np.kron(X_k.conj().T, X_k)
        
    return W_vis_x / d
