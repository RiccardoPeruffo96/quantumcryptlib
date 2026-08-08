import cvxpy as cp
import numpy as np
import numpy.typing as npt
import warnings

def solve_primal_sdp(
    d: int,
    observations: list[tuple[npt.NDArray[np.complex128], float]],
    C: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128] | None:
    """Solves the primal Semidefinite Program (SDP) to find the quantum state rho.

    This function formulates and solves an SDP to reconstruct or optimize a 
    bipartite quantum state density matrix (rho) under experimental or 
    theoretical observation constraints.

    Args:
        d: The dimension of the single qudit (local space dimension).
        observations: A list of tuples, where each tuple contains an operator 
            matrix W_k of shape (d^2, d^2) and its corresponding observed 
            real expectation value c_k (float). Represents Tr(W_k * rho) = c_k.
        C: The phase error cost matrix.

    Returns:
        npt.NDArray[np.complex128] | None: The optimal density matrix (rho) 
        of shape (d^2, d^2) that satisfies all quantum state constraints and 
        observation requirements, or None if the solver fails to converge.
    """
    # Total dimension of the bipartite joint Hilbert space (d^2 * d^2)
    dim = d * d

    # 1. Decision Variable: Density matrix (Hermitian matrix of size d^2 * d^2)
    rho = cp.Variable((dim, dim), complex=True, hermitian=True)

    # 2. Fundamental Quantum State Constraints
    constraints = [
        rho >> 0,                    # Positive Semidefinite constraint (rho >= 0)
        cp.trace(rho) == 1,          # Trace = 1
    ]

    # 3. Channel Observation Constraints
    for W_k, c_k in observations:
        # Match expected value Tr(W_k * rho) with observed data c_k
        constraints.append(cp.real(cp.trace(W_k @ rho)) == c_k)

    # 4. Objective Function Setup
    # Example objective: Minimizing the expectation value of a cost matrix C
    objective = cp.Minimize(cp.real(cp.trace(C @ rho)))

    # 5. Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    return rho.value

def solve_primal_sdp_W_from_U_bipartite(
    d: int,
    U_bipartite: dict[tuple[int, int, int, int], npt.NDArray[np.complex128]],
    observations: dict[tuple[int, int, int, int], float],
    C: npt.NDArray[np.complex128]
) -> npt.NDArray[np.complex128] | None:
    warnings.warn(
        "solve_primal_sdp_W_from_U_bipartite has been deprecated."
        "Use solve_primal_sdp(d, observations, C) instead.",
        category=DeprecationWarning,
        stacklevel=2  # Segnala la riga del chiamante, non l'interno della funzione
    )
    """Solves the primal Semidefinite Program (SDP) to find the quantum state rho.

    This function formulates and solves an SDP to reconstruct or optimize a 
    bipartite quantum state density matrix (rho) under experimental or 
    theoretical observation constraints.

    Args:
        d: The dimension of the single qudit (local space dimension).
        U_bipartite: A dictionary mapping 4-tuple keys (a1, b1, a2, b2) to 
            their corresponding (d^2 * d^2) Weyl-Heisenberg operator matrices 
            acting on the joint bipartite Hilbert space.
        observations: A dictionary mapping 4-tuple keys (a1, b1, a2, b2) to 
            their observed real expectation values (gamma) measured in the 
            quantum channel. Represents the constraint Tr(rho * W) = gamma.
        C: The phase error cost matrix

    Returns:
        npt.NDArray[np.complex128] | None: The optimal density matrix (rho) 
        of shape (d^2, d^2) that satisfies all quantum state constraints and 
        observation requirements, or None if the solver fails to converge.

    NOTE: This function is deprecated because it works only when 'observations'
    parameter contains the coordinates about correct matrix inside U_bipartite.
    With this structure the access to U_bipartite is not O(1) time.
    """
    # Total dimension of the bipartite joint Hilbert space (d^2 * d^2)
    dim = d * d

    # 1. Decision Variable: Density matrix (Hermitian matrix of size d^2 * d^2)
    rho = cp.Variable((dim, dim), complex=True, hermitian=True)

    # 2. Fundamental Quantum State Constraints
    constraints = [
        rho >> 0,                    # Positive Semidefinite constraint (rho >= 0)
        cp.trace(rho) == 1,          # Trace = 1
    ]

    # 3. Channel Observation Constraints
    for gamma_val, key in observations.items():
        W_v = U_bipartite[key]
        # Match expected value Tr(rho * W_v) with observed data gamma
        constraints.append(cp.real(cp.trace(rho @ W_v)) == gamma_val)

    # 4. Objective Function Setup
    # Example objective: Minimizing the expectation value of a cost matrix C
    objective = cp.Minimize(cp.real(cp.trace(C @ rho)))

    # 5. Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    return rho.value
