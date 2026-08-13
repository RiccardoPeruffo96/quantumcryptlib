# src/sdp/primal.py

import cvxpy as cp
import numpy as np
import numpy.typing as npt

def solve_primal_sdp(
    d: int,
    observations: list[tuple[npt.NDArray[np.complex128], float]],
    C: npt.NDArray[np.complex128],
    delta: float = 1e-10,
    N: int = 100000
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
        delta: The statistical confidence level for the Hoeffding's inequality 
            bound, used to define the margin of error in the constraints.
        N: The total number of experimental trials or measurements, used to
            calculate the statistical margin of error.

    Returns:
        npt.NDArray[np.complex128] | None: The optimal density matrix (rho) 
        of shape (d^2, d^2) that satisfies all quantum state constraints and 
        observation requirements, or None if the solver fails to converge.
    """
    # Total dimension of the bipartite joint Hilbert space (d^2 * d^2)
    dim = d * d

    # Statistical margin of error calculated from Hoeffding's inequality bound
    # delta = 2*exp(-2 * N * e^2) => 
    # e = sqrt(log(delta/2)/(-2*N))
    # e = sqrt(log(2/delta)/(2*N))
    e: float = np.sqrt(np.log(2/delta)/(2*N))

    # 1. Decision Variable: Density matrix (Hermitian matrix of size d^2 * d^2)
    rho = cp.Variable((dim, dim), complex=True, hermitian=True)

    # 2. Fundamental Quantum State Constraints
    constraints = [
        rho >> 0,                   # Positive Semidefinite constraint (rho >= 0)
        cp.trace(rho) == 1          # Trace = 1
    ]

    # 3. Channel Observation Constraints
    for W_k, c_k in observations:
        # Match expected value Tr(W_k * rho) with observed data c_k under the margin of error e
        constraints.append(cp.abs(cp.real(cp.trace(W_k @ rho)) - c_k) <= e)

    # 4. Objective Function Setup
    # Example objective: Minimizing the expectation value of a cost matrix C
    objective = cp.Minimize(cp.real(cp.trace(C @ rho)))

    # 5. Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, eps=1e-10, max_iters=100000)
    print("Solver Status: ", prob.status)

    return rho.value
