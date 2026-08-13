# src/sdp/dual.py

import cvxpy as cp
import numpy as np
import numpy.typing as npt

def solve_dual_sdp(
    observations: list[tuple[npt.NDArray[np.complex128], float]],
    C: npt.NDArray[np.complex128],
    delta: float = 1e-10,
    N: int = 100000
) -> float | None:
    """
    Solves the dual Semidefinite Program (SDP) to compute the secure key rate bound.

    Args:
        observations: A list of the witness operators W_k of shape (d^2, d^2)
        linked to s list of observed real expectation values c_k.
        C: The phase error cost matrix of shape (d^2, d^2).
        delta: The statistical confidence level for Hoeffding's inequality.
        N: The total number of experimental coincidences.

    Returns:
        float | None: The optimal dual objective value (upper bound on Eve's information),
        or None if the solver fails.
    """
    # Extract dim value from C matrix
    dim = C.shape[0]

    # Number of constraints/observations (k)
    num_constraints = len(observations)

    # Statistical margin of error calculated from Hoeffding's inequality bound
    # delta = 2*exp(-2 * N * e^2) => 
    # e = sqrt(log(delta/2)/(-2*N))
    # e = sqrt(log(2/delta)/(2*N))
    e: float = np.sqrt(np.log(2/delta)/(2*N))

    # 1. Dual Variables (Lagrange multipliers, one for each constraint)
    y = cp.Variable(num_constraints)

    W_matrices: list[npt.NDArray[np.complex128]] = []
    c_values: list[float] = []

    for W_k, c_k in observations:
        c_values.append(c_k)
        W_matrices.append(W_k)

    c_vector = np.array(c_values)

    # 2. Objective Function (Minimize)
    # y^T * c + penalty for finite-key statistics
    objective_expr = y.T @ c_vector + cp.sum(cp.abs(y) * e)
    objective = cp.Minimize(objective_expr)

    # 3. The Dual Constraint (Linear Matrix Inequality)
    # In the primal, we had rho >> 0. In the dual, rho is replaced by a condition 
    # on the sum of the witness operators weighted by the dual variables.
    y_0 = cp.Variable()
    M = y_0 * np.eye(dim, dtype=np.complex128)

    for k in range(num_constraints):
        M += y[k] * W_matrices[k]
    
    # The resulting matrix must be Positive Semidefinite (PSD)
    constraints = [
        M - C >> 0
    ]

    # 4. Solve the Problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS) # SCS or MOSEK are recommended for SDPs
    return problem.value
