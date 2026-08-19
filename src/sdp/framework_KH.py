# src/sdp/framework_KH.py

import cvxpy as cp
import numpy as np
import numpy.typing as npt

class framework_KH:
    def __init__(self, d: int, delta: float = 1e-10, N: int = 100000):
        """
        Initializes the framework_KH class with statistical parameters.
        
        Args:
            d: The dimension of the single qudit (local space dimension).
            delta: The statistical confidence level for the Hoeffding's inequality 
                    bound, used to define the margin of error in the constraints.
            N: The total number of experimental trials or measurements, used to
                    calculate the statistical margin of error.
        """
        self.d: int = d
        self.omega: complex = self.omega()  # d-th root of unity
        self.delta: float = delta
        self.N: int = N
        # Statistical margin of error calculated from Hoeffding's inequality bound
        # delta = 2*exp(-2 * N * e^2) => 
        # e = sqrt(log(delta/2)/(-2*N))
        # e = sqrt(log(2/delta)/(2*N))
        self.e: float = np.sqrt(np.log(2/delta)/(2*N))

    def solve_primal_sdp(
        self,
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
        dim = self.d * self.d

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
            constraints.append(cp.abs(cp.real(cp.trace(W_k @ rho)) - c_k) <= self.e)

        # 4. Objective Function Setup
        # Example objective: Minimizing the expectation value of a cost matrix C
        objective = cp.Minimize(cp.real(cp.trace(C @ rho)))

        # 5. Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, eps=1e-10, max_iters=100000)

        return rho.value

    def solve_dual_sdp(
        self,
        observations: list[tuple[npt.NDArray[np.complex128], float]],
        C: npt.NDArray[np.complex128]
    ) -> float | None:
        """
        Solves the dual SDP formulation for the secure key rate.

        Args:
            observations: A list of the witness operators W_k of shape (d^2, d^2)
            linked to s list of observed real expectation values c_k.
            C: The phase error cost matrix of shape (d^2, d^2)

        Returns:
            float | None: The optimal dual objective value (upper bound on Eve's information),
            or None if the solver fails.

        NOTE:
            This function implements the dual SDP formulation as described in the paper:
            arXiv:2406.08544, Section 'Key Rate Calculation', Example 1, Eq. (X).
        """
        # Extract dim value from C matrix
        dim = C.shape[0]

        # Number of constraints/observations (k)
        num_constraints = len(observations)

        # 1. Dual Variables (Lagrange multipliers, one for each constraint)
        y = cp.Variable(num_constraints)

        # Define the trace constraint
        y_0 = cp.Variable()

        # Extract W_k and c_k from observations list
        W_matrices: list[npt.NDArray[np.complex128]] = []
        c_values: list[float] = []

        for W_k, c_k in observations:
            c_values.append(c_k)
            W_matrices.append(W_k)

        c_vector = np.array(c_values)

        # 2. Objective Function (Minimize)
        # y^T * c + penalty for finite-key statistics
        objective_expr = (y.T @ c_vector + cp.sum(cp.abs(y) * self.e)) + y_0
        objective = cp.Minimize(objective_expr)

        # 3. The Dual Constraint (Linear Matrix Inequality)
        # In the primal, we had rho >> 0. In the dual, rho is replaced by a condition 
        # on the sum of the witness operators weighted by the dual variables.
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
    
    def omega(self) -> np.complex128:
        """
        Generate omega
        Omega is a complex number that represents the d-th root of unity.
        It is defined as exp(2πi/d), where i is the imaginary unit.
        This function returns the value of omega for the instance's d.

        Returns:
            np.complex128: The d-th root of unity, omega = exp(2πi/d).
        """
        return np.exp(2j * np.pi / self.d)

    def genShiftMatrix(self) -> npt.NDArray[np.complex128]:
        """
        Generate shift matrix
        es:
        X = [[0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]]
        Is a unitary matrix that shifts the elements of a vector to the right by one position,
        with the last element wrapping around to the first position.

        Returns:
            npt.NDArray[np.complex128]: The d x d shift matrix.
        """
        X = np.eye(self.d, k=-1)
        X[0, self.d-1] = 1.0
        return np.array(X)

    def genPhaseMatrix(self) -> npt.NDArray[np.complex128]:
        """
        Generate phase matrix
        es:
        Z = [[1, 0, 0, 0],
            [0, 1j, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1j]]
        Each value of the diagonal is an omega fraction of movement from -1 to 1 in the complex plane.

        Returns:
            npt.NDArray[np.complex128]: The d x d phase matrix.
        """
        Z = np.diag([self.omega**j for j in range(self.d)])
        return np.array(Z)

    def genWeylHeisenbergOperators(self,
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
            Xa_d: Optional; the shift matrix. If None, it will be generated.
            Zb_d: Optional; the phase matrix. If None, it will be generated.

        Returns:
            dict[tuple[int, int], npt.NDArray[np.complex128]]: A dictionary mapping (a, b) pairs to their corresponding Weyl-Heisenberg operator matrices Uab_d.
        """
        if Xa_d is None:
            Xa_d = self.genShiftMatrix()
        
        if Zb_d is None:
            Zb_d = self.genPhaseMatrix()
        
        U: dict[tuple[int, int], npt.NDArray[np.complex128]] = {}
        for a in range(self.d):
            Xa = np.linalg.matrix_power(Xa_d, a)
            for b in range(self.d):
                Zb = np.linalg.matrix_power(Zb_d, b)
                U[(a, b)] = Xa @ Zb
        return U

    def genBipartiteWeylHeisenbergOperators(U1_d: dict[tuple[int, int], npt.NDArray[np.complex128]],
                                U2_d: dict[tuple[int, int], npt.NDArray[np.complex128]]
                                ) -> dict[tuple[int, int, int, int], npt.NDArray[np.complex128]]:
        """
        Combine both Weyl Heisenberg operators using Kronecker product to create the d^4 bipartite operators

        Args:
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
    def gen_phase_error_cost_matrix(self) -> npt.NDArray[np.complex128]:
        """
        Generates the cost matrix C representing the Phase Error operator.
        
        In QKD SDP formulations, minimizing Tr(C * rho) finds the maximum possible 
        fidelity or minimum phase error under Eve's attack.
                    
        Returns:
            npt.NDArray[np.complex128]: Hermitian matrix C of size (d^2, d^2).
        """
        dim = self.d * self.d

        # 1. Define the ideal maximally entangled state |Phi+> = (1/sqrt(d)) * sum(|i,i>)
        phi_plus = np.zeros((dim, 1), dtype=np.complex128)
        for i in range(self.d):
            # Index in bipartite basis corresponding to |i, i> -> i*d + i
            phi_plus[i * self.d + i] = 1.0 / np.sqrt(self.d)

        # 2. Build the projector onto the ideal state: P_ideal = |Phi+><Phi+|
        P_ideal = phi_plus @ phi_plus.conj().T

        # 3. Cost matrix C = I - P_ideal (Phase error operator)
        # Minimizing Tr(C @ rho) is equivalent to maximizing Fidelity Tr(P_ideal @ rho)
        C = np.eye(dim, dtype=np.complex128) - P_ideal

        return C

    def gen_W_QBER_Z(self) -> npt.NDArray[np.complex128]:
        """
        Generate W_QBER_Z operator.
        Total dimension of the matrix: (d^2) x (d^2)

        Returns:
            npt.NDArray[np.complex128]: Hermitian matrix W_QBER_Z of size (d^2, d^2).
        """
        # This operator projects onto the subspace spanned by the states |i,i> for i in {0, 1, ..., d-1}.
        proj_correct = np.zeros((self.d**2, self.d**2), dtype=complex)
        for i in range(self.d):
            # State |i>
            ket_i = np.zeros((self.d, 1), dtype=complex)
            ket_i[i] = 1.0
            proj_i = ket_i @ ket_i.T.conj() # |i><i|
            # Tensor product of proj_i with itself to get the bipartite projector
            proj_correct += np.kron(proj_i, proj_i)
        
        # W_QBER_Z = I_total - proj_correct (for each i != j)
        W_qber_z = np.eye(self.d**2, dtype=complex) - proj_correct
        return W_qber_z

    def gen_W_visibility_X(self,
                        Xa_d: npt.NDArray[np.complex128] | None = None) -> npt.NDArray[np.complex128]:
        """
        Generate the Visibility operator along the X basis for two qudits.
        
        Args:
            Xa_d: Optional; the shift matrix. If None, it will be generated.
            
        Returns:
            npt.NDArray[np.complex128]: Hermitian matrix W_VisibilityX of size (d^2, d^2).    
        """

        X_d = Xa_d
        if(Xa_d is None):
            X_d = self.genShiftMatrix()

        W_vis_x = np.zeros((self.d**2, self.d**2), dtype=complex)
        
        for k in range(0, self.d):
            X_k = np.linalg.matrix_power(X_d, k)
            W_vis_x += np.kron(X_k, X_k)
        
        return W_vis_x / self.d
