# src/main.py

import argparse
import cvxpy as cp
import json
import numpy as np
import numpy.typing as npt
import os
import scipy as sp

import sdp.dual
import sdp.primal
import sdp.matrix
import sdp.utils

def main():
    parser = argparse.ArgumentParser(description="Program for developing Kanitschar Huber framework for quantum cryptography")
    parser.add_argument("--d", type=int, default=4, help="Qudit dimension (d >= 2)")
    parser.add_argument("--qberZ", type=float, default=0.05, help="QBER (Quantum Bit Error Rate) on Z (0 <= qberZ <= 1)")
    parser.add_argument("--tc", type=int, default=100000, help="Total coincidences (number of photons exchanged in a small slice of time) (tc >= 1)")
    parser.add_argument("--vX", type=float, default=0.88, help="Total visibility (how much noise or interception are in the channel, best near 1.0) (0 <= visibilityX <= 1)")

    if parser.parse_args().d < 2:
        parser.error("Qudit dimension d must be at least 2.")
    if parser.parse_args().qberZ < 0.0 or parser.parse_args().qberZ > 1.0:
        parser.error("Quantum Bit Error Rate must be between 0.0 and 1.0.")
    if parser.parse_args().tc <= 0:
        parser.error("Total coincidences must be a natural number not zero.")
    if parser.parse_args().vX < 0.0 or parser.parse_args().vX > 1.0:
        parser.error("Total visibility must be between 0.0 and 1.0.")

    d: int = parser.parse_args().d
    QBER_Z: float = parser.parse_args().qberZ
    total_coincidences: int = parser.parse_args().tc
    visibility_X: float = parser.parse_args().vX

    # Define c_k about constrains observation
    #c_QBER_Z: float = 1.0 - QBER_Z # ERROR
    c_QBER_Z: float = QBER_Z
    c_visibility_X: float = visibility_X

    # Default values about print view
    decimals: int = 2
    precision: float = 1e-8
    output_filename: str = "out/report.txt"

    delta: float = 1e-10  # statistical confidence level for the Hoeffding's inequality bound

    if os.path.exists("config.json"):
        with open("config.json", "r") as file:
            config = json.load(file)
            # Import SYSTEM_PARAMS
            if config.get("OVERWRITE_PARAMS", True).get("d", True):
                d = config.get("SYSTEM_PARAMS", d).get("d", d)
            if config.get("OVERWRITE_PARAMS", True).get("total_coincidences", True):
                total_coincidences = config.get("SYSTEM_PARAMS", total_coincidences).get("total_coincidences", total_coincidences)
            # Import SDP_SETTINGS
            decimals = config.get("SDP_SETTINGS", decimals).get("decimals", decimals)
            delta = config.get("SDP_SETTINGS", delta).get("delta", delta)
            output_filename = config.get("SDP_SETTINGS", output_filename).get("output_filename", output_filename)
            precision = config.get("SDP_SETTINGS", precision).get("precision", precision)
            # TODO: Add read parameters QBER_Z and visibility_X from JSON
            ## DRAFT CODE
            if False:
                for ES in config.get("EXPERIMENTAL_STATISTICS", []):
                    if config.get("OVERWRITE_PARAMS", True).get(ES.type, True):
                        if ES.type == "qber_z":
                            QBER_Z = ES.value
                            c_QBER_Z = 1.0 - QBER_Z
                        if ES.type == "visibility_X":
                            visibility_X = ES.value
                            c_visibility_X = ES.value

    # Base matrix (d x d)
    Xa_d = sdp.matrix.genShiftMatrix(d)

    # Phase matrix (d x d)
    Zb_d = sdp.matrix.genPhaseMatrix(d)

    # Generate Weyl Heisenberg local Operators (d^2 * d^2)
    U_local = sdp.matrix.genWeylHeisenbergOperators(d, Xa_d, Zb_d)

    # Generate bipartite operators (d^4 * d^4)
    U_bipartite = sdp.matrix.genBipartiteWeylHeisenbergOperators(d, U_local, U_local)

    # TODO: check the value
    # Generate the phase error cost matrix
    C = sdp.matrix.gen_phase_error_cost_matrix(d)

    # Generate observation constraints: list of (W_k matrix, c_k value)
    observations: list[tuple[npt.NDArray[np.complex128], float]] = []

    # Add first observation W_k to observations
    W_QBER_Z = sdp.matrix.gen_W_QBER_Z(d)
    observations.append((W_QBER_Z, c_QBER_Z))
    
    # Add second observation W_k to observations
    W_visibility_X = sdp.matrix.gen_W_visibility_X(d, Xa_d)
    observations.append((W_visibility_X, c_visibility_X))

    # Calcolate primal SDP
    rho_primal = sdp.primal.solve_primal_sdp(d, observations, C, delta, total_coincidences)

    # TODO: test it - Actual return infinite
    # Calcolate dual SDP
    y_dual = sdp.dual.solve_dual_sdp(observations, C, delta, total_coincidences)

    if output_filename.endswith('.txt'):
        sdp.utils.export_results_txt(
            d=d,
            qber_z=QBER_Z,
            visibility_x=visibility_X,
            total_coincidences=total_coincidences,
            Xa_d=Xa_d,
            Zb_d=Zb_d,
            C=C,
            rho=rho_primal,
            y=y_dual,
            decimals=decimals,
            precision=precision,
            output_filename=output_filename
        )
    else:
        sdp.utils.export_results_json(
            d=d,
            qber_z=QBER_Z,
            visibility_x=visibility_X,
            total_coincidences=total_coincidences,
            Xa_d=Xa_d,
            Zb_d=Zb_d,
            C=C,
            rho=rho_primal,
            y=y_dual,
            decimals=decimals,
            precision=precision,
            output_filename=output_filename
        )

    #sdp.utils.save_quantum_state(Xa_d, "data\\matrix\\Xa_d.safetensors")
    #sdp.utils.save_quantum_state(Zb_d, "data\\matrix\\Zb_d.safetensors")
    #sdp.utils.save_quantum_state(U_local, "data\\matrix\\U_local.safetensors")
    #sdp.utils.save_quantum_state(U_bipartite, "data\\matrix\\U_bipartite.safetensors")
    #sdp.utils.save_quantum_state(C, "data\\matrix\\C.safetensors")

if __name__ == "__main__":
    main()