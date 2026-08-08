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

def export_results(
    d: int,
    qber_z: float,
    visibility_x: float,
    total_coincidences: int,
    Xa_d: npt.NDArray[np.complex128],
    Zb_d: npt.NDArray[np.complex128],
    C: npt.NDArray[np.complex128],
    rho: npt.NDArray[np.complex128] | None,
    decimals: int = 2,
    precision: float = 1e-8,
    output_filename: str = "result.txt"
) -> None:
    """
    Generates a clean, human-readable execution report and saves it to a file.

    Args:
        d: Qudit space dimension.
        qber_z: Quantum Bit Error Rate along the Z basis.
        visibility_x: Channel visibility along the X basis.
        total_coincidences: Total photon coincidence count.
        Xa_d: Shift operator matrix (d x d).
        Zb_d: Phase operator matrix (d x d).
        C: Phase error cost matrix (d^2 x d^2).
        rho: Optimized density matrix resulting from the primal SDP.
        decimals: Number of decimal places to format floating point values.
        precision: Absolute threshold below which matrix entries are zeroed out.
        output_filename: Path to the target text file for report export.
    """
    report_lines = []
    header_border = "=" * 70
    section_border = "-" * 70

    report_lines.append(header_border)
    report_lines.append("        KANITSCHAR-HUBER FRAMEWORK - SDP EXECUTION REPORT")
    report_lines.append(header_border)

    # 1. System Parameters Summary
    report_lines.append("\n[1] EXPERIMENTAL & SYSTEM PARAMETERS")
    report_lines.append(section_border)
    report_lines.append(f"  * Qudit Dimension (d):      {d}")
    report_lines.append(f"  * QBER (Z Basis):           {qber_z:.4f}")
    report_lines.append(f"  * Visibility (X Basis):     {visibility_x:.4f}")
    report_lines.append(f"  * Total Coincidences (tc):  {total_coincidences}")

    # 2. Local Quantum Operators
    report_lines.append("\n[2] LOCAL OPERATORS")
    report_lines.append(section_border)
    report_lines.append("Shift Operator Xa_d (d x d):")
    report_lines.append(sdp.utils.format_complex_matrix(Xa_d, precision=decimals, eps=precision))
    
    report_lines.append("\nPhase Operator Zb_d (d x d):")
    report_lines.append(sdp.utils.format_complex_matrix(Zb_d, precision=decimals, eps=precision))

    # 3. Cost Matrix Summary
    report_lines.append("\n[3] PHASE ERROR COST MATRIX C (d^2 x d^2)")
    report_lines.append(section_border)
    if C is not None:
        report_lines.append(f"  * Matrix Shape: {C.shape}")
        report_lines.append("  * Diagonal Preview:")
        c_diag = np.diag(sdp.utils.clean_matrix(C, eps=precision))
        report_lines.append(f"    {np.array2string(c_diag, precision=decimals, suppress_small=True)}")

    # 4. SDP Optimization Outcome
    report_lines.append("\n[4] PRIMAL SDP SOLUTION (Density Matrix rho)")
    report_lines.append(section_border)
    if rho is None:
        report_lines.append("  * SOLVER STATUS: INFEASIBLE / FAILED")
        report_lines.append("  * Result: None")
        report_lines.append("  * Diagnostic: The provided observation constraints are mutually")
        report_lines.append("                incompatible with a positive semidefinite state (rho >= 0).")
    else:
        rho_cleaned = sdp.utils.clean_matrix(rho, eps=precision)
        trace_val = np.trace(rho_cleaned).real
        report_lines.append("  * SOLVER STATUS: OPTIMAL")
        report_lines.append(f"  * Matrix Shape: {rho.shape}")
        report_lines.append(f"  * State Trace Tr(rho): {trace_val:.6f}")
        report_lines.append("  * Reconstructed Density Matrix rho:")
        report_lines.append(sdp.utils.format_complex_matrix(rho, precision=decimals, eps=precision))

    report_lines.append("\n" + header_border)
    final_report = "\n".join(report_lines)

    # Output to stdout and export file
    print(final_report)

    # NOTE: redirect the final_report with pipe operator in start.ps1 XOR try open(), not BOTH
    
    #with open(output_filename, "w", encoding="utf-8") as f:
    #    f.write(final_report)

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
    output_filename: str = "result.txt"

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
    rho = sdp.primal.solve_primal_sdp(d, observations, C)

    #print("Xa_d:\n", Xa_d)
    #print("\nZb_d:\n", Zb_d)
    #print("\nU_local (Weyl Heisenberg Operators):\n", U_local)
    #print("\nU_bipartite (Bipartite Weyl Heisenberg Operators):\n", U_bipartite)
    #print("\nC (Phase Error Cost Matrix):\n", C)
    #print("\nrho (primal sdp):\n", rho)

    export_results(
        d=d,
        qber_z=QBER_Z,
        visibility_x=visibility_X,
        total_coincidences=total_coincidences,
        Xa_d=Xa_d,
        Zb_d=Zb_d,
        C=C,
        rho=rho,
        decimals=decimals,
        precision=precision,
        output_filename=output_filename
    )

if __name__ == "__main__":
    main()