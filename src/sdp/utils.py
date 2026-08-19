# src/sdp/utils.py

import json

import numpy as np
import numpy.typing as npt

from safetensors.numpy import save_file, load_file

def clean_matrix(arr: npt.NDArray[np.complex128], eps: float = 1e-10) -> npt.NDArray[np.complex128]:
    """
    Resets real and imaginary components below the eps tolerance.
    Works safely with read-only arrays (e.g., CVXPY result matrices).
    
    Args:
        arr: The target complex matrix to clean
        eps: Tolerance value
            
    Returns:
        npt.NDArray[np.complex128]: A new cleaned complex matrix
    """
    if arr is None:
        return None
    
    # Extract real and imaginary components and zero out values below tolerance
    real_part = np.where(np.abs(arr.real) < eps, 0.0, arr.real)
    imag_part = np.where(np.abs(arr.imag) < eps, 0.0, arr.imag)
    
    # Construct a brand new writable complex array
    return real_part + 1j * imag_part

def format_complex_matrix(arr: npt.NDArray[np.complex128], precision: int = 2, eps: float = 1e-8) -> str:
    """
    Returns a readable string of the complex array.
    
    Args:
        arr: The target complex matrix to clean
        precision: Decimals number
        eps: Tolerance value
                
    Returns:
        str: Formatted string representation of the cleaned matrix
    """
    if arr is None:
        return "None"
    cleaned = clean_matrix(arr, eps)
    
    # Set NumPy formatting to suppress unnecessary scientific notation
    with np.printoptions(precision=precision, suppress=True, linewidth=120):
        return str(cleaned)
class NumpyComplexEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy arrays, scalar types, and complex numbers.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (complex, np.complex128)):
            # If imaginary part is virtually zero, return a float for readability
            if abs(obj.imag) < 1e-10:
                return round(float(obj.real), 6)
            return {
                "real": round(float(obj.real), 6),
                "imag": round(float(obj.imag), 6)
            }
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

def export_results_json(
    d: int,
    qber_z: float,
    visibility_x: float,
    total_coincidences: int,
    Xa_d: npt.NDArray[np.complex128],
    Zb_d: npt.NDArray[np.complex128],
    C: npt.NDArray[np.complex128],
    rho: npt.NDArray[np.complex128] | None,
    y: float | None,
    precision: float = 1e-8,
    output_filename: str = "out/report.json"
) -> None:
    """
    Exports execution parameters, operators, and SDP results to a structured JSON file.
    """
    Xa_clean = clean_matrix(Xa_d, eps=precision)
    Zb_clean = clean_matrix(Zb_d, eps=precision)
    C_clean = clean_matrix(C, eps=precision) if C is not None else None
    rho_clean = clean_matrix(rho, eps=precision) if rho is not None else None

    report_data = {
        "framework": "KANITSCHAR-HUBER FRAMEWORK",
        "system_parameters": {
            "qudit_dimension": d,
            "qber_z": qber_z,
            "visibility_x": visibility_x,
            "total_coincidences": total_coincidences
        },
        "operators": {
            "Xa_d": Xa_clean,
            "Zb_d": Zb_clean
        },
        "cost_matrix_C": {
            "shape": list(C.shape) if C is not None else None,
            "diagonal": np.diag(C_clean) if C_clean is not None else None
        },
        "primal_sdp_solution": {
            "solver_status": "OPTIMAL" if rho is not None else "INFEASIBLE / FAILED",
            "trace": float(np.trace(rho_clean).real) if rho_clean is not None else None,
            "density_matrix_rho": rho_clean
        },
        "dual_sdp_solution": {
            "solver_status": "OPTIMAL" if y is not None else "INFEASIBLE / FAILED",
            "density_matrix_rho": y
        }
    }

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, cls=NumpyComplexEncoder)

    print(f"\n[+] Results successfully exported to '{output_filename}'")

def export_results_txt(
    d: int,
    qber_z: float,
    visibility_x: float,
    total_coincidences: int,
    Xa_d: npt.NDArray[np.complex128],
    Zb_d: npt.NDArray[np.complex128],
    C: npt.NDArray[np.complex128],
    rho: npt.NDArray[np.complex128] | None,
    y: float | None,
    decimals: int = 2,
    precision: float = 1e-8,
    output_filename: str = "out/report.txt"
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
        y: Lagrange variable lamba resulting from the dual SDP.
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
    report_lines.append(format_complex_matrix(Xa_d, precision=decimals, eps=precision))
    
    report_lines.append("\nPhase Operator Zb_d (d x d):")
    report_lines.append(format_complex_matrix(Zb_d, precision=decimals, eps=precision))

    # 3. Cost Matrix Summary
    report_lines.append("\n[3] PHASE ERROR COST MATRIX C (d^2 x d^2)")
    report_lines.append(section_border)
    if C is not None:
        report_lines.append(f"  * Matrix Shape: {C.shape}")
        report_lines.append("  * Diagonal Preview:")
        c_diag = np.diag(clean_matrix(C, eps=precision))
        report_lines.append(f"    {np.array2string(c_diag, precision=decimals, suppress_small=True)}")

    # 4. SDP Primal Optimization Outcome
    report_lines.append("\n[4] PRIMAL SDP SOLUTION (Density Matrix rho)")
    report_lines.append(section_border)
    if rho is None:
        report_lines.append("  * SOLVER STATUS: INFEASIBLE / FAILED")
        report_lines.append("  * Result: None")
        report_lines.append("  * Diagnostic: The provided observation constraints are mutually")
        report_lines.append("                incompatible with a positive semidefinite state (rho >= 0).")
    else:
        rho_cleaned = clean_matrix(rho, eps=precision)
        trace_val = np.trace(rho_cleaned).real
        report_lines.append("  * SOLVER STATUS: OPTIMAL")
        report_lines.append(f"  * Matrix Shape: {rho.shape}")
        report_lines.append(f"  * State Trace Tr(rho): {trace_val:.6f}")
        report_lines.append("  * Reconstructed Density Matrix rho:")
        report_lines.append(format_complex_matrix(rho, precision=decimals, eps=precision))

    # 5. SDP Dual Optimization Outcome
    report_lines.append("\n[5] DUAL SDP SOLUTION (Lagrange Operator lamba)")
    report_lines.append(section_border)
    if y is None:
        report_lines.append("  * SOLVER STATUS: INFEASIBLE / FAILED")
        report_lines.append("  * Result: None")
        report_lines.append("  * Diagnostic: The provided observation constraints are mutually")
        report_lines.append("                incompatible with a positive semidefinite state (y >= 0).")
    else:
        report_lines.append("  * SOLVER STATUS: OPTIMAL")
        report_lines.append(f"  * Result: {y:.6f}")
    
    report_lines.append("\n" + header_border)
    final_report = "\n".join(report_lines)

    # Output to stdout and export file
    print(final_report)
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_report)

#TODO: fix
def save_quantum_state(matrix: np.ndarray, filename: str):
    """
    Save the target matrix in a satensors format.
    
    Args:
        matrix: matrix to store(np.ndarray complex).
        filename: Il percorso del file di destinazione.
    
    Note:
        Safetensors doesn't support np.complex128 type, so the matrix require
        a view trasnformation.
    """
    data = {"matrix": matrix.view(np.float64)} 
    save_file(data, filename)

#TODO: fix
def load_quantum_state(filename: str, shape: tuple) -> np.ndarray:
    """
    Read target matrix from safetensors.

    Args:
        filename: Path to the source file.
        shape: Target shape of the matrix to reconstruct.

    Returns:
        np.ndarray: The reconstructed complex matrix.
    """
    tensors = load_file(filename)
    return tensors["rho"].view(np.complex128).reshape(shape)