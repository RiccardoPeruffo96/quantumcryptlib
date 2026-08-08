# src/sdp/utils.py

import numpy as np
import numpy.typing as npt

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
