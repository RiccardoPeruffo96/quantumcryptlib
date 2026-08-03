# src/sdp/matrix.py

import numpy as np
import numpy.typing as npt

def genShiftMatrix(d) -> npt.NDArray[np.float64]:
    """
    Generate shift matrix
    """

    X = [[0, 0, 0, 1],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0]]

    return np.array(X)

def genPhaseMatrix(d) -> npt.NDArray[np.complex128]:
    """
    Generate phase matrix
    """

    Z = [[1, 0, 0, 0],
         [0, 1j, 0, 0],
         [0, 0, -1, 0],
         [0, 0, 0, -1j]]

    return np.array(Z)