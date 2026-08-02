# src/sdp/primal.py

import numpy as np
import numpy.typing as npt

def primal_problem(d) -> npt.NDArray[np.float64]:
    """
    Risolve il problema primale
    """

    return np.eye(d)