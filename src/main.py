# src/main.py

import argparse
import numpy as np
import scipy as sp

import sdp.dual
import sdp.primal
import sdp.matrix

def main():
    parser = argparse.ArgumentParser(description="Esegue il circuito CNOT con ancilla usando un qubit di controllo generico.")
    parser.add_argument("--d", type=int, default=4, help="Qudit dimensionality")

    # d equals qudit dimensionality
    d = parser.parse_args().d

    # Base matrix (d x d)
    Xa_d = sdp.matrix.genShiftMatrix(d)

    # Phase matrix (d x d)
    Zb_d = sdp.matrix.genPhaseMatrix(d)

    print("Xa_d (zeros):\n", Xa_d)
    print("\nZb_d (identity):\n", Zb_d)

if __name__ == "__main__":
    main()