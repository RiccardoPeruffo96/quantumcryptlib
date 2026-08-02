# src/main.py

import argparse
import numpy as np
import scipy as sp

import sdp.dual
import sdp.primal

def main():
    parser = argparse.ArgumentParser(description="Esegue il circuito CNOT con ancilla usando un qubit di controllo generico.")
    parser.add_argument("--d", type=int, default=4, help="Qudit dimensionality")

    # d equals qudit dimensionality
    d = parser.parse_args().d

    # Base matrix (d x d)
    Xd = np.zeros((d, d)) # TODO: generate dinamically

    # Phase matrix (d x d)
    Zd = np.eye(d) # TODO: generate dinamically

    print("Xd (zeros):\n", Xd)
    print("\nZd (identity):\n", Zd)

if __name__ == "__main__":
    main()