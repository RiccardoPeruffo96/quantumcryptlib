# src/main.py

import argparse
import numpy as np
import scipy as sp

import sdp.dual
import sdp.primal
import sdp.matrix

def main():
    parser = argparse.ArgumentParser(description="Program for developing Kanitschar Huber framework for quantum cryptography")
    parser.add_argument("--d", type=int, default=4, help="Qudit dimension (d >= 2)")

    if parser.parse_args().d < 2:
        parser.error("Qudit dimension d must be at least 2.")

    # d equals qudit dimensionality
    d = parser.parse_args().d

    # Base matrix (d x d)
    Xa_d = sdp.matrix.genShiftMatrix(d)

    # Phase matrix (d x d)
    Zb_d = sdp.matrix.genPhaseMatrix(d)

    # Generate Weyl Heisenberg local Operators
    U_local = sdp.matrix.genWeylHeisenbergOperators(d, Xa_d, Zb_d)

    # Generate bipartite operators
    U_bipartite = sdp.matrix.genBipartiteWeylHeisenbergOperators(d, U_local, U_local)

    print("Xa_d (zeros):\n", Xa_d)
    print("\nZb_d (identity):\n", Zb_d)
    print("\nU_local (Weyl Heisenberg Operators):\n", U_local)
    print("\nU_bipartite (Bipartite Weyl Heisenberg Operators):\n", U_bipartite)

if __name__ == "__main__":
    main()