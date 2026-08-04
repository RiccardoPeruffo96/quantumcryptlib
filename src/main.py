# src/main.py

import argparse
import numpy as np
import scipy as sp
import json
import os

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
    
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
            if config.get("FORCE_OVERWRITE_d_PARAM", False):
                d = config.get("d", d)

    # Base matrix (d x d)
    Xa_d = sdp.matrix.genShiftMatrix(d)

    # Phase matrix (d x d)
    Zb_d = sdp.matrix.genPhaseMatrix(d)

    # Generate Weyl Heisenberg local Operators (d^2 * d^2)
    U_local = sdp.matrix.genWeylHeisenbergOperators(d, Xa_d, Zb_d)

    # Generate bipartite operators (d^4 * d^4)
    U_bipartite = sdp.matrix.genBipartiteWeylHeisenbergOperators(d, U_local, U_local)

    print("Xa_d:\n", Xa_d)
    print("\nZb_d:\n", Zb_d)
    print("\nU_local (Weyl Heisenberg Operators):\n", U_local)
    print("\nU_bipartite (Bipartite Weyl Heisenberg Operators):\n", U_bipartite)

if __name__ == "__main__":
    main()