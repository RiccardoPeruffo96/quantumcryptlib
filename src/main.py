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
    parser.add_argument("--qber", type=float, default=0.05, help="QBER (Quantum Bit Error Rate) (0 <= qber <= 1)")
    parser.add_argument("--tc", type=int, default=100000, help="Total coincidences (number of photons exchanged in a small slice of time) (tc >= 1)")
    parser.add_argument("--vX", type=float, default=0.88, help="Total visibility (how much noise or interception are in the channel, best near 1.0) (0 <= visibilityX <= 1)")

    # TODO: test parsing parameters
    if parser.parse_args().d < 2:
        parser.error("Qudit dimension d must be at least 2.")
    if parser.parse_args().qberZ < 0.0 or parser.parse_args().qberZ > 1.0:
        parser.error("Quantum Bit Error Rate must be between 0.0 and 1.0.")
    if parser.parse_args().tc <= 0:
        parser.error("Total coincidences must be a natural number not zero.")
    if parser.parse_args().visibilityX < 0.0 or parser.parse_args().visibilityX > 1.0:
        parser.error("Total visibility must be between 0.0 and 1.0.")

    d = parser.parse_args().d
    QBER_Z = parser.parse_args().qber
    c_QBER_Z = 1.0 - QBER_Z
    total_coincidences = parser.parse_args().tc
    visibility_X = parser.parse_args().vX
    c_visibility_X = visibility_X

    # TODO: Test reading config.json
    if os.path.exists("config.json"):
        with open("config.json", "r") as file:
            config = json.load(file)
            if config.get("OVERWRITE_PARAMS", True).get("d", True):
                d = config.get("SYSTEM_PARAMS", d).get("d", d)
            if config.get("OVERWRITE_PARAMS", True).get("total_coincidences", True):
                total_coincidences = config.get("SYSTEM_PARAMS", total_coincidences).get("total_coincidences", total_coincidences)
            # TODO: Add read parameters QBER_Z and visibility_X from JSON
            ## DRAFT CODE
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

    print("Xa_d:\n", Xa_d)
    print("\nZb_d:\n", Zb_d)
    print("\nU_local (Weyl Heisenberg Operators):\n", U_local)
    print("\nU_bipartite (Bipartite Weyl Heisenberg Operators):\n", U_bipartite)
    print("\nC (Phase Error Cost Matrix):\n", C)

if __name__ == "__main__":
    main()