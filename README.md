# High-Dimensional QKD Setup Analysis: Variable-Length Devetak-Winter Dual Problem

## Overview
This repository contains the ongoing work for a Master's Thesis project in Computer Science, specializing in Cybersecurity. The research framework is fundamentally based on the study detailed in "A Practical Framework for Analyzing High-Dimensional QKD Setups". 

The current scope of this repository is the computational analysis and code generation for Quantum Key Distribution (QKD) protocols, specifically addressing high-dimensional quantum systems.

## Research Objectives
The primary objective of this phase of the project is the generation of computational code related to the dual problem for the Devetak-Winter formula, specifically applied to variable-length contexts. 

Key milestones include:
* Mathematical modeling and automated code generation for the dual problem.
* Implementation of the variable-length Devetak-Winter key rate calculation.
* Validation and performance analysis of the numerical results.

## Future Developments
Pending satisfactory analytical results from the initial implementation, the computational framework will be evaluated for hardware acceleration. If the architectural and mathematical conditions permit, the algorithmic structures will be refactored to execute natively on a GPU environment, aiming to optimize the efficiency of high-dimensional matrix operations.

## How setup python
```bash
git clone https://github.com/RiccardoPeruffo96/quantumcryptlib
cd quantumcryptlib
pip install -r requirements.txt
```

## How to setup config.json
TODO

## Run the script
* Windows (x64): `.\start.ps1`

## Python version used during test and development
* Windows (x64): Python 3.12.10

## Author
**Riccardo Peruffo**
riccardo.peruffo@studenti.univr.it