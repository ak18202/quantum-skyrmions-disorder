# Quantum Skyrmions under Random Bond Disorder

This repositery provides a qualitative framework to model quantum skyrmions under random bond disorder based on [arXiv:2510.16929v1].
The implementation uses a neural network quantum state approach to model a spin-half quantum Heisenberg hamiltonian on square lattice with Dzyaloshinskii-Moriya Interaction, Heisenberg anisotropy and boundary pinned mangetic field, built using the [NetKet](https://www.netket.org) framework.

## Files
├── src/ # Source code
│ ├── main.py # Main execution script
│ ├── hamiltonian.py # Constructs the system Hamiltonian
│ ├── model.py # Defines the neural network models for modulus and phase
│ ├── training.py # Implements the training loop with separate learning rate schedules
│ ├── observables.py # Computes physical observables
│ └── skyrmion_texture.wl # Mathematica script to visualize skyrmion spin textures
│
├── data/ # Output data (CSV files for spin textures, entropies, etc.)
├── plots/ # Output plots (PDF: energy, skyrmion number, skyrmion stability, entropy)
│
├── requirements.txt # Dependencies
├── LICENSE
├── .gitignore
└── README.md

## Usage
python src/main.py
Run a short test (default: 10 iterations):
Adjust parameters system size, disorder strength, number of iterations, learning rates etc. according to need

## Output
Results are saved into:

data/ — CSV files with:
- spin textures
- Rényi entropies

plots/ — PDF figures showing plots for:
- energy convergence
- skyrmion number density
- skyrmion stability density
- Rényi entropies

## References & Acknowledgement
This code is impossible without the following referances:

[1] Carleo, G. et al., NetKet: A machine learning toolkit for many-body quantum systems, SoftwareX 10, 100311 (2019). DOI: 10.1016/j.softx.2019.100311

[2] Vicentini, F. et al., NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems, SciPost Phys. Codebases (2022). DOI: 10.21468/SciPostPhysCodeb.7

[3] Joshi, A., Peters, R., Posske, T., Ground state properties of quantum skyrmions described by neural network quantum states, Phys. Rev. B 108, 094410 (2023). DOI: 10.1103/PhysRevB.108.094410

[4] Joshi, A., Peters, R., Posske, T., Quantum skyrmion dynamics studied by neural network quantum states, arXiv:2403.08184 (2024).

## Contact
If you have questions, feedback, or encounter any issues, please feel free to reach out or open a GitHub issue.