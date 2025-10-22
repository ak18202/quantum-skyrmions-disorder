import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from hamiltonian import build_hamiltonian
from model import RealImagFFNN
from training import run_training
from observables import compute_spin_expectations, compute_skyrmion_number, compute_renyi_entropies, compute_topological_entropy

import netket as nk

def main():
    start_time = time.time()

    # Directories
    workdir = os.getcwd()
    plots_dir = os.path.join(workdir, "plots")
    data_dir = os.path.join(workdir, "data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    np.random.seed(42)
    # Parameters
    L = 5 # Length of the lattice 
    J = 1.0 # Heisenberg exchange constant
    A = 0.3 # Anisotropic constant
    D = 0.7 # Dzyaloshinskii-Moriya interaction constant
    Bz = 10.0 # External magnetic field
    delta = 0.0 # Disorder strength
    output_prefix = f"delta={delta},A={A},D={D},Bz={Bz}"

    # Build Hamiltonian
    ham, hi, g = build_hamiltonian(L=L, J=J, A=A, D=D, Bz=Bz, delta=delta)

    # Setup sampler & model
    sampler = nk.sampler.MetropolisLocal(hi)
    model = RealImagFFNN(L=L, alpha=2)

    # Train
    vstate, energy_iter, energy_means, energy_sem, energy_var = run_training(
        ham, sampler, model, n_samples=1000, n_iter=10, L=L, out_prefix="ground_state")

    # Plot energy convergence
    plt.errorbar(energy_iter, energy_means, yerr=energy_sem, fmt='-o', capsize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Energy per site")
    plt.title("Energy vs Iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{output_prefix}_energy.pdf"))
    print(f"Saved energy convergence plot to {output_prefix}_energy.pdf")
    plt.close() 

    # Extract x, y positions
    positions = np.array(g.positions) 
    x = positions[:, 0]
    y = positions[:, 1]

    # Spin expectations
    sx, sy, sz, sx_e, sy_e, sz_e = compute_spin_expectations(vstate, hi, g)
    # Saving file
    spin_csv = os.path.join(data_dir, f"{output_prefix}_spin_texture.csv")
    spin_out = np.vstack([x, y, sx, sy, sz, sx_e, sy_e, sz_e]).T
    np.savetxt(spin_csv, spin_out, delimiter=",", header="x,y,Sx,Sy,Sz,Sx_e,Sy_e,Sz_e", comments="")
    print(f"Saved spin-texture data to {spin_csv}")

    # Skyrmion number and stability
    C, Q, c_density, q_density, triangles = compute_skyrmion_number(sx, sy, sz, L, g)
    print(f"Skyrmion number: {C:.6f}, Skyrmion stability: {Q:.6f}")

    triang = mtri.Triangulation(x, y, triangles)

    plt.tripcolor(triang, c_density, shading='flat', cmap='plasma', )
    plt.colorbar(label="C")
    plt.gca().set_aspect('equal')
    plt.title("C on Triangular Plaquettes")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{output_prefix}_cden_grid.pdf"))
    print(f"Saved C plot to {output_prefix}_cden_grid.pdf")
    plt.close() 

    plt.tripcolor(triang, q_density, shading='flat', cmap='inferno')
    plt.colorbar(label="Q")
    plt.gca().set_aspect('equal')
    plt.title("Q on Triangular Plaquettes")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{output_prefix}_qden_grid.pdf"))
    print(f"Saved Q plot to {output_prefix}_qden_grid.pdf")
    plt.close() 

    # Renyi entropies
    entropies, renyi, renyi_err = compute_renyi_entropies(vstate, hi, g, L)

    plt.imshow(entropies, cmap='viridis', origin='lower')
    plt.colorbar(label="Rényi-2 Entropy")
    plt.title("Rényi-2 Entropy at Sites")
    plt.savefig(os.path.join(plots_dir, f"{output_prefix}_renyi.pdf"))
    print(f"Saved Renyi-2 entropies plot to {output_prefix}_renyi.pdf")
    plt.close()

    renyi = np.array(renyi)
    renyi_err = np.array(renyi_err)
    renyi_csv = os.path.join(data_dir, f"{output_prefix}_renyi_grid.csv")
    renyi_out = np.vstack([x, y, renyi, renyi_err]).T
    np.savetxt(renyi_csv, renyi_out, delimiter=",")
    print(f"Saved Renyi-2 entropy grid to {renyi_csv}")

    # Topological Entanglement Entropy
    Stopo, Stopo_err = compute_topological_entropy(vstate, hi, g, L)
    print(f"Topological Entanglement Entropy (TEE): {Stopo:.6f} ± {Stopo_err:.7f}")

    # Timing
    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"Execution time: {int(mins)} min {secs:.2f} sec")

if __name__ == "__main__":
    main()