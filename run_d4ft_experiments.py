#!/usr/bin/env python3
"""
Runner for the three D4FT paper benchmark experiments (Li et al., ICLR 2023):
  - Experiment 1: Ground-state energy comparison on H2, CH4, H2O, O2, C2H6O, C6H6
  - Experiment 2: Spin-polarized LSDA magnetism prediction on O2 (triplet) vs CO2 (singlet)
  - Experiment 3: Fullerene scalability (C20, C60, C80) with mini-batch grid quadrature
"""

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import jax
jax.config.update("jax_enable_x64", True)

from d4ft_meta.molecules import get_pyscf_baselines
from d4ft_meta.hamiltonian import build_energy_fn
from d4ft_meta.solvers import solve_dft_core
from d4ft_meta.config import MOLECULES


def run_spin_magnetism_experiment(basis: str = "6-31g"):
    print("\n=== Experiment 2: Magnetism Prediction (O2 vs CO2) ===")
    for mol_name in ("O2", "CO2"):
        energies = {}
        for spin in (0, 2):
            orig_spin = MOLECULES[mol_name]["spin"]
            MOLECULES[mol_name]["spin"] = spin
            try:
                mol, S, T, V, ERI, E_nuc, W, phi, phi_g, lda_e, _ = get_pyscf_baselines(
                    mol_name, basis=basis, grid_level=3, compute_ccsd_t=False
                )
                engine = build_energy_fn((mol, S, T, V, ERI, E_nuc, W, phi, phi_g), neural_model=None)
                _, _, e_d4ft, _ = solve_dft_core(engine, None, use_nn=False, max_steps=400)
                energies[spin] = (lda_e, float(e_d4ft))
            finally:
                MOLECULES[mol_name]["spin"] = orig_spin

        de_pyscf = energies[2][0] - energies[0][0]
        de_d4ft = energies[2][1] - energies[0][1]
        state = "Paramagnetic (Triplet Ground State)" if de_d4ft < 0 else "Diamagnetic (Singlet Ground State)"
        print(
            f"  {mol_name:<4} | D4FT E(S=0)={energies[0][1]:.5f} Ha, E(S=1)={energies[2][1]:.5f} Ha "
            f"| dE={de_d4ft:+.5f} Ha (PySCF dE={de_pyscf:+.5f} Ha) -> {state}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run D4FT Benchmarks")
    parser.add_argument("--basis", type=str, default="6-31g")
    args = parser.parse_args()
    run_spin_magnetism_experiment(basis=args.basis)


if __name__ == "__main__":
    main()
