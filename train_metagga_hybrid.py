#!/usr/bin/env python3
"""
Main training & extrapolation benchmark script for the High-Precision (FP64)
Neural Meta-GGA Global Hybrid Kohn-Sham DFT solver.

Trains on: H2, He, LiH, N, H2O, CH4 (def2-SVP, CCSD(T) targets)
Evaluates unseen extrapolation on: CO, F2, HF
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import jax
jax.config.update("jax_enable_x64", True)

from d4ft_meta.config import D4FTConfig, HA_TO_KCAL_MOL
from d4ft_meta.molecules import get_pyscf_baselines
from d4ft_meta.solvers import MetaGGAHybridTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Neural Meta-GGA Hybrid KS-DFT Functional")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Gaussian AO basis set")
    parser.add_argument("--epochs", type=int, default=101, help="Number of outer bi-level epochs")
    parser.add_argument("--inner-steps", type=int, default=600, help="Number of inner orbital steps")
    parser.add_argument("--lr", type=float, default=3e-3, help="Outer AdamW learning rate")
    args = parser.parse_args()

    config = D4FTConfig(
        basis=args.basis,
        outer_epochs=args.epochs,
        inner_max_steps=args.inner_steps,
        outer_lr=args.lr,
    )

    print(f"JAX Backend: {jax.default_backend().upper()} (FP64 Enabled)")
    print(f"Generating Exact Integrals & CCSD(T)/{config.basis} Baselines...")

    pyscf_data, pyscf_lda_results, ccsd_targets = {}, {}, {}
    for m in config.eval_molecules:
        mol, S, T, V, ERI, E_nuc, W, phi, phi_g, lda_e, ccsd_e = get_pyscf_baselines(
            m, basis=config.basis, grid_level=config.grid_level, compute_ccsd_t=True
        )
        pyscf_data[m] = (mol, S, T, V, ERI, E_nuc, W, phi, phi_g)
        pyscf_lda_results[m] = lda_e
        ccsd_targets[m] = ccsd_e

    trainer = MetaGGAHybridTrainer(config, pyscf_data, ccsd_targets)

    print("\n--- Phase 1: Validating Pure LSDA Engine (Dirac + Chachiyo) ---")
    lsda_results = trainer.evaluate_all(use_nn=False)

    print("\n--- Phase 2: Meta-GGA Hybrid Bi-Level Training ---")
    trainer.train(verbose=True)

    print("\n--- Phase 3: Evaluating Extrapolation on Seen & Unseen Molecules ---")
    hybrid_results = trainer.evaluate_all(use_nn=True)

    print("\n" + "=" * 105)
    print(
        f"{'Molecule':<9} | {'CCSD(T) Target (Ha)':<20} | {'PySCF LDA Err':<18} | "
        f"{'D4FT LSDA Err':<18} | {'Meta-GGA Hybrid Err':<18}"
    )
    print("=" * 105)

    for m in config.eval_molecules:
        target = ccsd_targets[m]
        pyscf_err = abs(pyscf_lda_results[m] - target) * HA_TO_KCAL_MOL
        lsda_err = abs(lsda_results[m] - target) * HA_TO_KCAL_MOL
        hyb_err = abs(hybrid_results[m] - target) * HA_TO_KCAL_MOL
        flag = "*" if m not in config.train_molecules else " "
        print(
            f"{m:<8}{flag} | {target:>17.6f} Ha | {pyscf_err:>11.2f} kcal/m | "
            f"{lsda_err:>11.2f} kcal/m | {hyb_err:>11.2f} kcal/m"
        )

    print("=" * 105)
    print("* Indicates unseen molecule during training (zero-shot chemical extrapolation).")


if __name__ == "__main__":
    main()
