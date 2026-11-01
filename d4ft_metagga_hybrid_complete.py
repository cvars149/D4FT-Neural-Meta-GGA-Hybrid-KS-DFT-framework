#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d4ft_metagga_hybrid_complete.py
==============================================================================
Complete, Bug-Fixed, High-Precision (FP64) Differentiable Kohn-Sham DFT Engine
with Neural Meta-GGA Enhancement + Learnable Exact Hartree-Fock Exchange.

Includes fixes for:
  1. Full LSDA baseline (Dirac Exchange + Chachiyo Correlation) when use_nn=False
  2. Regularized iso-orbital indicator alpha_iso in [0, 1] in exponential density tails
  3. Sign-stabilized QR Stiefel manifold projection (diag(R) >= 0)
  4. Safe handling of empty spin channels (e.g. N_beta = 0 for H atom)
==============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import value_and_grad, jit
import flax.linen as nn
import optax
from pyscf import gto, scf, dft, cc
from pyscf.dft import numint

print(f"JAX Backend: {jax.default_backend().upper()} (FP64 Enabled)")

# ==============================================================================
# 1. DATASET & EXACT BASELINE GENERATION
# ==============================================================================
BASIS = "def2-SVP"
HA_TO_KCAL_MOL = 627.5094740631

TRAIN_MOLS = ["H2", "He", "LiH", "N", "H2O", "CH4"]
MOLECULES = {
    "H2": {"coords": "H 0 0 0; H 0 0 0.74", "spin": 0},
    "He": {"coords": "He 0 0 0", "spin": 0},
    "LiH": {"coords": "Li 0 0 0; H 0 0 1.59", "spin": 0},
    "N": {"coords": "N 0 0 0", "spin": 3},
    "H2O": {"coords": "O 0 0 0.12; H 0 0.76 -0.48; H 0 -0.76 -0.48", "spin": 0},
    "CH4": {"coords": "C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H 0.63 -0.63 -0.63; H -0.63 0.63 -0.63", "spin": 0},
    "CO": {"coords": "C 0 0 0; O 0 0 1.13", "spin": 0},
    "F2": {"coords": "F 0 0 0; F 0 0 1.41", "spin": 0},
    "HF": {"coords": "F 0 0 0; H 0 0 0.92", "spin": 0},
}


def get_pyscf_baselines(name):
    mol = gto.Mole(atom=MOLECULES[name]["coords"], spin=MOLECULES[name]["spin"], basis=BASIS, verbose=0).build()

    mf_lda = dft.UKS(mol) if mol.spin > 0 else dft.RKS(mol)
    mf_lda.xc = "lda,vwn"
    mf_lda.kernel()

    mf_hf = scf.UHF(mol) if mol.spin > 0 else scf.RHF(mol)
    mf_hf.kernel()

    mycc = cc.UCCSD(mf_hf) if mol.spin > 0 else cc.CCSD(mf_hf)
    mycc.kernel()
    ccsd_e = float(mf_hf.e_tot + mycc.e_corr + mycc.ccsd_t())

    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    E_nuc = float(mol.energy_nuc())
    ERI = mol.intor("int2e")

    grids = dft.gen_grid.Grids(mol)
    grids.level = 4
    grids.build()
    ao_vals = numint.eval_ao(mol, grids.coords, deriv=1)

    return (
        mol,
        S,
        T,
        V,
        ERI,
        E_nuc,
        grids.weights,
        ao_vals[0],
        ao_vals[1:].transpose(1, 2, 0),
        float(mf_lda.e_tot),
        ccsd_e,
    )


# ==============================================================================
# 2. META-GGA HYBRID ARCHITECTURE (TAU + EXACT EXCHANGE)
# ==============================================================================
class NeuralMetaGGA(nn.Module):
    @nn.compact
    def __call__(self, rho_a, rho_b, grad_rho_a, grad_rho_b, tau_a, tau_b):
        rho_tot = jnp.clip(rho_a + rho_b, 1e-12, None)
        zeta = (jnp.clip(rho_a, 1e-12, None) - jnp.clip(rho_b, 1e-12, None)) / rho_tot

        gamma_a = jnp.sum(grad_rho_a ** 2, axis=-1)
        gamma_b = jnp.sum(grad_rho_b ** 2, axis=-1)

        tau_tot = jnp.clip(tau_a + tau_b, 1e-12, None)
        tau_w = 0.125 * jnp.sum((grad_rho_a + grad_rho_b) ** 2, axis=-1) / rho_tot
        # Fixed: Clip iso-orbital indicator alpha_iso to physical [0, 1] range in low-density tails
        alpha_iso = jnp.clip((tau_tot - tau_w) / tau_tot, 0.0, 1.0)

        x = jnp.stack([
            jnp.log1p(rho_tot),
            zeta,
            jnp.log1p(gamma_a),
            jnp.log1p(gamma_b),
            jnp.log1p(jnp.clip(tau_a, 0.0, None)),
            jnp.log1p(jnp.clip(tau_b, 0.0, None)),
            alpha_iso,
        ], axis=-1)

        for _ in range(3):
            x = nn.gelu(nn.Dense(128)(x))

        Fx_nn = nn.Dense(1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)[..., 0]
        Fc_nn = nn.Dense(1, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros)(x)[..., 0]

        alpha_ex_raw = self.param("alpha_ex_raw", jax.nn.initializers.constant(-0.847), ())
        alpha_ex = jax.nn.sigmoid(alpha_ex_raw) * 0.5

        return 1.0 + Fx_nn, 1.0 + Fc_nn, alpha_ex


def compute_exact_dirac_exchange(rho_a, rho_b):
    Cx = -0.7385587663820224
    ra, rb = jnp.clip(rho_a, 1e-12, None), jnp.clip(rho_b, 1e-12, None)
    return Cx * (ra * (2.0 * ra) ** (1.0 / 3.0) + rb * (2.0 * rb) ** (1.0 / 3.0)) / (ra + rb)


def compute_chachiyo_correlation(rho_a, rho_b):
    rho = jnp.clip(rho_a + rho_b, 1e-12, None)
    rs = (0.75 / (jnp.pi * rho)) ** (1.0 / 3.0)
    a, b = -0.01554535, 20.64336
    return a * jnp.log(1.0 + b / rs + b / (rs ** 2))


def stiefel_qr(W, D):
    if W.shape[0] == 0:
        return jnp.zeros((0, D.shape[0]), dtype=jnp.float64)
    Q, R = jnp.linalg.qr(W.T, mode="reduced")
    phases = jnp.where(jnp.diagonal(R) >= 0, 1.0, -1.0)
    Q = Q * phases[None, :]
    return Q.T @ D.T


def build_energy_fn(mol_name, pyscf_data, neural_model):
    mol, S, T, V, ERI, E_nuc, weights, phi, phi_grad = pyscf_data[mol_name]
    evals, evecs = np.linalg.eigh(S)
    D = jnp.array(evecs @ np.diag(np.clip(evals, 1e-12, None) ** -0.5), dtype=jnp.float64)

    T_j, V_j, ERI_j = map(lambda x: jnp.array(x, dtype=jnp.float64), (T, V, ERI))
    W_j, phi_j, phi_gj = map(lambda x: jnp.array(x, dtype=jnp.float64), (weights, phi, phi_grad))

    def energy_fn(W_a, W_b, nn_params, use_nn=True):
        Ca = stiefel_qr(W_a, D)
        Cb = stiefel_qr(W_b, D)

        Pa = Ca.T @ Ca if Ca.shape[0] > 0 else jnp.zeros_like(T_j)
        Pb = Cb.T @ Cb if Cb.shape[0] > 0 else jnp.zeros_like(T_j)
        P_tot = Pa + Pb

        E_core = jnp.sum(P_tot * (T_j + V_j))
        E_H = 0.5 * jnp.einsum("ij,kl,ijkl->", P_tot, P_tot, ERI_j)

        Ex_exact_a = -0.5 * jnp.einsum("il,jk,ijkl->", Pa, Pa, ERI_j)
        Ex_exact_b = -0.5 * jnp.einsum("il,jk,ijkl->", Pb, Pb, ERI_j)
        E_exact_exchange = Ex_exact_a + Ex_exact_b

        psi_a = phi_j @ Ca.T
        psi_b = phi_j @ Cb.T

        grad_psi_a = jnp.einsum("gbx,ib->xgi", phi_gj, Ca)
        grad_psi_b = jnp.einsum("gbx,ib->xgi", phi_gj, Cb)

        rho_a = jnp.sum(psi_a ** 2, axis=-1)
        rho_b = jnp.sum(psi_b ** 2, axis=-1)

        grad_rho_a = 2.0 * jnp.sum(psi_a[..., None] * jnp.einsum("xgi->gix", grad_psi_a), axis=1)
        grad_rho_b = 2.0 * jnp.sum(psi_b[..., None] * jnp.einsum("xgi->gix", grad_psi_b), axis=1)

        tau_a = 0.5 * jnp.sum(grad_psi_a ** 2, axis=(0, 2))
        tau_b = 0.5 * jnp.sum(grad_psi_b ** 2, axis=(0, 2))

        eps_x_lsda = compute_exact_dirac_exchange(rho_a, rho_b)
        eps_c_lsda = compute_chachiyo_correlation(rho_a, rho_b)

        if use_nn:
            Fx, Fc, alpha_ex = neural_model.apply(nn_params, rho_a, rho_b, grad_rho_a, grad_rho_b, tau_a, tau_b)
            E_x_nn = jnp.sum(W_j * (rho_a + rho_b) * (eps_x_lsda * Fx))
            E_c_nn = jnp.sum(W_j * (rho_a + rho_b) * (eps_c_lsda * Fc))
            E_xc_total = (alpha_ex * E_exact_exchange) + ((1.0 - alpha_ex) * E_x_nn) + E_c_nn
        else:
            # Fixed: Include eps_c_lsda in the pure LSDA baseline
            E_xc_total = jnp.sum(W_j * (rho_a + rho_b) * (eps_x_lsda + eps_c_lsda))

        return E_core + E_H + E_xc_total + E_nuc

    return energy_fn, D, (mol.nelectron + mol.spin) // 2, (mol.nelectron - mol.spin) // 2, phi.shape[1]


# ==============================================================================
# 3. DEEP CONVERGENCE SOLVER & TRAINING ENTRYPOINT
# ==============================================================================
def main():
    print("Generating Exact Integrals & CCSD(T) Baselines...")
    pyscf_data, pyscf_lda_results, ccsd_targets = {}, {}, {}
    for m in MOLECULES:
        mol, S, T, V, ERI, E_nuc, W, phi, phi_g, lda_e, ccsd_e = get_pyscf_baselines(m)
        pyscf_data[m] = (mol, S, T, V, ERI, E_nuc, W, phi, phi_g)
        pyscf_lda_results[m] = lda_e
        ccsd_targets[m] = ccsd_e

    model = NeuralMetaGGA()
    rng = jax.random.PRNGKey(42)
    init_params = model.init(
        rng,
        jnp.ones(10, dtype=jnp.float64),
        jnp.ones(10, dtype=jnp.float64),
        jnp.ones((10, 3), dtype=jnp.float64),
        jnp.ones((10, 3), dtype=jnp.float64),
        jnp.ones(10, dtype=jnp.float64),
        jnp.ones(10, dtype=jnp.float64),
    )
    engines = {m: build_energy_fn(m, pyscf_data, model) for m in MOLECULES}

    def solve_dft_core(mol_name, nn_params, use_nn=False, max_steps=600):
        e_fn, D, na, nb, n_basis = engines[mol_name]
        Wa = jax.random.normal(jax.random.PRNGKey(0), (na, n_basis), dtype=jnp.float64)
        Wb = jax.random.normal(jax.random.PRNGKey(1), (nb, n_basis), dtype=jnp.float64)

        schedule = optax.cosine_decay_schedule(0.05, max_steps)
        inner_opt = optax.adam(schedule)

        @jit
        def run_optimization(wa_init, wb_init):
            def body_fn(carry, _):
                wa, wb, sa, sb = carry
                E, (ga, gb) = value_and_grad(e_fn, argnums=(0, 1))(wa, wb, nn_params, use_nn)
                ua, sa = inner_opt.update(ga, sa)
                ub, sb = inner_opt.update(gb, sb)
                return (optax.apply_updates(wa, ua), optax.apply_updates(wb, ub), sa, sb), E

            carry_init = (wa_init, wb_init, inner_opt.init(wa_init), inner_opt.init(wb_init))
            final_carry, history = jax.lax.scan(body_fn, carry_init, jnp.arange(max_steps))
            return final_carry[0], final_carry[1], history[-1]

        return run_optimization(Wa, Wb)

    print("\n--- Phase 1: Validating Pure LSDA Engine ---")
    d4ft_results = {m: solve_dft_core(m, init_params, use_nn=False)[2] for m in MOLECULES}

    print("--- Phase 2: Meta-GGA Hybrid Training ---")
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=3e-3, weight_decay=1e-5))
    opt_state = optimizer.init(init_params)
    nn_params = init_params

    for epoch in range(101):
        epoch_loss = 0.0
        accumulated_grads = jax.tree_util.tree_map(jnp.zeros_like, nn_params)

        for m in TRAIN_MOLS:
            Wa_opt, Wb_opt, _ = solve_dft_core(m, nn_params, use_nn=True)

            def meta_loss(params):
                return jnp.abs(engines[m][0](Wa_opt, Wb_opt, params, use_nn=True) - ccsd_targets[m])

            loss, grads = value_and_grad(meta_loss)(nn_params)
            epoch_loss += loss
            accumulated_grads = jax.tree_util.tree_map(lambda x, y: x + y, accumulated_grads, grads)

        avg_grads = jax.tree_util.tree_map(lambda x: x / len(TRAIN_MOLS), accumulated_grads)
        updates, opt_state = optimizer.update(avg_grads, opt_state, nn_params)
        nn_params = optax.apply_updates(nn_params, updates)

        if epoch % 20 == 0:
            current_alpha = jax.nn.sigmoid(nn_params["params"]["alpha_ex_raw"]) * 0.5
            print(
                f"  Epoch {epoch:3d} | Avg Train Error: {(epoch_loss / len(TRAIN_MOLS)) * HA_TO_KCAL_MOL:8.3f} kcal/mol "
                f"| Exact Exchange: {current_alpha * 100:.1f}%"
            )

    print("\n--- Phase 3: Evaluating Extrapolation ---")
    hybrid_results = {m: solve_dft_core(m, nn_params, use_nn=True)[2] for m in MOLECULES}

    print("\n" + "=" * 95)
    print(f"{'Molecule':<8} | {'LSDA Base Err':<20} | {'Meta-GGA Hybrid Err':<15}")
    print("=" * 95)

    for m in MOLECULES:
        target = ccsd_targets[m]
        d4ft_err = abs(d4ft_results[m] - target) * HA_TO_KCAL_MOL
        hyb_err = abs(hybrid_results[m] - target) * HA_TO_KCAL_MOL
        flag = "*" if m not in TRAIN_MOLS else " "
        print(f"{m:<8}{flag} | {d4ft_err:>17.2f} kcal/mol | {hyb_err:>12.2f} kcal/mol")
    print("=" * 95)
    print("* Indicates unseen molecule during training.")


if __name__ == "__main__":
    main()
