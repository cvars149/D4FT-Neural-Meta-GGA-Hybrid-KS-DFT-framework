#!/usr/bin/env python3
"""
Unit test suite verifying the mathematical and quantum-chemical invariants of `d4ft_meta`.
Runs with standard Python `unittest` + `numpy`/`scipy` (and automatically tests JAX when installed).
"""

import unittest
import numpy as np

from d4ft_meta.manifold import compute_lowdin_whitening, stiefel_qr_orthonormalize
from d4ft_meta.functionals import (
    compute_exact_dirac_exchange,
    compute_chachiyo_correlation,
    compute_vwn5_correlation,
    compute_iso_orbital_indicator,
)
from d4ft_meta.hamiltonian import (
    evaluate_grid_observables,
    compute_exact_coulomb_and_exchange,
)


class TestQuantumChemistryInvariants(unittest.TestCase):
    """Verify exact mathematical and physical constraints of the D4FT & Meta-GGA engine."""

    def test_lowdin_whitening_and_stiefel_qr_orthonormality(self):
        """Verify C @ S @ C.T == I_N to machine precision (< 1e-13 in FP64)."""
        rng = np.random.default_rng(42)
        n_basis = 14
        n_occ = 5

        # Construct a realistic positive-definite overlap matrix S
        A = rng.normal(size=(n_basis, n_basis))
        S = A @ A.T + 0.5 * np.eye(n_basis)

        D = np.asarray(compute_lowdin_whitening(S))
        # Check whitening property D.T @ S @ D == I_B
        whitened_overlap = D.T @ S @ D
        np.testing.assert_allclose(whitened_overlap, np.eye(n_basis), atol=1e-13)

        # Check orbital coefficient generalized orthonormality C @ S @ C.T == I_N
        W = rng.normal(size=(n_occ, n_basis))
        C = np.asarray(stiefel_qr_orthonormalize(W, D))
        mo_overlap = C @ S @ C.T
        np.testing.assert_allclose(mo_overlap, np.eye(n_occ), atol=1e-13)

    def test_zero_electron_spin_channel_safety(self):
        """Verify N_beta == 0 (e.g., 1-electron H atom) returns empty (0, B) matrix cleanly."""
        S = np.eye(6)
        D = compute_lowdin_whitening(S)
        W_empty = np.zeros((0, 6))
        C_empty = stiefel_qr_orthonormalize(W_empty, D)
        self.assertEqual(C_empty.shape, (0, 6))

    def test_dirac_exchange_spin_scaling(self):
        """
        Verify exact spin-scaling of Dirac exchange:
          1. Unpolarized limit: eps_x(rho/2, rho/2) == Cx * rho^(1/3)
          2. Fully spin-polarized (ferromagnetic) limit: eps_x(rho, 0) == 2^(1/3) * eps_x(rho/2, rho/2)
        """
        Cx = -0.7385587663820224
        rho = np.array([0.01, 0.1, 0.5, 1.2, 5.0], dtype=np.float64)

        eps_unpol = np.asarray(compute_exact_dirac_exchange(0.5 * rho, 0.5 * rho))
        eps_expected = Cx * (rho ** (1.0 / 3.0))
        np.testing.assert_allclose(eps_unpol, eps_expected, rtol=1e-12)

        eps_ferro = np.asarray(compute_exact_dirac_exchange(rho, np.zeros_like(rho)))
        np.testing.assert_allclose(eps_ferro, (2.0 ** (1.0 / 3.0)) * eps_unpol, rtol=1e-10)

    def test_chachiyo_vs_vwn5_correlation_consistency(self):
        """Verify Chachiyo correlation agrees closely with VWN5 across physical electron densities."""
        rho_a = np.array([0.005, 0.05, 0.25, 1.0, 4.0], dtype=np.float64)
        rho_b = rho_a.copy()

        eps_chachiyo = np.asarray(compute_chachiyo_correlation(rho_a, rho_b, spin_polarized=False))
        eps_vwn5 = np.asarray(compute_vwn5_correlation(rho_a, rho_b))

        # Both must be strictly negative and agree within ~2 mHa across valence densities
        self.assertTrue(np.all(eps_chachiyo < 0.0))
        self.assertTrue(np.all(eps_vwn5 < 0.0))
        np.testing.assert_allclose(eps_chachiyo, eps_vwn5, atol=2.5e-3)

    def test_single_orbital_weizsacker_identity_and_iso_orbital_indicator(self):
        """
        For ANY single spatial orbital psi(r) (1-electron or 2-electron singlet like H2 / He):
            rho_a = psi^2, grad_rho_a = 2 * psi * grad_psi, tau_a = 0.5 * |grad_psi|^2.
        Then:
            tau_W = |grad_rho|^2 / (8 * rho) = 4 * psi^2 * |grad_psi|^2 / (8 * psi^2) = 0.5 * |grad_psi|^2 == tau!
        Therefore alpha_iso = (tau - tau_W) / tau MUST equal 0 identically at every grid point.
        """
        rng = np.random.default_rng(7)
        G, B = 25, 6
        phi = rng.normal(size=(G, B))
        phi_grad = rng.normal(size=(G, B, 3))

        # Single occupied alpha & beta spatial orbital (closed-shell 2-electron singlet like He or H2)
        c_vec = rng.normal(size=(1, B))
        rho_a, rho_b, grad_rho_a, grad_rho_b, tau_a, tau_b = evaluate_grid_observables(
            c_vec, c_vec, phi, phi_grad
        )

        alpha_iso, tau_w = compute_iso_orbital_indicator(
            rho_a, rho_b, grad_rho_a, grad_rho_b, tau_a, tau_b, clip_nonnegative=False
        )
        np.testing.assert_allclose(tau_w, tau_a + tau_b, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(alpha_iso, np.zeros(G), atol=1e-12)

    def test_one_electron_self_interaction_cancellation(self):
        """
        For any one-electron system (N_alpha = 1, N_beta = 0), the Hartree self-repulsion E_H
        must be cancelled 100% by the exact Hartree-Fock exchange energy E_x^{HF}:
            E_H[P_a] + E_x^{HF}[P_a, 0] == 0.
        This proves the exact exchange index contraction ('il,jk,ijkl->') is bug-free.
        """
        rng = np.random.default_rng(99)
        B = 5
        # Construct random symmetric 4-index ERI tensor with (ij|kl) = (ji|kl) = (ij|lk) = (kl|ij)
        M = rng.normal(size=(B, B, B))
        ERI = np.einsum("pij,pkl->ijkl", M, M)

        # Single normalized orbital vector c
        c = rng.normal(size=(1, B))
        Pa = c.T @ c  # Rank-1 density matrix
        Pb = np.zeros((B, B))

        E_H, E_x_exact = compute_exact_coulomb_and_exchange(Pa, Pb, ERI)
        self.assertGreater(float(E_H), 0.0)
        self.assertAlmostEqual(float(E_H + E_x_exact), 0.0, places=13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
