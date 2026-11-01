# `d4ft-neural-ksdft`: Differentiable Kohn-Sham DFT & Neural Meta-GGA Global Hybrid in JAX

A modular, high-precision (`float64`) research framework for **Differentiable Kohn-Sham Density Functional Theory (D4FT)** and **Bi-Level Learned Neural Meta-GGA Global Hybrid Functionals** built on **JAX**, **Flax**, **Optax**, and **PySCF**.

This repository unifies and extends two complementary deep-learning approaches to electronic structure theory:
1. **Direct Variational Stiefel-Manifold Kohn-Sham Optimization ([Li et al., ICLR 2023, arXiv:2303.00399](https://arxiv.org/abs/2303.00399))**: Replaces the $O(B^3)$ iterative Fock diagonalization and heuristic DIIS mixing of traditional Self-Consistent Field (SCF) solvers with direct gradient descent on unconstrained parameters $W$ projected onto the generalized Stiefel manifold via Löwdin canonical whitening and QR decomposition.
2. **Differentiable Meta-GGA Global Hybrid Functional Learning ([Kasim & Vinko, PRL 2021](https://link.aps.org/doi/10.1103/PhysRevLett.127.126403); [Kirkpatrick et al., Science 2021 (DM21)](https://pubmed.ncbi.nlm.nih.gov/34882476/))**: Ascends from Rung 1 (LSDA) to Rung 4 (Global Hybrid Meta-GGA) on Jacob's Ladder by combining exact Dirac spin exchange, Chachiyo uniform-electron-gas correlation ([Chachiyo, JCP 2016](https://pubs.aip.org/aip/jcp/article/145/2/021101/907773/Communication-Simple-and-accurate-uniform-electron)), the von Weizsäcker iso-orbital indicator $\alpha_{\text{iso}}(\mathbf{r})$, and a learnable fraction of exact non-local Hartree-Fock exchange ($\alpha_{\text{ex}} E_x^{\text{HF}}$), trained end-to-end against gold-standard $\text{CCSD(T)}/\text{def2-SVP}$ energies via the Hellmann-Feynman theorem.

## Repository Structure

| Path | Description |
| :--- | :--- |
| `d4ft_meta/config.py` | Hyperparameters, `def2-SVP` / `6-31g` settings, and molecular geometry registry (`H2`, `He`, `LiH`, `N`, `H2O`, `CH4`, `CO`, `F2`, `HF`, `O2`, `CO2`, `C2H6O`, `C6H6`, `C20`, `C60`, `C80`) |
| `d4ft_meta/manifold.py` | Löwdin canonical whitening ($D = U \Lambda^{-1/2}$) and sign-stabilized QR Stiefel projection ($C = Q_N^\top D^\top \implies C S C^\top = I_N$) |
| `d4ft_meta/functionals.py` | Analytical Dirac LSDA exchange, Chachiyo (2016) & VWN5 (1980) correlation, iso-orbital indicator $\alpha_{\text{iso}}$, `NeuralMetaGGA`, `NeuralXC`, and `NeuralBasisScaler` |
| `d4ft_meta/hamiltonian.py` | Exact analytical 1e/2e tensor contractions ($E_{\text{core}}, E_H, E_x^{\text{HF}}$) and grid observable evaluators ($\rho_\sigma, \nabla\rho_\sigma, \tau_\sigma$) |
| `d4ft_meta/molecules.py` | PySCF integral extractor (`int1e_ovlp`, `int1e_kin`, `int1e_nuc`, `int2e`), Becke grids, and RKS/UKS & CCSD(T) reference solvers |
| `d4ft_meta/solvers.py` | Fused XLA `jax.lax.scan` inner orbital solver and Hellmann-Feynman outer bi-level `MetaGGAHybridTrainer` |
| `scripts/train_metagga_hybrid.py` | End-to-end CLI script for Phase 1 (LSDA validation), Phase 2 (Meta-GGA Hybrid training on 6 molecules), and Phase 3 (zero-shot extrapolation on `CO`, `F2`, `HF`) |
| `scripts/run_d4ft_experiments.py` | CLI runner for D4FT ground-state, spin magnetism (`O2` vs `CO2`), and fullerene scaling benchmarks |
| `tests/test_physics_and_math.py` | Automated test suite verifying Stiefel orthonormality, Dirac spin scaling, Weizsäcker single-orbital identity ($\alpha_{\text{iso}}=0$), and 1e self-interaction cancellation |
| `docs/THEORY_AND_CODE_ANALYSIS.md` | Full mathematical derivations, line-by-line comparison of both scripts, and technical bug audit |

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run Physics & Mathematical Invariant Unit Tests

```bash
python3 -m unittest discover -s tests -v
```

### 3. Train the High-Precision (FP64) Neural Meta-GGA Hybrid Functional

```bash
python3 scripts/train_metagga_hybrid.py --basis def2-SVP --epochs 101 --inner-steps 600
```

### 4. Run Spin-Polarized Magnetism Benchmarks (`O2` vs `CO2`)

```bash
python3 scripts/run_d4ft_experiments.py --basis 6-31g
```

## Key Architectural Upgrades from `d4ft_kohn_sham_dft_final.py` to `NeuralMetaGGA`

| Feature / Physical Dimension | Script 1 (`d4ft_kohn_sham_dft_final.py`) | Script 2 (`NeuralMetaGGA` Hybrid Engine) | Physical & Numerical Impact |
| :--- | :--- | :--- | :--- |
| **Floating-Point Precision** | `jnp.float32` (~$10^{-7}$ relative precision) | `jax_enable_x64=True` (`jnp.float64`, ~$10^{-16}$) | Eliminates gradient floor noise below $0.1\text{ mHa}$ ($0.06\text{ kcal/mol}$, chemical accuracy) |
| **Gaussian Basis Set** | Pople split-valence `6-31g` (no polarization) | Karlsruhe polarized `def2-SVP` ($d$ on heavy, $p$ on H) | Captures angular distortion and $\pi$/polar covalent bonding in $\text{CO}, \text{HF}, \text{H}_2\text{O}$ |
| **1e Core Integrals ($T + V_{\text{ext}}$)** | Numerical grid quadrature ($\frac{1}{2}\int |\nabla\psi|^2 + \int \rho v_{\text{ext}}$) | Exact analytical Gaussian integrals (`int1e_kin` + `int1e_nuc`) | Removes grid-discretization error near nuclear cusps ($1/|\mathbf{r}-\mathbf{R}_A|$) |
| **Jacob's Ladder Rung** | Rung 1 Local Density (`NeuralXC` on $\rho$ only, closed-shell) | Rung 3/4 Meta-GGA + Global Hybrid (`NeuralMetaGGA` on $\rho_\sigma, \nabla\rho_\sigma, \tau_\sigma, \alpha_{\text{iso}}$ + $\alpha_{\text{ex}} E_x^{\text{HF}}$) | Distinguishes single covalent bonds ($\alpha_{\text{iso}}=0$) from metallic/delocalized regions and mitigates Self-Interaction Error (SIE) |
| **Inner Orbital Solver** | Python `for` loop calling `@jit` single step | Fully fused `@jit` `jax.lax.scan` over 600 cosine-decay steps | Eliminates Python dispatch overhead; runs 600 SCF-equivalent steps inside a single XLA kernel |
| **Open-Shell Spin Support** | `Method1Solver` restricted to closed-shell ($S=0$) | Full Unrestricted Kohn-Sham ($W_\alpha, W_\beta$) in training & inference | Enables training on open-shell Quartet Nitrogen ($\text{N}, 2S=3$) alongside closed-shell molecules |
