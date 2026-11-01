# Differentiable Kohn-Sham DFT (D4FT) & Neural Meta-GGA Global Hybrid Functionals in JAX

**A Self-Contained Deep Learning & Quantum Chemistry Framework for Direct Variational Electronic Structure Optimization and Bi-Level Exchange-Correlation Functional Discovery**

[https://img.shields.io/badge/Python-3.10%2B-blue.svg]
[https://img.shields.io/badge/JAX-FP64%20%2F%20XLA-orange.svg]
[https://img.shields.io/badge/Flax%20%26%20Optax-Neural%20DFT-green.svg]
[https://img.shields.io/badge/Integrals%20%26%20CCSD(T]-PySCF-purple.svg)
[https://img.shields.io/badge/License-MIT-lightgrey.svg]

## Overview

**Kohn-Sham Density Functional Theory (KS-DFT)** is the workhorse of modern quantum chemistry, materials science, and computational physics. Despite its ubiquity, standard KS-DFT implementations face two foundational barriers:

1. **The Algorithmic Solver Bottleneck (Iterative SCF Diagonalization)**: Traditional quantum chemistry packages solve the Kohn-Sham equations via the **Self-Consistent Field (SCF)** loop—repeatedly constructing the Fock matrix $F(\mathbf{C})$ and solving the generalized eigenvalue problem $F \mathbf{C}^\top = S \mathbf{C}^\top \boldsymbol{\epsilon}$ at $O(B^3)$ to $O(B^4)$ cost per iteration. Because the effective potential depends nonlinearly on the orbitals it generates, SCF iterations are not guaranteed to decrease total energy monotonically; they frequently suffer from charge sloshing and convergence failures in open-shell radicals or stretched/narrow-gap molecules (such as $\text{N}$, $\text{O}_2$, and $\text{F}_2$). Furthermore, differentiating through eigendecompositions to train machine-learned functionals introduces numerical instabilities whenever orbital eigenvalues become nearly degenerate ($\partial \mathbf{c}_i / \partial F \propto (\epsilon_i - \epsilon_j)^{-1}$).
2. **The Physical Accuracy Bottleneck (Exchange-Correlation Approximations & Self-Interaction Error)**: While Kohn-Sham theory is exact in principle ([Hohenberg & Kohn, 1964](https://link.aps.org/doi/10.1103/PhysRev.136.B864); [Kohn & Sham, 1965](https://link.aps.org/doi/10.1103/PhysRev.140.A1133)), its accuracy in practice is dictated entirely by the approximation chosen for the **Exchange-Correlation functional ($E_{\text{xc}}[\rho]$)**. Lower-rung functionals on John Perdew's **Jacob's Ladder**—such as the Local Spin Density Approximation (**LSDA**, Rung 1)—suffer from severe **Self-Interaction Error (SIE)** because the classical Hartree Coulomb integral $E_H[\rho]$ includes an unphysical repulsion of each electron with itself that local exchange cannot fully cancel. Moving toward **chemical accuracy** ($\le 1\text{ kcal/mol}$) requires climbing to **Rung 3 (Meta-GGA)** and **Rung 4 (Global Hybrid DFT)** by incorporating orbital kinetic energy densities $\tau_\sigma(\mathbf{r})$, the iso-orbital indicator $\alpha_{\text{iso}}(\mathbf{r})$, and non-local exact Hartree-Fock exchange $E_x^{\text{HF}}$.

This repository tackles **both** problems simultaneously inside an end-to-end differentiable **JAX + Flax + Optax + PySCF** architecture:

* **Direct Variational Stiefel-Manifold Optimization ([D4FT, Li et al., ICLR 2023](https://arxiv.org/abs/2303.00399))**: Replaces iterative Fock matrix diagonalization and heuristic DIIS mixing with direct gradient descent on unconstrained orbital parameters $W_\sigma$, mapped deterministically onto the generalized Stiefel manifold ($\mathbf{C}_\sigma S \mathbf{C}_\sigma^\top = I_{N_\sigma}$) via **Löwdin canonical whitening** and **QR decomposition**.
* **Bi-Level Learned Neural Meta-GGA Global Hybrid Functional ([Kasim & Vinko, PRL 2021](https://link.aps.org/doi/10.1103/PhysRevLett.127.126403); [Kirkpatrick et al., Science 2021 (DM21)](https://pubmed.ncbi.nlm.nih.gov/34882476/))**: Embeds a spin-polarized neural network (`NeuralMetaGGA`) directly inside the high-precision (`FP64`) Kohn-Sham Hamiltonian. The network observes local spin densities ($\rho_\alpha, \rho_\beta$), density gradients ($\gamma_\alpha, \gamma_\beta$), positive-definite kinetic energy densities ($\tau_\alpha, \tau_\beta$), and the von Weizsäcker iso-orbital indicator ($\alpha_{\text{iso}}$) to predict point-wise enhancement factors $(F_x(\mathbf{r}), F_c(\mathbf{r}))$ over exact Dirac exchange and [Chachiyo (2016) uniform-electron-gas correlation](https://pubs.aip.org/aip/jcp/article/145/2/021101/907773/Communication-Simple-and-accurate-uniform-electron), while simultaneously learning a global fraction of exact Hartree-Fock exchange ($\alpha_{\text{ex}} E_x^{\text{HF}}$).
* **Zero-Unroll Implicit Differentiation via the Hellmann-Feynman Theorem**: Trains the neural functional against gold-standard Coupled Cluster **$\text{CCSD(T)}/\text{def2-SVP}$** reference energies on small atoms and molecules ($\text{H}_2, \text{He}, \text{LiH}, \text{N}, \text{H}_2\text{O}, \text{CH}_4$) and evaluates zero-shot chemical extrapolation on unseen bonds ($\text{CO}, \text{F}_2, \text{HF}$) using $O(1)$-memory variational gradients without unrolling the 600-step inner XLA optimization loop.

## Repository Contents & Architecture

This repository is organized around two complete, self-contained executable scripts that capture both stages of the research workflow, accompanied by a full mathematical derivation and code audit document:

| File | Role in the Project | Key Features & Capabilities |
| :--- | :--- | :--- |
| `d4ft_metagga_hybrid_complete.py` | **Generation 2 Flagship Engine**: High-Precision (`FP64`) Neural Meta-GGA Global Hybrid Solver & Bi-Level Trainer | Full Unrestricted Kohn-Sham (`UKS`) in `def2-SVP`, exact analytical 1e/2e integrals (`int1e_kin`, `int1e_nuc`, `int2e`), `NeuralMetaGGA` ($\rho_\sigma, \nabla\rho_\sigma, \tau_\sigma, \alpha_{\text{iso}}$ + learnable exact HF exchange $\alpha_{\text{ex}}$), fused 600-step `@jit` `jax.lax.scan` solver, and bi-level training + unseen molecule extrapolation ($\text{CO}, \text{F}_2, \text{HF}$) against $\text{CCSD(T)}$ |
| `d4ft_kohn_sham_dft_final.py` | **Generation 1 Benchmark Suite**: Full D4FT Paper Reproduction ([Li et al., ICLR 2023](https://arxiv.org/abs/2303.00399)) | Compares **Method 1** (`NeuralXC` + `NeuralBasisScaler`), **Method 2** (Standard D4FT + LDA/LSDA), and **Method 3** (PySCF SCF Baseline) in `6-31g` across **Exp 1** (Ground-state accuracy on $\text{H}_2, \text{CH}_4, \text{H}_2\text{O}, \text{O}_2, \text{C}_2\text{H}_6\text{O}, \text{C}_6\text{H}_6$), **Exp 2** ($\text{O}_2$ paramagnetic vs. $\text{CO}_2$ diamagnetic spin prediction), and **Exp 3** (Fullerene $\text{C}_{20}, \text{C}_{60}, \text{C}_{80}$ scalability via mini-batch grid SGD) |
| `THEORY_AND_CODE_ANALYSIS.md` | **Theoretical Reference & Code Audit** | First-principles derivations of Löwdin/QR Stiefel projection, Jacob's Ladder, exact HF exchange tensor contractions, the Hellmann-Feynman bi-level theorem, and a line-by-line audit of all numerical/physical fixes |
| `requirements.txt` | **Environment Dependencies** | Pinned package requirements for `jax`, `jaxlib`, `flax`, `optax`, `pyscf`, `numpy`, `scipy`, `pandas`, and `matplotlib` |
| `LICENSE` | **Open-Source License** | MIT License |

## Detailed Comparison: Generation 1 (`d4ft_kohn_sham_dft_final.py`) vs. Generation 2 (`d4ft_metagga_hybrid_complete.py`)

Understanding how the two scripts differ illuminates why each architectural decision in `d4ft_metagga_hybrid_complete.py` is necessary to reach chemical accuracy:

| Physical / Algorithmic Dimension | Generation 1 (`d4ft_kohn_sham_dft_final.py`) | Generation 2 (`d4ft_metagga_hybrid_complete.py`) | Why the Upgrade Matters |
| :--- | :--- | :--- | :--- |
| **Floating-Point Precision** | `jnp.float32` ($\sim 10^{-7}$ relative precision) | `jax_enable_x64 = True` (`jnp.float64`, $\sim 10^{-16}$ precision) | Total molecular energies are $O(10^1\text{–}10^2\text{ Ha})$, while $1\text{ kcal/mol} \approx 0.00159\text{ Ha}$. In `float32`, QR and Cholesky/eigh rounding errors create gradient noise floors around $10^{-4}\text{ Ha}$, whereas `float64` converges smoothly to $< 10^{-12}\text{ Ha}$ |
| **Atomic Orbital Basis Set** | Pople split-valence `6-31g` (no $d$ or $p$ polarization orbitals) | Karlsruhe polarized split-valence `def2-SVP` ($d$ functions on $\text{Li–F}$, $p$ on $\text{H, He}$) | Polarization functions are physically mandatory to describe angular distortion of electron clouds in polar and triple bonds ($\text{CO}, \text{HF}, \text{H}_2\text{O}$) |
| **Kinetic & Nuclear Attraction ($T + V_{\text{nuc}}$)** | Numerical quadrature on Becke grids ($\frac{1}{2}\int \Vert{}\nabla\psi\Vert{}^2 d\mathbf{r} - \sum_A \int \frac{Z_A \rho(\mathbf{r})}{\Vert{}\mathbf{r}-\mathbf{R}_A\Vert{}} d\mathbf{r}$) | Exact analytical Gaussian integrals (`mol.intor("int1e_kin")` + `mol.intor("int1e_nuc")`) via $\text{Tr}[P(T+V)]$ | Analytical 1e integrals eliminate radial/angular grid integration errors near the singular $1/\Vert{}\mathbf{r}-\mathbf{R}_A\Vert{}$ nuclear cusp and reduce per-step compute to a tiny $B \times B$ matrix trace |
| **Jacob's Ladder Rung** | **Rung 1 (Local Density)**: `NeuralXC` takes $[\rho, \rho^{1/3}, \ln(1+\rho)]$ on spin-unpolarized density | **Rung 3/4 (Meta-GGA Global Hybrid)**: `NeuralMetaGGA` takes 7 spin-polarized features $[\ln(1+\rho_{\text{tot}}), \zeta, \ln(1+\gamma_\alpha), \ln(1+\gamma_\beta), \ln(1+\tau_\alpha), \ln(1+\tau_\beta), \alpha_{\text{iso}}]$ + exact HF exchange $\alpha_{\text{ex}} E_x^{\text{HF}}$ | Pure density functionals cannot distinguish a covalent single bond from a stretched radical or cancel 1-electron self-interaction error; combining $\alpha_{\text{iso}}$ with exact HF exchange resolves both |
| **Correlation Functional Baseline** | 6-parameter Padé approximant [VWN5 (Vosko, Wilk & Nusair, 1980)](https://cdnsciencepub.com/doi/10.1139/p80-159) | MP2 high-density derived [Chachiyo correlation (JCP 2016)](https://pubs.aip.org/aip/jcp/article/145/2/021101/907773/Communication-Simple-and-accurate-uniform-electron) | Chachiyo's analytical logarithmic formula $a \ln(1 + b/r_s + b/r_s^2)$ avoids expensive inverse-tangent evaluations in XLA and matches Ceperley-Alder QMC across all densities |
| **Inner Loop Execution Graph** | Python `for` loop calling single-step `@jit` function 150–200 times | Fully fused `@jit` `jax.lax.scan` over 600 steps with cosine-decay Adam (`0.05 -> 0.0`) | Compiles all 600 orbital optimization iterations into a single XLA kernel call with zero Python interpreter overhead |
| **Open-Shell Spin Support in Neural Solver** | `Method1Solver` restricted to closed-shell molecules ($\text{H}_2, \text{CH}_4, \text{H}_2\text{O}$) | Full spin-polarized $(W_\alpha, W_\beta)$ optimization across all molecules | Allows training on open-shell Quartet Nitrogen ($\text{N}, 2S=3$, 3 unpaired $2p$ electrons) alongside closed-shell singlets |
| **Outer Bi-Level Loss & Optimizer** | Mean Squared Error ($\text{Ha}^2$) on frozen density $\rho$, Adam (`lr=1e-3`) | Mean Absolute Error ($\Vert{}E - E_{\text{CCSD(T)}}\Vert{}$) through full `energy_fn`, global-norm gradient clipping (`1.0`) + `AdamW` (`lr=3e-3, wd=1e-5`) | Differentiating through `energy_fn` updates both the semi-local neural weights and the global exact exchange fraction $\alpha_{\text{ex}}$, while $L_1$ loss prevents outlier molecules from dominating gradients |

## Mathematical & Physical Foundations

### 1. The Unrestricted Kohn-Sham Energy in a Non-Orthogonal Gaussian Basis

Consider a molecular system with nuclear coordinates $\{\mathbf{R}_A\}$, atomic numbers $\{Z_A\}$, $N_\alpha$ spin-up electrons, and $N_\beta$ spin-down electrons ($N = N_\alpha + N_\beta$, spin $2S = N_\alpha - N_\beta$). We expand the occupied spatial Kohn-Sham molecular orbitals $\{\psi_{i\sigma}(\mathbf{r})\}_{i=1}^{N_\sigma}$ in a basis of $B$ real atom-centered Gaussian basis functions $\{\phi_\mu(\mathbf{r})\}_{\mu=1}^B$:

$$\psi_{i\sigma}(\mathbf{r}) = \sum_{\mu=1}^B C^\sigma_{i\mu}\,\phi_\mu(\mathbf{r}), \qquad \sigma \in \{\alpha, \beta\}, \quad C^\sigma \in \mathbb{R}^{N_\sigma \times B}$$

Because atomic orbitals centered on different atoms are non-orthogonal, their overlap matrix $S_{\mu\nu} = \int \phi_\mu(\mathbf{r})\phi_\nu(\mathbf{r})\,d\mathbf{r}$ is symmetric positive-definite ($S \succ 0$) rather than the identity matrix. Fermi-Dirac statistics require the Kohn-Sham orbitals within each spin channel to remain mutually orthonormal:

$$\int \psi_{i\sigma}(\mathbf{r})\psi_{j\sigma}(\mathbf{r})\,d\mathbf{r} = \sum_{\mu,\nu=1}^B C^\sigma_{i\mu} S_{\mu\nu} C^\sigma_{j\nu} = \delta_{ij} \quad \Longleftrightarrow \quad C^\sigma S (C^\sigma)^\top = I_{N_\sigma}$$

From the orbital coefficient matrices $(C^\alpha, C^\beta)$, we construct the one-particle spin density matrices $P^\alpha, P^\beta \in \mathbb{R}^{B \times B}$ and total density matrix $P^{\text{tot}}$:

$$P^\alpha = (C^\alpha)^\top C^\alpha, \qquad P^\beta = (C^\beta)^\top C^\beta, \qquad P^{\text{tot}} = P^\alpha + P^\beta$$

The variational Kohn-Sham ground-state problem is the constrained minimization of the total electronic + nuclear repulsion energy:

$$\min_{C^\alpha, C^\beta} \; E_{\text{total}}(C^\alpha, C^\beta) = E_{\text{core}}[P^{\text{tot}}] + E_H[P^{\text{tot}}] + E_{\text{xc}}[C^\alpha, C^\beta] + E_{\text{nuc}} \qquad \text{subject to} \quad C^\sigma S (C^\sigma)^\top = I_{N_\sigma}$$

where:
* **Core 1-Electron Energy ($E_{\text{core}}$)**:
  $$E_{\text{core}}[P^{\text{tot}}] = \text{Tr}\!\left[P^{\text{tot}}(T + V_{\text{nuc}})\right] = \sum_{\mu,\nu=1}^B P^{\text{tot}}_{\mu\nu} (T_{\mu\nu} + V_{\mu\nu})$$
  with analytical kinetic integrals $T_{\mu\nu} = -\frac{1}{2}\int \phi_\mu(\mathbf{r})\nabla^2\phi_\nu(\mathbf{r})\,d\mathbf{r}$ (`int1e_kin`) and nuclear attraction integrals $V_{\mu\nu} = -\sum_A Z_A \int \frac{\phi_\mu(\mathbf{r})\phi_\nu(\mathbf{r})}{\|\mathbf{r}-\mathbf{R}_A\|}\,d\mathbf{r}$ (`int1e_nuc`).
* **Classical Coulomb / Hartree Energy ($E_H$)**:
  $$E_H[P^{\text{tot}}] = \frac{1}{2}\iint \frac{\rho_{\text{tot}}(\mathbf{r}_1)\rho_{\text{tot}}(\mathbf{r}_2)}{\|\mathbf{r}_1 - \mathbf{r}_2\|}\,d\mathbf{r}_1 d\mathbf{r}_2 = \frac{1}{2}\sum_{\mu,\nu,\lambda,\kappa=1}^B P^{\text{tot}}_{\mu\nu} P^{\text{tot}}_{\lambda\kappa} (\mu\nu|\lambda\kappa)$$
  where $(\mu\nu|\lambda\kappa) = \iint \phi_\mu(\mathbf{r}_1)\phi_\nu(\mathbf{r}_1) r_{12}^{-1} \phi_\lambda(\mathbf{r}_2)\phi_\kappa(\mathbf{r}_2)\,d\mathbf{r}_1 d\mathbf{r}_2$ is the 4-center 2-electron repulsion integral tensor (`ERI = mol.intor("int2e")` in Chemist's notation).
* **Nuclear Repulsion Energy ($E_{\text{nuc}}$)**:
  $$E_{\text{nuc}} = \sum_{A < B} \frac{Z_A Z_B}{\|\mathbf{R}_A - \mathbf{R}_B\|}$$

### 2. Differentiable Stiefel Manifold Projection (Löwdin Whitening + QR Reparameterization)

How do we enforce the quadratic matrix constraint $C^\sigma S (C^\sigma)^\top = I_{N_\sigma}$ during gradient descent without Lagrange multipliers or projection steps that drift off the manifold?

Following [D4FT (Li et al., ICLR 2023)](https://arxiv.org/abs/2303.00399), we perform a two-stage differentiable reparameterization:

1. **Löwdin Canonical Whitening of the Overlap Metric**:
   Before optimization begins, we compute the eigendecomposition of the symmetric positive-definite overlap matrix $S = U \Lambda U^\top$ (where $U^\top U = I_B$ and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_B) > 0$) and form the whitening transformation matrix:
   $$D = U \Lambda^{-1/2} \in \mathbb{R}^{B \times B}$$
   Notice that $D$ transforms $S$ into the Euclidean identity metric:
   $$D^\top S D = \Lambda^{-1/2} U^\top (U \Lambda U^\top) U \Lambda^{-1/2} = \Lambda^{-1/2} \Lambda \Lambda^{-1/2} = I_B$$
2. **Sign-Stabilized QR Projection onto the Euclidean Stiefel Manifold**:
   Let $W_\sigma \in \mathbb{R}^{N_\sigma \times B}$ be an **unconstrained** matrix of trainable parameters initialized from $\mathcal{N}(0, 1)$. At every forward pass, we compute the reduced QR decomposition of $W_\sigma^\top \in \mathbb{R}^{B \times N_\sigma}$:
   $$Q_\sigma, R_\sigma = \text{QR}(W_\sigma^\top), \qquad Q_\sigma \in \mathbb{R}^{B \times N_\sigma}, \quad R_\sigma \in \mathbb{R}^{N_\sigma \times N_\sigma}$$
   To prevent discontinuous sign flips ($q_i \to -q_i$) when columns of $W_\sigma^\top$ cross Householder reflection hyperplanes during optimization, we enforce a unique gauge with positive diagonal entries on $R_\sigma$:
   $$\tilde{Q}_\sigma = Q_\sigma \cdot \text{diag}\!\left(\text{sign}(\text{diag}(R_\sigma))\right)$$
   Finally, we map $\tilde{Q}_\sigma$ into the non-orthogonal AO basis via $D^\top$:
   $$C^\sigma(W_\sigma) = \tilde{Q}_\sigma^\top D^\top \in \mathbb{R}^{N_\sigma \times B}$$
   **Proof of Exact Generalized Orthonormality**:
   $$C^\sigma S (C^\sigma)^\top = \tilde{Q}_\sigma^\top \underbrace{(D^\top S D)}_{=\,I_B} \tilde{Q}_\sigma = \tilde{Q}_\sigma^\top \tilde{Q}_\sigma = I_{N_\sigma}$$
   This holds identically for **any** unconstrained matrix $W_\sigma \in \mathbb{R}^{N_\sigma \times B}$, converting constrained Kohn-Sham optimization into standard unconstrained deep-learning optimization with `optax.adam`.

### 3. Grid Observables: Spin Densities, Gradients, and Kinetic Energy Densities

Semi-local exchange-correlation functionals are integrated numerically on atom-centered **Becke-Lebedev quadrature grids** $\{\mathbf{r}_g, w_g\}_{g=1}^G$ (`grids.level = 4`). PySCF evaluates the atomic basis functions and their Cartesian spatial derivatives on all $G$ grid points simultaneously via `ao_vals = numint.eval_ao(mol, grids.coords, deriv=1)`:
* `phi = ao_vals[0]` has shape $(G, B)$, representing $\phi_\mu(\mathbf{r}_g)$.
* `phi_grad = ao_vals[1:].transpose(1, 2, 0)` permutes PySCF's $(3, G, B)$ derivative array into shape $(G, B, 3)$, representing $\nabla_{\mathbf{r}}\phi_\mu(\mathbf{r}_g) = (\partial_x \phi_\mu, \partial_y \phi_\mu, \partial_z \phi_\mu)$.

From `phi` and `phi_grad`, `d4ft_metagga_hybrid_complete.py` constructs the three fundamental spin-resolved fields required up to Rung 3 of Jacob's Ladder:

1. **Molecular Orbitals & Spatial Gradients on the Grid**:
   $$\psi_{i\sigma}(\mathbf{r}_g) = \sum_{\mu=1}^B \phi_\mu(\mathbf{r}_g) C^\sigma_{i\mu} \quad (\text{shape } G \times N_\sigma), \qquad \nabla\psi_{i\sigma}(\mathbf{r}_g) = \sum_{\mu=1}^B \nabla\phi_\mu(\mathbf{r}_g) C^\sigma_{i\mu} \quad (\text{shape } 3 \times G \times N_\sigma)$$
2. **Spin-Resolved Electron Densities ($\rho_\alpha, \rho_\beta$)**:
   $$\rho_\sigma(\mathbf{r}_g) = \sum_{i=1}^{N_\sigma} \left|\psi_{i\sigma}(\mathbf{r}_g)\right|^2, \qquad \rho_{\text{tot}}(\mathbf{r}_g) = \rho_\alpha(\mathbf{r}_g) + \rho_\beta(\mathbf{r}_g)$$
3. **Spin Density Gradients ($\nabla\rho_\alpha, \nabla\rho_\beta$) and Reduced Gradient Norms ($\gamma_\alpha, \gamma_\beta$)**:
   By the chain rule $\nabla(\psi_{i\sigma}^2) = 2\psi_{i\sigma}\nabla\psi_{i\sigma}$:
   $$\nabla\rho_\sigma(\mathbf{r}_g) = 2 \sum_{i=1}^{N_\sigma} \psi_{i\sigma}(\mathbf{r}_g)\,\nabla\psi_{i\sigma}(\mathbf{r}_g) \in \mathbb{R}^3, \qquad \gamma_\sigma(\mathbf{r}_g) = \left\|\nabla\rho_\sigma(\mathbf{r}_g)\right\|_2^2$$
4. **Positive-Definite Kohn-Sham Kinetic Energy Densities ($\tau_\alpha, \tau_\beta$)**:
   $$\tau_\sigma(\mathbf{r}_g) = \frac{1}{2}\sum_{i=1}^{N_\sigma} \left\|\nabla\psi_{i\sigma}(\mathbf{r}_g)\right\|_2^2, \qquad \tau_{\text{tot}}(\mathbf{r}_g) = \tau_\alpha(\mathbf{r}_g) + \tau_\beta(\mathbf{r}_g)$$
   Note that integrating $\tau_{\text{tot}}(\mathbf{r})$ over all space recovers the exact non-interacting Kohn-Sham kinetic energy: $\int \tau_{\text{tot}}(\mathbf{r})\,d\mathbf{r} = \text{Tr}[P^{\text{tot}} T]$.

### 4. Climbing Jacob's Ladder: From Analytical LSDA to Neural Meta-GGA Global Hybrids

#### Rung 1 Baseline: Exact Spin-Polarized Dirac Exchange + Chachiyo (2016) Correlation
Instead of training a neural network from random noise—which would produce wild unphysical potentials on Epoch 0 and destabilize the orbital solver—we anchor `NeuralMetaGGA` to exact uniform-electron-gas physics:

* **Spin-Polarized Dirac/Slater Exchange Energy per Electron**:
  $$\varepsilon_x^{\text{LSDA}}(\rho_\alpha, \rho_\beta) = C_x \frac{\rho_\alpha(2\rho_\alpha)^{1/3} + \rho_\beta(2\rho_\beta)^{1/3}}{\rho_\alpha + \rho_\beta}, \qquad C_x = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3} \approx -0.7385587663820224$$
  In the unpolarized limit ($\rho_\alpha = \rho_\beta = \rho/2$), this reduces to $\varepsilon_x(\rho) = C_x \rho^{1/3}$; in the fully spin-polarized ferromagnetic limit ($\rho_\alpha = \rho, \rho_\beta = 0$), it scales by the exact factor $2^{1/3}$.
* **Chachiyo (2016) Uniform Electron Gas Correlation Energy per Electron**:
  Derived from second-order Møller-Plesset (MP2) perturbation theory in the high-density limit ([T. Chachiyo, J. Chem. Phys. 145, 021101 (2016)](https://pubs.aip.org/aip/jcp/article/145/2/021101/907773/Communication-Simple-and-accurate-uniform-electron)):
  $$\varepsilon_c^{\text{Chachiyo}}(r_s) = a \ln\!\left(1 + \frac{b}{r_s} + \frac{b}{r_s^2}\right), \qquad r_s = \left(\frac{3}{4\pi(\rho_\alpha + \rho_\beta)}\right)^{1/3}$$
  where $r_s$ is the Wigner-Seitz radius, $a = \frac{\ln 2 - 1}{2\pi^2} \approx -0.01554535$, and $b = 20.64336$.

#### Rung 3 Ingredient: The von Weizsäcker Kinetic Energy Density $\tau_W$ and Iso-Orbital Indicator $\alpha_{\text{iso}}$
How can a semi-local functional know what kind of chemical bond it is looking at? By Cauchy-Schwarz, the total Kohn-Sham kinetic energy density $\tau_{\text{tot}}(\mathbf{r})$ is bounded below everywhere by the **von Weizsäcker kinetic energy density** $\tau_W(\mathbf{r})$—the exact kinetic energy density of a hypothetical boson/single-orbital system sharing the same total density $\rho_{\text{tot}}(\mathbf{r})$:

$$\tau_W(\mathbf{r}) = \frac{\left\|\nabla\rho_{\text{tot}}(\mathbf{r})\right\|_2^2}{8\,\rho_{\text{tot}}(\mathbf{r})} \le \tau_{\text{tot}}(\mathbf{r})$$

From $\tau_{\text{tot}}$ and $\tau_W$, we form the dimensionless **iso-orbital indicator** ([TPSS](https://link.aps.org/doi/10.1103/PhysRevLett.91.146401); [$r^2\text{SCAN}$, Furness et al., JPCL 2020](https://pubs.acs.org/doi/pdf/10.1021/acs.jpclett.0c02405)):

$$\alpha_{\text{iso}}(\mathbf{r}) = \text{clip}\!\left(\frac{\tau_{\text{tot}}(\mathbf{r}) - \tau_W(\mathbf{r})}{\tau_{\text{tot}}(\mathbf{r})},\; 0,\; 1\right) \in [0, 1]$$

**Why $\alpha_{\text{iso}}(\mathbf{r})$ is a physical game-changer**:
* **Single-Orbital / Covalent Single-Bond Regions ($\alpha_{\text{iso}} = 0$)**: Whenever the density at $\mathbf{r}$ is dominated by a single spatial orbital $\psi(\mathbf{r})$ (true everywhere in 1-electron $\text{H}$, 2-electron singlets like $\text{H}_2$ and $\text{He}$, atomic $1s$ cores, and localized covalent single bonds), $\rho_{\text{tot}} = N\psi^2$ and $\nabla\rho_{\text{tot}} = 2N\psi\nabla\psi$. Substituting into $\tau_W$ gives:
  $$\tau_W = \frac{4N^2\psi^2\|\nabla\psi\|^2}{8N\psi^2} = \frac{N}{2}\|\nabla\psi\|^2 = \tau_{\text{tot}} \quad \Longrightarrow \quad \alpha_{\text{iso}}(\mathbf{r}) = 0 \text{ identically!}$$
  Thus, $\alpha_{\text{iso}} = 0$ tells `NeuralMetaGGA` that it is inside an iso-orbital region where same-spin correlation must vanish to prevent self-correlation error.
* **Slowly Varying / Metallic / Uniform Gas Regions ($\alpha_{\text{iso}} \to 1$)**: Where $\nabla\rho_{\text{tot}} \to 0$ while $\tau_{\text{tot}} > 0$, $\tau_W \to 0$ and $\alpha_{\text{iso}} \to 1$.

#### Rung 4 Ingredient: Exact Non-Local Hartree-Fock Exchange ($E_x^{\text{HF}}$)
To eliminate non-local Self-Interaction Error (SIE), we compute the exact spin-polarized Hartree-Fock exchange energy directly from the 4-center 2-electron repulsion integrals:

$$E_x^{\text{HF}}[P^\alpha, P^\beta] = -\frac{1}{2}\sum_{i,j,k,l=1}^B P^\alpha_{il}\,P^\alpha_{jk}\,(ij|kl) \;-\; \frac{1}{2}\sum_{i,j,k,l=1}^B P^\beta_{il}\,P^\beta_{jk}\,(ij|kl)$$

In JAX, this is evaluated via:
```python
Ex_exact_a = -0.5 * jnp.einsum("il,jk,ijkl->", Pa, Pa, ERI_j)
Ex_exact_b = -0.5 * jnp.einsum("il,jk,ijkl->", Pb, Pb, ERI_j)
E_exact_exchange = Ex_exact_a + Ex_exact_b
```
Notice the crucial difference between Coulomb (`'ij,kl,ijkl->'`) and Exact Exchange (`'il,jk,ijkl->'`): for any single-electron system ($N_\alpha = 1, N_\beta = 0$), $P^\alpha = \mathbf{c}^\top \mathbf{c}$ is rank-1, so $P^\alpha_{ij} P^\alpha_{kl} = c_i c_j c_k c_l = c_i c_l c_j c_k = P^\alpha_{il} P^\alpha_{jk}$, which proves that $E_H[P^\alpha] + E_x^{\text{HF}}[P^\alpha, 0] = 0$ identically—100% cancellation of 1-electron self-interaction!

#### Inside `NeuralMetaGGA`: Feature Vector, Zero-Initialized Heads & Hybrid Mixing
At every grid point $\mathbf{r}_g$, `NeuralMetaGGA` constructs a 7-dimensional rotationally and translationally invariant feature vector compressed with $\ln(1 + \cdot)$ (`jnp.log1p`) so that core densities ($\rho \sim 10^2\text{–}10^3\text{ Bohr}^{-3}$) and valence tails ($\rho \sim 10^{-4}\text{ Bohr}^{-3}$) map onto a well-conditioned numerical scale:

$$\mathbf{x}(\mathbf{r}_g) = \begin{bmatrix} \ln(1 + \rho_{\text{tot}}) \\ \zeta = (\rho_\alpha - \rho_\beta)/\rho_{\text{tot}} \\ \ln(1 + \gamma_\alpha) \\ \ln(1 + \gamma_\beta) \\ \ln(1 + \tau_\alpha) \\ \ln(1 + \tau_\beta) \\ \alpha_{\text{iso}} \end{bmatrix} \in \mathbb{R}^7$$

This vector passes through 3 hidden layers of width 128 with `GELU` activations (`nn.gelu(nn.Dense(128)(x))`), followed by two **zero-initialized** linear output heads:

$$F_x(\mathbf{r}_g) = 1.0 + F_x^{\text{NN}}(\mathbf{x}(\mathbf{r}_g)), \qquad F_c(\mathbf{r}_g) = 1.0 + F_c^{\text{NN}}(\mathbf{x}(\mathbf{r}_g))$$

Alongside the MLP, a trainable scalar logit `alpha_ex_raw` (initialized to $-0.847$) defines the bounded global exact exchange fraction:

$$\alpha_{\text{ex}} = 0.5 \cdot \sigma(\alpha_{\text{ex,raw}}) \in (0, 0.5), \qquad \alpha_{\text{ex}}^{(0)} = 0.5 \cdot \sigma(-0.847) \approx 0.150 \;\;(15\% \text{ exact HF exchange})$$

The total Meta-GGA Global Hybrid exchange-correlation energy is assembled as:

$$E_{\text{xc}}^{\text{Hybrid}} = \alpha_{\text{ex}}\,E_x^{\text{HF}} + (1 - \alpha_{\text{ex}})\sum_{g=1}^G w_g\,\rho_{\text{tot}}(\mathbf{r}_g)\,\varepsilon_x^{\text{LSDA}}(\mathbf{r}_g)\,F_x(\mathbf{r}_g) + \sum_{g=1}^G w_g\,\rho_{\text{tot}}(\mathbf{r}_g)\,\varepsilon_c^{\text{Chachiyo}}(\mathbf{r}_g)\,F_c(\mathbf{r}_g)$$

### 5. Bi-Level Optimization via the Hellmann-Feynman Theorem

Training a density functional inside a Kohn-Sham solver is a **bi-level optimization problem**:
* **Inner Problem (Physics / Orbital Relaxation)**: For current neural parameters $\theta$, find the ground-state orbital parameters $W^*(\theta) = (W_\alpha^*(\theta), W_\beta^*(\theta))$ that minimize the Kohn-Sham energy:
  $$W^*(\theta) = \underset{W_\alpha, W_\beta}{\arg\min}\; E(W_\alpha, W_\beta; \theta)$$
* **Outer Problem (Meta-Learning against Coupled Cluster Reference)**: Minimize the discrepancy between the converged Kohn-Sham energy $E(W^*(\theta); \theta)$ and the gold-standard $\text{CCSD(T)}$ energy across the training set $\mathcal{M}_{\text{train}}$:
  $$\min_\theta \; \mathcal{L}(\theta) = \frac{1}{|\mathcal{M}_{\text{train}}|} \sum_{m \in \mathcal{M}_{\text{train}}} \left| E_m(W_m^*(\theta); \theta) - E_m^{\text{CCSD(T)}} \right|$$

**Why We Do Not Need to Backpropagate Through the 600-Step `jax.lax.scan` Loop**:
Naively differentiating $E(W^*(\theta); \theta)$ with respect to $\theta$ via reverse-mode autodiff through 600 steps of Adam (`jax.lax.scan`) would require storing 600 intermediate QR decompositions, density matrices, and grid evaluations in memory, and can suffer from exploding/oscillating unrolled optimization gradients.

Instead, because $W^*(\theta)$ is a stationary minimum of the inner variational problem, the first derivative of the energy with respect to the orbital parameters vanishes at convergence:

$$\left.\frac{\partial E(W; \theta)}{\partial W}\right|_{W = W^*(\theta)} = \mathbf{0}$$

Applying the multivariable chain rule to $E^*( \theta ) \equiv E(W^*(\theta); \theta)$ yields the **Hellmann-Feynman Theorem**:

$$\frac{d E^*(\theta)}{d\theta} = \underbrace{\left.\frac{\partial E(W; \theta)}{\partial W}\right|_{W^*}}_{=\,\mathbf{0}\text{ (Stationary Condition)}} \cdot \frac{d W^*(\theta)}{d\theta} + \left.\frac{\partial E(W; \theta)}{\partial \theta}\right|_{W = W^*(\theta)} = \left.\frac{\partial E(W^*(\theta); \theta)}{\partial \theta}\right|_{W^* \text{ held fixed}}$$

In code, this reduces the outer gradient calculation per molecule to three clean lines:
```python
Wa_opt, Wb_opt, _ = solve_dft_core(m, nn_params, use_nn=True)

def meta_loss(params):
    return jnp.abs(engines[m][0](Wa_opt, Wb_opt, params, use_nn=True) - ccsd_targets[m])

loss, grads = value_and_grad(meta_loss)(nn_params)
```
This computes the exact first-order meta-gradient in **1 forward/backward evaluation** ($O(1)$ memory overhead), while flowing gradients into both the semi-local MLP parameters (`Fx`, `Fc`) and the global exact exchange mixing parameter (`alpha_ex_raw`).

## Summary of Engineering & Physical Bug Fixes

Both scripts in this repository include critical fixes discovered during mathematical and numerical auditing:

| # | Bug / Subtle Pitfall | Affected Component | Impact Before Fix | Fix Implemented |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Flax Scalar Parameter Initialization Shape** | `NeuralMetaGGA.__call__` | Passing `0.0` or omitting shape in `self.param('alpha_ex_raw', ...)` raises a `TypeError` in Flax | Passed empty tuple `()` as the shape argument: `self.param('alpha_ex_raw', jax.nn.initializers.constant(-0.847), ())` |
| **2** | **Exact Hartree-Fock Exchange Contraction Indices** | `build_energy_fn` | Using `'ij,kl,ijkl->'` computes the direct Coulomb (Hartree) integral $J$ instead of the Fock exchange integral $K$ | Used exact Fock exchange index pairing `'il,jk,ijkl->'`: `-0.5 * jnp.einsum('il,jk,ijkl->', Pa, Pa, ERI_j)` |
| **3** | **PySCF AO Gradient Axis Alignment** | `get_pyscf_baselines` & `build_energy_fn` | `numint.eval_ao(..., deriv=1)[1:]` has shape `(3, G, B)`; contracting directly with `Ca` `(N_occ, B)` mismatches dimensions | Transposed AO gradients to `(G, B, 3)` (`ao_vals[1:].transpose(1, 2, 0)`) and contracted via `jnp.einsum('gbx,ib->xgi', phi_gj, Ca)` |
| **4** | **Missing Correlation in `use_nn=False` Baseline** | `build_energy_fn` (`else` branch) | `else: E_xc_total = jnp.sum(W_j * (rho_a + rho_b) * eps_x_lsda)` dropped `eps_c_lsda`, turning Phase 1 into Slater Exchange-Only and inflating baseline error by $150\text{–}350\text{ kcal/mol}$ | Added `+ eps_c_lsda` in `d4ft_metagga_hybrid_complete.py`: `E_xc_total = jnp.sum(W_j * (rho_a + rho_b) * (eps_x_lsda + eps_c_lsda))` and set `mf_lda.xc = 'lda,vwn'` |
| **5** | **Unbounded Negative $\alpha_{\text{iso}}$ in Density Tails** | `NeuralMetaGGA.__call__` | Far from nuclei ($r > 6\text{ Bohr}$), $\rho_{\text{tot}}$ hits the $10^{-12}$ clip floor before $\Vert{}\nabla\rho_{\text{tot}}\Vert{}^2$ vanishes, causing $\tau_W > \tau_{\text{tot}}$ and $\alpha_{\text{iso}} \ll 0$ | Clipped `alpha_iso = jnp.clip((tau_tot - tau_w) / tau_tot, 0.0, 1.0)` and enforced non-negative `tau_a, tau_b` inside `jnp.log1p` |
| **6** | **Householder QR Sign-Flip Gauge Discontinuity** | `stiefel_qr` / `energy_fn` | Standard `jnp.linalg.qr` allows arbitrary sign flips on columns of $Q$ when $W$ crosses reflection hyperplanes, perturbing Adam momentum buffers | Multiplied columns of $Q$ by `phases = jnp.where(jnp.diagonal(R) >= 0, 1.0, -1.0)` in `stiefel_qr` |
| **7** | **Empty Beta Spin Channel ($N_\beta = 0$) Safety** | `stiefel_qr` & `build_energy_fn` | Single-electron systems (e.g., $\text{H}$ atom, $N_\alpha=1, N_\beta=0$) have `Wb.shape == (0, B)` | Added explicit `if W.shape[0] == 0:` guards returning zero-shaped orbital and density arrays cleanly |

## Benchmark Molecules & Experimental Protocols

### Benchmark 1: High-Precision Meta-GGA Global Hybrid Training & Zero-Shot Extrapolation (`d4ft_metagga_hybrid_complete.py`)
* **Basis Set**: `def2-SVP` (polarized split-valence) with `grids.level = 4` Becke quadrature and `FP64` precision.
* **Gold-Standard Reference**: Unrestricted/Restricted Coupled Cluster with Single, Double, and Perturbative Triple excitations ($\text{CCSD(T)}/\text{def2-SVP}$).
* **Training Set (6 Molecules/Atoms spanning diverse spin states and bond orders)**:
  * $\text{H}_2$ ($d = 0.74\text{ \AA}$, singlet covalent $\sigma$ bond)
  * $\text{He}$ (closed-shell $1s^2$ noble gas atom, pure iso-orbital $\alpha_{\text{iso}} = 0$)
  * $\text{LiH}$ ($d = 1.59\text{ \AA}$, ionic/polar $\sigma$ bond with core-valence separation)
  * $\text{N}$ ($2S = 3$, open-shell quartet atom with three half-filled $2p$ orbitals)
  * $\text{H}_2\text{O}$ ($C_{2v}$ bent polar covalent molecule with two lone pairs)
  * $\text{CH}_4$ ($T_d$ tetrahedral $sp^3$ hydrocarbon)
* **Zero-Shot Extrapolation Test Set (3 Unseen Molecules with distinct chemical environments)**:
  * $\text{CO}$ ($d = 1.13\text{ \AA}$, polar triple bond—never seen during training)
  * $\text{F}_2$ ($d = 1.41\text{ \AA}$, notoriously difficult lone-pair-rich weak covalent bond where standard LDA/GGA heavily overbinds and HF underbinds)
  * $\text{HF}$ ($d = 0.92\text{ \AA}$, strongly polar hydrogen-fluoride bond)

### Benchmark 2: Original D4FT 3-Experiment Suite (`d4ft_kohn_sham_dft_final.py`)
* **Experiment 1 (Ground-State Energy Accuracy)**: Evaluates Method 1 (`NeuralXC`), Method 2 (Standard D4FT + LDA), and Method 3 (PySCF RKS LDA) across $\text{H}_2, \text{CH}_4, \text{H}_2\text{O}, \text{O}_2, \text{C}_2\text{H}_6\text{O}$ (ethanol), and $\text{C}_6\text{H}_6$ (benzene) in `6-31g`, generating `exp1_energy_comparison.png`.
* **Experiment 2 (Magnetism & Spin-State Ordering)**: Compares singlet ($2S=0$) vs. triplet ($2S=2$) total energies on $\text{O}_2$ and $\text{CO}_2$ using spin-polarized LSDA, verifying that D4FT correctly predicts $\text{O}_2$ to be **paramagnetic** ($E_{S=1} < E_{S=0}$) and $\text{CO}_2$ to be **diamagnetic** ($E_{S=0} < E_{S=1}$), generating `exp2_magnetism.png`.
* **Experiment 3 (Fullerene Scalability via Mini-Batch Quadrature)**: Benchmarks convergence wall-clock time and epoch trajectories on $\text{C}_{20}$ (120 electrons), $\text{C}_{60}$ (360 electrons), and $\text{C}_{80}$ (480 electrons) using stochastic mini-batch sampling (`batch_size = 4096`) over the 3D quadrature grid, generating `exp3_scalability.png` and `exp3_time_comparison.png`.

## Installation & Quickstart

### 1. Clone the Repository & Install Dependencies

```bash
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

*(If running on Google Colab or a CUDA GPU machine, install the GPU-accelerated wheel for JAX via `pip install -U "jax[cuda12]"` alongside `pyscf optax flax numpy scipy pandas matplotlib`.)*

### 2. Run the Flagship FP64 Neural Meta-GGA Global Hybrid Benchmark

```bash
python3 d4ft_metagga_hybrid_complete.py
```

This script executes three sequential phases:
1. **Exact Integral & $\text{CCSD(T)}/\text{def2-SVP}$ Baseline Generation**: Builds all 9 molecules in PySCF, runs RKS/UKS and RHF/UHF $\to$ CCSD(T), and extracts analytical `(S, T, V, ERI, E_nuc)` tensors and level-4 Becke grids.
2. **Phase 1 (Pure LSDA Validation) & Phase 2 (Bi-Level Meta-GGA Hybrid Training)**: Validates the baseline D4FT solver (`use_nn=False`), compiles the XLA `jax.lax.scan` graphs, and trains `NeuralMetaGGA` for 101 outer epochs (600 inner orbital steps per molecule per epoch), logging the training Mean Absolute Error (in `kcal/mol`) and learned exact exchange percentage (`alpha_ex * 100%`) every 20 epochs.
3. **Phase 3 (Zero-Shot Extrapolation Report)**: Evaluates the trained functional across both training molecules and unseen test molecules ($\text{CO}, \text{F}_2, \text{HF}$, marked with `*`) and prints the final comparison table in `kcal/mol`.

### 3. Run the Original D4FT Multi-Experiment Suite

```bash
python3 d4ft_kohn_sham_dft_final.py
```

## References & Foundational Literature

1. **D4FT (Direct Variational KS-DFT & Stiefel QR Reparameterization)**:
   T. Li, M. Lin, Z. Hu, K. Zheng, G. Vignale, K. Kawaguchi, A. H. Castro Neto, K. S. Novoselov, and S. Yan, *"D4FT: A Deep Learning Approach to Kohn-Sham Density Functional Theory"*, International Conference on Learning Representations (ICLR 2023). [arXiv:2303.00399](https://arxiv.org/abs/2303.00399) | [GitHub: sail-sg/d4ft](https://github.com/sail-sg/d4ft)
2. **Differentiable KS-DFT & Implicit Exchange-Correlation Learning**:
   M. F. Kasim and S. M. Vinko, *"Learning the Exchange-Correlation Functional from Nature with Fully Differentiable Density Functional Theory"*, Physical Review Letters **127**, 126403 (2021). [DOI: 10.1103/PhysRevLett.127.126403](https://link.aps.org/doi/10.1103/PhysRevLett.127.126403)
3. **DeepMind DM21 Neural Meta-GGA Hybrid Functional**:
   J. Kirkpatrick *et al.*, *"Pushing the frontiers of density functionals by solving the fractional electron problem"*, Science **374**, 1385–1389 (2021). [PubMed: 34882476](https://pubmed.ncbi.nlm.nih.gov/34882476/)
4. **Chachiyo Uniform Electron Gas Correlation Functional**:
   T. Chachiyo, *"Communication: Simple and accurate uniform electron gas correlation energy for the full range of densities"*, The Journal of Chemical Physics **145**, 021101 (2016). [DOI: 10.1063/1.4958669](https://pubs.aip.org/aip/jcp/article/145/2/021101/907773/Communication-Simple-and-accurate-uniform-electron)
5. **Vosko-Wilk-Nusair (VWN5) Correlation Functional**:
   S. H. Vosko, L. Wilk, and M. Nusair, *"Accurate spin-dependent electron liquid correlation energies for local spin density calculations: a critical analysis"*, Canadian Journal of Physics **58**, 1200–1211 (1980). [DOI: 10.1139/p80-159](https://cdnsciencepub.com/doi/10.1139/p80-159)
6. **Role of Exact Exchange in Hybrid DFT**:
   A. D. Becke, *"Density-functional thermochemistry. III. The role of exact exchange"*, The Journal of Chemical Physics **98**, 5648–5652 (1993). [Paper PDF](http://hanicka.uochb.cas.cz/~bour/prednaska/becke.pdf)
7. **Regularized Meta-GGA & Iso-Orbital Indicator ($\alpha_{\text{iso}}$)**:
   J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew, and J. Sun, *"Accurate and Numerically Efficient $r^2\text{SCAN}$ Meta-Generalized Gradient Approximation"*, The Journal of Physical Chemistry Letters **11**, 8208–8215 (2020). [DOI: 10.1021/acs.jpclett.0c02405](https://pubs.acs.org/doi/pdf/10.1021/acs.jpclett.0c02405)
8. **PySCF Electronic Structure Package**:
   Q. Sun *et al.*, *"Recent developments in the PySCF program package"*, The Journal of Chemical Physics **153**, 024109 (2020). [DOI: 10.1063/5.0006074](https://pubs.aip.org/aip/jcp/article-abstract/153/2/024109/1061482)

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.
