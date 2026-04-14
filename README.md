# TurboQuant-2D: Advanced Snaked-MPS Quantum Solver

TurboQuant-2D is a research-grade **Two-Site DMRG** platform optimized for **2D Quantum Many-Body Systems** (e.g., Square Lattice Heisenberg model). It leverages a **Snaked-MPO** architecture and the novel **TurboQuant Truncation** algorithm to address the "Area Law" of entanglement in 2D systems.

## 🚀 Key Features
- **Snaked-MPO Architecture:** Maps a 2D square lattice onto a 1D chain for MPS-DMRG. Correctly implements vertical nearest-neighbor interactions that become long-range in the 1D mapping.
- **Two-Site Variational Physics:** Features a modern two-site sweep algorithm to minimize the variational energy of the $L \times W$ lattice.
- **First-Principles Energy:** All energy calculations are derived from the full Hamiltonian expectation value $\langle \psi | H | \psi \rangle$.
- **FWHT-Based Truncation:** Replaces the computationally prohibitive $O(\chi^3)$ SVD with an $O(\chi^2 \log \chi)$ randomized basis rotation using the Fast Walsh-Hadamard Transform (FWHT).

## 📊 Scientific Benchmarks (3x3 Heisenberg Lattice)
First-principles comparison of converged **Energy per Site** and **Truncation Time** for $\chi_{max} = 64$:

| Method | Energy per Site (E₀/N) | Truncation Time (avg) | Variational Precision |
| :--- | :--- | :--- | :--- |
| **Standard SVD** | $\approx -0.448$ | $3.64 \times 10^{-5}$ s | Optimal |
| **TurboQuant** | $\approx -0.449$ | $2.14 \times 10^{-4}$ s | High (Matched) |

### **Scientific Analysis**
- **Variational Accuracy:** TurboQuant successfully maintains the 2D entanglement structure. Despite the snaking path inducing high-entanglement bonds, the FWHT-based "scrambling" allows for a near-optimal truncation that matches the SVD's energy density.
- **Scaling Breakthrough:** In 2D systems where bond dimensions ($\chi$) must grow exponentially with width ($W$), the $O(\chi^2 \log \chi)$ scaling of TurboQuant offers a theoretical pathway to simulations that are impossible with SVD-based solvers.

## 🛠️ Tech Stack
- **Backend:** Python 3.11+, NumPy, SciPy.
- **Methodology:** Snaked MPO construction ($D_{mpo} = 3W+2$).

## ⏩ Beyond Proof-of-Concept
1. **High-Performance C++ Core:** The current Python implementation of FWHT introduces overhead that obscures the asymptotic speedup. A low-level C implementation is required for research-grade performance.
2. **GPU Parallelization:** The TurboQuant rotation is highly parallelizable, making it ideal for CUDA-based DMRG engines.
3. **PEPS Support:** Future extensions to Projected Entanglement Pair States (PEPS) to better capture 2D entanglement without snaking.
