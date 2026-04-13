# 🌌 TurboQuant-2D — High-Performance 2D Quantum Solver

**TurboQuant-2D** is an advanced scientific simulation platform designed to solve **2D Quantum Many-Body Hamiltonians** (e.g., 2D Heisenberg Square Lattice) with unprecedented computational efficiency. By leveraging the **TurboQuant** vector quantization algorithm, this solver bypasses the $O(\chi^3)$ bottleneck of traditional DMRG/PEPS methods, achieving near-linear $O(\chi \log \chi)$ scaling for bond dimension truncation.

[![Tech Stack](https://img.shields.io/badge/Stack-Python%20%2B%20NumPy-blue?style=for-the-badge&logo=python)](https://github.com/fredm23579/turboquant-2d-quantum)
[![Performance](https://img.shields.io/badge/Speedup-6.3x%20at%20chi=512-success?style=for-the-badge)](backend/EMPIRICAL_RESULTS_2D.md)
[![Tests](https://img.shields.io/badge/Tests-80+%20Passing-green?style=for-the-badge)](backend/tests/test_solver_2d.py)

---

## 📈 Empirical Complexity Advantage (2D)

In 2D quantum systems, the "Area Law" of entanglement forces bond dimensions ($\chi$) to be significantly higher than in 1D. Traditional SVD methods scale cubically, making large-scale 2D simulations impossible on standard hardware. **TurboQuant-2D** fundamentally changes this curve.

### 2D Truncation Speedup ($\chi$ vs Time)
```text
Time (ms)
  ^
  |                                      Standard SVD O(χ³)
  |                                            /
  |                                           /
300|                                         /
  |                                         /
  |                                        /
  |                                       /
150|                                     /
  |                                     /
  |                                    /
  |                                   /   TurboQuant O(χ log χ)
  |                                  /  _______------
  |_________________________________/_/________________> Bond Dim (χ)
  0        128      256      384      512
```

### Quantitative Highlights
- **6.3x Faster** truncation at $\chi=512$.
- **Sub-cubic scaling** confirmed empirically across LxW lattices.
- **Numerical Stability** maintained across 80+ rigorous unit tests.

---

## 🚀 Key Features

- **2D Lattice Support**: Solves LxW square lattices using an optimized snaked-MPS strategy.
- **Turbo-Charged Truncation**: Replaces expensive SVD with vectorized Fast Walsh-Hadamard Transforms (FWHT).
- **Complexity Benchmarking**: Integrated tools to measure and report time complexity scaling in real-time.
- **Robust Physics**: Implements local spin-1/2 operators and energy convergence monitoring.

## 🛠️ Built With

- **Python 3.13**: Core simulation logic.
- **NumPy**: High-performance vectorized linear algebra.
- **PyTest**: Comprehensive testing suite (parameterized for 80+ edge cases).
- **Vectorized FWHT**: Custom $O(N \log N)$ implementation for maximum throughput.

---

## 🏁 Getting Started

### 1. Prerequisites
- Python 3.10+
- NumPy & PyTest

### 2. Run Benchmarks
```bash
cd backend
python benchmark_2d.py
```

### 3. Run Unit Tests
```bash
cd backend
python -m pytest tests/test_solver_2d.py
```

---

## 📄 License
MIT License.

*Pushing the boundaries of 2D quantum simulation with vector quantization.*
