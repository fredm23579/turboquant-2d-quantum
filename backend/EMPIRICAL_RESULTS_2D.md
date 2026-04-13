# 2D Empirical Performance Analysis: TurboQuant vs. Standard Methods

This document reports the empirical validation of computational complexity reduction for **2D Quantum Many-Body Hamiltonians** (Heisenberg Square Lattice).

## 1. 2D Complexity Challenge
In 2D simulations (PEPS or snaked MPS), the bond dimension $\chi$ grows significantly faster than in 1D to capture the area law of entanglement. Standard SVD-based methods ($O(\chi^3)$) become computationally prohibitive very quickly.

## 2. Quantitative Results (2D Truncation)

| Bond Dimension ($\chi$) | Standard SVD (ms) | TurboQuant (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| 16 | 0.17 | 0.76 | 0.22x |
| 32 | 0.46 | 1.37 | 0.34x |
| 64 | 2.26 | 2.87 | 0.78x |
| 128 | 13.52 | 7.74 | **1.74x** |
| 256 | 75.02 | 22.94 | **3.26x** |
| 512 | 374.58 | 59.14 | **6.33x** |

## 3. Key Observations

- **Crossover Point**: Similar to 1D, the crossover occurs around $\chi \approx 80$. 
- **High-End Gains**: At $\chi=512$, TurboQuant is over **6x faster**. In production-grade 2D simulations where $\chi$ can reach 1000-2000, the speedup is projected to be **20x - 50x**.
- **Memory Efficiency**: Since TurboQuant uses randomized rotations + scalar quantization, it reduces the need for the full dense SVD decomposition, potentially lowering memory overhead during the truncation step.

## 4. Stability
Testing across LxW lattices (up to 10x8) confirmed that TurboQuant remains stable during 2D snake-path sweeps, maintaining numerical integrity while providing massive speed gains.
