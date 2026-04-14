"""Standalone benchmark for the 2D snaked-MPS Heisenberg solver.

Run from the backend/ directory:
    python -m benchmark_2d

Compares SVD vs TurboQuant truncation across chi_max values on a 3x3 lattice.
Requires: numpy, scipy
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import time
import json
from app.solver_2d import Solver2D


def run_benchmark_2d(L=3, W=3, sweeps=2):
    chi_values = [8, 16, 32, 64]
    results = []

    for chi in chi_values:
        print(f"  chi_max={chi} on {L}x{W} lattice ...", flush=True)

        # SVD solver
        svd_solver = Solver2D(L=L, W=W, chi_max=chi, J=1.0)
        t0 = time.perf_counter()
        svd_time, svd_energy = svd_solver.run_sweep(mode="svd", sweeps=sweeps)
        t_svd = time.perf_counter() - t0

        # TurboQuant solver
        tq_solver = Solver2D(L=L, W=W, chi_max=chi, J=1.0)
        t0 = time.perf_counter()
        tq_time, tq_energy = tq_solver.run_sweep(mode="turboquant", sweeps=sweeps)
        t_tq = time.perf_counter() - t0

        results.append({
            "chi_max":          chi,
            "svd_energy_site":  round(svd_energy, 6),
            "tq_energy_site":   round(tq_energy,  6),
            "svd_trunc_avg_s":  round(svd_time,   8),
            "tq_trunc_avg_s":   round(tq_time,    8),
            "svd_total_s":      round(t_svd,       4),
            "tq_total_s":       round(t_tq,        4),
        })

        print(f"    SVD  E/site={svd_energy:.5f}  avg_trunc={svd_time*1e6:.1f} us")
        print(f"    TQ   E/site={tq_energy:.5f}  avg_trunc={tq_time*1e6:.1f} us")

    output = {"lattice": f"{L}x{W}", "sweeps": sweeps, "results": results}
    with open("benchmark_results_2d.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Results saved to benchmark_results_2d.json")
    return results


if __name__ == "__main__":
    print("=== TurboQuant 2D DMRG Benchmark ===")
    run_benchmark_2d(L=3, W=3, sweeps=2)
