import numpy as np
import time
import json
from app.solver_2d import Solver2D

def benchmark_2d():
    print("🚀 Starting 2D Complexity Benchmark...")
    results = []
    # In 2D, bond dimension chi grows much faster than 1D.
    # Typical values for small lattices are 32-256.
    chi_values = [16, 32, 64, 128, 256, 512]
    
    for chi in chi_values:
        print(f"Testing Bond Dimension chi={chi} on 4x4 Lattice...")
        solver = Solver2D(L=4, W=4, chi_max=chi)
        # Mock a 2D tensor (chi, d, chi)
        tensor = np.random.randn(chi, 2, chi)
        
        # SVD Benchmark
        start = time.perf_counter()
        for _ in range(5):
            solver.svd_truncate(tensor, chi // 2)
        t_svd = (time.perf_counter() - start) / 5.0
        
        # TurboQuant Benchmark
        start = time.perf_counter()
        for _ in range(5):
            solver.turboquant_truncate(tensor, chi // 2)
        t_tq = (time.perf_counter() - start) / 5.0
        
        results.append({
            "chi": chi,
            "t_svd": t_svd,
            "t_tq": t_tq,
            "speedup": t_svd / t_tq
        })
        
    with open("benchmark_results_2d.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ 2D Benchmark Complete. Results saved.")

if __name__ == "__main__":
    benchmark_2d()
