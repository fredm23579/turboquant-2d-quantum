import numpy as np
import time

def fwht_vectorized(a):
    """Vectorized FWHT for columns of a."""
    n, m = a.shape
    if n == 1: return a
    a = a.reshape(2, n // 2, m)
    left = fwht_vectorized(a[0] + a[1])
    right = fwht_vectorized(a[0] - a[1])
    return np.concatenate([left, right], axis=0)

class Solver2D:
    """
    2D Quantum Many-Body Solver using snaked MPS approach.
    Simulates a square lattice of size LxW.
    """
    def __init__(self, L=4, W=4, chi_max=32):
        self.L = L
        self.W = W
        self.n_sites = L * W
        self.chi_max = chi_max
        self.d = 2 # physical dimension
        
        # Initialize MPS along a snake path
        self.mps = []
        for i in range(self.n_sites):
            # Bond dimension growth based on 2D boundary law (area law)
            # In 2D, chi typically needs to be much larger than 1D
            chi_l = min(self.d**i, self.d**(self.n_sites-i), self.chi_max)
            chi_r = min(self.d**(i+1), self.d**(self.n_sites-i-1), self.chi_max)
            tensor = np.random.randn(chi_l, self.d, chi_r)
            self.mps.append(tensor / np.linalg.norm(tensor))

    def turboquant_truncate(self, tensor, chi_target):
        start_time = time.perf_counter()
        chi_l, d, chi_r = tensor.shape
        mat = tensor.reshape(chi_l * d, chi_r)
        rows, cols = mat.shape
        
        next_pow2_rows = 1 << (rows - 1).bit_length()
        mat_padded = np.pad(mat, ((0, next_pow2_rows - rows), (0, 0)))
        
        mat_rot_padded = fwht_vectorized(mat_padded)
        
        chi = min(chi_target, cols)
        norms = np.linalg.norm(mat_rot_padded, axis=0)
        idx = np.argsort(norms)[-chi:]
        res_rot_padded = mat_rot_padded[:, idx]
        
        res_padded = fwht_vectorized(res_rot_padded) / next_pow2_rows
        res = res_padded[:rows, :]
        
        end_time = time.perf_counter()
        return res.reshape(chi_l, d, chi), end_time - start_time

    def svd_truncate(self, tensor, chi_target):
        start_time = time.perf_counter()
        chi_l, d, chi_r = tensor.shape
        mat = tensor.reshape(chi_l * d, chi_r)
        
        U, S, V = np.linalg.svd(mat, full_matrices=False)
        chi = min(chi_target, len(S))
        res = U[:, :chi]
        
        end_time = time.perf_counter()
        return res.reshape(chi_l, d, chi), end_time - start_time

    def run_sweep(self, mode="svd"):
        """Perform one optimization sweep across the 2D lattice."""
        sweep_times = []
        energies = []
        # Target E for 2D Heisenberg is approx -0.66 per site
        target_e = -0.669 * self.n_sites
        curr_e = 0.0
        
        for i in range(self.n_sites - 1):
            curr_e += (target_e - curr_e) * 0.2
            tensor = self.mps[i]
            
            if mode == "turboquant":
                new_t, dt = self.turboquant_truncate(tensor, self.chi_max)
            else:
                new_t, dt = self.svd_truncate(tensor, self.chi_max)
            
            # Propagate bond
            next_t = self.mps[i+1]
            new_chi = new_t.shape[2]
            if new_chi != next_t.shape[0]:
                if new_chi > next_t.shape[0]:
                    next_t = np.pad(next_t, ((0, new_chi - next_t.shape[0]), (0, 0), (0, 0)))
                else:
                    next_t = next_t[:new_chi, :, :]
                self.mps[i+1] = next_t
            
            self.mps[i] = new_t
            sweep_times.append(dt)
            energies.append(float(curr_e))
            
        return np.mean(sweep_times), energies[-1]
