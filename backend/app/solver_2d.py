import numpy as np
import time
from scipy.sparse.linalg import eigsh, LinearOperator

def fwht_vectorized(a):
    """Vectorized FWHT for columns of a."""
    n, m = a.shape
    if n == 1: return a
    next_pow2 = 1 << (n - 1).bit_length()
    if next_pow2 > n:
        a = np.pad(a, ((0, next_pow2 - n), (0, 0)))
        n = next_pow2
    a = a.reshape(2, n // 2, m)
    left = fwht_vectorized(a[0] + a[1])
    right = fwht_vectorized(a[0] - a[1])
    return np.concatenate([left, right], axis=0)

class Solver2D:
    def __init__(self, L=4, W=4, chi_max=32, J=1.0):
        self.L = L
        self.W = W
        self.n_sites = L * W
        self.chi_max = chi_max
        self.J = J
        self.d = 2 
        
        self.Sz = 0.5 * np.array([[1, 0], [0, -1]])
        self.Sp = np.array([[0, 1], [0, 0]])
        self.Sm = np.array([[0, 0], [1, 0]])
        self.I = np.eye(2)
        
        # Build 2D MPO along snake path
        self.mpos = self._build_2d_snaked_mpo()
        
        # Initialize MPS
        self.mps = [np.random.randn(1, self.d, 1) for _ in range(self.n_sites)]
        self._right_canonicalize()
        
        self.L_env = [None] * (self.n_sites + 1)
        self.R_env = [None] * (self.n_sites + 1)
        self.L_env[0] = np.ones((1, 1, 1))
        self.R_env[self.n_sites] = np.ones((1, 1, 1))

    def _build_2d_snaked_mpo(self):
        # Systematic MPO for 2D Heisenberg on a snake path.
        # Bond dimension of MPO grows with W to handle vertical bonds.
        mpos = []
        for i in range(self.n_sites):
            D_mpo = 3 * self.W + 2
            W_mpo = np.zeros((D_mpo, D_mpo, 2, 2))
            W_mpo[0, 0] = self.I
            W_mpo[-1, -1] = self.I
            
            # Local terms and horizontal neighbors
            W_mpo[1, 0] = self.Sz
            W_mpo[-1, 1] = self.J * self.Sz
            W_mpo[2, 0] = self.Sp
            W_mpo[-1, 3] = self.J * 0.5 * self.Sm
            W_mpo[3, 0] = self.Sm
            W_mpo[-1, 2] = self.J * 0.5 * self.Sp
            
            # Vertical neighbors (delay lines in MPO)
            for k in range(1, self.W):
                W_mpo[3*k+1, 3*(k-1)+1] = self.I
                W_mpo[3*k+2, 3*(k-1)+2] = self.I
                W_mpo[3*k+3, 3*(k-1)+3] = self.I
            
            # Close the vertical loop at the correct distance
            W_mpo[-1, 3*self.W-2] = self.J * self.Sz
            W_mpo[-1, 3*self.W-1] = self.J * 0.5 * self.Sm
            W_mpo[-1, 3*self.W] = self.J * 0.5 * self.Sp
            
            mpos.append(W_mpo)
            
        mpos[0] = mpos[0][-1:, :, :, :]
        mpos[-1] = mpos[-1][:, :1, :, :]
        return mpos

    def _right_canonicalize(self):
        for i in range(self.n_sites - 1, 0, -1):
            cl, d, cr = self.mps[i].shape
            mat = self.mps[i].reshape(cl, d * cr)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            self.mps[i] = vh.reshape(-1, d, cr)
            self.mps[i-1] = np.einsum('ijk,kl,l->ijl', self.mps[i-1], u, s)

    def update_env_right(self, site):
        self.R_env[site] = np.einsum('ijk, lmnj, pnr, rmk -> pli', self.mps[site], self.mpos[site], self.mps[site].conj(), self.R_env[site+1], optimize=True)

    def update_env_left(self, site):
        self.L_env[site+1] = np.einsum('pli, ijk, lmnj, pnr -> rmk', self.L_env[site], self.mps[site], self.mpos[site], self.mps[site].conj(), optimize=True)

    def solve_local_2site(self, i, j):
        L, R = self.L_env[i], self.R_env[j+1]
        W1, W2 = self.mpos[i], self.mpos[j]
        shape = (self.mps[i].shape[0], self.d, self.d, self.mps[j].shape[2])
        size = np.prod(shape)

        def matvec(v):
            psi = v.reshape(shape)
            res = np.einsum('pqi, qrmj, rsnk, usl, ijkl -> pmnu', L, W1, W2, R, psi, optimize=True)
            return res.ravel()

        H_eff = LinearOperator((size, size), matvec=matvec)
        psi_init = np.einsum('ijk, klm -> ijlm', self.mps[i], self.mps[j], optimize=True).ravel()
        vals, vecs = eigsh(H_eff, k=1, which='SA', v0=psi_init, tol=1e-5)
        return vecs[:, 0].reshape(shape), vals[0]

    def turboquant_truncate(self, mat, chi_target):
        rows, cols = mat.shape
        next_pow2 = 1 << (rows - 1).bit_length()
        mat_padded = np.pad(mat, ((0, next_pow2 - rows), (0, 0)))
        mat_rot = fwht_vectorized(mat_padded)
        chi = min(chi_target, cols)
        norms = np.linalg.norm(mat_rot, axis=0)
        idx = np.argsort(norms)[-chi:]
        res_rot = mat_rot[:, idx]
        res = fwht_vectorized(res_rot) / next_pow2
        return res[:rows, :]

    def run_sweep(self, mode="svd"):
        for i in range(self.n_sites - 1, 0, -1):
            self.update_env_right(i)
            
        sweep_times = []
        energy_sum = 0
        
        for i in range(self.n_sites - 1):
            psi_2site, e = self.solve_local_2site(i, i+1)
            cl, d1, d2, cr = psi_2site.shape
            mat = psi_2site.reshape(cl * d1, d2 * cr)
            
            t0 = time.perf_counter()
            if mode == "turboquant":
                U_hat = self.turboquant_truncate(mat, self.chi_max)
                U, _ = np.linalg.qr(U_hat)
                V = U.T @ mat
                S = np.ones(U.shape[1])
            else:
                U, S, V = np.linalg.svd(mat, full_matrices=False)
                chi = min(self.chi_max, len(S))
                U, S, V = U[:, :chi], S[:chi], V[:chi, :]
            
            sweep_times.append(time.perf_counter() - t0)
            self.mps[i] = U.reshape(cl, d1, -1)
            self.mps[i+1] = (np.diag(S) @ V).reshape(-1, d2, cr)
            self.update_env_left(i)
            energy_sum = e
            
        return np.mean(sweep_times), energy_sum / self.n_sites
