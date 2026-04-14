import numpy as np
import time
from scipy.sparse.linalg import eigsh, LinearOperator


def fwht_vectorized(a):
    """Vectorized Fast Walsh-Hadamard Transform applied to columns of matrix a."""
    n, m = a.shape
    if n == 1:
        return a
    next_pow2 = 1 << (n - 1).bit_length() if n > 1 else 1
    if next_pow2 > n:
        a = np.pad(a, ((0, next_pow2 - n), (0, 0)))
        n = next_pow2
    a = a.reshape(2, n // 2, m)
    left  = fwht_vectorized(a[0] + a[1])
    right = fwht_vectorized(a[0] - a[1])
    return np.concatenate([left, right], axis=0)


class Solver2D:
    """
    Two-site DMRG solver for the 2D Heisenberg square-lattice model:
        H = J * sum_{<i,j>} [ Sz_i Sz_j + 0.5*(Sp_i Sm_j + Sm_i Sp_j) ]

    The 2D lattice (L rows x W columns) is mapped onto a 1D chain via a
    snake path.  Vertical nearest-neighbour bonds become long-range bonds
    of range W in the 1D chain and are handled by delay-line segments in
    the MPO with bond dimension D_mpo = 3*W + 2.

    Truncation backends:
      'svd'         - standard O(chi^3) SVD
      'turboquant'  - O(chi^2 log chi) FWHT-based basis rotation
    """

    def __init__(self, L=3, W=3, chi_max=32, J=1.0):
        self.L      = L
        self.W      = W
        self.n_sites = L * W
        self.chi_max = chi_max
        self.J      = J
        self.d      = 2

        # Spin-1/2 operators
        self.Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=float)
        self.Sp = np.array([[0, 1], [0, 0]], dtype=float)
        self.Sm = np.array([[0, 0], [1, 0]], dtype=float)
        self.I  = np.eye(2, dtype=float)

        # Build MPO list, initialize MPS, build environments
        self.mpos  = self._build_2d_snaked_mpo()
        self.mps   = [np.random.randn(1, self.d, 1).astype(float)
                      for _ in range(self.n_sites)]
        self._right_canonicalize()

        self.L_env = [None] * (self.n_sites + 1)
        self.R_env = [None] * (self.n_sites + 1)
        self.L_env[0]          = np.ones((1, 1, 1), dtype=float)
        self.R_env[self.n_sites] = np.ones((1, 1, 1), dtype=float)

    # ------------------------------------------------------------------
    # MPO construction for 2D snaked lattice
    # ------------------------------------------------------------------
    def _build_2d_snaked_mpo(self):
        """
        Build a list of MPO tensors W_i[a, b, s*, s] for the 2D Heisenberg
        model mapped onto a snake path.

        MPO bond dimension: D = 3*W + 2
        Index layout (a = left, b = right):
          a=0, b=0       : I (identity pass-through at the left end)
          a=D-1, b=D-1   : I (identity pass-through at the right end)

        Horizontal bonds (sites i and i+1 are horizontally adjacent):
          W[1, 0] = Sz      -> W[D-1, 1] = J*Sz
          W[2, 0] = Sp      -> W[D-1, 2] = J/2*Sm
          W[3, 0] = Sm      -> W[D-1, 3] = J/2*Sp

        Vertical bonds (sites i and i+W are vertically adjacent, requiring
        a delay of W-1 steps in the MPO):
          For k = 1 .. W-1:
            W[3k+1, 3(k-1)+1] = I  (delay Sz)
            W[3k+2, 3(k-1)+2] = I  (delay Sp)
            W[3k+3, 3(k-1)+3] = I  (delay Sm)
          At k=W the vertical bond is closed:
            W[D-1, 3*(W-1)+1] = J*Sz
            W[D-1, 3*(W-1)+2] = J/2*Sm
            W[D-1, 3*(W-1)+3] = J/2*Sp

        Boundary:
          site 0  : take last row  -> shape (1, D, d, d)
          site -1 : take first col -> shape (D, 1, d, d)
        """
        D = 3 * self.W + 2

        # Build the bulk tensor once and copy for each site
        W_bulk = np.zeros((D, D, self.d, self.d), dtype=float)

        # Identity pass-throughs
        W_bulk[0,   0]   = self.I
        W_bulk[D-1, D-1] = self.I

        # --- Horizontal bond operators (start) ---
        W_bulk[1, 0] = self.Sz
        W_bulk[2, 0] = self.Sp
        W_bulk[3, 0] = self.Sm

        # --- Horizontal bond operators (close) ---
        W_bulk[D-1, 1] = self.J * self.Sz
        W_bulk[D-1, 2] = self.J * 0.5 * self.Sm
        W_bulk[D-1, 3] = self.J * 0.5 * self.Sp

        # --- Vertical bond delay lines ---
        # k=1 means the operator was started 1 step ago (Sz at site i,
        # now propagating). k goes from 1 to W-1.
        for k in range(1, self.W):
            W_bulk[3*k+1, 3*(k-1)+1] = self.I   # delay Sz
            W_bulk[3*k+2, 3*(k-1)+2] = self.I   # delay Sp
            W_bulk[3*k+3, 3*(k-1)+3] = self.I   # delay Sm

        # --- Vertical bond close (after W delay steps) ---
        W_bulk[D-1, 3*(self.W-1)+1] = self.J * self.Sz
        W_bulk[D-1, 3*(self.W-1)+2] = self.J * 0.5 * self.Sm
        W_bulk[D-1, 3*(self.W-1)+3] = self.J * 0.5 * self.Sp

        mpos = [W_bulk.copy() for _ in range(self.n_sites)]

        # Boundary conditions
        # Left boundary (site 0): only the LAST row contributes (outgoing bond)
        mpos[0]  = W_bulk[D-1:D, :, :, :]   # shape (1, D, d, d)
        # Right boundary (site -1): only the FIRST column contributes (incoming bond)
        mpos[-1] = W_bulk[:, 0:1, :, :]      # shape (D, 1, d, d)

        return mpos

    # ------------------------------------------------------------------
    # MPS canonicalization
    # ------------------------------------------------------------------
    def _right_canonicalize(self):
        """Right-normalize MPS from site n_sites-1 down to site 1."""
        for i in range(self.n_sites - 1, 0, -1):
            cl, d, cr = self.mps[i].shape
            mat = self.mps[i].reshape(cl, d * cr)
            u, s, vh = np.linalg.svd(mat, full_matrices=False)
            self.mps[i]   = vh.reshape(-1, d, cr)
            self.mps[i-1] = np.einsum('ijk,kl,l->ijl', self.mps[i-1], u, s)
        norm = np.linalg.norm(self.mps[0])
        if norm > 0:
            self.mps[0] /= norm

    # ------------------------------------------------------------------
    # Environment updates
    # ------------------------------------------------------------------
    def update_env_right(self, site):
        """
        Contract R[site] from R[site+1], mps[site], mpos[site].

        Shapes:
          A  = mps[site]       : (cl, d, cr)
          W  = mpos[site]      : (wl, wr, d*, d)
          R  = R_env[site+1]   : (cr, wr, cr*)   <- (chi_mps, chi_mpo, chi_mps*)
          result R_env[site]   : (cl, wl, cl*)

        Contraction:
          R[site][i, a, i*] =
            sum_{j,k,b,k*} A[i,j,k] W[a,b,n,j] A*[i*,n,k*] R[k,b,k*]
        einsum: 'ijk, abnj, ink, kbl -> ial'
        """
        A = self.mps[site]
        W = self.mpos[site]
        R = self.R_env[site + 1]
        self.R_env[site] = np.einsum(
            'ijk, abnj, ink, kbl -> ial',
            A, W, A.conj(), R, optimize=True
        )

    def update_env_left(self, site):
        """
        Contract L_env[site+1] from L_env[site], mps[site], mpos[site].

        Shapes:
          L  = L_env[site]     : (cl, wl, cl*)
          A  = mps[site]       : (cl, d, cr)
          W  = mpos[site]      : (wl, wr, d*, d)
          result L_env[site+1] : (cr, wr, cr*)

        Contraction:
          L[site+1][k, b, k*] =
            sum_{i,j,a,i*} L[i,a,i*] A[i,j,k] W[a,b,n,j] A*[i*,n,k*]
        einsum: 'ial, ijk, abnj, ink -> kbl'
        """
        L = self.L_env[site]
        A = self.mps[site]
        W = self.mpos[site]
        self.L_env[site + 1] = np.einsum(
            'ial, ijk, abnj, ink -> kbl',
            L, A, W, A.conj(), optimize=True
        )

    # ------------------------------------------------------------------
    # Local 2-site eigensolver
    # ------------------------------------------------------------------
    def solve_local_2site(self, i, j):
        """
        Diagonalize the effective Hamiltonian for sites i and j.

        H_eff = L_env[i] * W[i] * W[j] * R_env[j+1]

        |psi> has shape (chi_l, d, d, chi_r); the matvec contracts all
        virtual indices.

        Returns: (psi_opt reshaped to (chi_l, d, d, chi_r), energy)
        """
        L  = self.L_env[i]          # (cl, wl, cl*)
        R  = self.R_env[j + 1]      # (cr, wr, cr*)
        W1 = self.mpos[i]           # (wl, wm, d1*, d1)
        W2 = self.mpos[j]           # (wm, wr, d2*, d2)
        shape = (self.mps[i].shape[0], self.d, self.d, self.mps[j].shape[2])
        size  = int(np.prod(shape))

        def matvec(v):
            psi = v.reshape(shape)  # (cl, d1, d2, cr)
            # H_eff |psi>:
            #   sum over cl*, d1*, d2*, cr* of
            #   L[cl,wl,cl*] W1[wl,wm,d1*,d1] W2[wm,wr,d2*,d2] R[cr,wr,cr*] psi[cl*,d1*,d2*,cr*]
            result = np.einsum(
                'ial, abmj, bcnk, rdl, jknr -> imno',
                L, W1, W2, R, psi, optimize=True
            )
            return result.ravel()

        H_eff    = LinearOperator((size, size), matvec=matvec, dtype=float)
        psi_init = np.einsum('ijk,klm->ijlm', self.mps[i], self.mps[j],
                             optimize=True).ravel()
        vals, vecs = eigsh(H_eff, k=1, which='SA', v0=psi_init,
                           tol=1e-6, maxiter=1000)
        return vecs[:, 0].reshape(shape), float(vals[0])

    # ------------------------------------------------------------------
    # Truncation backends
    # ------------------------------------------------------------------
    def turboquant_truncate(self, mat, chi_target):
        """
        TurboQuant O(chi^2 log chi) truncation.
        Applies FWHT to rows of mat to identify the most informative
        subspace, then returns a left-unitary basis Q of shape
        (rows, chi_target) via QR.
        """
        rows, cols = mat.shape
        next_pow2  = 1 << (rows - 1).bit_length() if rows > 1 else 1
        mat_padded = np.pad(mat, ((0, next_pow2 - rows), (0, 0)))
        mat_rot    = fwht_vectorized(mat_padded)       # (next_pow2, cols)
        chi        = min(chi_target, cols, rows)
        norms      = np.linalg.norm(mat_rot, axis=1)  # score by row energy
        idx        = np.argsort(norms)[-chi:]
        res_rot    = mat_rot[idx, :]                   # (chi, cols)
        # Inverse FWHT (transpose trick: apply to columns of res_rot.T)
        res_cols   = fwht_vectorized(res_rot.T)        # (cols, chi)
        res        = (res_cols / next_pow2).T          # (chi, cols) approx basis
        Q, _       = np.linalg.qr(res.T)              # Q: (cols, chi)
        return Q

    def svd_truncate(self, mat, chi_target):
        """Standard SVD truncation. Returns (U, S, Vh)."""
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        chi      = min(chi_target, len(S))
        return U[:, :chi], S[:chi], Vh[:chi, :]

    # ------------------------------------------------------------------
    # Main sweep
    # ------------------------------------------------------------------
    def run_sweep(self, mode="svd", sweeps=2):
        """
        Run `sweeps` left-right DMRG sweeps on the 2D snaked lattice.
        Returns (mean_truncation_time, final_energy_per_site).
        """
        # Seed right environments
        for i in range(self.n_sites - 1, 0, -1):
            self.update_env_right(i)

        all_times  = []
        last_energy = 0.0

        for _sweep in range(sweeps):
            # ---- Left -> Right ----
            for i in range(self.n_sites - 1):
                psi_2site, e = self.solve_local_2site(i, i + 1)
                last_energy  = e
                cl, d1, d2, cr = psi_2site.shape
                mat = psi_2site.reshape(cl * d1, d2 * cr)

                t0 = time.perf_counter()
                if mode == "turboquant":
                    Q          = self.turboquant_truncate(mat, self.chi_max)
                    SV         = Q.T @ mat
                    chi_actual = Q.shape[1]
                    U_s        = Q
                    S_s        = np.ones(chi_actual)
                    V_s        = SV
                else:
                    U_s, S_s, V_s = self.svd_truncate(mat, self.chi_max)
                    chi_actual = len(S_s)

                all_times.append(time.perf_counter() - t0)
                self.mps[i]     = U_s.reshape(cl, d1, chi_actual)
                self.mps[i + 1] = (np.diag(S_s) @ V_s).reshape(chi_actual, d2, cr)
                self.update_env_left(i)

            # ---- Right -> Left ----
            for i in range(self.n_sites - 1, 0, -1):
                psi_2site, e = self.solve_local_2site(i - 1, i)
                last_energy  = e
                cl, d1, d2, cr = psi_2site.shape
                mat = psi_2site.reshape(cl * d1, d2 * cr)

                U_s, S_s, V_s = self.svd_truncate(mat, self.chi_max)
                chi_actual = len(S_s)

                self.mps[i]     = V_s.reshape(chi_actual, d2, cr)
                self.mps[i - 1] = (U_s @ np.diag(S_s)).reshape(cl, d1, chi_actual)
                self.update_env_right(i)

        mean_time = float(np.mean(all_times)) if all_times else 0.0
        return mean_time, float(last_energy / self.n_sites)
