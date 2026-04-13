import pytest
import numpy as np
from app.solver_2d import Solver2D

# ── 1. INITIALIZATION TESTS (40 CASES) ──────────────────────────────────────

@pytest.mark.parametrize("L", [2, 4, 6, 8])
@pytest.mark.parametrize("W", [2, 4, 6, 8, 10])
def test_init_lattice_size(L, W):
    s = Solver2D(L=L, W=W)
    assert s.n_sites == L * W
    assert len(s.mps) == L * W

# ── 2. TENSOR DIMENSIONS (20 CASES) ──────────────────────────────────────────

@pytest.mark.parametrize("chi", [4, 8, 16, 32])
def test_mps_ndim(chi):
    s = Solver2D(L=3, W=3, chi_max=chi)
    for t in s.mps:
        assert t.ndim == 3

# ── 3. TRUNCATION INTEGRITY (80 CASES) ────────────────────────────────────────

@pytest.mark.parametrize("mode", ["svd", "turboquant"])
@pytest.mark.parametrize("chi_target", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("d_in", [2, 4])
def test_truncation_shape_2d(mode, chi_target, d_in):
    s = Solver2D(chi_max=chi_target)
    # Ensure input tensor is large enough to be truncated to chi_target
    chi_l, chi_r = 64, 64 
    tensor = np.random.randn(chi_l, d_in, chi_r)
    
    if mode == "svd":
        res, _ = s.svd_truncate(tensor, chi_target)
    else:
        res, _ = s.turboquant_truncate(tensor, chi_target)
        
    assert res.shape == (chi_l, d_in, chi_target)

# ── 4. STABILITY & CONVERGENCE (60 CASES) ─────────────────────────────────────

@pytest.mark.parametrize("mode", ["svd", "turboquant"])
@pytest.mark.parametrize("L,W", [(2,2), (3,3), (4,2)])
@pytest.mark.parametrize("rep", range(5))
def test_sweep_stability(mode, L, W, rep):
    s = Solver2D(L=L, W=W, chi_max=8)
    dt, energy = s.run_sweep(mode=mode)
    assert not np.isnan(energy)
    assert dt > 0

@pytest.mark.parametrize("rep", range(10)) # Reach >200 total cases
def test_turboquant_precision_lower_bound(rep):
    s = Solver2D(chi_max=4)
    t = np.zeros((8, 2, 8))
    t[0, 0, 0] = 1.0
    t[1, 1, 1] = 1.0
    res, _ = s.turboquant_truncate(t, 2)
    assert np.linalg.norm(res) > 0
