"""Small k-point test: does ISDF of muffin-tin-like orbital products need
an adaptive (atom-refined) real-space grid?

Model
-----
Cubic cell a = 6 bohr, one atom at the origin, muffin-tin radius R_MT = 2.
Six "LAPW-flavored" localized orbitals (unit-normalized, smoothly cut at
R_cut = 2.9 < a/2 so Bloch sums need only nearest images):

  1s core-like  e^{-Z1 r},  Z1 = 20  (sharp, all-electron-like cusp scale)
  2s valence    r e^{-Z2 r}, Z2 = 2
  2p x,y,z      {x,y,z} e^{-Z3 r}, Z3 = 2
  diffuse s     e^{-alpha r^2},  alpha = 0.25   (interstitial-like)

Gamma-centered 2x2x2 k-mesh; q on the same mesh.  Cell-periodic Bloch parts
u_{ik}(r); pair densities at momentum transfer q:

  rho^q_{k,ij}(r) = conj(u_{i,k-q}(r)) u_{j,k}(r)     (periodic in r)

ISDF with k-points (Yeh & Morales, JCTC 19, 6197 (2023)): one q-independent
set of interpolation points r_mu; per-q interpolation vectors zeta^q_mu(r).

Experiment
----------
For each *candidate grid* (uniform N^3, or MT-adaptive = exponential radial
mesh x Fibonacci sphere inside R_MT + coarse uniform interstitial shell):
  1. evaluate u_{ik} with quadrature weights,
  2. select r_mu by randomized-sketch QRCP on the sqrt(w)-scaled pair
     collocation, using either the q=0 block or all q blocks; the random
     projection is shared across candidate grids and the selected points are
     then shared by every q,
  3. fit zeta^q on a *dense reference grid* (log-radial x angular, weighted)
     against the exact row values at r_mu, and report the relative weighted
     L2 fitting error of all pair densities, max over q.
The reference-grid fit makes the comparison honest: a coarse candidate grid
cannot hide its aliasing because errors are always measured on the same
dense reference set.  The SVD of the reference pair matrix gives the best
possible error at each rank (lower bound for any point selection).
This is a point-selection and pair-density interpolation experiment, not an
end-to-end tensor-hypercontraction test: no Coulomb kernel or two-electron
integral factorization is constructed here.
"""

import numpy as np
from numpy.linalg import lstsq, norm, svd
from scipy.linalg import qr

A = 6.0          # lattice constant (bohr)
RMT = 2.0        # muffin-tin radius
RCUT = 2.9       # strict localization radius (< a/2)
Z1, Z2, Z3, ALP = 20.0, 2.0, 2.0, 0.25
NORB = 6
SKETCH = 160     # sketch rows for randomized QRCP point selection
POINT_CHUNK = 4096
ALPHAS = [2, 4, 6, 8, 10, 12, 14]        # N_mu = alpha * NORB
HEADLINE_SEED = 7
STRATEGY_SEEDS = (7, 19, 43)
RANDOM_SHIFT_SEED = 29

# ---------------------------------------------------------------- orbitals
def chi_all(disp):
    """Localized orbitals evaluated at displacements from the atom, (n,3)->(n,NORB)."""
    r = norm(disp, axis=1)
    fc = np.where(r < RCUT, (1.0 - (np.minimum(r, RCUT) / RCUT) ** 2) ** 2, 0.0)
    out = np.empty((len(r), NORB))
    out[:, 0] = np.exp(-Z1 * r)
    out[:, 1] = r * np.exp(-Z2 * r)
    pr = np.exp(-Z3 * r)
    out[:, 2] = disp[:, 0] * pr
    out[:, 3] = disp[:, 1] * pr
    out[:, 4] = disp[:, 2] * pr
    out[:, 5] = np.exp(-ALP * r * r)
    return out * fc[:, None]

# ------------------------------------------------------------------ grids
def fib_sphere(n):
    i = np.arange(n) + 0.5
    ct = 1.0 - 2.0 * i / n
    st = np.sqrt(1.0 - ct**2)
    th = np.pi * (1.0 + 5.0**0.5) * i
    return np.column_stack([st * np.cos(th), st * np.sin(th), ct])

def log_radial(r0, r1, n):
    """Exponential mesh r_j = r0 e^{jh} with trapezoid weights for int f r^2 dr."""
    h = np.log(r1 / r0) / (n - 1)
    r = r0 * np.exp(h * np.arange(n))
    w = r**3 * h                      # dr = r h on the log mesh
    w[0] *= 0.5
    w[-1] *= 0.5
    return r, w

def atom_grid(r0, r1, nrad, nang):
    r, wr = log_radial(r0, r1, nrad)
    ang = fib_sphere(nang)
    pts = (r[:, None, None] * ang[None, :, :]).reshape(-1, 3)
    w = np.repeat(wr * 4.0 * np.pi / nang, nang)
    return pts, w

def fold(pts):
    """Fold absolute coords to the nearest-image displacement from the atom at 0."""
    return pts - A * np.round(pts / A)

def uniform_grid(n, shift="half", seed=None):
    """Uniform grid with origin-, half-grid-, or reproducibly random shift."""
    if shift == "origin":
        offset = np.zeros(3)
    elif shift == "half":
        offset = np.full(3, 0.5)
    elif shift == "random":
        if seed is None:
            raise ValueError("random uniform-grid shift requires an explicit seed")
        offset = np.random.default_rng(seed).random(3)
    else:
        raise ValueError(f"unknown uniform-grid shift: {shift!r}")
    axes = [(np.arange(n) + offset[d]) / n * A for d in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    pts = fold(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    w = np.full(len(pts), A**3 / n**3)
    return pts, w

def adaptive_grid(nrad, nang, ninter):
    """MT-style: exponential radial x angular inside R_MT + coarse uniform shell."""
    mt_pts, mt_w = atom_grid(2e-3, RMT, nrad, nang)
    up, uw = uniform_grid(ninter)
    d = norm(up, axis=1)
    keep = (d > RMT) & (d <= RCUT)     # interstitial where orbitals still live
    return np.vstack([mt_pts, up[keep]]), np.concatenate([mt_w, uw[keep]])

# ----------------------------------------------------------- Bloch orbitals
KFRAC = np.array([[i, j, k] for i in (0, 0.5) for j in (0, 0.5) for k in (0, 0.5)])
NK = len(KFRAC)
KIDX = {tuple(np.mod(np.round(2 * kf).astype(int), 2)): i for i, kf in enumerate(KFRAC)}

def kminus(ik, iq):
    unwrapped = KFRAC[ik] - KFRAC[iq]
    d = np.mod(np.round(2 * unwrapped).astype(int), 2)
    index = KIDX[tuple(d)]
    reciprocal_shift = np.rint(unwrapped - KFRAC[index]).astype(int)
    return index, reciprocal_shift

IMAGES = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])

def bloch_u(pts, orb_norm):
    """Cell-periodic parts u_{ik}(r): (npts, NK, NORB)."""
    U = np.zeros((len(pts), NK, NORB), dtype=complex)
    for T in IMAGES:
        chi = chi_all(pts - A * T) / orb_norm
        ph = np.exp(2j * np.pi * (KFRAC @ T))          # e^{i k . T}
        U += ph[None, :, None] * chi[:, None, :]
    ph_r = np.exp(-2j * np.pi / A * (pts @ KFRAC.T))    # e^{-i k . r}
    return U * ph_r[:, :, None]

def pair_matrix(U, pts, iq):
    """Columns rho^q_{k,ij} at the grid points of U: (npts, NK*NORB^2)."""
    cols = []
    for ik in range(NK):
        left, reciprocal_shift = kminus(ik, iq)
        umklapp = np.exp(2j * np.pi / A * (pts @ reciprocal_shift))
        cols.append(umklapp[:, None, None]
                    * np.conj(U[:, left, :, None]) * U[:, ik, None, :])
    return np.stack(cols, axis=1).reshape(len(U), NK * NORB * NORB)

# ------------------------------------------------------------- ISDF pieces
def sketch_factors(seed):
    """Structured random pair-density projection, indexed by (row, k, orb)."""
    local_rng = np.random.default_rng(seed)
    shape = (SKETCH, NK, NORB)
    G1 = local_rng.normal(size=shape) + 1j * local_rng.normal(size=shape)
    G2 = local_rng.normal(size=shape) + 1j * local_rng.normal(size=shape)
    return G1, G2

def select_points(U, pts, w, nmax, factors, strategy="all-q"):
    """QRCP pivots from a shared q=0 or all-q randomized pair-density sketch."""
    if strategy not in ("q=0", "all-q"):
        raise ValueError(f"unknown selection strategy: {strategy!r}")
    G1, G2 = factors
    S = np.zeros((SKETCH, len(U)), dtype=complex)
    for start in range(0, len(U), POINT_CHUNK):
        stop = min(start + POINT_CHUNK, len(U))
        if strategy == "q=0":
            block = np.zeros((SKETCH, stop - start), dtype=complex)
            for ik in range(NK):
                L = np.conj(U[start:stop, ik, :]) @ G1[:, ik, :].T
                R = U[start:stop, ik, :] @ G2[:, ik, :].T
                L *= R
                block += L.T
        else:
            block = np.zeros((SKETCH, stop - start), dtype=complex)
            for iq in range(NK):
                for ik in range(NK):
                    left, reciprocal_shift = kminus(ik, iq)
                    L = np.conj(U[start:stop, left, :]) @ G1[:, left, :].T
                    R = U[start:stop, ik, :] @ G2[:, ik, :].T
                    phase = np.exp(2j * np.pi / A
                                   * (pts[start:stop] @ reciprocal_shift))
                    block += (phase[:, None] * L * R).T
        S[:, start:stop] = block * np.sqrt(w[start:stop])[None, :]
    _, _, piv = qr(S, mode="economic", pivoting=True)
    return piv[:nmax]

# columns rho^q_{k,ij} with i or j equal to the sharp core orbital (index 0)
_ij_core = np.array([(i == 0) or (j == 0)
                     for _ in range(NK) for i in range(NORB) for j in range(NORB)])
_column_groups = {"all": np.ones(NK * NORB * NORB, dtype=bool),
                  "core": _ij_core,
                  "valence": ~_ij_core}

def fit_errors(Zref_w, rows, ranks):
    """Aggregate Frobenius and maximum nonzero-column relative fit errors."""
    metrics = {group: {"fro": [], "maxcol": []} for group in _column_groups}
    column_norms = norm(Zref_w, axis=0)
    scale = column_norms.max(initial=0.0)
    nonzero = column_norms > np.finfo(float).eps * max(scale, 1.0)
    for nmu in ranks:
        R = rows[:nmu]                                  # (nmu, ncols), unweighted
        X, *_ = lstsq(R.T, Zref_w.T, rcond=None)        # min ||R^T X - Zref^T||
        E = Zref_w.T - R.T @ X
        error_norms = norm(E, axis=1)
        relative = np.zeros_like(error_norms)
        relative[nonzero] = error_norms[nonzero] / column_norms[nonzero]
        for group, mask in _column_groups.items():
            active = mask & nonzero
            denominator = norm(Zref_w[:, active])
            metrics[group]["fro"].append(norm(E[active]) / denominator)
            metrics[group]["maxcol"].append(relative[active].max(initial=0.0))
    return metrics

def evaluate_selection(U, pts, w, Zref_w, ranks, factors, strategy):
    """Select shared points, then take the worst fit metric over q."""
    piv = select_points(U, pts, w, max(ranks), factors, strategy)
    worst = {group: {metric: np.zeros(len(ranks))
                     for metric in ("fro", "maxcol")}
             for group in _column_groups}
    for iq in range(NK):
        metrics = fit_errors(Zref_w[iq], pair_matrix(U[piv], pts[piv], iq), ranks)
        for group in _column_groups:
            for metric in ("fro", "maxcol"):
                worst[group][metric] = np.maximum(worst[group][metric],
                                                   metrics[group][metric])
    return piv, worst

# ------------------------------------------- SPEX-style MT product basis size
def mt_product_basis_count(tol=1e-4):
    """Count MT-MT product-basis functions a la SPEX mixedbasis.f: normalized
    radial products per L channel, overlap diagonalization, keep eig > tol."""
    r, wr = log_radial(5e-4, RCUT, 400)
    fc = (1.0 - (r / RCUT) ** 2) ** 2
    u = {"s1": np.exp(-Z1 * r) * fc, "s2": r * np.exp(-Z2 * r) * fc,
         "p": r * np.exp(-Z3 * r) * fc, "s3": np.exp(-ALP * r * r) * fc}
    s_set = ["s1", "s2", "s3"]
    channels = {0: [], 1: [], 2: []}
    for a in range(len(s_set)):                 # s x s -> L = 0
        for b in range(a, len(s_set)):
            channels[0].append(u[s_set[a]] * u[s_set[b]])
    for a in s_set:                             # s x p -> L = 1
        channels[1].append(u[a] * u["p"])
    pp = u["p"] * u["p"]                        # p x p -> L = 0, 2 (parity)
    channels[0].append(pp)
    channels[2].append(pp)
    total, detail = 0, []
    for L, fns in channels.items():
        B = np.array([f / np.sqrt((f * f * wr).sum()) for f in fns])
        eig = np.linalg.eigvalsh((B * wr) @ B.T)
        kept = int((eig > tol).sum())
        detail.append(f"L={L}: {len(fns)} products -> {kept} kept")
        total += kept * (2 * L + 1)
    return total, detail

# ------------------------------------------------------------------- main
def main():
    probe = np.array([[0.0, A / 4.0, 0.0]])
    constant_orbitals = np.ones((1, NK, NORB), dtype=complex)
    wrapped_pair = pair_matrix(constant_orbitals, probe, iq=2)[0, 0]
    if abs(wrapped_pair + 1j) > 2e-14:
        raise AssertionError("canonical-q Umklapp phase regression")

    # dense reference (log-radial covers the cusp AND the smooth tail to RCUT)
    ref_pts, ref_w = atom_grid(5e-4, RCUT, 72, 78)
    onorm = np.sqrt((chi_all(ref_pts) ** 2 * ref_w[:, None]).sum(axis=0))
    Uref = bloch_u(ref_pts, onorm)
    sw = np.sqrt(ref_w)[:, None]
    Zref_w = {iq: pair_matrix(Uref, ref_pts, iq) * sw for iq in range(NK)}

    # best possible error at each rank (SVD of the reference pair matrix)
    svd_best = {}
    for iq in range(NK):
        s = svd(Zref_w[iq], compute_uv=False)
        tot = np.sqrt((s**2).sum())
        svd_best[iq] = np.sqrt(np.maximum(np.cumsum(s[::-1] ** 2)[::-1], 0)) / tot
    ranks = [a * NORB for a in ALPHAS]
    best = [max(svd_best[iq][min(nmu, len(svd_best[iq]) - 1)] for iq in range(NK))
            for nmu in ranks]

    grids = [("uniform 16 half", *uniform_grid(16, "half")),
             ("uniform 24 origin", *uniform_grid(24, "origin")),
             ("uniform 24 half", *uniform_grid(24, "half")),
             (f"uniform 24 random({RANDOM_SHIFT_SEED})",
              *uniform_grid(24, "random", RANDOM_SHIFT_SEED)),
             ("uniform 32 half", *uniform_grid(32, "half")),
             ("MT-adaptive nrad=12", *adaptive_grid(12, 26, 12)),
             ("MT-adaptive nrad=20", *adaptive_grid(20, 26, 12)),
             ("MT-adaptive nrad=30", *adaptive_grid(30, 26, 12))]

    lines = []
    hdr = f"{'candidate grid':<25}{'npts':>7}{'min|r|':>9}{'quad.err':>10}" + \
          "".join(f"  a={a:<7}" for a in ALPHAS)
    headline_factors = sketch_factors(HEADLINE_SEED)
    rows_fro_all, rows_fro_core, rows_maxcol = [], [], []
    strategy_grid = None
    for name, pts, w in grids:
        U = bloch_u(pts, onorm)
        # secondary metric: can the grid even integrate sum_cols |rho|^2 (q=0)?
        t_grid = (np.abs(pair_matrix(U, pts, 0)) ** 2 * w[:, None]).sum()
        t_ref = (np.abs(Zref_w[0]) ** 2).sum()
        quad_err = abs(t_grid / t_ref - 1.0)
        piv, metrics = evaluate_selection(U, pts, w, Zref_w, ranks,
                                          headline_factors, "all-q")
        mind = norm(pts[piv[: max(ranks)]], axis=1).min()
        pre = f"{name:<25}{len(pts):>7}{mind:>9.4f}{quad_err:>10.1e}"
        rows_fro_all.append(pre + "".join(
            f"  {e:<8.1e}" for e in metrics["all"]["fro"]))
        rows_fro_core.append(pre + "".join(
            f"  {e:<8.1e}" for e in metrics["core"]["fro"]))
        compact = zip(metrics["all"]["maxcol"], metrics["core"]["maxcol"],
                      metrics["valence"]["maxcol"])
        rows_maxcol.append(f"{name:<25}{len(pts):>7}" +
                           "".join(f"  {ea:.1e}/{ec:.1e}/{ev:.1e}"
                                   for ea, ec, ev in compact))
        if name == "MT-adaptive nrad=20":
            strategy_grid = (U, pts, w)
    svd_row = (f"{'SVD lower bound':<25}{'--':>7}{'--':>9}{'--':>10}" +
               "".join(f"  {e:<8.1e}" for e in best))
    lines.append("=== Aggregate Frobenius ISDF fit error, ALL columns (max over q) ===")
    lines += [hdr, "-" * len(hdr), *rows_fro_all, "-" * len(hdr), svd_row, ""]
    lines.append("=== Aggregate Frobenius ISDF fit error, CORE-involving columns ===")
    lines += [hdr, "-" * len(hdr), *rows_fro_core, ""]

    maxcol_hdr = f"{'candidate grid':<25}{'npts':>7}" + \
                 "".join(f"  a={a:<20}" for a in ALPHAS)
    lines.append("=== Maximum per-column relative error (all/core/valence-only; max over q) ===")
    lines += [maxcol_hdr, "-" * len(maxcol_hdr), *rows_maxcol, ""]
    lines.append(f"headline selection: all-q structured sketch, seed={HEADLINE_SEED}; "
                 f"N_mu = a * {NORB} interpolation points; "
                 f"min|r| = closest selected point to the nucleus (bohr); "
                 f"quad.err = relative error of the grid quadrature of "
                 f"sum_ij |rho_ij|^2 at q=0.")
    lines.append("Per-column ratios exclude reference columns at numerical zero norm.")

    strategy_rank = 8 * NORB
    samples = {strategy: {group: {metric: [] for metric in ("fro", "maxcol")}
                                  for group in _column_groups}
               for strategy in ("q=0", "all-q")}
    U_strategy, pts_strategy, w_strategy = strategy_grid
    for seed in STRATEGY_SEEDS:
        factors = sketch_factors(seed)
        for strategy in samples:
            _, metrics = evaluate_selection(U_strategy, pts_strategy, w_strategy, Zref_w,
                                            [strategy_rank], factors, strategy)
            for group in _column_groups:
                for metric in ("fro", "maxcol"):
                    samples[strategy][group][metric].append(
                        metrics[group][metric][0])
    strategy_hdr = (f"{'selected from':<14}{'seeds':>8}{'N_mu':>7}  "
                    f"{'Frob all mean/max':>21}  {'maxcol all mean/max':>23}  "
                    f"{'maxcol core mean/max':>24}  {'maxcol val mean/max':>23}")
    lines += ["", "=== Shared-point selection strategy on MT-adaptive nrad=20 (max over q) ===",
              strategy_hdr, "-" * len(strategy_hdr)]
    for strategy in samples:
        values = []
        for group, metric in (("all", "fro"), ("all", "maxcol"),
                              ("core", "maxcol"), ("valence", "maxcol")):
            x = np.asarray(samples[strategy][group][metric])
            values.append(f"{x.mean():.1e}/{x.max():.1e}")
        lines.append(f"{strategy:<14}{len(STRATEGY_SEEDS):>8}{strategy_rank:>7}  "
                     f"{values[0]:>21}  {values[1]:>23}  {values[2]:>24}  "
                     f"{values[3]:>23}")
    lines.append("For each seed, q=0 restricts the same random factor arrays to "
                 "k_left=k_right; all-q includes every k_left/k_right pair. "
                 "The factor arrays are shared by candidate grids.")
    nmt, detail = mt_product_basis_count(1e-4)
    lines.append("")
    lines.append("=== SPEX-style MT x MT product basis for the same model "
                 "(overlap diag., TOL=1e-4) ===")
    lines += [f"  {d}" for d in detail]
    lines.append(f"  total MT product-basis functions (with 2L+1): {nmt} "
                 f"(compare with the ISDF N_mu above; the selected interpolation "
                 f"points are shared by all q)")
    lines.append("  This count and the ISDF fit study do not constitute an end-to-end THC test.")
    out = "\n".join(lines)
    print(out)
    with open("thc_mt_kpoint_results.txt", "w") as f:
        f.write(out + "\n")

if __name__ == "__main__":
    main()
