"""End-to-end adaptive-grid ISDF/THC test for a synthetic periodic LAPW basis.

This is deliberately one step above ``thc_mt_kpoint_test.py``.  The basis is
periodic and has a genuine two-region form:

* plane waves in the interstitial;
* atom-centred s/p/d/f augmentation corrections inside two non-overlapping
  muffin-tin spheres;
* augmentation corrections vanish with their first derivative at R_MT, so
  value and slope match the interstitial plane wave exactly.

For each candidate grid, interpolation points and zeta are constructed using
only values and weights on that grid.  The candidate-grid zeta functions are
then transformed to reciprocal space by the same quadrature and used to build
the periodic Coulomb metric and a THC ERI block.  A separately converged
composite grid supplies the reference ERI block; it is never used to fit the
candidate-grid zeta functions.

The reciprocal Coulomb sum is evaluated at a fixed finite G cutoff.  The q=0,
G=0 component is omitted, corresponding to the conventional neutralizing-
background definition of the periodic Coulomb kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.linalg import eigh, lstsq, norm
from scipy.linalg import qr
from scipy.special import sph_harm_y


RNG_SEED = 19
A = 5.0
VOLUME = A**3
RMT = 0.82
LMAX = 3

ATOM_POS = np.array(
    [
        [-1.18, -0.31, 0.17],
        [1.07, 0.43, -0.29],
    ]
)
ATOM_SCALE = np.array([1.0, 0.73])

# Four APW envelopes plus two confined local orbitals.  Reference-grid
# orthonormalization below turns them into six k-dependent orbital combinations.
G_APW = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=int,
)
LO_SPECS = ((0, 0, 0, 18.0), (1, 3, 2, 3.0))
NORB = len(G_APW) + len(LO_SPECS)

KFRAC = np.array(
    [[i, j, 0.0] for i in (0.0, 0.5) for j in (0.0, 0.5)]
)
NK = len(KFRAC)
KIDX = {
    tuple(np.mod(np.rint(2 * k).astype(int), 2)): i
    for i, k in enumerate(KFRAC)
}


@dataclass(frozen=True)
class Grid:
    name: str
    points: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class Metrics:
    name: str
    npoints: int
    pair_fourier: float
    eri_frobenius: float
    eri_max_element: float
    action_max: float
    candidate_fit_frobenius: float
    candidate_fit_column_max: float


def fold_displacement(displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-image displacement and the selected lattice image."""
    image = np.rint(displacement / A).astype(int)
    return displacement - A * image, image


def fib_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    cos_theta = 1.0 - 2.0 * i / n
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.column_stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
    )


def log_radial(r0: float, r1: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    h = np.log(r1 / r0) / (n - 1)
    radius = r0 * np.exp(h * np.arange(n))
    # Point-centred logarithmic shells.  Finite-volume weights integrate a
    # constant over [0, r1] exactly, including the small central ball.
    boundaries = np.empty(n + 1)
    boundaries[0] = 0.0
    boundaries[1:n] = np.sqrt(radius[:-1] * radius[1:])
    boundaries[n] = r1
    weights = (boundaries[1:] ** 3 - boundaries[:-1] ** 3) / 3.0
    return radius, weights


def atom_grid(nrad: int, nang: int) -> tuple[np.ndarray, np.ndarray]:
    radius, radial_weights = log_radial(2.0e-4, RMT, nrad)
    directions = fib_sphere(nang)
    local = (radius[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    local_weights = np.repeat(radial_weights * 4.0 * np.pi / nang, nang)
    points = np.vstack([local + position for position in ATOM_POS])
    weights = np.tile(local_weights, len(ATOM_POS))
    return points, weights


def uniform_grid(n: int, shift: tuple[float, float, float] = (0.5, 0.5, 0.5)) -> Grid:
    axes = [((np.arange(n) + s) / n - 0.5) * A for s in shift]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    weights = np.full(len(points), VOLUME / n**3)
    return Grid(f"uniform {n}^3", points, weights)


def outside_all_spheres(points: np.ndarray) -> np.ndarray:
    outside = np.ones(len(points), dtype=bool)
    for position in ATOM_POS:
        displacement, _ = fold_displacement(points - position)
        outside &= norm(displacement, axis=1) >= RMT
    return outside


def composite_grid(name: str, nrad: int, nang: int, ninter: int) -> Grid:
    mt_points, mt_weights = atom_grid(nrad, nang)
    interstitial = uniform_grid(ninter)
    keep = outside_all_spheres(interstitial.points)
    interstitial_weights = interstitial.weights[keep].copy()
    exact_interstitial_volume = (
        VOLUME - len(ATOM_POS) * 4.0 * np.pi * RMT**3 / 3.0
    )
    interstitial_weights *= exact_interstitial_volume / interstitial_weights.sum()
    points = np.vstack([mt_points, interstitial.points[keep]])
    weights = np.concatenate([mt_weights, interstitial_weights])
    return Grid(name, points, weights)


def augmentation_radial(radius: np.ndarray, l: int, atom_scale: float) -> np.ndarray:
    """Regular localized correction with value and slope zero at R_MT."""
    x = radius / RMT
    amplitude = np.array([3.2, 2.3, 1.55, 1.05])[l]
    sharpness = np.array([10.0, 7.0, 4.5, 2.5])[l]
    return (
        atom_scale
        * amplitude
        * x**l
        * (1.0 - x) ** 2
        * np.exp(-sharpness * x)
    )


def local_orbital_radial(radius: np.ndarray, l: int, sharpness: float) -> np.ndarray:
    """Confined LAPW local orbital; both value and slope vanish at R_MT."""
    x = radius / RMT
    return x**l * (1.0 - x) ** 2 * np.exp(-sharpness * x)


def raw_cell_periodic_orbitals(points: np.ndarray) -> np.ndarray:
    """Evaluate synthetic APWs, returning (npoint, nk, norb)."""
    result = np.empty((len(points), NK, NORB), dtype=complex)
    reciprocal_factor = 2.0 * np.pi / A

    for ik, k_fractional in enumerate(KFRAC):
        k_vector = reciprocal_factor * k_fractional
        bloch_to_periodic = np.exp(-1j * (points @ k_vector))

        for ib, g_integer in enumerate(G_APW):
            wave_vector = reciprocal_factor * (k_fractional + g_integer)
            psi = np.exp(1j * (points @ wave_vector))
            wave_norm = norm(wave_vector)
            if wave_norm > 0.0:
                wave_theta = np.arccos(np.clip(wave_vector[2] / wave_norm, -1.0, 1.0))
                wave_phi = np.arctan2(wave_vector[1], wave_vector[0])
            else:
                wave_theta = 0.0
                wave_phi = 0.0

            for atom, (position, atom_scale) in enumerate(zip(ATOM_POS, ATOM_SCALE)):
                displacement, image = fold_displacement(points - position)
                radius = norm(displacement, axis=1)
                inside = radius < RMT
                if not np.any(inside):
                    continue

                d = displacement[inside]
                r = radius[inside]
                theta = np.zeros_like(r)
                nonzero = r > 0.0
                theta[nonzero] = np.arccos(np.clip(d[nonzero, 2] / r[nonzero], -1.0, 1.0))
                phi = np.arctan2(d[:, 1], d[:, 0])
                image_center = position + A * image[inside]
                center_phase = np.exp(1j * (image_center @ wave_vector))

                correction = np.zeros(len(r), dtype=complex)
                for l in range(LMAX + 1):
                    radial = augmentation_radial(r, l, atom_scale)
                    if wave_norm == 0.0 and l > 0:
                        continue
                    for m in range(-l, l + 1):
                        coefficient = (
                            4.0
                            * np.pi
                            * (1j**l)
                            * np.conj(sph_harm_y(l, m, wave_theta, wave_phi))
                        )
                        correction += (
                            coefficient
                            * radial
                            * sph_harm_y(l, m, theta, phi)
                        )
                psi[inside] += center_phase * correction

            result[:, ik, ib] = psi * bloch_to_periodic

        for offset, (atom, l, m, sharpness) in enumerate(LO_SPECS):
            ib = len(G_APW) + offset
            position = ATOM_POS[atom]
            displacement, image = fold_displacement(points - position)
            radius = norm(displacement, axis=1)
            inside = radius < RMT
            psi = np.zeros(len(points), dtype=complex)
            if np.any(inside):
                d = displacement[inside]
                r = radius[inside]
                theta = np.zeros_like(r)
                nonzero = r > 0.0
                theta[nonzero] = np.arccos(
                    np.clip(d[nonzero, 2] / r[nonzero], -1.0, 1.0)
                )
                phi = np.arctan2(d[:, 1], d[:, 0])
                lattice_phase = np.exp(
                    1j * ((A * image[inside]) @ k_vector)
                )
                psi[inside] = (
                    lattice_phase
                    * local_orbital_radial(r, l, sharpness)
                    * sph_harm_y(l, m, theta, phi)
                )
            result[:, ik, ib] = psi * bloch_to_periodic

    return result


def orthonormalizers(raw_reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    transforms = np.empty((NK, NORB, NORB), dtype=complex)
    for ik in range(NK):
        values = raw_reference[:, ik, :]
        overlap = values.conj().T @ (weights[:, None] * values)
        eigenvalues, eigenvectors = eigh(overlap)
        if eigenvalues[0] < 1.0e-10:
            raise RuntimeError(f"ill-conditioned synthetic APW overlap at k={ik}")
        transforms[ik] = (
            eigenvectors * eigenvalues[None, :] ** -0.5
        ) @ eigenvectors.conj().T
    return transforms


def apply_orbital_transform(raw: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    return np.einsum("pki,kij->pkj", raw, transforms, optimize=True)


def kminus(ik: int, iq: int) -> tuple[int, np.ndarray]:
    """Return folded k-q index and its reciprocal-lattice gauge shift."""
    unwrapped = KFRAC[ik] - KFRAC[iq]
    key = np.mod(np.rint(2 * unwrapped).astype(int), 2)
    index = KIDX[tuple(key)]
    reciprocal_shift = np.rint(unwrapped - KFRAC[index]).astype(int)
    return index, reciprocal_shift


def pair_matrix(orbitals: np.ndarray, points: np.ndarray, iq: int) -> np.ndarray:
    columns = []
    for ik in range(NK):
        left_index, reciprocal_shift = kminus(ik, iq)
        # u_{k-q}=exp(-i G_shift.r) u_{folded(k-q)}.  Conjugation gives the
        # positive phase below and keeps rho^q in the canonical q gauge.
        umklapp = np.exp(
            2j * np.pi / A * (points @ reciprocal_shift)
        )
        columns.append(
            umklapp[:, None, None]
            * np.conj(orbitals[:, left_index, :, None])
            * orbitals[:, ik, None, :]
        )
    return np.stack(columns, axis=1).reshape(len(orbitals), NK * NORB * NORB)


def select_shared_points(
    orbitals: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Select one point set from the union of all q pair-density spaces."""
    all_q_pairs = np.hstack(
        [pair_matrix(orbitals, points, iq) for iq in range(NK)]
    )
    weighted_transpose = (all_q_pairs * np.sqrt(weights)[:, None]).T
    _, _, pivots = qr(weighted_transpose, mode="economic", pivoting=True)
    return pivots[:rank]


def fit_candidate_zeta(
    pair_values: np.ndarray,
    selected_rows: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    rows = pair_values[selected_rows]
    zeta_transpose, *_ = lstsq(rows.T, pair_values.T, rcond=1.0e-12)
    reconstructed = (rows.T @ zeta_transpose).T
    sqrt_weights = np.sqrt(weights)[:, None]
    weighted_values = sqrt_weights * pair_values
    weighted_error = sqrt_weights * (pair_values - reconstructed)
    column_norm = norm(weighted_values, axis=0)
    column_error = norm(weighted_error, axis=0)
    relative = np.divide(
        column_error,
        column_norm,
        out=np.zeros_like(column_error),
        where=column_norm > 1.0e-14,
    )
    frobenius = norm(weighted_error) / norm(weighted_values)
    return zeta_transpose.T, float(frobenius), float(relative.max())


def reciprocal_vectors(cutoff_squared: int = 12) -> tuple[np.ndarray, np.ndarray]:
    limit = int(np.sqrt(cutoff_squared)) + 1
    integer_vectors = np.array(
        [
            [i, j, k]
            for i in range(-limit, limit + 1)
            for j in range(-limit, limit + 1)
            for k in range(-limit, limit + 1)
            if i * i + j * j + k * k <= cutoff_squared
        ],
        dtype=int,
    )
    return integer_vectors, 2.0 * np.pi / A * integer_vectors


def fourier_coefficients(
    values: np.ndarray,
    grid: Grid,
    reciprocal: np.ndarray,
) -> np.ndarray:
    phase = np.exp(-1j * (grid.points @ reciprocal.T))
    return phase.T @ (grid.weights[:, None] * values) / VOLUME


def coulomb_factors(q_fractional: np.ndarray, g_integer: np.ndarray) -> np.ndarray:
    q_plus_g = 2.0 * np.pi / A * (g_integer + q_fractional[None, :])
    squared = np.sum(q_plus_g**2, axis=1)
    factors = np.zeros_like(squared)
    nonzero = squared > 1.0e-14
    factors[nonzero] = 4.0 * np.pi / squared[nonzero]
    return factors


def coulomb_gram(pair_fourier: np.ndarray, factors: np.ndarray) -> np.ndarray:
    weighted = factors[:, None] * pair_fourier
    return VOLUME * (pair_fourier.conj().T @ weighted)


def compare_candidate(
    grid: Grid,
    reference_pair_fourier: dict[int, np.ndarray],
    reference_gram: dict[int, np.ndarray],
    g_integer: np.ndarray,
    reciprocal: np.ndarray,
    transforms: np.ndarray,
    rank: int,
) -> Metrics:
    orbitals = apply_orbital_transform(raw_cell_periodic_orbitals(grid.points), transforms)
    selected = select_shared_points(orbitals, grid.points, grid.weights, rank)
    if grid.name.startswith("adaptive"):
        selected_points = grid.points[selected]
        selected_in_mt = []
        for position in ATOM_POS:
            displacement, _ = fold_displacement(selected_points - position)
            selected_in_mt.append(norm(displacement, axis=1) < RMT)
        interstitial_count = int((~np.logical_or.reduce(selected_in_mt)).sum())
        region_counts = [int(mask.sum()) for mask in selected_in_mt]
        if min(region_counts + [interstitial_count]) < 4:
            raise AssertionError(
                f"shared points miss a grid region: MT={region_counts}, "
                f"interstitial={interstitial_count}"
            )
    rng = np.random.default_rng(RNG_SEED)

    pair_errors = []
    eri_errors = []
    max_element_errors = []
    action_errors = []
    fit_frobenius_errors = []
    fit_column_errors = []

    for iq in range(NK):
        pairs = pair_matrix(orbitals, grid.points, iq)
        zeta, fit_frobenius, fit_column_max = fit_candidate_zeta(
            pairs, selected, grid.weights
        )
        rows = pairs[selected]
        zeta_fourier = fourier_coefficients(zeta, grid, reciprocal)
        approximated_pair_fourier = zeta_fourier @ rows
        exact_pair_fourier = reference_pair_fourier[iq]
        pair_errors.append(
            norm(approximated_pair_fourier - exact_pair_fourier)
            / norm(exact_pair_fourier)
        )

        factors = coulomb_factors(KFRAC[iq], g_integer)
        approximate_gram = coulomb_gram(approximated_pair_fourier, factors)
        exact_gram = reference_gram[iq]
        for label, gram in (("THC", approximate_gram), ("reference", exact_gram)):
            hermitian_error = norm(gram - gram.conj().T) / norm(gram)
            if hermitian_error > 2.0e-12:
                raise AssertionError(
                    f"{label} Coulomb block is not Hermitian: {hermitian_error:.3e}"
                )
            eigenvalues = np.linalg.eigvalsh((gram + gram.conj().T) * 0.5)
            if eigenvalues[0] < -1.0e-10 * eigenvalues[-1]:
                raise AssertionError(
                    f"{label} Coulomb block is not positive semidefinite"
                )
        difference = approximate_gram - exact_gram
        eri_errors.append(norm(difference) / norm(exact_gram))
        max_element_errors.append(np.max(np.abs(difference)) / np.max(np.abs(exact_gram)))

        q_action_errors = []
        for _ in range(8):
            vector = rng.normal(size=exact_gram.shape[0]) + 1j * rng.normal(
                size=exact_gram.shape[0]
            )
            denominator = norm(exact_gram @ vector)
            if denominator > 1.0e-14:
                q_action_errors.append(norm(difference @ vector) / denominator)
        action_errors.append(max(q_action_errors))
        fit_frobenius_errors.append(fit_frobenius)
        fit_column_errors.append(fit_column_max)

    return Metrics(
        name=grid.name,
        npoints=len(grid.points),
        pair_fourier=float(max(pair_errors)),
        eri_frobenius=float(max(eri_errors)),
        eri_max_element=float(max(max_element_errors)),
        action_max=float(max(action_errors)),
        candidate_fit_frobenius=float(max(fit_frobenius_errors)),
        candidate_fit_column_max=float(max(fit_column_errors)),
    )


def verify_basis_structure(transforms: np.ndarray) -> None:
    rng = np.random.default_rng(5)
    points = rng.uniform(-A / 2, A / 2, size=(24, 3))
    shifted = points + np.array([A, 0.0, 0.0])
    periodic = apply_orbital_transform(raw_cell_periodic_orbitals(points), transforms)
    periodic_shifted = apply_orbital_transform(
        raw_cell_periodic_orbitals(shifted), transforms
    )
    covariance_error = norm(periodic_shifted - periodic) / norm(periodic)
    if covariance_error > 2.0e-12:
        raise AssertionError(f"cell-periodic covariance failed: {covariance_error:.3e}")

    epsilon = 2.0e-6
    radius = np.array([RMT - epsilon, RMT, RMT + epsilon])
    correction = np.where(
        radius <= RMT,
        augmentation_radial(np.minimum(radius, RMT), 0, 1.0),
        0.0,
    )
    value_jump = abs(correction[1])
    slope_jump = abs((correction[1] - correction[0]) / epsilon)
    if value_jump > 1.0e-13 or slope_jump > 2.0e-4:
        raise AssertionError(
            f"MT matching failed: value={value_jump:.3e}, slope={slope_jump:.3e}"
        )

    probe = np.array([[0.0, A / 4.0, 0.0]])
    constant_orbitals = np.ones((1, NK, NORB), dtype=complex)
    wrapped_pair = pair_matrix(constant_orbitals, probe, iq=1)[0, 0]
    if abs(wrapped_pair + 1j) > 2.0e-14:
        raise AssertionError(
            f"canonical-q Umklapp phase failed: got {wrapped_pair}, expected -1j"
        )


def build_reference(
    reference: Grid,
    transforms: np.ndarray,
    g_integer: np.ndarray,
    reciprocal: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    orbitals = apply_orbital_transform(
        raw_cell_periodic_orbitals(reference.points), transforms
    )
    pair_fourier = {}
    gram = {}
    for iq in range(NK):
        coefficients = fourier_coefficients(
            pair_matrix(orbitals, reference.points, iq), reference, reciprocal
        )
        pair_fourier[iq] = coefficients
        gram[iq] = coulomb_gram(
            coefficients, coulomb_factors(KFRAC[iq], g_integer)
        )
    return pair_fourier, gram


def format_metrics(metrics: list[Metrics], rank: int) -> str:
    lines = ["=== Synthetic periodic LAPW adaptive-grid ISDF/THC ==="]
    lines.append(
        f"two atoms, s/p/d/f augmentation, {NK} k/q points, "
        f"Norb={NORB}, Nmu={rank}; integer |G|^2 <= 12; q=0 G=0 omitted"
    )
    lines.append(
        f"{'grid':<24}{'npts':>7}{'pair-G':>12}{'ERI-F':>12}"
        f"{'ERI-max':>12}{'action':>12}{'fit-F':>12}{'fit-col':>12}"
    )
    for item in metrics:
        lines.append(
            f"{item.name:<24}{item.npoints:>7d}"
            f"{item.pair_fourier:>12.3e}{item.eri_frobenius:>12.3e}"
            f"{item.eri_max_element:>12.3e}{item.action_max:>12.3e}"
            f"{item.candidate_fit_frobenius:>12.3e}"
            f"{item.candidate_fit_column_max:>12.3e}"
        )
    return "\n".join(lines)


def main() -> None:
    reference = composite_grid("reference", nrad=38, nang=110, ninter=20)
    raw_reference = raw_cell_periodic_orbitals(reference.points)
    transforms = orthonormalizers(raw_reference, reference.weights)
    verify_basis_structure(transforms)

    g_integer, reciprocal = reciprocal_vectors(cutoff_squared=12)
    reference_pair_fourier, reference_gram = build_reference(
        reference, transforms, g_integer, reciprocal
    )
    medium_reference = composite_grid(
        "medium reference", nrad=30, nang=86, ninter=18
    )
    _, medium_gram = build_reference(
        medium_reference, transforms, g_integer, reciprocal
    )
    reference_convergence = max(
        norm(medium_gram[iq] - reference_gram[iq]) / norm(reference_gram[iq])
        for iq in range(NK)
    )
    if reference_convergence > 5.0e-2:
        raise AssertionError("independent Coulomb/ERI reference is not converged")

    rank = 16 * NORB
    candidates = [
        composite_grid("adaptive 14x50 + 14^3", 14, 50, 14),
        composite_grid("adaptive 26x86 + 18^3", 26, 86, 18),
        uniform_grid(16),
    ]
    metrics = [
        compare_candidate(
            grid,
            reference_pair_fourier,
            reference_gram,
            g_integer,
            reciprocal,
            transforms,
            rank,
        )
        for grid in candidates
    ]
    output = (
        f"independent-reference ERI convergence: {reference_convergence:.3e}\n"
        + format_metrics(metrics, rank)
    )
    print(output)

    coarse, fine, uniform = metrics
    if fine.eri_frobenius > coarse.eri_frobenius * 1.05:
        raise AssertionError("refining the adaptive grid did not improve the ERI block")
    if fine.action_max > 8.0e-2:
        raise AssertionError(
            f"adaptive-grid THC action error is too large: {fine.action_max:.3e}"
        )
    if fine.eri_frobenius > 8.0e-2 or fine.eri_max_element > 8.0e-2:
        raise AssertionError(
            "adaptive-grid THC failed a deterministic ERI error threshold"
        )
    if coarse.eri_frobenius > uniform.eri_frobenius:
        raise AssertionError("adaptive grid did not beat the similar-size uniform grid")
    Path(__file__).with_name("thc_lapw_end_to_end_results.txt").write_text(
        output + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
