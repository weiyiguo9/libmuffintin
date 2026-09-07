"""CoQui-style THC threshold scan on the synthetic periodic LAPW model.

The interpolation points are ordered by a matrix-free pivoted Cholesky
factorization of the q=0 pair-density metric.  As in CoQui, each orbital is
scaled by ``(Norb^2 * Nk)^(-1/4)`` before forming the metric and ``thresh`` is
an absolute maximum-residual-diagonal cutoff.  Candidate-grid quadrature
weights are used for fit diagnostics and Coulomb integration, not point
selection.

Unlike the stricter end-to-end smoke test, the primary outputs here are fixed-
occupation Hartree and exchange contractions.  ERI norms remain diagnostics.
The reciprocal Coulomb cutoff is held fixed so this script isolates the ISDF
``thresh`` / N_mu tradeoff from the separate ``ecut`` convergence problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.linalg import norm

import thc_lapw_end_to_end_test as lapw


THRESHOLDS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-8)
VALENCE_OCCUPIED = (0, 1, 2)


@dataclass(frozen=True)
class ScanResult:
    grid: str
    window: str
    norb: int
    threshold: float
    npoints: int
    nmu: int
    residual: float
    fit_frobenius: float
    eri_frobenius: float
    hartree_error_over_exchange: float
    exchange_relative: float
    occupied_action: float


def pivoted_cholesky_order(
    pair_values: np.ndarray,
    minimum_threshold: float,
    norb: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return pivot order and CoQui-normalized residual after each pivot."""
    pair_scale = 1.0 / (norb * np.sqrt(lapw.NK))
    metric_pairs = pair_scale * pair_values
    diagonal = np.sum(np.abs(metric_pairs) ** 2, axis=1).real
    initial_max = float(diagonal.max())
    if initial_max <= 0.0:
        raise RuntimeError("zero q=0 pair-density metric")

    max_rank = min(pair_values.shape)
    cholesky = np.empty((len(pair_values), max_rank), dtype=complex)
    pivots = []
    residuals = []

    for rank in range(max_rank):
        pivot = int(np.argmax(diagonal))
        pivot_value = float(diagonal[pivot])
        if pivot_value <= minimum_threshold:
            break

        column = metric_pairs @ metric_pairs[pivot].conj()
        if rank:
            column -= cholesky[:, :rank] @ cholesky[pivot, :rank].conj()
        vector = column / np.sqrt(pivot_value)
        cholesky[:, rank] = vector
        pivots.append(pivot)

        diagonal -= np.abs(vector) ** 2
        np.maximum(diagonal, 0.0, out=diagonal)
        diagonal[pivot] = 0.0
        residuals.append(float(diagonal.max()))

    if not residuals or residuals[-1] > minimum_threshold:
        raise RuntimeError(
            f"pivoted Cholesky did not reach {minimum_threshold:.1e}; "
            f"last residual={residuals[-1] if residuals else 1.0:.3e}"
        )
    return np.asarray(pivots), np.asarray(residuals), initial_max


def rank_for_threshold(
    residuals: np.ndarray, initial_max: float, threshold: float
) -> int:
    # CoQui checks the stopping condition before taking the first pivot.
    if threshold > initial_max:
        return 0
    reached = np.flatnonzero(residuals <= threshold)
    if not len(reached):
        raise RuntimeError(f"threshold {threshold:.1e} was not reached")
    return int(reached[0] + 1)


def density_vector(norb: int, occupied: tuple[int, ...]) -> np.ndarray:
    vector = np.zeros(lapw.NK * norb * norb, dtype=complex)
    for ik in range(lapw.NK):
        for orbital in occupied:
            index = ik * norb * norb + orbital * norb + orbital
            vector[index] = 1.0
    return vector


def occupied_pair_vector(norb: int, occupied: tuple[int, ...]) -> np.ndarray:
    occupations = np.zeros(norb)
    occupations[list(occupied)] = 1.0
    vector = np.zeros(lapw.NK * norb * norb, dtype=complex)
    for ik in range(lapw.NK):
        offset = ik * norb * norb
        for i in range(norb):
            for j in range(norb):
                vector[offset + i * norb + j] = np.sqrt(
                    occupations[i] * occupations[j] / lapw.NK
                )
    return vector


def physical_contractions(
    grams: dict[int, np.ndarray], norb: int, occupied: tuple[int, ...]
) -> tuple[float, float]:
    occupations = np.zeros(norb)
    occupations[list(occupied)] = 1.0
    density = density_vector(norb, occupied)
    # Closed-shell normalization used by CoQui's compare_eri diagnostic.
    hartree = 2.0 / lapw.NK * np.vdot(density, grams[0] @ density).real

    exchange = 0.0
    for iq in range(lapw.NK):
        diagonal = np.diag(grams[iq]).real
        for ik in range(lapw.NK):
            offset = ik * norb * norb
            for i in range(norb):
                for j in range(norb):
                    exchange -= (
                        1.0
                        / lapw.NK
                        * occupations[i]
                        * occupations[j]
                        * diagonal[offset + i * norb + j]
                    )
    return float(hartree), float(exchange)


def build_candidate_grams(
    grid: lapw.Grid,
    orbitals: np.ndarray,
    selected: np.ndarray,
    g_integer: np.ndarray,
    reciprocal: np.ndarray,
    pair_columns: np.ndarray,
) -> tuple[dict[int, np.ndarray], float]:
    grams = {}
    fit_errors = []
    for iq in range(lapw.NK):
        pairs = lapw.pair_matrix(orbitals, grid.points, iq)[:, pair_columns]
        if not len(selected):
            grams[iq] = np.zeros(
                (pairs.shape[1], pairs.shape[1]), dtype=complex
            )
            fit_errors.append(1.0)
            continue
        zeta, fit_frobenius, _ = lapw.fit_candidate_zeta(
            pairs, selected, grid.weights
        )
        rows = pairs[selected]
        zeta_fourier = lapw.fourier_coefficients(zeta, grid, reciprocal)
        pair_fourier = zeta_fourier @ rows
        grams[iq] = lapw.coulomb_gram(
            pair_fourier, lapw.coulomb_factors(lapw.KFRAC[iq], g_integer)
        )
        fit_errors.append(fit_frobenius)
    return grams, float(max(fit_errors))


def scan_grid(
    grid: lapw.Grid,
    transforms: np.ndarray,
    reference_grams: dict[int, np.ndarray],
    g_integer: np.ndarray,
    reciprocal: np.ndarray,
    window: str,
    active_orbitals: tuple[int, ...],
    occupied_orbitals: tuple[int, ...],
) -> list[ScanResult]:
    norb = len(active_orbitals)
    local_index = {orbital: i for i, orbital in enumerate(active_orbitals)}
    occupied = tuple(local_index[orbital] for orbital in occupied_orbitals)
    pair_columns = np.asarray(
        [
            ik * lapw.NORB * lapw.NORB + left * lapw.NORB + right
            for ik in range(lapw.NK)
            for left in active_orbitals
            for right in active_orbitals
        ]
    )
    orbitals = lapw.apply_orbital_transform(
        lapw.raw_cell_periodic_orbitals(grid.points), transforms
    )
    q0_pairs = lapw.pair_matrix(orbitals, grid.points, iq=0)[:, pair_columns]
    pivots, residuals, initial_max = pivoted_cholesky_order(
        q0_pairs, min(THRESHOLDS), norb
    )

    active_reference_grams = {
        iq: reference_grams[iq][np.ix_(pair_columns, pair_columns)]
        for iq in range(lapw.NK)
    }
    reference_hartree, reference_exchange = physical_contractions(
        active_reference_grams, norb, occupied
    )
    occupied_pairs = occupied_pair_vector(norb, occupied)
    results = []
    for threshold in THRESHOLDS:
        nmu = rank_for_threshold(residuals, initial_max, threshold)
        grams, fit_error = build_candidate_grams(
            grid,
            orbitals,
            pivots[:nmu],
            g_integer,
            reciprocal,
            pair_columns,
        )
        hartree, exchange = physical_contractions(grams, norb, occupied)
        eri_error = max(
            norm(grams[iq] - active_reference_grams[iq])
            / norm(active_reference_grams[iq])
            for iq in range(lapw.NK)
        )
        action_numerator = 0.0
        action_denominator = 0.0
        for iq in range(lapw.NK):
            difference_action = (
                grams[iq] - active_reference_grams[iq]
            ) @ occupied_pairs
            reference_action = active_reference_grams[iq] @ occupied_pairs
            action_numerator += norm(difference_action) ** 2
            action_denominator += norm(reference_action) ** 2
        results.append(
            ScanResult(
                grid=grid.name,
                window=window,
                norb=norb,
                threshold=threshold,
                npoints=len(grid.points),
                nmu=nmu,
                residual=float(residuals[nmu - 1] if nmu else initial_max),
                fit_frobenius=fit_error,
                eri_frobenius=float(eri_error),
                hartree_error_over_exchange=(
                    abs(hartree - reference_hartree) / abs(reference_exchange)
                ),
                exchange_relative=abs(exchange / reference_exchange - 1.0),
                occupied_action=float(np.sqrt(action_numerator / action_denominator)),
            )
        )
    return results


def format_results(results: list[ScanResult]) -> str:
    lines = [
        "=== CoQui-style threshold scan on synthetic periodic LAPW ===",
        "q=0 CoQui-normalized pivoted-Cholesky selection; fixed integer |G|^2 <= 12",
        (
            "windows: all->valence selects with bands 0..5 and evaluates 0,1,2; "
            "valence-only selects and evaluates 0,1,2"
        ),
        (
            f"{'grid':<25}{'window':<16}{'thresh':>10}{'npts':>7}{'Nmu':>7}{'alpha':>8}"
            f"{'resid':>11}{'fit-F':>11}{'ERI-F':>11}"
            f"{'dEH/|EX|':>11}{'dEX':>11}{'occ-act':>11}"
        ),
    ]
    for item in results:
        lines.append(
            f"{item.grid:<25}{item.window:<16}{item.threshold:>10.1e}"
            f"{item.npoints:>7d}{item.nmu:>7d}{item.nmu / item.norb:>8.2f}"
            f"{item.residual:>11.2e}{item.fit_frobenius:>11.2e}"
            f"{item.eri_frobenius:>11.2e}"
            f"{item.hartree_error_over_exchange:>11.2e}"
            f"{item.exchange_relative:>11.2e}{item.occupied_action:>11.2e}"
        )
    return "\n".join(lines)


def main() -> None:
    reference = lapw.composite_grid("reference", 38, 110, 20)
    raw_reference = lapw.raw_cell_periodic_orbitals(reference.points)
    transforms = lapw.orthonormalizers(raw_reference, reference.weights)
    lapw.verify_basis_structure(transforms)

    g_integer, reciprocal = lapw.reciprocal_vectors(cutoff_squared=12)
    _, reference_grams = lapw.build_reference(
        reference, transforms, g_integer, reciprocal
    )

    candidates = [
        lapw.composite_grid("adaptive 14x50 + 14^3", 14, 50, 14),
        lapw.composite_grid("adaptive 26x86 + 18^3", 26, 86, 18),
        lapw.uniform_grid(16),
    ]
    windows = [
        ("all->valence", tuple(range(lapw.NORB)), VALENCE_OCCUPIED),
        ("valence-only", VALENCE_OCCUPIED, VALENCE_OCCUPIED),
    ]
    results = []
    for grid in candidates:
        for window, active_orbitals, occupied_orbitals in windows:
            results.extend(
                scan_grid(
                    grid,
                    transforms,
                    reference_grams,
                    g_integer,
                    reciprocal,
                    window,
                    active_orbitals,
                    occupied_orbitals,
                )
            )

    for grid in candidates:
        for window, _, _ in windows:
            series = [
                item
                for item in results
                if item.grid == grid.name and item.window == window
            ]
            if any(
                tighter.nmu < looser.nmu
                for looser, tighter in zip(series, series[1:])
            ):
                raise AssertionError("Nmu decreased when thresh was tightened")

    refined_valence = [
        item
        for item in results
        if item.grid == "adaptive 26x86 + 18^3"
        and item.window == "valence-only"
    ]
    loose, tight = refined_valence[1], refined_valence[-1]
    if loose.exchange_relative < 0.5:
        raise AssertionError("synthetic loose-threshold counterexample vanished")
    if tight.exchange_relative > 5.0e-3 or tight.occupied_action > 3.0e-2:
        raise AssertionError("tight active-window observable check failed")

    output = format_results(results)
    print(output)
    Path(__file__).with_name("thc_lapw_coqui_threshold_results.txt").write_text(
        output + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
