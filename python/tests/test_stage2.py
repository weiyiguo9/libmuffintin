from pathlib import Path

import numpy as np
import pytest

import libmuffintin as mt


FIXTURES = Path(__file__).with_name("fixtures")


def _parent_grid(product: mt.ScalarProductInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometry = product.export_geometry()
    radials = product.export_radials()
    origin = geometry["site_cartesian"][0]
    radii = radials["mesh_radii"]
    middle = len(radii) // 2

    def on_shell(radius: float, direction: list[float]) -> np.ndarray:
        direction_array = np.asarray(direction, dtype=np.float64)
        return origin + radius * direction_array / np.linalg.norm(direction_array)

    coordinates = np.asarray(
        [
            on_shell(radii[0], [0.4, -0.3, 0.2]),
            on_shell(radii[middle], [1.0, 0.0, 0.0]),
            on_shell(radii[middle], [0.0, 1.0, 0.0]),
            [0.2, 0.2, 0.2],
            [5.0, 4.0, 4.0],
            [2.0, 6.5, 4.0],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([0.35, 0.0, 0.45, 0.8, 0.15, 0.25], dtype=np.float64)
    regions = np.asarray(
        [[0, 0, 0], [0, 0, middle], [0, 0, middle], [1, -1, -1], [1, -1, -1], [1, -1, -1]],
        dtype=np.int64,
    )
    return coordinates, weights, regions


def test_stage2_scalar_handles_exports_and_acceptance_gates() -> None:
    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    physics = mt.CheckpointPhysics(checkpoint)
    product = physics.scalar_product_input(
        FIXTURES / "hydrogen_input.toml", q=[0.0, 0.0, 0.0]
    )
    product_slice = physics.scalar_q_slice(FIXTURES / "hydrogen_input.toml")
    coordinates, weights, regions = _parent_grid(product)

    samples = mt.sample_scalar_orbitals(product, coordinates, weights, regions, spin=0)
    assert samples["schema"] == "libmuffintin.pyexport"
    assert samples["version"] == 1
    assert samples["large"].dtype == np.complex128
    assert samples["small"].dtype == np.complex128
    assert samples["large"].shape == samples["small"].shape == (6, 2, 1)

    selections = np.asarray([[0, 0, 0, 0]], dtype=np.int64)
    mpb = mt.build_scalar_mpb(product, selections, 2, 1.5, 1.0e-4)
    auxiliary = mpb.export_auxiliary()
    vertices = mpb.export_vertices()
    assert auxiliary["regions"].dtype == np.int64
    assert auxiliary["q_umklapp_index"].dtype == np.int32
    assert auxiliary["interstitial_g_index"].dtype == np.int32
    assert vertices["labels"].dtype == np.int64
    assert vertices["coefficients"].dtype == np.complex128
    mpb_coulomb = mt.build_scalar_mpb_coulomb(mpb, lexp=2).export_matrix()
    assert mpb_coulomb["q_index"] is None
    assert mpb_coulomb["spin"] is None
    assert mpb_coulomb["matrix"].shape == (
        auxiliary["dimension"],
        auxiliary["dimension"],
    )

    candidates = np.asarray([0, 2, 3, 4, 5], dtype=np.int64)
    thc = mt.build_scalar_thc(
        product_slice,
        coordinates,
        weights,
        regions,
        spin=0,
        rank=1,
        engine="qrcp",
        candidates=candidates,
    )
    repeated = mt.build_scalar_thc(
        product_slice,
        coordinates,
        weights,
        regions,
        spin=0,
        rank=1,
        engine="qrcp",
        candidates=candidates,
    )
    selection = thc.export_selection()
    repeated_selection = repeated.export_selection()
    np.testing.assert_array_equal(selection["pivots"], repeated_selection["pivots"])
    assert selection["effective_rank"] == 1

    exported = thc.export_records()
    assert len(exported["records"]) == 2
    for record in exported["records"]:
        pair_samples = record["pair_samples"]
        point_ids = record["point_ids"]
        np.testing.assert_allclose(
            record["vertices"], pair_samples[point_ids, :].T, rtol=0.0, atol=0.0
        )
        reconstructed = record["zeta"] @ record["vertices"].T
        weighted_difference = (pair_samples - reconstructed) * np.sqrt(weights)[:, None]
        weighted_reference = pair_samples * np.sqrt(weights)[:, None]
        relative = np.linalg.norm(weighted_difference) / np.linalg.norm(weighted_reference)
        np.testing.assert_allclose(relative, record["l2_all"][0], rtol=1.0e-12, atol=1.0e-14)

    coulomb = mt.build_scalar_coulomb(
        product_slice,
        thc,
        lexp=2,
        interpolation_pw_cutoff=1.5,
        interpolation_l_max=1,
        comparisons=[(0, mpb, 0)],
    )
    matrix = coulomb.export_matrix(0)
    diagnostics = coulomb.export_diagnostics()
    column = int(diagnostics["column"][0])
    coefficient = exported["records"][0]["vertices"][column]
    quadratic = coefficient.conj() @ matrix["matrix"] @ coefficient
    np.testing.assert_allclose(
        quadratic,
        diagnostics["thc_quadratic"][0],
        rtol=0.0,
        atol=1.0e-12,
    )

    # Gate 2 (same-engine THC reproduction) is scipy-only; a missing scipy
    # shows as a visible skip here without hiding the scipy-independent
    # assertions above.
    scipy_linalg = pytest.importorskip("scipy.linalg")
    stacked = np.concatenate(
        [record["pair_samples"][candidates, :].T for record in exported["records"]],
        axis=0,
    )
    stacked *= np.sqrt(weights[candidates])[None, :]
    _, _, scipy_pivots = scipy_linalg.qr(stacked, mode="economic", pivoting=True)
    np.testing.assert_array_equal(selection["pivots"], candidates[scipy_pivots[:1]])
