from pathlib import Path

import numpy as np

import libmuffintin as mt


FIXTURES = Path(__file__).with_name("fixtures")
SCHEMA_KEYS = {"schema", "version"}


def _assert_export(export: dict, keys: set[str]) -> None:
    assert set(export) == SCHEMA_KEYS | keys
    assert export["schema"] == "libmuffintin.pyexport"
    assert export["version"] == 1


def test_v1_hydrogen_checkpoint_exports_full_regions_and_exact_radial_trace() -> None:
    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    physics = mt.CheckpointPhysics(checkpoint)

    potential = physics.export_frozen_potential()
    _assert_export(
        potential,
        {
            "angular_basis",
            "g_vectors",
            "components",
            "mt_mesh_site",
            "mt_mesh_first",
            "mt_mesh_increment",
            "mt_mesh_count",
            "mt_mesh_offsets",
            "mt_mesh_radii",
            "mt_mesh_weights",
            "mt_channel_labels",
            "mt_sample_offsets",
            "mt_components",
        },
    )
    assert potential["angular_basis"] == "complex-condon-shortley"
    assert potential["g_vectors"].dtype == np.int32
    assert potential["g_vectors"].shape == (1, 3)
    assert potential["components"].dtype == np.complex128
    assert potential["components"].shape == (4, 1)
    np.testing.assert_array_equal(potential["mt_mesh_site"], [0])
    np.testing.assert_array_equal(potential["mt_mesh_count"], [61])
    np.testing.assert_array_equal(potential["mt_mesh_offsets"], [0, 61])
    assert potential["mt_mesh_radii"].shape == (61,)
    assert potential["mt_mesh_weights"].shape == (61,)
    np.testing.assert_array_equal(potential["mt_channel_labels"], [[0, 0, 0]])
    np.testing.assert_array_equal(potential["mt_sample_offsets"], [0, 61])
    assert potential["mt_components"].dtype == np.complex128
    assert potential["mt_components"].shape == (4, 61)
    np.testing.assert_allclose(
        potential["mt_components"][0, [0, -1]],
        np.sqrt(4.0 * np.pi) * np.array([-10000.0, -1.0]),
    )
    np.testing.assert_array_equal(potential["mt_components"][1:], 0.0)
    assert physics.export_restart_density() is None

    step = 1.0e-5
    energies = np.array([-0.3 - step, -0.3, -0.3 + step])
    radials = physics.sample_frozen_scalar_radials("H-1", 0, energies)
    _assert_export(
        radials,
        {
            "site_index",
            "site_id",
            "l",
            "energies",
            "mesh_first",
            "mesh_increment",
            "mesh_count",
            "mesh_radii",
            "radial_samples",
            "boundary_radius",
            "boundary_radial",
            "log_derivative",
            "energy_derivative_boundary_radial",
        },
    )
    assert (radials["site_index"], radials["site_id"], radials["l"]) == (0, "H-1", 0)
    np.testing.assert_array_equal(radials["energies"], energies)
    assert radials["mesh_count"] == 61
    assert radials["mesh_radii"].shape == (61,)
    assert radials["radial_samples"].shape == (3, 61)
    assert radials["boundary_radial"].shape == (3, 2)
    assert radials["energy_derivative_boundary_radial"].shape == (3, 2)
    np.testing.assert_allclose(
        radials["radial_samples"][:, -1], radials["boundary_radial"][:, 0]
    )
    np.testing.assert_allclose(
        radials["log_derivative"],
        radials["boundary_radial"][:, 1] / radials["boundary_radial"][:, 0],
    )
    centered_boundary_derivative = (
        radials["boundary_radial"][2] - radials["boundary_radial"][0]
    ) / (2.0 * step)
    np.testing.assert_allclose(
        radials["energy_derivative_boundary_radial"][1],
        centered_boundary_derivative,
        rtol=2.0e-5,
        atol=2.0e-7,
    )
