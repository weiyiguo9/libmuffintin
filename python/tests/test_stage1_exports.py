from pathlib import Path

import numpy as np

import libmuffintin as mt


FIXTURES = Path(__file__).with_name("fixtures")
SCHEMA_KEYS = {"schema", "version"}


def _assert_export(export: dict, keys: set[str]) -> None:
    assert set(export) == SCHEMA_KEYS | keys
    assert export["schema"] == "libmuffintin.pyexport"
    assert export["version"] == 1


def _assert_array(value: object, dtype: np.dtype, shape: tuple[int, ...]) -> None:
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.dtype(dtype)
    assert value.shape == shape


def test_stage1_hydrogen_exports_follow_pyexport_v1() -> None:
    assert {"Checkpoint", "CheckpointPhysics", "ScalarProductInput", "load_checkpoint"} <= set(
        mt.__all__
    )
    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    assert isinstance(checkpoint, mt.Checkpoint)
    physics = mt.CheckpointPhysics(checkpoint)
    product = physics.scalar_product_input(
        FIXTURES / "hydrogen_input.toml", q=[0.0, 0.0, 0.0]
    )
    assert isinstance(product, mt.ScalarProductInput)

    orbitals = product.export_orbitals()
    _assert_export(
        orbitals,
        {"k_fractional", "band_window_start", "band_window_count", "channels"},
    )
    _assert_array(orbitals["k_fractional"], np.float64, (2, 3))
    assert type(orbitals["band_window_start"]) is int
    assert orbitals["band_window_start"] == 0
    assert type(orbitals["band_window_count"]) is int
    assert orbitals["band_window_count"] == 1
    assert isinstance(orbitals["channels"], list)
    assert len(orbitals["channels"]) == 2
    for spin, channel in enumerate(orbitals["channels"]):
        assert set(channel) == {"spin", "energies", "eigenvectors", "available_bands"}
        assert type(channel["spin"]) is int
        assert channel["spin"] == spin
        _assert_array(channel["energies"], np.float64, (2, 1))
        _assert_array(channel["available_bands"], np.int64, (2,))
        assert isinstance(channel["eigenvectors"], list)
        assert len(channel["eigenvectors"]) == 2
        for eigenvectors, shape in zip(
            channel["eigenvectors"], [(1, 1), (2, 1)], strict=True
        ):
            _assert_array(eigenvectors, np.complex128, shape)
            assert eigenvectors.flags["F_CONTIGUOUS"]

    for spin in (0, 1):
        for k_index, basis_dimension in enumerate((1, 2)):
            basis = product.export_basis(k_index, spin)
            _assert_export(
                basis,
                {
                    "k_index",
                    "spin",
                    "basis_dimension",
                    "plane_wave_count",
                    "plane_wave_g",
                    "plane_wave_k_cartesian",
                    "plane_wave_k_plus_g",
                    "apw_labels",
                    "apw_coefficients",
                    "local_orbital_rows",
                },
            )
            assert type(basis["k_index"]) is int
            assert basis["k_index"] == k_index
            assert type(basis["spin"]) is int
            assert basis["spin"] == spin
            assert type(basis["basis_dimension"]) is int
            assert basis["basis_dimension"] == basis_dimension
            assert type(basis["plane_wave_count"]) is int
            assert basis["plane_wave_count"] == basis_dimension
            _assert_array(basis["plane_wave_g"], np.int32, (basis_dimension, 3))
            _assert_array(
                basis["plane_wave_k_cartesian"], np.float64, (basis_dimension, 3)
            )
            _assert_array(
                basis["plane_wave_k_plus_g"], np.float64, (basis_dimension, 3)
            )
            _assert_array(basis["apw_labels"], np.int64, (4 * basis_dimension, 4))
            _assert_array(
                basis["apw_coefficients"],
                np.complex128,
                (4 * basis_dimension, 2),
            )
            _assert_array(basis["local_orbital_rows"], np.int64, (0, 6))

    radials = product.export_radials()
    _assert_export(
        radials,
        {
            "mesh_site",
            "mesh_first",
            "mesh_increment",
            "mesh_count",
            "mesh_offsets",
            "mesh_radii",
            "mesh_weights",
            "radial_labels",
            "sample_offsets",
            "large",
            "small_present",
            "small_offsets",
            "small",
        },
    )
    _assert_array(radials["mesh_site"], np.int64, (1,))
    _assert_array(radials["mesh_first"], np.float64, (1,))
    _assert_array(radials["mesh_increment"], np.float64, (1,))
    _assert_array(radials["mesh_count"], np.int64, (1,))
    _assert_array(radials["mesh_offsets"], np.int64, (2,))
    _assert_array(radials["mesh_radii"], np.float64, (61,))
    _assert_array(radials["mesh_weights"], np.float64, (61,))
    _assert_array(radials["radial_labels"], np.int64, (8, 5))
    _assert_array(radials["sample_offsets"], np.int64, (9,))
    _assert_array(radials["large"], np.float64, (488,))
    _assert_array(radials["small_present"], np.bool_, (8,))
    _assert_array(radials["small_offsets"], np.int64, (9,))
    _assert_array(radials["small"], np.float64, (488,))

    geometry = product.export_geometry()
    _assert_export(
        geometry,
        {
            "site_id",
            "atomic_number",
            "site_fractional",
            "site_cartesian",
            "muffin_tin_radius",
            "direct_lattice",
            "reciprocal_lattice",
            "cell_volume",
        },
    )
    assert geometry["site_id"] == ["H-1"]
    _assert_array(geometry["atomic_number"], np.int64, (1,))
    _assert_array(geometry["site_fractional"], np.float64, (1, 3))
    _assert_array(geometry["site_cartesian"], np.float64, (1, 3))
    _assert_array(geometry["muffin_tin_radius"], np.float64, (1,))
    _assert_array(geometry["direct_lattice"], np.float64, (3, 3))
    _assert_array(geometry["reciprocal_lattice"], np.float64, (3, 3))
    assert type(geometry["cell_volume"]) is float

    kq_map = product.export_kq_map()
    _assert_export(
        kq_map,
        {
            "k_index",
            "kq_index",
            "g_wrap_index",
            "g_wrap_cartesian",
            "transfer_cartesian",
            "global_transfer_index",
        },
    )
    _assert_array(kq_map["k_index"], np.int64, (2,))
    _assert_array(kq_map["kq_index"], np.int64, (2,))
    _assert_array(kq_map["g_wrap_index"], np.int32, (2, 3))
    _assert_array(kq_map["g_wrap_cartesian"], np.float64, (2, 3))
    _assert_array(kq_map["transfer_cartesian"], np.float64, (3,))
    _assert_array(kq_map["global_transfer_index"], np.int32, (3,))

    pair_support = product.export_pair_support()
    _assert_export(
        pair_support,
        {"g_relative_index", "g_relative_cartesian", "g_relative_norm"},
    )
    _assert_array(pair_support["g_relative_index"], np.int32, (3, 3))
    _assert_array(pair_support["g_relative_cartesian"], np.float64, (3, 3))
    _assert_array(pair_support["g_relative_norm"], np.float64, (3,))

    pair_layout = product.export_pair_layout()
    _assert_export(
        pair_layout,
        {"n_k", "n_orb", "n_columns", "core_orbital", "pair_order"},
    )
    assert type(pair_layout["n_k"]) is int
    assert pair_layout["n_k"] == 2
    assert type(pair_layout["n_orb"]) is int
    assert pair_layout["n_orb"] == 1
    assert type(pair_layout["n_columns"]) is int
    assert pair_layout["n_columns"] == 2
    assert pair_layout["core_orbital"] is None
    assert pair_layout["pair_order"] == "k*n_orb^2 + i*n_orb + j"
