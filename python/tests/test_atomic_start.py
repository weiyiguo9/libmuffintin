import math

import libmuffintin as mt


def test_atomic_start_closes_charge_and_writes_a_loadable_checkpoint(tmp_path) -> None:
    first = 1.0e-4
    radius = 1.5
    point_count = 61
    log_increment = math.log(radius / first) / (point_count - 1)
    structure = mt.Structure(
        lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        site_ids=["H-1"],
        atomic_numbers=[1],
        fractional_positions=[[0.5, 0.5, 0.5]],
        radial_meshes=[(first, log_increment, point_count)],
        radial_equations=["scalar-koelling-harmon"],
        linearization_energies=[[(0, -0.3)]],
    )
    layout = mt.RegionalFieldLayout.from_g_cutoff(
        structure,
        g_cutoff=4.0,
        muffin_tin_l_max=2,
    )
    controls = mt.FreeAtomControls(
        mesh_first=1.0e-6,
        mesh_log_increment=0.01,
        mesh_point_count=1683,
        mixing=0.3,
        potential_tolerance=2.0e-5,
        tail_tolerance=1.0e-7,
        max_iterations=120,
        angular_points=50,
    )

    start = mt.materialize_atomic_start(
        structure,
        layout,
        xc="lda-pw92",
        free_atom_controls=controls,
    )
    assert set(start.charge_closure) == {
        "interstitial_fraction",
        "response_volume",
        "target_electron_count",
        "uncorrected_electron_count",
        "zero_mode_coefficient_correction",
        "represented_electron_count",
    }
    assert start.charge_closure["target_electron_count"] == 1.0
    assert abs(start.charge_closure["represented_electron_count"] - 1.0) < 1.0e-11

    path = tmp_path / "atomic_start.toml"
    start.checkpoint.write(path)
    loaded = mt.load_checkpoint(path)
    assert isinstance(loaded, mt.Checkpoint)
    assert mt.CheckpointPhysics(loaded).export_restart_density() is not None
