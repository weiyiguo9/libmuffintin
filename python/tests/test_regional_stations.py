import math

import numpy as np
import pytest

import libmuffintin as mt


def test_regional_density_roundtrip_and_potential_station() -> None:
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
    physics = mt.CheckpointPhysics(start.checkpoint)
    density = physics.restart_density()
    assert isinstance(density, mt.RegionalDensity)

    exported = density.export_interstitial()
    roundtrip = mt.RegionalDensity(
        structure,
        layout,
        exported["angular_basis"],
        exported["components"],
        exported["mt_channel_labels"],
        exported["mt_sample_offsets"],
        exported["mt_components"],
    )
    rebuilt = roundtrip.export_interstitial()
    for key in ("g_vectors", "components", "mt_channel_labels", "mt_sample_offsets", "mt_components"):
        np.testing.assert_array_equal(rebuilt[key], exported[key])
    assert roundtrip.difference_rms(density) == 0.0
    assert roundtrip.difference(density).residual_rms() == 0.0
    assert roundtrip.add_scaled(0.0, density).difference_rms(roundtrip) == 0.0

    potential = mt.build_regional_potential(roundtrip, "lda-pw92")
    frozen = physics.export_frozen_potential()
    rebuilt_potential = potential.export_interstitial()
    np.testing.assert_allclose(rebuilt_potential["components"], frozen["components"])
    np.testing.assert_allclose(rebuilt_potential["mt_components"], frozen["mt_components"])
    assert all(
        math.isfinite(value)
        for value in (
            potential.madelung,
            potential.coulomb,
            potential.exchange_correlation,
            potential.exchange_correlation_potential,
        )
    )

    with pytest.raises(ValueError, match="mt_sample_offsets"):
        mt.RegionalDensity(
            structure,
            layout,
            exported["angular_basis"],
            exported["components"],
            exported["mt_channel_labels"],
            exported["mt_sample_offsets"][:-1],
            exported["mt_components"],
        )
