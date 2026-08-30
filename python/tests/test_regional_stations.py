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
        site_ids=["C-1"],
        atomic_numbers=[6],
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

    core = mt.CoreStation(
        [mt.CoreSite(0, "C-1", [mt.CoreState(1, -1, occupation=2.0)])]
    ).solve(potential)
    assert math.isfinite(core.core_eigenvalue_sum)
    assert core.core_eigenvalue_sum < 0.0
    assert core.site_ids() == ["C-1"]
    assert core.requested_charges() == pytest.approx([2.0])
    assert core.represented_charges() == pytest.approx([2.0])
    core_density = core.density()
    core_export = core_density.export_interstitial()
    assert np.isfinite(core_export["components"]).all()
    assert np.isfinite(core_export["mt_components"]).all()
    combined = roundtrip.add_scaled(1.0, core_density)
    assert np.isfinite(combined.export_interstitial()["components"]).all()

    band_energy = -1.25
    occupation_correction = 0.02
    energy = mt.evaluate_total_energy(
        potential,
        combined,
        band_energy=band_energy,
        core_eigenvalue_sum=core.core_eigenvalue_sum,
        occupation_correction=occupation_correction,
    )
    expected_total = (
        band_energy
        + core.core_eigenvalue_sum
        + 0.5 * (potential.madelung - potential.coulomb)
        + potential.exchange_correlation
        - potential.exchange_correlation_potential
        + occupation_correction
    )
    assert energy.total == pytest.approx(expected_total)
    assert energy.density_rms == pytest.approx(roundtrip.difference_rms(combined))
    assert energy.energy_change is None
    changed = mt.evaluate_total_energy(
        potential,
        combined,
        band_energy=band_energy,
        core_eigenvalue_sum=core.core_eigenvalue_sum,
        occupation_correction=occupation_correction,
        previous_total=energy.total + 0.001,
    )
    assert changed.energy_change == pytest.approx(0.001)

    mixer_output = roundtrip.add_scaled(-0.25, roundtrip)
    linear_mixer = mt.DensityMixer.linear(0.4)
    linear_step = linear_mixer.step(roundtrip, mixer_output)
    assert linear_step.status == "linear"
    expected_mixed = roundtrip.add_scaled(-0.1, roundtrip)
    linear_density = linear_step.density()
    linear_export = linear_density.export_interstitial()
    expected_export = expected_mixed.export_interstitial()
    np.testing.assert_allclose(linear_export["components"], expected_export["components"])
    np.testing.assert_allclose(
        linear_export["mt_components"], expected_export["mt_components"]
    )

    broyden = mt.DensityMixer.broyden2(0.4, 4)
    broyden_step = broyden.step(roundtrip, mixer_output)
    assert broyden_step.status == "nonlinear-warmup"
    assert broyden.history_length == 1
    pulay = mt.DensityMixer.pulay_anderson(0.4, 4)
    pulay_step = pulay.step(roundtrip, mixer_output)
    assert pulay_step.status == "nonlinear-warmup"
    assert pulay.history_length == 1
    np.testing.assert_array_equal(pulay.last_pulay_coefficients(), [1.0])
    for step in (linear_step, broyden_step, pulay_step):
        assert math.isfinite(step.density().residual_rms())

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
