from pathlib import Path

import numpy as np

import libmuffintin as mt


FIXTURES = Path(__file__).with_name("fixtures")
INPUT = FIXTURES / "hydrogen_input.toml"


def test_hydrogen_global_and_staged_scf_share_one_transition_loop() -> None:
    direct = mt.run_dft_scf(INPUT)
    planned = mt.prepare_dft_scf(INPUT).session().run()

    assert direct.converged and planned.converged
    assert direct.iterations == planned.iterations == 2
    np.testing.assert_array_equal(direct.energy_history(), planned.energy_history())
    np.testing.assert_array_equal(
        direct.convergence_history(), planned.convergence_history()
    )
    assert isinstance(direct.restart_checkpoint(), mt.Checkpoint)

    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    session = mt.CheckpointPhysics(checkpoint).scf_session(INPUT)
    density = session.initial_density()
    assert density.iteration == 1
    exported_density = density.export_interstitial()
    assert exported_density["schema"] == "libmuffintin.pyexport"
    assert exported_density["version"] == 1
    assert exported_density["g_vectors"].dtype == np.int32
    assert exported_density["components"].dtype == np.complex128
    assert exported_density["components"].shape[0] == 4
    potential = session.potential(density)
    core = session.core(potential)
    lapw = session.lapw(core)
    occupations = session.occupations(lapw)
    assert isinstance(occupations.values(), np.ndarray)
    assembled = session.density(occupations)
    energy = session.energy(assembled)
    decision = session.convergence(energy)
    assert decision.iteration == 1
    assert not decision.converged
    next_density = session.mix(decision)
    assert next_density.iteration == 2
