from pathlib import Path

import numpy as np
import pytest

import libmuffintin as mt


FIXTURES = Path(__file__).with_name("fixtures")
INPUT = FIXTURES / "hydrogen_input.toml"


def _run_one_iteration(session: mt.ScfSession, density: mt.RegionalDensity) -> mt.ConvergenceDecision:
    potential = session.potential(density)
    core = session.core(potential)
    lapw = session.lapw(core)
    occupations = session.occupations(lapw)
    assembled = session.density(occupations)
    energy = session.energy(assembled)
    return session.convergence(energy)


def _converged_decision(session: mt.ScfSession) -> mt.ConvergenceDecision:
    decision = _run_one_iteration(session, session.initial_density())
    while not decision.converged:
        decision = _run_one_iteration(session, session.mix(decision))
    return decision


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


def test_stage4_reusing_a_consumed_transition_handle_raises() -> None:
    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    session = mt.CheckpointPhysics(checkpoint).scf_session(INPUT)
    density = session.initial_density()
    session.potential(density)
    with pytest.raises(ValueError, match="already been consumed"):
        session.potential(density)


def test_stage4_mixing_a_converged_decision_raises() -> None:
    checkpoint = mt.load_checkpoint(FIXTURES / "hydrogen_checkpoint.toml")
    session = mt.CheckpointPhysics(checkpoint).scf_session(INPUT)
    decision = _converged_decision(session)
    assert decision.converged
    with pytest.raises(ValueError, match="cannot be mixed"):
        session.mix(decision)


def test_stage4_foreign_session_handle_raises() -> None:
    plan = mt.prepare_dft_scf(INPUT)
    session_a = plan.session()
    session_b = plan.session()
    density_a = session_a.initial_density()
    with pytest.raises(ValueError, match="another staged loop"):
        session_b.potential(density_a)
