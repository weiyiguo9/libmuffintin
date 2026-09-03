//! Bounded neutral closed-shell relaxed-core HF production-path fixture.

use muffintin::{
    CheckpointPhysics, FockMixing, GammaExchangeTreatment, RelaxedCoreHfSpec,
    run_gamma_relaxed_core_hf, run_relaxed_core_hf,
};
use muffintin_core::{Hartree, InverseBohr};
use muffintin_coulomb::CoulombRequest;
use muffintin_dft::{
    CoreFixedPotentialSpec, ScfChannelIdentity, ScfConvergence, ScfCoreState, ScfMixing,
};
use muffintin_io::InitialV2;
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{LATTICE, hydrogen_spinor_checkpoint, spinor_config};

fn closed_shell_core_valence_setup() -> (muffintin_io::CheckpointV2, RelaxedCoreHfSpec) {
    let mut checkpoint = hydrogen_spinor_checkpoint();
    checkpoint.meta.title = "neutral closed-shell core+valence HF smoke".to_owned();
    checkpoint.geometry.sites[0].atomic_number = 4;
    let InitialV2::FrozenPotential { potential } = &mut checkpoint.initial else {
        unreachable!("shared fixture is a frozen-potential checkpoint")
    };
    for channel in &mut potential.v0.muffin_tins[0].channels {
        for value in &mut channel.real {
            *value *= 4.0;
        }
    }

    let mut config = spinor_config([1, 1, 1], 0.5);
    config.electron_count = 4.0;
    config.basis.channels[0].identity = ScfChannelIdentity::ScalarL { n: 2, l: 0 };
    config.core_sites[0].states = vec![ScfCoreState {
        principal_quantum_number: 1,
        kappa: -1,
        occupation: 2.0,
    }];
    config.mixing = ScfMixing::Linear { alpha: 0.5 };
    config.convergence = ScfConvergence {
        energy_tolerance: Hartree(1.0e100),
        density_tolerance: 1.0e100,
        max_iterations: 2,
    };
    let spec = RelaxedCoreHfSpec {
        config,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: CoulombRequest::cubic(LATTICE, 2).unwrap(),
        gamma: GammaExchangeTreatment::FiniteBody,
        max_fock_iterations: 32,
        fock_density_tolerance: 1.0e-7,
        fock_feedback_tolerance: Hartree(1.0e-8),
        fock_mixing: FockMixing::CommutatorDiis { history: 8 },
        core: CoreFixedPotentialSpec {
            action_mixing: 1.0,
            energy_tolerance: Hartree(1.0e100),
            radial_tolerance: 1.0e100,
            vc_imaginary_tolerance: 1.0e-8,
            max_iterations: 2,
        },
        sector_numerical_tolerance: Hartree(1.0e-8),
        maximum_core_shell_spill: 1.0,
    };
    (checkpoint, spec)
}

#[test]
fn neutral_closed_shell_replaces_core_mixes_only_valence_and_keeps_gamma_parity() {
    let (checkpoint, spec) = closed_shell_core_valence_setup();
    let mut gamma_physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let gamma = run_gamma_relaxed_core_hf(&mut gamma_physics, &spec).unwrap();

    assert!(
        gamma
            .diagnostics
            .iter()
            .any(|item| item.fresh_core_replacement_rms > 0.0)
    );
    assert!(gamma.diagnostics.iter().all(|item| {
        item.valence_eigenvalue_identity_residual <= 1.0e-8
            && (item.valence_feedback_vv_cv_trace.get() - item.trace_vv.get() - item.trace_cv.get())
                .abs()
                <= 1.0e-12
    }));
    assert_eq!(gamma.diagnostics[0].core_inner_iterations, vec![1]);
    assert!(gamma.fock_feedback_residual <= spec.fock_feedback_tolerance);
    assert!((gamma.diagnostics[0].valence_electron_count - 2.0).abs() <= 1.0e-8);
    assert!((gamma.diagnostics[0].core_electron_count - 2.0).abs() <= 1.0e-8);
    assert!((gamma.diagnostics[0].total_electron_count - 4.0).abs() <= 1.0e-8);

    let mut generic_physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let generic = run_relaxed_core_hf(&mut generic_physics, &spec).unwrap();
    assert_eq!(generic.k_fractional, vec![[0.0; 3]]);
    assert_eq!(generic.q_fractional, vec![[0.0; 3]]);
    assert_eq!(generic.valence_density, gamma.valence_density);
    assert_eq!(generic.core_density, gamma.core_density);
    assert_eq!(generic.total_density, gamma.total_density);
    assert_eq!(generic.occupations, gamma.occupations);
    assert_eq!(generic.sector_exchange, gamma.sector_exchange);
    assert_eq!(generic.total_energy, gamma.total_energy);
}
