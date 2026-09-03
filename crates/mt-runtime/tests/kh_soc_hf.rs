//! Scalar KH Hartree–Fock followed by SOC second-variation regression.

use muffintin::{
    CheckpointPhysics, FockMixing, GammaExchangeTreatment, KhSocCoreTreatment, KhSocValenceHfSpec,
    run_gamma_kh_soc_valence_hf,
};
use muffintin_core::{Hartree, InverseBohr};
use muffintin_dft::{
    FirstVariationWindow, LinearizationEnergyGenerator, ScfChannelIdentity, ScfChannelProvenance,
    ScfChannelRecipe, ScfChannelTreatment, ScfConvergence, ScfCoreState, ScfMixing, ScfRelativity,
    electron_count,
};
use muffintin_io::InitialV2;
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;

#[path = "scalar_hydrogen.rs"]
mod scalar_hydrogen;

#[test]
fn gamma_scalar_hf_then_soc_preserves_closed_shell_density_and_exchange() {
    let checkpoint = scalar_hydrogen::hydrogen_checkpoint();
    let mut physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let mut config = scalar_hydrogen::scalar_config([1, 1, 1], 0.5);
    config.relativity = ScfRelativity::SocSecondVariation {
        window: FirstVariationWindow::new(0, 1).unwrap(),
    };
    config.mixing = ScfMixing::Linear { alpha: 0.5 };
    config.convergence = ScfConvergence {
        energy_tolerance: Hartree(1.0e100),
        density_tolerance: 1.0e100,
        max_iterations: 2,
    };
    let spec = KhSocValenceHfSpec {
        config,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: scalar_hydrogen::coulomb_spec().request,
        gamma: GammaExchangeTreatment::FiniteBody,
        max_fock_iterations: 32,
        fock_density_tolerance: 1.0e-7,
        fock_feedback_tolerance: Hartree(1.0e-8),
        fock_mixing: FockMixing::QuasiNewtonDiis {
            history: 4,
            level_shift: Hartree(0.25),
        },
        core_treatment: KhSocCoreTreatment::ValenceOnly,
    };

    let result = run_gamma_kh_soc_valence_hf(&mut physics, &spec).unwrap();

    assert_eq!(result.occupations.len(), 1);
    assert_eq!(result.orbital_energies.len(), 1);
    assert_eq!(result.orbital_energies[0].len(), 2);
    assert_eq!(result.second_variation_diagnostics.len(), 1);
    assert_eq!(result.second_variation_diagnostics[0].len(), 2);
    assert!(
        (result.orbital_energies[0][0].get() - result.orbital_energies[0][1].get()).abs()
            <= 1.0e-10
    );
    assert!(result.fock_fixed_point_residual <= spec.fock_density_tolerance);
    assert!(result.fock_feedback_residual <= spec.fock_feedback_tolerance);
    assert!(result.second_variation_density_rms <= 1.0e-10);
    assert!(result.exchange_energy_change.get() <= 1.0e-10);
    assert_eq!(result.k_fractional, vec![[0.0; 3]]);
    assert_eq!(result.q_fractional, vec![[0.0; 3]]);
    assert_eq!(result.k_weights, vec![1.0]);
    assert!(result.core_orbitals.is_empty());
}

#[test]
fn frozen_core_enters_scalar_fock_soc_and_total_density_without_expanding_vv() {
    let mut checkpoint = scalar_hydrogen::hydrogen_checkpoint();
    checkpoint.geometry.sites[0].atomic_number = 4;
    let InitialV2::FrozenPotential { potential } = &mut checkpoint.initial else {
        unreachable!()
    };
    for channel in &mut potential.v0.muffin_tins[0].channels {
        for value in &mut channel.real {
            *value *= 4.0;
        }
    }
    let mut physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let mut config = scalar_hydrogen::scalar_config([1, 1, 1], 0.5);
    config.electron_count = 4.0;
    config.basis.channels[0].identity = ScfChannelIdentity::ScalarL { n: 2, l: 0 };
    config.basis.channels.push(ScfChannelRecipe {
        site: "H-1".to_owned(),
        identity: ScfChannelIdentity::ScalarL { n: 3, l: 0 },
        treatment: ScfChannelTreatment::Hdlo,
        derivative_order: 2,
        generator: LinearizationEnergyGenerator::Explicit,
        seed: Some(Hartree(-0.3)),
        provenance: ScfChannelProvenance::Site,
    });
    config.core_sites[0].states = vec![ScfCoreState {
        principal_quantum_number: 1,
        kappa: -1,
        occupation: 2.0,
    }];
    config.relativity = ScfRelativity::SocSecondVariation {
        window: FirstVariationWindow::new(0, 2).unwrap(),
    };
    config.mixing = ScfMixing::Linear { alpha: 0.5 };
    config.convergence = ScfConvergence {
        energy_tolerance: Hartree(1.0e100),
        density_tolerance: 1.0e100,
        max_iterations: 2,
    };
    let spec = KhSocValenceHfSpec {
        config,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: scalar_hydrogen::coulomb_spec().request,
        gamma: GammaExchangeTreatment::FiniteBody,
        max_fock_iterations: 32,
        fock_density_tolerance: 1.0e-7,
        fock_feedback_tolerance: Hartree(1.0e-8),
        fock_mixing: FockMixing::PulayAnderson {
            alpha: 0.5,
            history: 4,
        },
        core_treatment: KhSocCoreTreatment::Frozen,
    };

    let result = run_gamma_kh_soc_valence_hf(&mut physics, &spec).unwrap();

    assert_eq!(result.core_orbitals.len(), 1);
    assert!((electron_count(&result.valence_density).unwrap() - 2.0).abs() < 1.0e-8);
    assert!((electron_count(&result.core_density).unwrap() - 2.0).abs() < 1.0e-8);
    assert!((electron_count(&result.total_density).unwrap() - 4.0).abs() < 1.0e-8);
    assert!(result.core_h0_trace.get().is_finite());
    assert!(result.core_core_exchange.get() < 0.0);
    assert!(result.core_valence_exchange.get() < 0.0);
}
