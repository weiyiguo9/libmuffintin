//! Scalar KH Hartree–Fock followed by SOC second-variation regression.

use muffintin::{
    CheckpointPhysics, FockMixing, GammaExchangeTreatment, KhSocValenceHfSpec,
    run_gamma_kh_soc_valence_hf,
};
use muffintin_core::{Hartree, InverseBohr};
use muffintin_dft::{FirstVariationWindow, ScfConvergence, ScfMixing, ScfRelativity};
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
        fock_mixing: FockMixing::PulayAnderson {
            alpha: 0.5,
            history: 4,
        },
    };

    let result = run_gamma_kh_soc_valence_hf(&mut physics, &spec).unwrap();

    assert_eq!(result.occupations.len(), 1);
    assert_eq!(result.orbital_energies.len(), 1);
    assert_eq!(result.orbital_energies[0].len(), 2);
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
}
