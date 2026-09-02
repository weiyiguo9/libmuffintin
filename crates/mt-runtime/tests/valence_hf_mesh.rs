//! Focused full-BZ valence-HF topology gates.

use muffintin::{CheckpointPhysics, ValenceHfError, ValenceHfSpec, run_valence_hf};
use muffintin_core::{Bohr, Hartree, InverseBohr};
use muffintin_coulomb::CoulombRequest;
use muffintin_dft::{ScfConvergence, ScfKReduction, ScfMixing};
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{LATTICE, hydrogen_spinor_checkpoint, spinor_config};

#[test]
fn shifted_2x2x1_uses_unshifted_q_permutations_and_rejects_symmetry_reduction() {
    let checkpoint = hydrogen_spinor_checkpoint();
    let physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let mut config = spinor_config([2, 2, 1], 0.5);
    config.k_mesh.shift = [0.5, 0.5, 0.0];
    let q_fractional = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
    ];
    let inputs = q_fractional
        .iter()
        .map(|&q| physics.spinor_product_input(&config, q).unwrap())
        .collect::<Vec<_>>();
    let k_fractional = &inputs[0].orbitals.k_fractional;
    assert_eq!(
        k_fractional,
        &[
            [0.25, 0.25, 0.0],
            [0.75, 0.25, 0.0],
            [0.25, 0.75, 0.0],
            [0.75, 0.75, 0.0],
        ]
    );
    for (q_index, input) in inputs.iter().enumerate() {
        assert_eq!(input.orbitals, inputs[0].orbitals);
        let mut targets = input
            .k_minus_q
            .iter()
            .map(|mapped| mapped.kq_index)
            .collect::<Vec<_>>();
        targets.sort_unstable();
        assert_eq!(targets, vec![0, 1, 2, 3]);
        for mapped in &input.k_minus_q {
            let k = k_fractional[mapped.k_index];
            let kq = k_fractional[mapped.kq_index];
            for axis in 0..3 {
                let residual = k[axis]
                    - q_fractional[q_index][axis]
                    - kq[axis]
                    - f64::from(mapped.umklapp.index[axis]);
                assert!(residual.abs() <= 1.0e-12);
            }
        }
    }

    config.k_mesh.reduction = ScfKReduction::Symmetry {
        symprec: Bohr(1.0e-6),
        include_time_reversal: true,
    };
    config.mixing = ScfMixing::Linear { alpha: 0.5 };
    let spec = ValenceHfSpec {
        config,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: CoulombRequest::cubic(LATTICE, 2).unwrap(),
        max_fock_iterations: 2,
        fock_density_tolerance: 1.0e-7,
        fock_mixing: 0.5,
    };
    let mut rejected_physics = CheckpointPhysics::new(&checkpoint).unwrap();
    assert!(matches!(
        run_valence_hf(&mut rejected_physics, &spec),
        Err(ValenceHfError::SymmetryReduction)
    ));
}

#[test]
fn shifted_2x1x1_executes_complete_q_slice_and_per_k_feedback() {
    let checkpoint = hydrogen_spinor_checkpoint();
    let mut config = spinor_config([2, 1, 1], 0.5);
    config.k_mesh.shift = [0.5, 0.0, 0.0];
    config.mixing = ScfMixing::Linear { alpha: 0.5 };
    config.convergence = ScfConvergence {
        energy_tolerance: Hartree(1.0e-6),
        density_tolerance: 1.0e-5,
        max_iterations: 32,
    };
    let spec = ValenceHfSpec {
        config,
        product_l_max: 2,
        product_g_max: InverseBohr(1.0),
        overlap_tolerance: DEFAULT_TOLERANCE,
        coulomb: CoulombRequest::cubic(LATTICE, 2).unwrap(),
        max_fock_iterations: 24,
        fock_density_tolerance: 1.0e-7,
        fock_mixing: 0.5,
    };
    let mut physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let result = run_valence_hf(&mut physics, &spec).unwrap();

    assert_eq!(
        result.k_fractional,
        vec![[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]
    );
    assert_eq!(result.q_fractional, vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]);
    assert_eq!(result.k_weights, vec![0.5, 0.5]);
    assert_eq!(result.bands.points().len(), 2);
    assert_eq!(result.occupations.len(), 2);
    assert_eq!(result.orbital_energies.len(), 2);
    assert_eq!(result.first_one_shot_exchange.band_matrices.len(), 2);
    for (k_index, matrix) in result
        .first_one_shot_exchange
        .band_matrices
        .iter()
        .enumerate()
    {
        assert_eq!(matrix.k_index(), k_index);
    }
    assert!(result.exchange_rebuilds >= 2);
    assert!(result.fock_fixed_point_residual <= spec.fock_density_tolerance);
    assert!(result.regional_density_rms <= spec.config.convergence.density_tolerance);
    assert!(
        result.diagnostics[0]
            .first_global_solve_identity_residual
            .is_some_and(|residual| residual <= 1.0e-8)
    );
    assert!(
        result
            .diagnostics
            .iter()
            .all(|item| item.lifting_identity_residual <= 1.0e-8)
    );
}
