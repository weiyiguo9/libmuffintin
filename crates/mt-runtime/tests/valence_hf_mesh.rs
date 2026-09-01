//! Focused full-BZ valence-HF topology gates.

use muffintin::{
    CheckpointPhysics, ValenceHfError, ValenceHfSpec, run_valence_hf,
};
use muffintin_core::{Bohr, InverseBohr};
use muffintin_coulomb::CoulombRequest;
use muffintin_dft::{ScfKReduction, ScfMixing};
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
