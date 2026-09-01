//! Bounded production-path fixture for Gamma valence-only spinor HF.

use muffintin::{
    CheckpointPhysics, GammaExchangeTreatment, GammaValenceHfError, GammaValenceHfSpec,
    IsdfExchangeError, IsdfExchangeSpec, SpinorMpbSelection, SpinorMpbSpec, build_spinor_mpb,
    build_spinor_mpb_exchange, run_gamma_valence_hf,
};
use muffintin_core::InverseBohr;
use muffintin_coulomb::CoulombRequest;
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;
use muffintin_tensor::DenseEigenvectors;

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{LATTICE, hydrogen_spinor_checkpoint, spinor_config};

fn full_mpb_spec(n_k: usize, n_orb: usize) -> SpinorMpbSpec {
    SpinorMpbSpec {
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections: (0..n_k)
            .flat_map(|k| {
                (0..n_orb).flat_map(move |left_band| {
                    (0..n_orb).map(move |right_band| SpinorMpbSelection {
                        k,
                        left_band,
                        right_band,
                    })
                })
            })
            .collect(),
    }
}

#[test]
fn gamma_hydrogen_rebuilds_full_vv_feedback_and_rejects_stale_orbitals() {
    let checkpoint = hydrogen_spinor_checkpoint();
    let config = spinor_config([1, 1, 1], 0.5);
    let physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let input = physics.spinor_product_input(&config, [0.0; 3]).unwrap();
    let n_orb = input.pair_columns.n_orb;
    let mpb = build_spinor_mpb(&input, &full_mpb_spec(input.pair_columns.n_k, n_orb)).unwrap();
    assert_eq!(mpb.vertices.len(), input.pair_columns.n_columns().unwrap());
    let request = CoulombRequest::cubic(LATTICE, 2).unwrap();
    let exchange_spec = IsdfExchangeSpec {
        k_weights: vec![1.0],
        occupations: vec![
            (0..n_orb)
                .map(|band| if band == 0 { 1.0 } else { 0.0 })
                .collect(),
        ],
        gamma: GammaExchangeTreatment::FiniteBody,
    };
    let one_shot = build_spinor_mpb_exchange(
        std::slice::from_ref(&input),
        std::slice::from_ref(&mpb),
        &request,
        &exchange_spec,
    )
    .unwrap();
    let rebuilt_mpb =
        build_spinor_mpb(&input, &full_mpb_spec(input.pair_columns.n_k, n_orb)).unwrap();
    let rebuilt = build_spinor_mpb_exchange(
        std::slice::from_ref(&input),
        std::slice::from_ref(&rebuilt_mpb),
        &request,
        &exchange_spec,
    )
    .unwrap();
    assert_eq!(one_shot, rebuilt);
    assert!(one_shot.maximum_antihermitian_residual <= 1.0e-8);

    let mut rotated = input.clone();
    let source = &input.orbitals.eigenvectors[0];
    assert!(source.columns() >= 2);
    let mut values = source.to_host_column_major();
    let rows = source.rows();
    let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
    for row in 0..rows {
        let left = source.at(row, 0);
        let right = source.at(row, 1);
        values[row] = inverse_sqrt_two * (left + right);
        values[rows + row] = inverse_sqrt_two * (-left + right);
    }
    rotated.orbitals.eigenvectors[0] =
        DenseEigenvectors::from_host_column_major(rows, source.columns(), values).unwrap();
    assert!(matches!(
        build_spinor_mpb_exchange(&[rotated], &[mpb], &request, &exchange_spec),
        Err(IsdfExchangeError::MpbContext { index: 0 })
    ));

    let mut live_physics = CheckpointPhysics::new(&checkpoint).unwrap();
    let result = run_gamma_valence_hf(
        &mut live_physics,
        &GammaValenceHfSpec {
            config,
            product_l_max: 2,
            product_g_max: InverseBohr(1.5),
            overlap_tolerance: DEFAULT_TOLERANCE,
            coulomb: request,
            max_fock_iterations: 2,
            fock_density_tolerance: 1.0e100,
        },
    );
    match result {
        Ok(result) => {
            assert!(result.exchange_rebuilds >= 4);
            assert_eq!(result.occupations.len(), 1);
            assert_eq!(result.orbital_energies.len(), 1);
            assert!(result.maximum_antihermitian_residual <= 1.0e-8);
            assert!(
                result.diagnostics[0]
                    .first_one_shot_parity_residual
                    .is_some_and(|residual| residual <= 1.0e-8)
            );
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
        Err(GammaValenceHfError::FockNotConverged { iterations: 2, .. })
        | Err(GammaValenceHfError::NotConverged { iterations: 2, .. }) => {}
        Err(error) => panic!("unexpected Gamma HF failure: {error}"),
    }
}
