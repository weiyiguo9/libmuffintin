//! Molecule-in-box natural-grid ISDF exchange regressions.

use muffintin::{
    CheckpointPhysics, GammaExchangeTreatment, IsdfExchangeError, IsdfExchangeSpec,
    NaturalThcGridError, NaturalThcGridSpec, ScalarThcError, build_natural_thc_parent_grid,
    build_scalar_coulomb, build_scalar_isdf_exchange, build_scalar_thc, build_spinor_coulomb,
    build_spinor_isdf_exchange, build_spinor_thc,
};
use muffintin_core::{Bohr, Cell, ReciprocalLattice};
use muffintin_operators::lapw::Provenance;

#[path = "scalar_hydrogen.rs"]
#[allow(unused_imports)]
mod scalar_hydrogen;
#[path = "spinor_hydrogen.rs"]
#[allow(unused_imports)]
mod spinor_hydrogen;

fn occupied_first_band(n_k: usize, n_bands: usize) -> IsdfExchangeSpec {
    let mut occupations = vec![vec![0.0; n_bands]; n_k];
    for row in &mut occupations {
        row[0] = 1.0;
    }
    IsdfExchangeSpec {
        k_weights: vec![1.0 / n_k as f64; n_k],
        occupations,
        gamma: GammaExchangeTreatment::FiniteBody,
    }
}

#[test]
fn scalar_gamma_exchange_matches_the_single_pair_quadratic() {
    let physics = CheckpointPhysics::new(&scalar_hydrogen::hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_hydrogen::scalar_config([1, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let coulomb_spec = scalar_hydrogen::coulomb_spec();
    let meshes = input
        .source
        .radials
        .iter()
        .map(|site| site.mesh.clone())
        .collect::<Vec<_>>();
    let grid = build_natural_thc_parent_grid(
        input.source.partition.clone(),
        *coulomb_spec.request.cell(),
        input.reciprocal,
        &meshes,
        Provenance {
            recipe: Some("molecule-box-natural-grid".to_owned()),
            reference: None,
        },
        NaturalThcGridSpec {
            angular_points_per_shell: 6,
            interstitial_divisions: [3, 3, 3],
        },
    )
    .unwrap();
    assert!(
        grid.points()
            .iter()
            .any(|point| { matches!(point.region, muffintin::ThcRegion::MuffinTin { .. }) })
    );
    assert!(
        grid.points()
            .iter()
            .any(|point| matches!(point.region, muffintin::ThcRegion::Interstitial))
    );

    let thc = build_scalar_thc(
        std::slice::from_ref(&input),
        &grid,
        &scalar_hydrogen::thc_spec(),
    )
    .unwrap();
    let coulomb =
        build_scalar_coulomb(std::slice::from_ref(&input), &thc, &coulomb_spec, &[]).unwrap();
    let result = build_scalar_isdf_exchange(
        std::slice::from_ref(&input),
        &coulomb,
        &occupied_first_band(1, input.pair_columns.n_orb),
    )
    .unwrap();
    let record = &coulomb.records()[0];
    let column = input.pair_columns.encode(0, 0, 0);
    let direct = record
        .operator
        .quadratic_form(&record.vertices[column], &record.vertices[column])
        .unwrap();
    assert!((result.exchange_energy.get() + 0.5 * direct.re).abs() < 1.0e-10);
    assert!(result.exchange_energy.get().is_finite());
    assert!(result.maximum_antihermitian_residual < 1.0e-10);

    let mut reject = occupied_first_band(1, input.pair_columns.n_orb);
    reject.gamma = GammaExchangeTreatment::Reject;
    assert_eq!(
        build_scalar_isdf_exchange(std::slice::from_ref(&input), &coulomb, &reject),
        Err(IsdfExchangeError::GammaHead { q_index: 0 })
    );
}

#[test]
fn scalar_two_k_exchange_rejects_a_map_changed_after_coulomb_construction() {
    let physics = CheckpointPhysics::new(&scalar_hydrogen::hydrogen_checkpoint()).unwrap();
    let config = scalar_hydrogen::scalar_config([2, 1, 1], 0.5);
    let q0 = physics.scalar_product_input(&config, [0.0; 3]).unwrap();
    let q15 = physics
        .scalar_product_input(&config, [1.5, 0.0, 0.0])
        .unwrap();
    let inputs = vec![q0, q15];
    let coulomb_spec = scalar_hydrogen::coulomb_spec();
    let meshes = inputs[0]
        .source
        .radials
        .iter()
        .map(|site| site.mesh.clone())
        .collect::<Vec<_>>();
    let grid = build_natural_thc_parent_grid(
        inputs[0].source.partition.clone(),
        *coulomb_spec.request.cell(),
        inputs[0].reciprocal,
        &meshes,
        Provenance {
            recipe: Some("two-k-sealed-map".to_owned()),
            reference: None,
        },
        NaturalThcGridSpec {
            angular_points_per_shell: 6,
            interstitial_divisions: [3, 3, 3],
        },
    )
    .unwrap();
    let thc = build_scalar_thc(&inputs, &grid, &scalar_hydrogen::thc_spec()).unwrap();
    let coulomb = build_scalar_coulomb(&inputs, &thc, &coulomb_spec, &[]).unwrap();

    let mut altered = inputs.clone();
    altered[0].k_minus_q[1].umklapp.index[0] += 1;
    assert_eq!(
        build_scalar_isdf_exchange(
            &altered,
            &coulomb,
            &occupied_first_band(2, inputs[0].pair_columns.n_orb),
        ),
        Err(IsdfExchangeError::FrozenInputContext)
    );
}

#[test]
fn natural_grid_rejects_same_volume_shear_reciprocal_mismatch() {
    let physics = CheckpointPhysics::new(&scalar_hydrogen::hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_hydrogen::scalar_config([1, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let sheared = Cell::new([
        [Bohr(8.0), Bohr(0.0), Bohr(0.0)],
        [Bohr(2.0), Bohr(8.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(8.0)],
    ])
    .unwrap();
    assert_eq!(
        sheared.volume(),
        input.source.partition.interstitial().cell_volume()
    );
    let sheared_reciprocal = ReciprocalLattice::from_direct(*sheared.basis()).unwrap();
    let meshes = input
        .source
        .radials
        .iter()
        .map(|site| site.mesh.clone())
        .collect::<Vec<_>>();
    let provenance = Provenance {
        recipe: Some("same-volume-shear".to_owned()),
        reference: None,
    };
    let grid_spec = NaturalThcGridSpec {
        angular_points_per_shell: 6,
        interstitial_divisions: [3, 3, 3],
    };
    assert_eq!(
        build_natural_thc_parent_grid(
            input.source.partition.clone(),
            sheared,
            input.reciprocal,
            &meshes,
            provenance.clone(),
            grid_spec,
        ),
        Err(NaturalThcGridError::ReciprocalMismatch)
    );

    let sheared_grid = build_natural_thc_parent_grid(
        input.source.partition.clone(),
        sheared,
        sheared_reciprocal,
        &meshes,
        provenance,
        grid_spec,
    )
    .unwrap();
    assert_eq!(
        build_scalar_thc(
            std::slice::from_ref(&input),
            &sheared_grid,
            &scalar_hydrogen::thc_spec(),
        ),
        Err(ScalarThcError::GridReciprocalMismatch)
    );
}

#[test]
fn spinor_gamma_exchange_uses_the_same_natural_partition_contract() {
    let physics = CheckpointPhysics::new(&spinor_hydrogen::hydrogen_spinor_checkpoint()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_hydrogen::spinor_config([1, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let coulomb_spec = spinor_hydrogen::coulomb_spec();
    let meshes = input
        .source
        .radials
        .iter()
        .map(|site| site.mesh.clone())
        .collect::<Vec<_>>();
    let grid = build_natural_thc_parent_grid(
        input.source.partition.clone(),
        *coulomb_spec.request.cell(),
        input.reciprocal,
        &meshes,
        Provenance {
            recipe: Some("molecule-box-natural-grid".to_owned()),
            reference: None,
        },
        NaturalThcGridSpec {
            angular_points_per_shell: 6,
            interstitial_divisions: [3, 3, 3],
        },
    )
    .unwrap();
    let thc = build_spinor_thc(
        std::slice::from_ref(&input),
        &grid,
        &spinor_hydrogen::thc_spec(),
    )
    .unwrap();
    let coulomb =
        build_spinor_coulomb(std::slice::from_ref(&input), &thc, &coulomb_spec, &[]).unwrap();
    let result = build_spinor_isdf_exchange(
        std::slice::from_ref(&input),
        &coulomb,
        &occupied_first_band(1, input.pair_columns.n_orb),
    )
    .unwrap();
    let record = &coulomb.records()[0];
    let column = input.pair_columns.encode(0, 0, 0);
    let direct = record
        .operator
        .quadratic_form(&record.vertices[column], &record.vertices[column])
        .unwrap();
    assert!((result.exchange_energy.get() + 0.5 * direct.re).abs() < 1.0e-10);
    assert!(result.exchange_energy.get().is_finite());
    assert!(result.maximum_antihermitian_residual < 1.0e-10);
}
