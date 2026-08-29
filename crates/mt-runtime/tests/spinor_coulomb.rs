//! Public M-L5d spinor sampled-ζ Coulomb tests on frozen M-L5b/M-L5c/M-L5d output.

use std::f64::consts::PI;

use muffintin::{
    SPINOR_COULOMB_EXACTNESS_FLOOR, SnapshotDftPhysics, SpinorCoulombError, SpinorCoulombPairMatch,
    SpinorCoulombSpec, SpinorMpbSelection, SpinorMpbSpec, ThcParentGrid, build_spinor_coulomb,
    build_spinor_mpb, build_spinor_thc,
};
use muffintin_auxiliary_ir::{AuxiliaryLayout, OrbitalPair, PairVertex, TransferQ};
use muffintin_core::{Bohr, InverseBohr};
use muffintin_coulomb::{
    AuxiliaryKind, CoulombError, CoulombRequest, InterpolationProjection, SampledPointSupport,
    assemble_coulomb,
};
use muffintin_grid::Cell;
use muffintin_mpb::DEFAULT_TOLERANCE;
use muffintin_tensor::DenseEigenvectors;
use num_complex::Complex64;

#[path = "spinor_hydrogen.rs"]
mod spinor_hydrogen;

use spinor_hydrogen::{
    coulomb_spec, hydrogen_spinor_snapshot, parent_grid, spinor_config, thc_spec,
};

fn mpb_spec() -> SpinorMpbSpec {
    SpinorMpbSpec {
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections: vec![SpinorMpbSelection {
            k: 0,
            left_band: 0,
            right_band: 0,
        }],
    }
}

/// Rank-1 THC versus MPB quadratic $c^\dagger V c$ on this 6-point hydrogen /
/// LEXP=2 spinor fixture. The bound is three-ish times a typical algebraic
/// gap on this bounded fixture and is not a SPEX or material tolerance.
const FIXTURE_MPB_THC_QUADRATIC_ABS: f64 = 0.5;

fn dense_quadratic(matrix: &[Complex64], coefficients: &[Complex64]) -> Complex64 {
    let n = coefficients.len();
    assert_eq!(matrix.len(), n * n);
    let mut acc = Complex64::default();
    for row in 0..n {
        let mut applied = Complex64::default();
        for (column, coefficient) in coefficients.iter().enumerate() {
            applied += matrix[row * n + column] * coefficient;
        }
        acc += coefficients[row].conj() * applied;
    }
    acc
}

fn dense_action_norm(matrix: &[Complex64], coefficients: &[Complex64]) -> f64 {
    let n = coefficients.len();
    assert_eq!(matrix.len(), n * n);
    let mut norm_sq = 0.0;
    for row in 0..n {
        let mut applied = Complex64::default();
        for (column, coefficient) in coefficients.iter().enumerate() {
            applied += matrix[row * n + column] * coefficient;
        }
        norm_sq += applied.norm_sqr();
    }
    norm_sq.sqrt()
}

#[test]
fn gamma_sampled_coulomb_uses_full_parent_grid_and_keeps_head_as_metadata() {
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_spinor_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let result = build_spinor_coulomb(&[input], &thc, &coulomb_spec(), &[]).unwrap();
    assert_eq!(result.records().len(), 1);
    let record = &result.records()[0];
    assert_eq!(record.q_index, 0);
    assert_eq!(record.sampled.n_grid(), 6);
    assert_eq!(record.sampled.zeta().len(), 6 * record.operator.dimension());
    assert!(
        record
            .sampled
            .weights()
            .iter()
            .any(|weight| weight.get() == 0.0)
    );
    assert!(matches!(
        record.sampled.supports()[0],
        SampledPointSupport::MuffinTin {
            site: 0,
            radial_index: 0
        }
    ));
    assert!(matches!(
        record.sampled.supports()[3],
        SampledPointSupport::Interstitial
    ));
    assert_eq!(record.operator.kind(), AuxiliaryKind::InterpolationPoints);
    assert_eq!(record.operator.q(), record.q);
    let gamma = record.operator.gamma().expect("Gamma head metadata");
    assert!(gamma.spherical_average_subtracted);
    assert!((gamma.head_prefactor - 4.0 * PI).abs() < SPINOR_COULOMB_EXACTNESS_FLOOR);
    assert_eq!(
        gamma.constant_coefficients.len(),
        record.operator.dimension()
    );
    assert!(
        record
            .operator
            .matrix()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite()),
        "Gamma body must stay finite; the singular head is metadata only"
    );
}

#[test]
fn matched_pair_reports_quadratic_and_rejects_mismatch() {
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_spinor_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let mpb = build_spinor_mpb(&input, &mpb_spec()).unwrap();
    let result = build_spinor_coulomb(
        std::slice::from_ref(&input),
        &thc,
        &coulomb_spec(),
        &[SpinorCoulombPairMatch {
            q_index: 0,
            mpb: &mpb,
            mpb_vertex: 0,
        }],
    )
    .unwrap();
    assert_eq!(result.diagnostics.len(), 1);
    let diagnostic = &result.diagnostics[0];
    assert_eq!(diagnostic.q_index, 0);
    assert_eq!(
        diagnostic.pair,
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0,
        }
    );
    let vertex = &result.records()[0].vertices[diagnostic.column];
    let matrix = result.records()[0].operator.matrix();
    let coefficients = vertex.coefficients();
    let independent_quadratic = dense_quadratic(matrix, coefficients);
    let independent_thc_action = dense_action_norm(matrix, coefficients);
    let mpb_operator = assemble_coulomb(&mpb.auxiliary, &coulomb_spec().request).unwrap();
    let independent_mpb_quadratic =
        dense_quadratic(mpb_operator.matrix(), mpb.vertices[0].vertex.coefficients());
    let independent_mpb_action =
        dense_action_norm(mpb_operator.matrix(), mpb.vertices[0].vertex.coefficients());
    assert!(
        (independent_quadratic - diagnostic.thc_quadratic).norm() < SPINOR_COULOMB_EXACTNESS_FLOOR
    );
    assert!(
        (independent_mpb_quadratic - diagnostic.mpb_quadratic).norm()
            < SPINOR_COULOMB_EXACTNESS_FLOOR
    );
    assert!(
        (independent_thc_action - diagnostic.thc_action_norm).abs()
            < SPINOR_COULOMB_EXACTNESS_FLOOR
    );
    assert!(
        (independent_mpb_action - diagnostic.mpb_action_norm).abs()
            < SPINOR_COULOMB_EXACTNESS_FLOOR
    );
    assert!(
        diagnostic.quadratic_discrepancy.absolute < FIXTURE_MPB_THC_QUADRATIC_ABS,
        "hydrogen rank-1 spinor fixture MPB-vs-THC quadratic gap exceeded the algebraic bound: \
         mpb_quadratic={} thc_quadratic={} q_abs={} q_rel={} (non-material/non-SPEX)",
        diagnostic.mpb_quadratic,
        diagnostic.thc_quadratic,
        diagnostic.quadratic_discrepancy.absolute,
        diagnostic.quadratic_discrepancy.relative
    );
    assert!(matches!(
        build_spinor_coulomb(
            std::slice::from_ref(&input),
            &thc,
            &coulomb_spec(),
            &[SpinorCoulombPairMatch {
                q_index: 1,
                mpb: &mpb,
                mpb_vertex: 0,
            }],
        ),
        Err(SpinorCoulombError::ComparisonQIndex(1))
    ));
}

#[test]
fn matched_mpb_must_originate_from_the_frozen_input() {
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
    let input_a = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let mut input_b = input_a.clone();
    let rows = input_b.orbitals.eigenvectors[0].rows();
    let columns = input_b.orbitals.eigenvectors[0].columns();
    let mut values = input_b.orbitals.eigenvectors[0].to_host_column_major();
    values[0] += Complex64::new(0.25, -0.125);
    input_b.orbitals.eigenvectors[0] =
        DenseEigenvectors::from_host_column_major(rows, columns, values).unwrap();
    assert_eq!(input_b.source.partition, input_a.source.partition);
    assert_eq!(input_b.source.q, input_a.source.q);
    assert_eq!(input_b.reciprocal, input_a.reciprocal);
    assert_eq!(input_b.pair_columns, input_a.pair_columns);
    assert_eq!(input_b.orbitals.band_window, input_a.orbitals.band_window);
    assert_eq!(
        input_b.orbitals.available_bands,
        input_a.orbitals.available_bands
    );
    assert_ne!(
        input_b.orbitals.eigenvectors[0],
        input_a.orbitals.eigenvectors[0]
    );

    let grid = parent_grid(&input_a);
    let thc = build_spinor_thc(std::slice::from_ref(&input_a), &grid, &thc_spec()).unwrap();
    let mpb_a = build_spinor_mpb(&input_a, &mpb_spec()).unwrap();
    build_spinor_coulomb(
        std::slice::from_ref(&input_a),
        &thc,
        &coulomb_spec(),
        &[SpinorCoulombPairMatch {
            q_index: 0,
            mpb: &mpb_a,
            mpb_vertex: 0,
        }],
    )
    .unwrap();
    assert!(matches!(
        build_spinor_coulomb(
            std::slice::from_ref(&input_b),
            &thc,
            &coulomb_spec(),
            &[SpinorCoulombPairMatch {
                q_index: 0,
                mpb: &mpb_a,
                mpb_vertex: 0,
            }],
        ),
        Err(SpinorCoulombError::FrozenInputMismatch { q_index: 0 })
    ));
}

#[test]
fn reciprocal_grid_and_bloch_mismatches_are_rejected() {
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
    let input = physics
        .spinor_product_input(&spinor_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_spinor_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();

    let sheared = Cell::new([
        [Bohr(8.0), Bohr(0.0), Bohr(0.0)],
        [Bohr(2.0), Bohr(8.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(8.0)],
    ])
    .unwrap();
    let spec = SpinorCoulombSpec {
        request: CoulombRequest::new(sheared, 2).unwrap(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 1).unwrap(),
    };
    assert_ne!(spec.request.reciprocal(), &input.reciprocal);
    assert!(matches!(
        build_spinor_coulomb(std::slice::from_ref(&input), &thc, &spec, &[]),
        Err(SpinorCoulombError::ReciprocalMismatch)
    ));

    let mut permuted = thc.clone();
    let mut points = permuted.grid.points().to_vec();
    points.swap(3, 4);
    permuted.grid = ThcParentGrid::new(
        permuted.grid.partition().clone(),
        permuted.grid.provenance().clone(),
        points,
    )
    .unwrap();
    assert!(!permuted.records_match_parent_grid());
    assert!(matches!(
        build_spinor_coulomb(
            std::slice::from_ref(&input),
            &permuted,
            &coulomb_spec(),
            &[]
        ),
        Err(SpinorCoulombError::GridIdentity { index: 0 })
    ));

    assert!(
        thc.records[0].vertices.len() >= 2,
        "fixture must expose at least two pair columns"
    );
    let mut tampered = thc.clone();
    tampered.records[0].vertices.swap(0, 1);
    assert!(matches!(
        build_spinor_coulomb(
            std::slice::from_ref(&input),
            &tampered,
            &coulomb_spec(),
            &[]
        ),
        Err(SpinorCoulombError::VertexIdentity {
            index: 0,
            column: 0
        })
    ));
}

#[test]
fn finite_q_preserves_transfer_q() {
    let physics = SnapshotDftPhysics::new(&hydrogen_spinor_snapshot()).unwrap();
    let q0 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let q15 = physics
        .spinor_product_input(&spinor_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    assert_eq!(q15.source.q.umklapp.index, [1, 0, 0]);
    let grid = parent_grid(&q15);
    let thc = build_spinor_thc(&[q0.clone(), q15.clone()], &grid, &thc_spec()).unwrap();
    let result = build_spinor_coulomb(&[q0, q15.clone()], &thc, &coulomb_spec(), &[]).unwrap();
    let finite = &result.records()[1];
    assert_eq!(finite.q_index, 1);
    assert_eq!(finite.q, q15.source.q);
    assert_eq!(finite.operator.q(), q15.source.q);
    assert!(finite.operator.gamma().is_none());
    let dropped = TransferQ::from_cartesian(finite.q.cartesian).unwrap();
    assert_ne!(dropped, finite.q);
    let layout = AuxiliaryLayout::from_regions(dropped, finite.operator.regions().to_vec());
    let vertex = &finite.vertices[0];
    let rephased = PairVertex::new(
        layout,
        vertex.pair(),
        vertex.coefficients().to_vec(),
        vertex.provenance().clone(),
    )
    .unwrap();
    assert!(matches!(
        finite.operator.apply(&rephased),
        Err(CoulombError::VertexTransferQ)
    ));
}
