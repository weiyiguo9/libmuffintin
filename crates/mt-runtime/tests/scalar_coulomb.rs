//! Public scalar sampled-ζ Coulomb tests on frozen product-input, mixed-product, and THC output.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use muffintin::{
    CheckpointPhysics, RankPolicy, SCALAR_COULOMB_EXACTNESS_FLOOR, ScalarCoulombError,
    ScalarCoulombPairMatch, ScalarCoulombSpec, ScalarMpbSelection, ScalarMpbSpec, ScalarThcSpec,
    ThcCandidates, ThcEngine, ThcParentGrid, build_scalar_coulomb, build_scalar_mpb,
    build_scalar_thc,
};
use muffintin_core::Cell;
use muffintin_core::{Bohr, Hartree, InverseBohr};
use muffintin_coulomb::{
    AuxiliaryKind, CoulombError, CoulombRequest, InterpolationProjection, SampledPointSupport,
    assemble_coulomb,
};
use muffintin_dft::{
    LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations, ScfRelativity,
    XcFunctional,
};
use muffintin_io::{
    AngularBasis, BasisHints, CheckpointMeta, CheckpointV1, CheckpointV2, Complex64V1,
    EnergyParameterV1, EnergyUnit, ExponentialMeshSpec, FourierCoefficientV1, FourierNormalization,
    FourierPhase, GeometryV1, InterstitialV1, InverseLengthUnit, LatticeV1, LengthUnit,
    LinearizationV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTag, SiteSpinV1, SiteV1, SphericalChannelConvention, SpinTag,
};
use muffintin_operators::lapw::Provenance;
use muffintin_prodbasis::mpb::DEFAULT_TOLERANCE;
use muffintin_prodbasis::{AuxiliaryLayout, OrbitalPair, PairVertex, TransferQ};
use num_complex::Complex64;

#[path = "thc_fixture_common.rs"]
mod thc_fixture_common;

use thc_fixture_common::scalar_parent_grid as parent_grid;

fn hydrogen_checkpoint() -> CheckpointV2 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    CheckpointV1::new(
        CheckpointMeta {
            title: "scalar Coulomb hydrogen smoke".to_owned(),
            producer: "mt-runtime test".to_owned(),
            producer_version: None,
            energy_zero: "zero interstitial Fourier mean".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasis::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConvention::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        GeometryV1 {
            lattice: LatticeV1 {
                unit: LengthUnit::Bohr,
                vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            },
            sites: vec![SiteV1 {
                id: "H-1".to_owned(),
                atomic_number: 1,
                fractional_position: [1.25, -0.5, 0.5],
                muffin_tin_radius_unit: LengthUnit::Bohr,
                muffin_tin_radius: radius,
                spins: vec![SiteSpinV1 {
                    spin: SpinTag::Scalar,
                    mesh: ExponentialMeshSpec {
                        radius_unit: LengthUnit::Bohr,
                        first,
                        log_increment: increment,
                        point_count,
                        last: first * ((point_count - 1) as f64 * increment).exp(),
                        consistency_tolerance: 1.0e-12,
                    },
                    radial_equation: RadialEquationTag::ScalarKoellingHarmon,
                    potential_unit: EnergyUnit::Hartree,
                    potential_channels: vec![PotentialChannelV1 {
                        l: 0,
                        m: 0,
                        real: radii.iter().map(|radius| -1.0 / radius).collect(),
                        imaginary: Vec::new(),
                    }],
                    linearization: LinearizationV1 {
                        energy_unit: EnergyUnit::Hartree,
                        linearization_energies: vec![
                            EnergyParameterV1 { l: 0, energy: -0.3 },
                            EnergyParameterV1 {
                                l: 1,
                                energy: -0.15,
                            },
                        ],
                        local_orbital_energies: Vec::new(),
                    },
                }],
            }],
        },
        InterstitialV1 {
            coefficient_unit: EnergyUnit::Hartree,
            coefficients: vec![FourierCoefficientV1 {
                g: [0; 3],
                value: Complex64V1 {
                    real: 0.0,
                    imaginary: 0.0,
                },
            }],
            basis_hints: BasisHints {
                reciprocal_length_unit: InverseLengthUnit::BohrInverse,
                plane_wave_cutoff: Some(0.5),
                coefficient_cutoff: Some(1.0),
                normalization: FourierNormalization::CellNormalized,
                phase: FourierPhase::NegativeExponent,
            },
        },
    )
    .normalize_v2()
    .unwrap()
}

fn scalar_config(divisions: [usize; 3], cutoff: f64) -> ScfConfig {
    ScfConfig {
        electron_count: 1.0,
        k_mesh: ScfKMesh {
            divisions,
            shift: [0.0; 3],
            reduction: muffintin_dft::ScfKReduction::Full,
        },
        basis: ScfBasis {
            plane_wave_cutoff: InverseBohr(cutoff),
            l_max: 1,
            channels: vec![
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 1, l: 0 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenCheckpoint,
                    seed: None,
                    provenance: ScfChannelProvenance::BuiltIn,
                },
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 2, l: 1 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenCheckpoint,
                    seed: None,
                    provenance: ScfChannelProvenance::BuiltIn,
                },
            ],
            resolved_channels: Vec::new(),
        },
        occupations: ScfOccupations::FermiDirac {
            temperature: Hartree(0.02),
        },
        exchange_correlation: ScfExchangeCorrelation {
            functional: XcFunctional::LdaPw92,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        },
        mixing: ScfMixing::Linear { alpha: 1.0 },
        relativity: ScfRelativity::Scalar,
        convergence: ScfConvergence {
            energy_tolerance: Hartree(1.0e100),
            density_tolerance: 1.0e100,
            max_iterations: 2,
        },
        core_sites: vec![ScfCoreSite {
            id: "H-1".to_owned(),
            states: Vec::new(),
        }],
    }
}

fn thc_spec() -> ScalarThcSpec {
    ScalarThcSpec {
        spin: 0,
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::All,
        engine: ThcEngine::FullColumnPivotedQr,
    }
}

fn coulomb_spec() -> ScalarCoulombSpec {
    ScalarCoulombSpec {
        request: CoulombRequest::cubic(8.0, 2).unwrap(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 1).unwrap(),
    }
}

/// Rank-1 THC versus MPB quadratic $c^\dagger V c$ on this 6-point hydrogen /
/// LEXP=2 fixture. Observed algebraic gap ≈ 0.167 (not a SPEX/material
/// tolerance). The bound is three times that gap and sits well above the
/// $10^{-12}$ exactness floor.
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

fn mpb_spec(lattice: muffintin_core::ReciprocalLattice) -> ScalarMpbSpec {
    ScalarMpbSpec {
        lattice,
        product_l_max: 2,
        product_g_max: InverseBohr(1.5),
        overlap_tolerance: DEFAULT_TOLERANCE,
        selections: vec![ScalarMpbSelection {
            spin: 0,
            k: 0,
            left_band: 0,
            right_band: 0,
        }],
    }
}

#[test]
fn gamma_sampled_coulomb_uses_full_parent_grid_and_keeps_head_as_metadata() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let result = build_scalar_coulomb(&[input], &thc, &coulomb_spec(), &[]).unwrap();
    assert_eq!(result.spin, 0);
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
    assert!((gamma.head_prefactor - 4.0 * PI).abs() < SCALAR_COULOMB_EXACTNESS_FLOOR);
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
fn finite_q_preserves_transfer_q_and_rejects_dropped_umklapp() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let q0 = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let q15 = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    assert_eq!(q15.source.q.umklapp.index, [1, 0, 0]);
    let grid = parent_grid(&q15);
    let thc = build_scalar_thc(&[q0.clone(), q15.clone()], &grid, &thc_spec()).unwrap();
    let result = build_scalar_coulomb(&[q0, q15.clone()], &thc, &coulomb_spec(), &[]).unwrap();
    assert_eq!(result.records().len(), 2);
    let finite = &result.records()[1];
    assert_eq!(finite.q_index, 1);
    assert_eq!(finite.q, q15.source.q);
    assert_eq!(finite.operator.q(), q15.source.q);
    assert_eq!(finite.operator.q().umklapp.index, [1, 0, 0]);
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

#[test]
fn matched_pair_reports_quadratic_and_action_and_rejects_mismatch() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let mpb = build_scalar_mpb(&input, &mpb_spec(*physics.reciprocal())).unwrap();
    let result = build_scalar_coulomb(
        std::slice::from_ref(&input),
        &thc,
        &coulomb_spec(),
        &[ScalarCoulombPairMatch {
            q_index: 0,
            mpb: &mpb,
            mpb_vertex: 0,
        }],
    )
    .unwrap();
    assert_eq!(result.diagnostics.len(), 1);
    let diagnostic = &result.diagnostics[0];
    assert_eq!(diagnostic.q_index, 0);
    assert_eq!(diagnostic.spin, 0);
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
        (independent_quadratic - diagnostic.thc_quadratic).norm() < SCALAR_COULOMB_EXACTNESS_FLOOR,
        "stored THC quadratic {} must match independent c†Vc {}; mpb={} thc={} abs={} rel={}",
        diagnostic.thc_quadratic,
        independent_quadratic,
        diagnostic.mpb_quadratic,
        diagnostic.thc_quadratic,
        diagnostic.quadratic_discrepancy.absolute,
        diagnostic.quadratic_discrepancy.relative
    );
    assert!(
        (independent_mpb_quadratic - diagnostic.mpb_quadratic).norm()
            < SCALAR_COULOMB_EXACTNESS_FLOOR,
        "stored MPB quadratic {} must match independent c†Vc {}",
        diagnostic.mpb_quadratic,
        independent_mpb_quadratic
    );
    assert!(
        (independent_thc_action - diagnostic.thc_action_norm).abs()
            < SCALAR_COULOMB_EXACTNESS_FLOOR,
        "stored THC debug action {} must match independent ||Vc|| {} in the interpolation-point basis",
        diagnostic.thc_action_norm,
        independent_thc_action
    );
    assert!(
        (independent_mpb_action - diagnostic.mpb_action_norm).abs()
            < SCALAR_COULOMB_EXACTNESS_FLOOR,
        "stored MPB debug action {} must match independent ||Vc|| {} in the mixed-product basis",
        diagnostic.mpb_action_norm,
        independent_mpb_action
    );
    assert!(
        diagnostic.quadratic_discrepancy.absolute < FIXTURE_MPB_THC_QUADRATIC_ABS,
        "hydrogen rank-1 fixture MPB-vs-THC quadratic gap exceeded the algebraic bound: \
         mpb_quadratic={} thc_quadratic={} q_abs={} q_rel={} mpb_action={} thc_action={}",
        diagnostic.mpb_quadratic,
        diagnostic.thc_quadratic,
        diagnostic.quadratic_discrepancy.absolute,
        diagnostic.quadratic_discrepancy.relative,
        diagnostic.mpb_action_norm,
        diagnostic.thc_action_norm
    );
    assert!(matches!(
        build_scalar_coulomb(
            std::slice::from_ref(&input),
            &thc,
            &coulomb_spec(),
            &[ScalarCoulombPairMatch {
                q_index: 1,
                mpb: &mpb,
                mpb_vertex: 0,
            }],
        ),
        Err(ScalarCoulombError::ComparisonQIndex(1))
    ));
    let mut mismatched = mpb.clone();
    mismatched.vertices[0].spin = 1;
    assert!(matches!(
        build_scalar_coulomb(
            std::slice::from_ref(&input),
            &thc,
            &coulomb_spec(),
            &[ScalarCoulombPairMatch {
                q_index: 0,
                mpb: &mismatched,
                mpb_vertex: 0,
            }],
        ),
        Err(ScalarCoulombError::ComparisonContext { q_index: 0 })
    ));
}

#[test]
fn selected_nodes_or_permuted_parent_rows_fail_context_or_oracle() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let production =
        build_scalar_coulomb(std::slice::from_ref(&input), &thc, &coulomb_spec(), &[]).unwrap();

    let mut truncated = thc.clone();
    let selected = truncated
        .selection
        .points
        .iter()
        .map(|point| truncated.grid.points()[point.id])
        .collect::<Vec<_>>();
    truncated.grid = ThcParentGrid::new(
        truncated.grid.partition().clone(),
        truncated.grid.provenance().clone(),
        selected,
    )
    .unwrap();
    assert!(!truncated.records_match_parent_grid());
    assert!(matches!(
        build_scalar_coulomb(
            std::slice::from_ref(&input),
            &truncated,
            &coulomb_spec(),
            &[]
        ),
        Err(ScalarCoulombError::GridIdentity { index: 0 })
    ));

    let mut nodes = thc.clone();
    let ids = nodes
        .selection
        .points
        .iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();
    let selected = ids
        .iter()
        .map(|&id| nodes.grid.points()[id])
        .collect::<Vec<_>>();
    nodes.grid = ThcParentGrid::new(
        nodes.grid.partition().clone(),
        nodes.grid.provenance().clone(),
        selected,
    )
    .unwrap();
    for record in &mut nodes.records {
        let n_mu = record.fit.n_mu;
        let mut zeta = Vec::with_capacity(ids.len() * n_mu);
        for &id in &ids {
            let start = id * n_mu;
            zeta.extend_from_slice(&record.fit.zeta[start..start + n_mu]);
        }
        record.fit.zeta = zeta;
        record.fit.n_points = ids.len();
    }
    assert!(!nodes.records_match_parent_grid());
    assert!(matches!(
        build_scalar_coulomb(std::slice::from_ref(&input), &nodes, &coulomb_spec(), &[]),
        Err(ScalarCoulombError::GridIdentity { index: 0 })
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
    assert_eq!(permuted.records[0].fit.zeta, thc.records[0].fit.zeta);
    assert!(!permuted.records_match_parent_grid());
    assert!(matches!(
        build_scalar_coulomb(
            std::slice::from_ref(&input),
            &permuted,
            &coulomb_spec(),
            &[]
        ),
        Err(ScalarCoulombError::GridIdentity { index: 0 })
    ));
    assert_eq!(production.records()[0].sampled.n_grid(), 6);
}

#[test]
fn same_volume_skew_reciprocal_is_rejected_before_assembly() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    assert_eq!(input.reciprocal, *physics.reciprocal());
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let sheared = Cell::new([
        [Bohr(8.0), Bohr(0.0), Bohr(0.0)],
        [Bohr(2.0), Bohr(8.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(8.0)],
    ])
    .unwrap();
    assert_ne!(
        sheared.basis()[1][0].get(),
        0.0,
        "adversarial cell must be sheared, not merely anisotropic-diagonal"
    );
    assert!((sheared.volume().get() - 512.0).abs() < SCALAR_COULOMB_EXACTNESS_FLOOR);
    let spec = ScalarCoulombSpec {
        request: CoulombRequest::new(sheared, 2).unwrap(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 1).unwrap(),
    };
    assert_ne!(spec.request.reciprocal(), &input.reciprocal);
    assert!(matches!(
        build_scalar_coulomb(std::slice::from_ref(&input), &thc, &spec, &[]),
        Err(ScalarCoulombError::ReciprocalMismatch)
    ));
}

#[test]
fn tampered_vertex_layout_or_order_is_rejected() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    assert!(
        thc.records[0].vertices.len() >= 2,
        "fixture must expose at least two pair columns"
    );
    let mut tampered = thc.clone();
    tampered.records[0].vertices.swap(0, 1);
    assert!(matches!(
        build_scalar_coulomb(
            std::slice::from_ref(&input),
            &tampered,
            &coulomb_spec(),
            &[]
        ),
        Err(ScalarCoulombError::VertexIdentity {
            index: 0,
            column: 0
        })
    ));
}

#[test]
fn tampered_vertex_provenance_is_rejected() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let vertex = &thc.records[0].vertices[0];
    assert_eq!(
        vertex.provenance(),
        &thc.records[0].auxiliary.provenance,
        "auxiliaries and generated vertices must bind the same provenance at construction"
    );
    let forged = PairVertex::new(
        vertex.layout().clone(),
        vertex.pair(),
        vertex.coefficients().to_vec(),
        Provenance {
            recipe: Some("tampered-vertex-provenance".to_owned()),
            reference: vertex.provenance().reference.clone(),
        },
    )
    .unwrap();
    assert_ne!(forged.provenance(), &thc.records[0].auxiliary.provenance);
    let mut tampered = thc.clone();
    tampered.records[0].vertices[0] = forged;
    assert!(matches!(
        build_scalar_coulomb(
            std::slice::from_ref(&input),
            &tampered,
            &coulomb_spec(),
            &[]
        ),
        Err(ScalarCoulombError::VertexIdentity {
            index: 0,
            column: 0
        })
    ));
}

#[test]
fn malformed_bloch_index_is_rejected_without_encode() {
    let physics = CheckpointPhysics::new(&hydrogen_checkpoint()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &thc_spec()).unwrap();
    let vertex = &thc.records[0].vertices[0];
    let malformed = PairVertex::new(
        vertex.layout().clone(),
        OrbitalPair::Bloch {
            k_index: usize::MAX,
            left: 0,
            right: 0,
        },
        vertex.coefficients().to_vec(),
        vertex.provenance().clone(),
    )
    .unwrap();
    let mut tampered = thc.clone();
    tampered.records[0].vertices[0] = malformed;
    assert!(matches!(
        build_scalar_coulomb(
            std::slice::from_ref(&input),
            &tampered,
            &coulomb_spec(),
            &[]
        ),
        Err(ScalarCoulombError::VertexIdentity {
            index: 0,
            column: 0
        })
    ));
}
