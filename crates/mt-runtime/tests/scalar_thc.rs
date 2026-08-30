//! Public scalar adaptive-THC tests on frozen scalar product input.

use std::collections::BTreeMap;

use muffintin::{
    RankPolicy, SCALAR_RADIAL_U, SCALAR_RADIAL_UDOT, ScalarProductInput, ScalarThcError,
    ScalarThcSpec, SnapshotDftPhysics, ThcCandidates, ThcEngine, ThcParentGrid, ThcRegion,
    build_scalar_thc,
};
use muffintin_prodbasis::{AuxiliaryPartition, ProductOrbitalKind, ProductRadialId, TransferQ};
use muffintin_core::{
    Bohr, Hartree, VolumeBohr3, complex_spherical_harmonics, lm_from_index, lm_index,
};
use muffintin_dft::{
    LinearizationEnergyGenerator, NoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations, ScfRelativity,
    XcFunctional,
};
use muffintin_io::{
    AngularBasisV1, BasisHintsV1, Complex64V1, EnergyParameterV1, EnergyUnitV1,
    ExponentialMeshSpecV1, FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1,
    GeometryV1, InterstitialV1, InverseLengthUnitV1, LatticeV1, LengthUnitV1, LinearizationV1,
    MetaV1, PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTagV1, SiteSpinV1, SiteV1, SnapshotV1, SnapshotV2, SphericalChannelConventionV1,
    SpinTagV1,
};
use muffintin_operators::lapw::{CompiledBasis, Provenance};
use muffintin_operators::CompiledSiteProjection;
use muffintin_prodbasis::thc::{GridPath, L2Engine, SelectorStrategy, ThcError};
use num_complex::Complex64;

#[path = "thc_fixture_common.rs"]
mod thc_fixture_common;

use thc_fixture_common::{on_shell, scalar_parent_grid as parent_grid};

fn hydrogen_snapshot() -> SnapshotV2 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    SnapshotV1::new(
        MetaV1 {
            title: "scalar THC hydrogen smoke".to_owned(),
            producer: "mt-runtime test".to_owned(),
            producer_version: None,
            energy_zero: "zero interstitial Fourier mean".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasisV1::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConventionV1::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        GeometryV1 {
            lattice: LatticeV1 {
                unit: LengthUnitV1::Bohr,
                vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            },
            sites: vec![SiteV1 {
                id: "H-1".to_owned(),
                atomic_number: 1,
                fractional_position: [1.25, -0.5, 0.5],
                muffin_tin_radius_unit: LengthUnitV1::Bohr,
                muffin_tin_radius: radius,
                spins: vec![SiteSpinV1 {
                    spin: SpinTagV1::Scalar,
                    mesh: ExponentialMeshSpecV1 {
                        radius_unit: LengthUnitV1::Bohr,
                        first,
                        log_increment: increment,
                        point_count,
                        last: first * ((point_count - 1) as f64 * increment).exp(),
                        consistency_tolerance: 1.0e-12,
                    },
                    radial_equation: RadialEquationTagV1::ScalarKoellingHarmon,
                    potential_unit: EnergyUnitV1::Hartree,
                    potential_channels: vec![PotentialChannelV1 {
                        l: 0,
                        m: 0,
                        real: radii.iter().map(|radius| -1.0 / radius).collect(),
                        imaginary: Vec::new(),
                    }],
                    linearization: LinearizationV1 {
                        energy_unit: EnergyUnitV1::Hartree,
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
            coefficient_unit: EnergyUnitV1::Hartree,
            coefficients: vec![FourierCoefficientV1 {
                g: [0; 3],
                value: Complex64V1 {
                    real: 0.0,
                    imaginary: 0.0,
                },
            }],
            basis_hints: BasisHintsV1 {
                reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
                plane_wave_cutoff: Some(0.5),
                coefficient_cutoff: Some(1.0),
                normalization: FourierNormalizationV1::CellNormalized,
                phase: FourierPhaseV1::NegativeExponent,
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
        },
        basis: ScfBasis {
            plane_wave_cutoff: cutoff,
            l_max: 1,
            channels: vec![
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 1, l: 0 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenSnapshot,
                    seed: None,
                    provenance: ScfChannelProvenance::BuiltIn,
                },
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::ScalarL { n: 2, l: 1 },
                    treatment: ScfChannelTreatment::Valence,
                    derivative_order: 0,
                    generator: LinearizationEnergyGenerator::FrozenSnapshot,
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

fn spec_all(n_mu: usize) -> ScalarThcSpec {
    spec_all_engine(n_mu, ThcEngine::FullColumnPivotedQr)
}

fn spec_all_engine(n_mu: usize, engine: ThcEngine) -> ScalarThcSpec {
    ScalarThcSpec {
        spin: 0,
        rank: RankPolicy::Exact { n_mu },
        candidates: ThcCandidates::All,
        engine,
    }
}

fn site_id(
    compiled: &CompiledBasis,
    site: usize,
    spin: u8,
    coord: usize,
) -> Option<(ProductRadialId, i32)> {
    let n_lm = compiled
        .site_augmentations
        .get(site)
        .and_then(|waves| waves.first())
        .map_or(0, |wave| wave.coefficients.len());
    if coord < 2 * n_lm {
        let lm = lm_from_index(coord / 2);
        return Some((
            ProductRadialId {
                site,
                kind: ProductOrbitalKind::Valence,
                l: lm.l,
                n: if coord % 2 == 0 {
                    SCALAR_RADIAL_U
                } else {
                    SCALAR_RADIAL_UDOT
                },
                spin,
            },
            lm.m,
        ));
    }
    None
}

#[derive(Clone, Copy)]
enum MtKPhase {
    CellPeriodic,
    Missing,
    Opposite,
}

#[allow(clippy::too_many_arguments)]
fn independent_pq(
    input: &ScalarProductInput,
    grid: &ThcParentGrid,
    point: usize,
    k: usize,
    band: usize,
    skip_udot: bool,
    skip_q: bool,
    mt_k_phase: MtKPhase,
) -> (Complex64, Complex64) {
    let channel = &input.orbitals.channels[0];
    let compiled = &channel.bases[k];
    let evecs = &channel.eigenvectors[k];
    match grid.points()[point].region {
        ThcRegion::Interstitial => {
            let volume = input
                .source
                .partition
                .interstitial()
                .cell_volume()
                .get()
                .sqrt();
            let mut large = Complex64::default();
            let r = grid.points()[point].coordinate;
            for (row, wave) in compiled.plane_waves.iter().enumerate() {
                let argument = wave
                    .g
                    .cartesian
                    .iter()
                    .zip(r)
                    .map(|(g, x)| g.get() * x.get())
                    .sum();
                large += evecs.at(row, band) * Complex64::from_polar(1.0, argument) / volume;
            }
            (large, Complex64::default())
        }
        ThcRegion::MuffinTin { site, radial_index } => {
            let projected = CompiledSiteProjection::scalar(compiled, site)
                .unwrap()
                .project_eigenvectors(evecs)
                .unwrap();
            let radials = &input.source.radials[site];
            let origin = input.source.partition.sites()[site].position;
            let r = grid.points()[point].coordinate;
            let direction = [
                r[0].get() - origin[0].get(),
                r[1].get() - origin[1].get(),
                r[2].get() - origin[2].get(),
            ];
            let radius = radials.mesh.radii()[radial_index].get();
            let inv_r = 1.0 / radius;
            let l_max = (0..projected.coordinate_count())
                .filter_map(|coord| site_id(compiled, site, 0, coord).map(|(id, _)| id.l))
                .max()
                .unwrap_or(0);
            let harmonics = complex_spherical_harmonics(l_max, direction);
            let mut large = Complex64::default();
            let mut small = Complex64::default();
            for coord in 0..projected.coordinate_count() {
                let Some((id, m)) = site_id(compiled, site, 0, coord) else {
                    continue;
                };
                if skip_udot && id.n == SCALAR_RADIAL_UDOT {
                    continue;
                }
                let radial = radials
                    .valence
                    .iter()
                    .find(|radial| radial.l == id.l && radial.n == id.n && radial.spin == id.spin)
                    .unwrap();
                let y = harmonics[lm_index(id.l, m).unwrap()];
                let amplitude = projected.at(coord, band) * y * inv_r;
                large += amplitude * radial.samples.large[radial_index];
                if !skip_q && let Some(q) = radial.samples.small.as_ref() {
                    small += amplitude * q[radial_index];
                }
            }
            let argument: f64 = compiled.plane_waves[0]
                .k
                .iter()
                .zip(r)
                .map(|(component, point)| component.get() * point.get())
                .sum();
            let phase = match mt_k_phase {
                MtKPhase::CellPeriodic => Complex64::from_polar(1.0, -argument),
                MtKPhase::Missing => Complex64::new(1.0, 0.0),
                MtKPhase::Opposite => Complex64::from_polar(1.0, argument),
            };
            (large * phase, small * phase)
        }
    }
}

fn wrap_phase(input: &ScalarProductInput, k: usize, r: [Bohr; 3]) -> Complex64 {
    let mapped = input
        .k_minus_q
        .iter()
        .find(|mapped| mapped.k_index == k)
        .unwrap();
    let argument = mapped
        .umklapp
        .cartesian
        .iter()
        .zip(r)
        .map(|(g, x)| g.get() * x.get())
        .sum();
    Complex64::from_polar(1.0, argument)
}

#[allow(clippy::too_many_arguments)]
fn independent_pair(
    input: &ScalarProductInput,
    grid: &ThcParentGrid,
    point: usize,
    k: usize,
    left: usize,
    right: usize,
    skip_udot: bool,
    skip_q: bool,
    phase_sign: f64,
    extra_q: Option<TransferQ>,
    mt_k_phase: MtKPhase,
) -> Complex64 {
    let mapped = input
        .k_minus_q
        .iter()
        .find(|mapped| mapped.k_index == k)
        .unwrap();
    let (p_l, q_l) = independent_pq(
        input,
        grid,
        point,
        mapped.kq_index,
        left,
        skip_udot,
        skip_q,
        mt_k_phase,
    );
    let (p_r, q_r) = independent_pq(
        input,
        grid,
        point,
        mapped.k_index,
        right,
        skip_udot,
        skip_q,
        mt_k_phase,
    );
    let mut phase = wrap_phase(input, k, grid.points()[point].coordinate);
    if phase_sign < 0.0 {
        phase = phase.conj();
    }
    if let Some(q) = extra_q {
        let argument = q
            .umklapp
            .cartesian
            .iter()
            .zip(grid.points()[point].coordinate)
            .map(|(g, x)| g.get() * x.get())
            .sum();
        phase *= Complex64::from_polar(1.0, argument);
    }
    phase * (p_l.conj() * p_r + q_l.conj() * q_r)
}

fn selected_pair(
    result: &muffintin::ScalarThcResult,
    q: usize,
    parent: usize,
    column: usize,
) -> Complex64 {
    let mu = result
        .selection
        .points
        .iter()
        .position(|point| point.id == parent)
        .expect("parent point was selected");
    result.records[q].vertices[column].coefficients()[mu]
}

#[test]
fn q0_mt_and_interstitial_orbitals_match_independent_pp_qq_oracle() {
    let physics = SnapshotDftPhysics::new(&hydrogen_snapshot()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let n_parent = grid.points().len();
    let n_positive = grid
        .points()
        .iter()
        .filter(|point| point.weight > 0.0)
        .count();
    assert_eq!(n_parent, 6);
    assert_eq!(n_positive, 5);
    let result =
        build_scalar_thc(std::slice::from_ref(&input), &grid, &spec_all(n_positive)).unwrap();
    assert_eq!(result.records.len(), 1);
    assert_eq!(result.spin, 0);
    assert_eq!(result.effective_rank, n_positive);
    assert_eq!(result.records[0].fit.n_points, n_parent);
    assert_eq!(result.records[0].fit.zeta.len(), n_parent * n_positive);
    assert!(!result.selection.points.iter().any(|point| point.id == 1));
    assert!(!result.selection.pivots.contains(&1));
    assert_eq!(
        result.selection.provenance.grid_path,
        GridPath::External {
            n_points: n_parent,
            n_candidates: n_positive,
        }
    );
    assert_eq!(
        result.selection.provenance.strategy,
        SelectorStrategy::AllQL2
    );
    assert_eq!(
        result.selection.provenance.engine,
        L2Engine::FullColumnPivotedQr
    );
    let has_q = input.source.radials[0]
        .valence
        .iter()
        .any(|radial| radial.samples.small.is_some());
    let mt = 0;
    let interstitial = 3;
    let column = input.pair_columns.encode(0, 0, 0);
    let exact_mt = independent_pair(
        &input,
        &grid,
        mt,
        0,
        0,
        0,
        false,
        false,
        1.0,
        None,
        MtKPhase::CellPeriodic,
    );
    let got_mt = selected_pair(&result, 0, mt, column);
    assert!(
        (got_mt - exact_mt).norm() < 1.0e-8,
        "{got_mt} vs {exact_mt}"
    );
    let without_udot = independent_pair(
        &input,
        &grid,
        mt,
        0,
        0,
        0,
        true,
        false,
        1.0,
        None,
        MtKPhase::CellPeriodic,
    );
    assert!(
        (without_udot - exact_mt).norm() > 1.0e-8,
        "UDOT must contribute to the MT pair density"
    );
    if has_q {
        let without_q = independent_pair(
            &input,
            &grid,
            mt,
            0,
            0,
            0,
            false,
            true,
            1.0,
            None,
            MtKPhase::CellPeriodic,
        );
        assert!(
            (without_q - exact_mt).norm() > 1.0e-10,
            "KH small-component QQ must contribute"
        );
    }
    let exact_i = independent_pair(
        &input,
        &grid,
        interstitial,
        0,
        0,
        0,
        false,
        false,
        1.0,
        None,
        MtKPhase::CellPeriodic,
    );
    let got_i = selected_pair(&result, 0, interstitial, column);
    assert!((got_i - exact_i).norm() < 1.0e-8, "{got_i} vs {exact_i}");
}

#[test]
fn finite_q_pair_density_uses_stored_positive_wrap_not_global_umklapp() {
    let physics = SnapshotDftPhysics::new(&hydrogen_snapshot()).unwrap();
    let q0 = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let q15 = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [1.5, 0.0, 0.0])
        .unwrap();
    assert_eq!(q15.source.q.umklapp.index, [1, 0, 0]);
    assert_eq!(q15.k_minus_q[0].umklapp.index, [-1, 0, 0]);
    let grid = parent_grid(&q15);
    let spec = ScalarThcSpec {
        spin: 0,
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::Indices(vec![0]),
        engine: ThcEngine::FullColumnPivotedQr,
    };
    let result = build_scalar_thc(&[q0, q15.clone()], &grid, &spec).unwrap();
    assert_eq!(result.spin, 0);
    assert_eq!(result.records[1].q_index, 1);
    assert_eq!(result.records[1].q, q15.source.q);
    let parent = 0;
    let column = q15.pair_columns.encode(0, 0, 0);
    let exact = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        1.0,
        None,
        MtKPhase::CellPeriodic,
    );
    let got = selected_pair(&result, 1, parent, column);
    assert!((got - exact).norm() < 1.0e-8, "{got} vs {exact}");
    let flipped = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        -1.0,
        None,
        MtKPhase::CellPeriodic,
    );
    let doubled = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        1.0,
        Some(q15.source.q),
        MtKPhase::CellPeriodic,
    );
    let missing_k = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        1.0,
        None,
        MtKPhase::Missing,
    );
    let opposite_k = independent_pair(
        &q15,
        &grid,
        parent,
        0,
        0,
        0,
        false,
        false,
        1.0,
        None,
        MtKPhase::Opposite,
    );
    assert!((got - flipped).norm() > 1.0e-8);
    assert!((got - doubled).norm() > 1.0e-8);
    assert!(
        (got - missing_k).norm() > 1.0e-8,
        "cell-periodic MT factor must differ from the missing Bloch factor"
    );
    assert!(
        (got - opposite_k).norm() > 1.0e-8,
        "cell-periodic MT factor must differ from the opposite sign"
    );
    assert!((missing_k - opposite_k).norm() > 1.0e-8);
}

#[test]
fn multi_q_allq_l2_fits_heterogeneous_weights_with_independent_residual() {
    let physics = SnapshotDftPhysics::new(&hydrogen_snapshot()).unwrap();
    let q0 = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let qh = physics
        .scalar_product_input(&scalar_config([2, 1, 1], 0.5), [0.5, 0.0, 0.0])
        .unwrap();
    let grid = parent_grid(&q0);
    let inputs = [q0.clone(), qh.clone()];
    let engines = [
        ThcEngine::FullColumnPivotedQr,
        ThcEngine::FullPivotedCholesky,
    ];
    let n_col = q0.pair_columns.n_columns().unwrap();
    let n_pts = grid.points().len();
    let mut exact = vec![Complex64::default(); n_pts * n_col];
    for p in 0..n_pts {
        for k in 0..q0.pair_columns.n_k {
            for i in 0..q0.pair_columns.n_orb {
                for j in 0..q0.pair_columns.n_orb {
                    let column = q0.pair_columns.encode(k, i, j);
                    exact[p * n_col + column] = independent_pair(
                        &qh,
                        &grid,
                        p,
                        k,
                        i,
                        j,
                        false,
                        false,
                        1.0,
                        None,
                        MtKPhase::CellPeriodic,
                    );
                }
            }
        }
    }
    let weights: Vec<f64> = grid.points().iter().map(|point| point.weight).collect();
    let mut rank_one = Vec::new();
    let mut rank_two = Vec::new();
    for engine in engines {
        let one = build_scalar_thc(&inputs, &grid, &spec_all_engine(1, engine)).unwrap();
        let two = build_scalar_thc(&inputs, &grid, &spec_all_engine(2, engine)).unwrap();
        assert_eq!(two.records.len(), 2);
        assert_eq!(two.spin, 0);
        assert_eq!(two.records[0].q_index, 0);
        assert_eq!(two.records[1].q_index, 1);
        assert_eq!(two.effective_rank, 2);
        assert_eq!(two.requested_rank, RankPolicy::Exact { n_mu: 2 });
        assert_eq!(two.records[0].layout, q0.pair_columns);
        assert_eq!(two.records[0].vertices.len(), n_col);
        assert_eq!(two.selection.provenance.strategy, SelectorStrategy::AllQL2);
        assert_eq!(two.selection.provenance.engine, L2Engine::from(engine));
        assert_eq!(
            two.selection.provenance.grid_path,
            GridPath::External {
                n_points: n_pts,
                n_candidates: 5,
            }
        );
        assert!(!two.selection.points.iter().any(|point| point.id == 1));
        rank_one.push(one);
        rank_two.push(two);
    }
    let residuals_one: Vec<f64> = rank_one
        .iter()
        .map(|result| reconstruction_residual(result, &exact, &weights, n_col, 1))
        .collect();
    let residuals_two: Vec<f64> = rank_two
        .iter()
        .map(|result| reconstruction_residual(result, &exact, &weights, n_col, 1))
        .collect();
    let baseline = residuals_one[0];
    assert!(baseline.is_finite() && baseline > 0.0);
    let exactness_floor = 1.0e-12;
    for (engine, (&rank1, &rank2)) in engines
        .iter()
        .zip(residuals_one.iter().zip(residuals_two.iter()))
    {
        assert!(rank2.is_finite() && rank2 >= 0.0, "engine={engine:?}");
        assert!(
            rank2 < rank1.max(exactness_floor),
            "rank-2 must improve on rank-1 or stay below the exactness floor for {engine:?}: {rank2} vs {rank1}"
        );
        assert!(
            rank2 < baseline.max(exactness_floor),
            "rank-2 must improve on the shared QRCP rank-1 baseline or stay below the exactness floor for {engine:?}: {rank2} vs {baseline}"
        );
    }
    if rank_two[0].selection.pivots == rank_two[1].selection.pivots {
        assert_eq!(rank_two[0].selection.points, rank_two[1].selection.points);
    }
}

fn reconstruction_residual(
    result: &muffintin::ScalarThcResult,
    exact: &[Complex64],
    weights: &[f64],
    n_col: usize,
    q: usize,
) -> f64 {
    let n_pts = result.records[q].fit.n_points;
    let n_mu = result.records[q].fit.n_mu;
    let ids = result
        .selection
        .points
        .iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    for &id in &ids {
        selected.extend_from_slice(&exact[id * n_col..(id + 1) * n_col]);
    }
    let recon = reconstruct_pairs(&selected, n_mu, n_col, &result.records[q].fit.zeta, n_pts);
    weighted_frobenius(exact, &recon, weights, n_col)
}

fn reconstruct_pairs(
    selected: &[Complex64],
    n_mu: usize,
    n_col: usize,
    zeta: &[Complex64],
    n_pts: usize,
) -> Vec<Complex64> {
    let mut recon = vec![Complex64::default(); n_pts * n_col];
    for p in 0..n_pts {
        for col in 0..n_col {
            let mut acc = Complex64::default();
            for mu in 0..n_mu {
                acc += zeta[p * n_mu + mu] * selected[mu * n_col + col];
            }
            recon[p * n_col + col] = acc;
        }
    }
    recon
}

fn weighted_frobenius(
    exact: &[Complex64],
    recon: &[Complex64],
    weights: &[f64],
    n_col: usize,
) -> f64 {
    let mut acc = 0.0;
    for p in 0..weights.len() {
        for col in 0..n_col {
            acc += weights[p] * (exact[p * n_col + col] - recon[p * n_col + col]).norm_sqr();
        }
    }
    acc.sqrt()
}

#[test]
fn parent_grid_construction_identity_binds_q_fits() {
    let physics = SnapshotDftPhysics::new(&hydrogen_snapshot()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 1.0), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    let thc = build_scalar_thc(std::slice::from_ref(&input), &grid, &spec_all(1)).unwrap();
    assert!(thc.records_match_parent_grid());
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
    assert_eq!(permuted.records[0].fit.zeta, thc.records[0].fit.zeta);
}

#[test]
fn scalar_thc_rejects_empty_slice_or_partition_mismatch() {
    let physics = SnapshotDftPhysics::new(&hydrogen_snapshot()).unwrap();
    let input = physics
        .scalar_product_input(&scalar_config([1, 1, 1], 0.5), [0.0; 3])
        .unwrap();
    let grid = parent_grid(&input);
    assert!(matches!(
        build_scalar_thc(&[], &grid, &spec_all(2)),
        Err(ScalarThcError::EmptySlice)
    ));
    let other = AuxiliaryPartition::from_interstitial(
        muffintin_core::InterstitialGeometry::new(
            VolumeBohr3(1000.0),
            input.source.partition.interstitial().spheres().to_vec(),
        )
        .unwrap(),
    );
    let bad = ThcParentGrid::new(other, Provenance::default(), grid.points().to_vec()).unwrap();
    assert!(matches!(
        build_scalar_thc(std::slice::from_ref(&input), &bad, &spec_all(2)),
        Err(ScalarThcError::GridPartitionMismatch)
    ));
    assert!(matches!(
        build_scalar_thc(
            std::slice::from_ref(&input),
            &grid,
            &spec_all(grid.points().len())
        ),
        Err(ScalarThcError::Thc(ThcError::RankExceedsGrid {
            n_mu: 6,
            n_points: 5
        }))
    ));
    let zero_candidate = ScalarThcSpec {
        spin: 0,
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::Indices(vec![1]),
        engine: ThcEngine::FullColumnPivotedQr,
    };
    assert!(matches!(
        build_scalar_thc(std::slice::from_ref(&input), &grid, &zero_candidate),
        Err(ScalarThcError::Thc(ThcError::ZeroWeightCandidate(1)))
    ));
    let origin = input.source.partition.sites()[0].position;
    let mid = input.source.radials[0].mesh.radii().len() / 2;
    let r_mid = input.source.radials[0].mesh.radii()[mid].get();
    let mut off_shell = grid.points().to_vec();
    off_shell[0].coordinate = on_shell(origin, r_mid, [0.4, -0.3, 0.2]);
    off_shell[0].region = ThcRegion::MuffinTin {
        site: 0,
        radial_index: 0,
    };
    let bad_shell = ThcParentGrid::new(
        input.source.partition.clone(),
        Provenance::default(),
        off_shell,
    )
    .unwrap();
    assert!(matches!(
        build_scalar_thc(std::slice::from_ref(&input), &bad_shell, &spec_all(1)),
        Err(ScalarThcError::RadialShellMismatch {
            index: 0,
            site: 0,
            radial_index: 0
        })
    ));
}
