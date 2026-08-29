//! Shared 2-k scalar hydrogen q0 + q_input=1.5 fixture for MLDUMP materialization.

#![allow(dead_code)]

use std::collections::BTreeMap;

use muffintin::{
    ScalarCoulombSpec, ScalarProductInput, ScalarThcSpec, ThcCandidates, ThcEngine, ThcParentGrid,
    ThcPoint, ThcRegion,
};
use muffintin_core::{Bohr, Hartree, InverseBohr};
use muffintin_coulomb::{CoulombRequest, InterpolationProjection};
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
use muffintin_lapw::Provenance;
use muffintin_thc::RankPolicy;

pub fn hydrogen_snapshot() -> SnapshotV2 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    SnapshotV1::new(
        MetaV1 {
            title: "scalar MLDUMP hydrogen smoke".to_owned(),
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

pub fn scalar_config(divisions: [usize; 3], cutoff: f64) -> ScfConfig {
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

fn on_shell(origin: [Bohr; 3], radius: f64, direction: [f64; 3]) -> [Bohr; 3] {
    let norm = direction
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    [
        Bohr(origin[0].get() + radius * direction[0] / norm),
        Bohr(origin[1].get() + radius * direction[1] / norm),
        Bohr(origin[2].get() + radius * direction[2] / norm),
    ]
}

pub fn parent_grid(input: &ScalarProductInput) -> ThcParentGrid {
    let origin = input.source.partition.sites()[0].position;
    let mesh = &input.source.radials[0].mesh;
    let mid = mesh.radii().len() / 2;
    let r_mid = mesh.radii()[mid].get();
    let r0 = mesh.radii()[0].get();
    ThcParentGrid::new(
        input.source.partition.clone(),
        Provenance::default(),
        vec![
            ThcPoint {
                coordinate: on_shell(origin, r0, [0.4, -0.3, 0.2]),
                weight: 0.35,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: 0,
                },
            },
            ThcPoint {
                coordinate: on_shell(origin, r_mid, [1.0, 0.0, 0.0]),
                weight: 0.0,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: mid,
                },
            },
            ThcPoint {
                coordinate: on_shell(origin, r_mid, [0.0, 1.0, 0.0]),
                weight: 0.45,
                region: ThcRegion::MuffinTin {
                    site: 0,
                    radial_index: mid,
                },
            },
            ThcPoint {
                coordinate: [Bohr(0.2), Bohr(0.2), Bohr(0.2)],
                weight: 0.8,
                region: ThcRegion::Interstitial,
            },
            ThcPoint {
                coordinate: [Bohr(5.0), Bohr(4.0), Bohr(4.0)],
                weight: 0.15,
                region: ThcRegion::Interstitial,
            },
            ThcPoint {
                coordinate: [Bohr(2.0), Bohr(6.5), Bohr(4.0)],
                weight: 0.25,
                region: ThcRegion::Interstitial,
            },
        ],
    )
    .unwrap()
}

pub fn thc_spec() -> ScalarThcSpec {
    ScalarThcSpec {
        spin: 0,
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::All,
        engine: ThcEngine::FullColumnPivotedQr,
    }
}

pub fn coulomb_spec() -> ScalarCoulombSpec {
    ScalarCoulombSpec {
        request: CoulombRequest::cubic(8.0, 2).unwrap(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 1).unwrap(),
    }
}

pub const LATTICE: f64 = 8.0;
