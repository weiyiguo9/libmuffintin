//! Shared full-first-variation two-k hydrogen/HDLO fixture for spinor runtime tests.

#![allow(dead_code)]

use std::collections::BTreeMap;

use muffintin::{RankPolicy, SpinorCoulombSpec, SpinorThcSpec, ThcCandidates, ThcEngine};
use muffintin_core::{Hartree, InverseBohr};
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

#[path = "thc_fixture_common.rs"]
mod thc_fixture_common;

pub use thc_fixture_common::spinor_parent_grid as parent_grid;

pub const LATTICE: f64 = 8.0;

pub fn hydrogen_spinor_snapshot() -> SnapshotV2 {
    let point_count = 61;
    let first: f64 = 1.0e-4;
    let radius: f64 = 1.0;
    let increment = (radius / first).ln() / (point_count - 1) as f64;
    let radii = (0..point_count)
        .map(|index| first * (index as f64 * increment).exp())
        .collect::<Vec<_>>();
    SnapshotV1::new(
        MetaV1 {
            title: "spinor hydrogen smoke".to_owned(),
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
                vectors: [
                    [LATTICE, 0.0, 0.0],
                    [0.0, LATTICE, 0.0],
                    [0.0, 0.0, LATTICE],
                ],
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
                    radial_equation: RadialEquationTagV1::FullyRelativisticDirac,
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

pub fn spinor_config(divisions: [usize; 3], cutoff: f64) -> ScfConfig {
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
                ScfChannelRecipe {
                    site: "H-1".to_owned(),
                    identity: ScfChannelIdentity::Kappa { n: 2, kappa: 1 },
                    treatment: ScfChannelTreatment::Hdlo,
                    derivative_order: 2,
                    generator: LinearizationEnergyGenerator::FrozenSnapshot,
                    seed: None,
                    provenance: ScfChannelProvenance::Site,
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
        relativity: ScfRelativity::SpinorFirstVariation,
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

pub fn thc_spec() -> SpinorThcSpec {
    SpinorThcSpec {
        rank: RankPolicy::Exact { n_mu: 1 },
        candidates: ThcCandidates::All,
        engine: ThcEngine::FullColumnPivotedQr,
    }
}

pub fn coulomb_spec() -> SpinorCoulombSpec {
    SpinorCoulombSpec {
        request: CoulombRequest::cubic(LATTICE, 2).unwrap(),
        projection: InterpolationProjection::new(InverseBohr(1.5), 1).unwrap(),
    }
}
