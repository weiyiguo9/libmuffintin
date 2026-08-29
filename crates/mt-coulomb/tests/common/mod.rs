//! Shared toy fixtures for Coulomb integration tests.
#![allow(dead_code)]

use muffintin_auxiliary_ir::{
    AuxiliaryPartition, AuxiliaryRepresentation, AuxiliarySource, CompiledAuxiliaryBasis,
    InterpolationAuxiliaryPoint, InterpolationPointAuxiliary, InterpolationRegion, OrbitalPair,
    PairOrbital, PairVertex, ProductOrbitalKind, ProductRadial, ProductRadialId, RadialSamples,
    RawInterstitialPairComponent, RawInterstitialPairSupport, SiteRadialSet, TransferQ,
};
use muffintin_basis::Provenance;
use muffintin_core::{
    Bohr, GVector, InterstitialGeometry, InverseBohr, ReciprocalLattice, Sphere, VolumeBohr3,
};
use muffintin_coulomb::{SampledAuxiliaryFunctions, SampledPointSupport};
use muffintin_mpb::spex_mixed_product_basis;
use num_complex::Complex64;
use std::f64::consts::PI;

pub const LATTICE: f64 = 8.0;
pub const RADIUS: f64 = 0.8;
pub const POSITION: [Bohr; 3] = [Bohr(0.25), Bohr(0.0), Bohr(0.0)];

pub fn mesh() -> muffintin_core::ExponentialMesh {
    let first = 1.0e-5;
    let number = 73;
    let increment = (RADIUS / first).ln() / (number - 1) as f64;
    muffintin_core::ExponentialMesh::new(Bohr(first), increment, number).unwrap()
}

pub fn samples(kind: u8) -> RadialSamples {
    let mesh = mesh();
    let large = mesh
        .radii()
        .iter()
        .map(|radius| {
            let r = radius.get();
            match kind {
                0 => r * (-2.0 * r).exp(),
                1 => r * (1.0 - 0.4 * r) * (-2.0 * r).exp(),
                _ => r * r * (-2.2 * r).exp(),
            }
        })
        .collect();
    RadialSamples { large, small: None }
}

pub fn partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(LATTICE.powi(3)),
            vec![Sphere {
                center: POSITION,
                radius: Bohr(RADIUS),
            }],
        )
        .unwrap(),
    )
}

pub fn cubic_lattice() -> ReciprocalLattice {
    ReciprocalLattice::from_direct([
        [Bohr(LATTICE), Bohr(0.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(LATTICE), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(LATTICE)],
    ])
    .unwrap()
}

pub fn g_vector(lattice: &ReciprocalLattice, index: [i32; 3]) -> GVector {
    let cartesian = lattice.cartesian(index);
    let norm = InverseBohr(
        cartesian
            .iter()
            .map(|component| component.get().powi(2))
            .sum::<f64>()
            .sqrt(),
    );
    GVector {
        index,
        cartesian,
        norm,
    }
}

pub fn transfer_q(frac: [f64; 3]) -> TransferQ {
    let scale = 2.0 * PI / LATTICE;
    TransferQ::from_cartesian(std::array::from_fn(|axis| InverseBohr(scale * frac[axis]))).unwrap()
}

pub fn pair_support(q: TransferQ, lattice: &ReciprocalLattice) -> RawInterstitialPairSupport {
    let labels = [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]];
    RawInterstitialPairSupport::from_components(
        q,
        labels
            .into_iter()
            .map(|index| RawInterstitialPairComponent {
                g_relative: g_vector(lattice, index),
            })
            .collect(),
    )
    .unwrap()
}

pub fn product_source(q: TransferQ) -> AuxiliarySource {
    let lattice = cubic_lattice();
    AuxiliarySource::new(
        partition(),
        vec![SiteRadialSet {
            mesh: mesh(),
            valence: vec![
                ProductRadial {
                    l: 0,
                    n: 0,
                    spin: 0,
                    samples: samples(0),
                },
                ProductRadial {
                    l: 0,
                    n: 1,
                    spin: 0,
                    samples: samples(1),
                },
            ],
            cores: Vec::new(),
        }],
        q,
        pair_support(q, &lattice),
        Provenance::default(),
    )
    .unwrap()
}

pub fn mixed_product_auxiliary(q: TransferQ) -> (AuxiliarySource, CompiledAuxiliaryBasis) {
    let source = product_source(q);
    let lattice = cubic_lattice();
    let (_raw, auxiliary) =
        spex_mixed_product_basis(&source, 0, InverseBohr(1.6), &lattice).unwrap();
    (source, auxiliary)
}

pub fn interpolation_auxiliary(q: TransferQ) -> CompiledAuxiliaryBasis {
    let first_shell = mesh().first().get();
    let mut points = vec![
        InterpolationAuxiliaryPoint {
            id: 0,
            coordinate: [
                Bohr(POSITION[0].get() + first_shell),
                POSITION[1],
                POSITION[2],
            ],
            weight: VolumeBohr3(0.05),
            region: InterpolationRegion::MuffinTin { site: 0 },
        },
        InterpolationAuxiliaryPoint {
            id: 1,
            coordinate: [Bohr(2.0), Bohr(0.0), Bohr(0.0)],
            weight: VolumeBohr3(0.4),
            region: InterpolationRegion::Interstitial,
        },
        InterpolationAuxiliaryPoint {
            id: 2,
            coordinate: [Bohr(0.0), Bohr(2.0), Bohr(0.0)],
            weight: VolumeBohr3(0.4),
            region: InterpolationRegion::Interstitial,
        },
        InterpolationAuxiliaryPoint {
            id: 3,
            coordinate: [Bohr(0.0), Bohr(0.0), Bohr(2.0)],
            weight: VolumeBohr3(0.4),
            region: InterpolationRegion::Uniform,
        },
    ];
    muffintin_auxiliary_ir::sort_interpolation_points(&mut points);
    let auxiliary = CompiledAuxiliaryBasis {
        partition: partition(),
        q,
        representation: AuxiliaryRepresentation::InterpolationPoints(InterpolationPointAuxiliary {
            points,
        }),
        provenance: Provenance {
            recipe: Some("coulomb-interpolation-fixture".to_owned()),
            reference: Some("interpolation charge expansion".to_owned()),
        },
    };
    auxiliary.validate().unwrap();
    auxiliary
}

/// Identity $\zeta$ on the interpolation nodes themselves (plumbing, not THC).
pub fn identity_zeta(auxiliary: &CompiledAuxiliaryBasis) -> SampledAuxiliaryFunctions {
    let points = auxiliary.require_interpolation_points().unwrap();
    let n = points.len();
    let mut zeta = vec![Complex64::default(); n * n];
    for mu in 0..n {
        zeta[mu * n + mu] = Complex64::new(1.0, 0.0);
    }
    SampledAuxiliaryFunctions::new(
        auxiliary.layout(),
        vec![mesh()],
        points.iter().map(|point| point.coordinate).collect(),
        points.iter().map(|point| point.weight).collect(),
        points
            .iter()
            .map(|point| match point.region {
                InterpolationRegion::MuffinTin { site } => SampledPointSupport::MuffinTin {
                    site,
                    radial_index: 0,
                },
                InterpolationRegion::Interstitial => SampledPointSupport::Interstitial,
                InterpolationRegion::Uniform => SampledPointSupport::Uniform,
            })
            .collect(),
        zeta,
    )
    .unwrap()
}

pub fn unit_vertex(auxiliary: &CompiledAuxiliaryBasis, index: usize) -> PairVertex {
    let mut coefficients = vec![Complex64::default(); auxiliary.dimension()];
    coefficients[index] = Complex64::new(1.0, 0.0);
    PairVertex::from_auxiliary(
        auxiliary,
        OrbitalPair::MuffinTin {
            left: PairOrbital::Radial {
                id: ProductRadialId {
                    site: 0,
                    kind: ProductOrbitalKind::Valence,
                    l: 0,
                    n: 0,
                    spin: 0,
                },
                m: 0,
            },
            right: PairOrbital::Radial {
                id: ProductRadialId {
                    site: 0,
                    kind: ProductOrbitalKind::Valence,
                    l: 0,
                    n: 0,
                    spin: 0,
                },
                m: 0,
            },
        },
        coefficients,
    )
    .unwrap()
}
