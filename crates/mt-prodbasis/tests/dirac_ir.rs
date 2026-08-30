//! Parallel Dirac product-IR invariants.

use muffintin_core::{
    Bohr, ExponentialMesh, InterstitialGeometry, InverseBohr, Kappa, Sphere, VolumeBohr3,
};
use muffintin_envelope::Provenance;
use muffintin_prodbasis::{
    AuxiliaryPartition, CoupledChannel, DiracChargeSector, DiracPairChannel, DiracProductError,
    DiracProductSource, DiracRadial, DiracRadialId, DiracRadialSamples, DiracRawProductSpace,
    DiracRawRadialProduct, DiracSiteRadialSet, ProductOrbitalKind, RawInterstitialPairSupport,
    TransferQ,
};

fn mesh() -> ExponentialMesh {
    ExponentialMesh::new(Bohr(1.0e-4), 0.15, 24).unwrap()
}

fn partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(512.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(0.8),
            }],
        )
        .unwrap(),
    )
}

fn q_gamma() -> TransferQ {
    TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap()
}

fn samples(scale_q: f64) -> DiracRadialSamples {
    let mesh = mesh();
    let large = mesh
        .radii()
        .iter()
        .map(|radius| radius.get() * (-2.0 * radius.get()).exp())
        .collect::<Vec<_>>();
    let small = large.iter().map(|value| scale_q * value).collect();
    DiracRadialSamples { large, small }
}

fn source_with(radials: Vec<DiracRadial>) -> DiracProductSource {
    let q = q_gamma();
    DiracProductSource::new(
        partition(),
        vec![DiracSiteRadialSet {
            mesh: mesh(),
            valence: radials,
            cores: Vec::new(),
        }],
        q,
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap()
}

fn kappa(value: i32) -> Kappa {
    Kappa::new(value).unwrap()
}

#[test]
fn dirac_source_requires_equal_length_physical_p_and_q() {
    // Compile-time exhaustiveness: Dirac charge sectors are only PP and QQ.
    match DiracChargeSector::LargeLarge {
        DiracChargeSector::LargeLarge | DiracChargeSector::SmallSmall => {}
    }
    let n = mesh().len();
    let large = vec![1.0; n];
    let error = DiracProductSource::new(
        partition(),
        vec![DiracSiteRadialSet {
            mesh: mesh(),
            valence: vec![DiracRadial {
                kappa: kappa(-1),
                n: 0,
                samples: DiracRadialSamples {
                    large: large.clone(),
                    small: vec![0.1; n - 1],
                },
            }],
            cores: Vec::new(),
        }],
        q_gamma(),
        RawInterstitialPairSupport::empty(q_gamma()),
        Provenance::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracProductError::UnequalPqLength {
            site: 0,
            kappa: -1,
            n: 0,
            large,
            small,
        } if large == n && small == n - 1
    ));
}

#[test]
fn dirac_source_rejects_site_count_mismatch() {
    let error = DiracProductSource::new(
        partition(),
        Vec::new(),
        q_gamma(),
        RawInterstitialPairSupport::empty(q_gamma()),
        Provenance::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracProductError::SiteCount {
            expected: 1,
            actual: 0
        }
    ));
}

#[test]
fn dirac_raw_rejects_signed_kappa_identity_mismatch() {
    let source = source_with(vec![DiracRadial {
        kappa: kappa(-1),
        n: 0,
        samples: samples(0.3),
    }]);
    let n = mesh().len();
    let raw = DiracRawProductSpace::new(
        source.partition.clone(),
        source.q,
        vec![DiracRawRadialProduct {
            channel: DiracPairChannel {
                q: source.q,
                left: DiracRadialId {
                    site: 0,
                    kind: ProductOrbitalKind::Valence,
                    kappa: kappa(1),
                    n: 0,
                },
                right: DiracRadialId {
                    site: 0,
                    kind: ProductOrbitalKind::Valence,
                    kappa: kappa(-1),
                    n: 0,
                },
                coupled_l: 0,
                sector: DiracChargeSector::LargeLarge,
            },
            samples: vec![0.0; n],
        }],
        vec![CoupledChannel {
            site: 0,
            l: 0,
            m: 0,
            radial_index: 0,
        }],
        source.interstitial_pair_support.clone(),
        Provenance::default(),
    )
    .unwrap();
    let error = raw.validate_against_source(&source).unwrap_err();
    assert!(matches!(
        error,
        DiracProductError::UnknownDiracOrbital {
            site: 0,
            kind: ProductOrbitalKind::Valence,
            kappa: 1,
            n: 0
        }
    ));
}

fn valence_id(site: usize, raw_kappa: i32, n: usize) -> DiracRadialId {
    DiracRadialId {
        site,
        kind: ProductOrbitalKind::Valence,
        kappa: kappa(raw_kappa),
        n,
    }
}

fn dummy_product(
    q: TransferQ,
    left: DiracRadialId,
    right: DiracRadialId,
    coupled_l: u32,
    sector: DiracChargeSector,
) -> DiracRawRadialProduct {
    DiracRawRadialProduct {
        channel: DiracPairChannel {
            q,
            left,
            right,
            coupled_l,
            sector,
        },
        samples: vec![0.0],
    }
}

#[test]
fn dirac_raw_rejects_cross_site_mt_product() {
    let q = q_gamma();
    let error = DiracRawProductSpace::new(
        partition(),
        q,
        vec![dummy_product(
            q,
            valence_id(0, -1, 0),
            valence_id(1, -1, 0),
            0,
            DiracChargeSector::LargeLarge,
        )],
        vec![CoupledChannel {
            site: 0,
            l: 0,
            m: 0,
            radial_index: 0,
        }],
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracProductError::CrossSiteRawProduct {
            left_site: 0,
            right_site: 1
        }
    ));
}

#[test]
fn dirac_raw_rejects_swapped_duplicate_products() {
    let q = q_gamma();
    let left = valence_id(0, -1, 0);
    let right = valence_id(0, -2, 1);
    let error = DiracRawProductSpace::new(
        partition(),
        q,
        vec![
            dummy_product(q, left, right, 1, DiracChargeSector::LargeLarge),
            dummy_product(q, right, left, 1, DiracChargeSector::LargeLarge),
        ],
        vec![CoupledChannel {
            site: 0,
            l: 1,
            m: 0,
            radial_index: 0,
        }],
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        DiracProductError::DuplicateRawRadialProduct {
            coupled_l: 1,
            sector: DiracChargeSector::LargeLarge,
            ..
        }
    ));
    let exact = DiracRawProductSpace::new(
        partition(),
        q,
        vec![
            dummy_product(q, left, right, 1, DiracChargeSector::SmallSmall),
            dummy_product(q, left, right, 1, DiracChargeSector::SmallSmall),
        ],
        vec![CoupledChannel {
            site: 0,
            l: 1,
            m: 0,
            radial_index: 0,
        }],
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap_err();
    assert!(matches!(
        exact,
        DiracProductError::DuplicateRawRadialProduct {
            coupled_l: 1,
            sector: DiracChargeSector::SmallSmall,
            ..
        }
    ));
}
