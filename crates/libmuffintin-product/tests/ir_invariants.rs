//! Product-space IR invariants independent of mixed-product `TOL`.

use libmuffintin_basis::Provenance;
use libmuffintin_core::{
    Bohr, ExponentialMesh, GVector, InterstitialGeometry, InverseBohr, Sphere, VolumeBohr3,
};
use libmuffintin_product::{
    ProductError, ProductPartition, ProductRadial, ProductSource, RadialSamples,
    RawInterstitialPairComponent, RawInterstitialPairSupport, SiteRadialSet, TransferQ,
};

fn geometry() -> InterstitialGeometry {
    InterstitialGeometry::new(
        VolumeBohr3(512.0),
        vec![Sphere {
            center: [Bohr(0.1), Bohr(0.0), Bohr(0.0)],
            radius: Bohr(0.8),
        }],
    )
    .unwrap()
}

fn q_gamma() -> TransferQ {
    TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap()
}

#[test]
fn partition_follows_interstitial_spheres() {
    let interstitial = geometry();
    let partition = ProductPartition::from_interstitial(interstitial.clone());
    assert_eq!(partition.site_count(), 1);
    assert_eq!(
        partition.sites[0].position,
        [Bohr(0.1), Bohr(0.0), Bohr(0.0)]
    );
    assert_eq!(
        partition.interstitial.cell_volume(),
        interstitial.cell_volume()
    );
}

#[test]
fn transfer_q_records_explicit_umklapp() {
    let input = [InverseBohr(1.2), InverseBohr(0.0), InverseBohr(0.0)];
    let wrap = GVector {
        index: [1, 0, 0],
        cartesian: [InverseBohr(0.8), InverseBohr(0.0), InverseBohr(0.0)],
        norm: InverseBohr(0.8),
    };
    let q = TransferQ::fold_by_reciprocal_vector(input, wrap).unwrap();
    assert!((q.cartesian[0].get() - 0.4).abs() < 1.0e-15);
    assert_eq!(q.umklapp.index, [1, 0, 0]);
}

#[test]
fn product_source_requires_matching_radial_site_count() {
    let partition = ProductPartition::from_interstitial(geometry());
    let q = q_gamma();
    let error = ProductSource::new(
        partition,
        Vec::new(),
        q,
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        libmuffintin_product::ProductError::SiteCount {
            expected: 1,
            actual: 0
        }
    ));
}

#[test]
fn product_source_does_not_carry_compiled_basis_fields() {
    let mesh = ExponentialMesh::new(Bohr(1.0e-4), 0.2, 16).unwrap();
    let n = mesh.len();
    let q = q_gamma();
    let source = ProductSource::new(
        ProductPartition::from_interstitial(geometry()),
        vec![SiteRadialSet {
            mesh,
            valence: vec![ProductRadial {
                l: 0,
                n: 0,
                spin: 0,
                samples: RadialSamples {
                    large: vec![0.0; n],
                    small: None,
                },
            }],
            cores: Vec::new(),
        }],
        q,
        RawInterstitialPairSupport::empty(q),
        Provenance::default(),
    )
    .unwrap();
    assert_eq!(source.radials.len(), 1);
    assert!(source.partition.sites[0].radius.get() > 0.0);
    assert!(source.interstitial_pair_support.components.is_empty());
}

#[test]
fn raw_pair_support_rejects_duplicate_g_labels() {
    let q = q_gamma();
    let g = GVector {
        index: [1, 0, 0],
        cartesian: [InverseBohr(0.8), InverseBohr(0.0), InverseBohr(0.0)],
        norm: InverseBohr(0.8),
    };
    let error = RawInterstitialPairSupport::from_components(
        q,
        vec![
            RawInterstitialPairComponent { g_relative: g },
            RawInterstitialPairComponent { g_relative: g },
        ],
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ProductError::DuplicatePairComponent { index } if index == [1, 0, 0]
    ));
}

#[test]
fn product_source_rejects_pair_support_at_a_different_q() {
    let mesh = ExponentialMesh::new(Bohr(1.0e-4), 0.2, 16).unwrap();
    let n = mesh.len();
    let q = q_gamma();
    let other =
        TransferQ::from_cartesian([InverseBohr(0.2), InverseBohr(0.0), InverseBohr(0.0)]).unwrap();
    let error = ProductSource::new(
        ProductPartition::from_interstitial(geometry()),
        vec![SiteRadialSet {
            mesh,
            valence: vec![ProductRadial {
                l: 0,
                n: 0,
                spin: 0,
                samples: RadialSamples {
                    large: vec![0.0; n],
                    small: None,
                },
            }],
            cores: Vec::new(),
        }],
        q,
        RawInterstitialPairSupport::empty(other),
        Provenance::default(),
    )
    .unwrap_err();
    assert!(matches!(error, ProductError::PairSupportTransferQ));
}

#[test]
fn interpolation_points_are_not_empty_mixed_product_payloads() {
    use libmuffintin_core::{Bohr, VolumeBohr3};
    use libmuffintin_product::{
        AuxiliaryRegion, AuxiliaryRepresentation, CompiledAuxiliaryBasis,
        InterpolationAuxiliaryPoint, InterpolationPointAuxiliary, InterpolationRegion,
        MixedProductAuxiliary, OrbitalPair, PairVertex,
    };
    use num_complex::Complex64;

    let partition = ProductPartition::from_interstitial(geometry());
    let q = q_gamma();
    let points = vec![
        InterpolationAuxiliaryPoint {
            id: 2,
            coordinate: [Bohr(0.1), Bohr(0.0), Bohr(0.0)],
            weight: VolumeBohr3(0.01),
            region: InterpolationRegion::MuffinTin { site: 0 },
        },
        InterpolationAuxiliaryPoint {
            id: 0,
            coordinate: [Bohr(1.5), Bohr(0.0), Bohr(0.0)],
            weight: VolumeBohr3(0.02),
            region: InterpolationRegion::Interstitial,
        },
        InterpolationAuxiliaryPoint {
            id: 1,
            coordinate: [Bohr(2.0), Bohr(0.0), Bohr(0.0)],
            weight: VolumeBohr3(0.03),
            region: InterpolationRegion::Uniform,
        },
    ];
    let mut ordered = points.clone();
    libmuffintin_product::sort_interpolation_points(&mut ordered);
    let auxiliary = CompiledAuxiliaryBasis {
        partition: partition.clone(),
        q,
        representation: AuxiliaryRepresentation::InterpolationPoints(InterpolationPointAuxiliary {
            points: ordered,
        }),
        provenance: Provenance::default(),
    };
    auxiliary.validate().unwrap();
    assert!(auxiliary.mixed_product().is_none());
    assert_eq!(auxiliary.mt_dimension(), 1);
    assert_eq!(auxiliary.interstitial_dimension(), 2);
    assert_eq!(
        auxiliary.regions(),
        vec![
            AuxiliaryRegion::InterpolationPoint {
                id: 2,
                region: InterpolationRegion::MuffinTin { site: 0 },
            },
            AuxiliaryRegion::InterpolationPoint {
                id: 0,
                region: InterpolationRegion::Interstitial,
            },
            AuxiliaryRegion::InterpolationPoint {
                id: 1,
                region: InterpolationRegion::Uniform,
            },
        ]
    );
    let vertex = PairVertex::new(
        q,
        OrbitalPair::Bloch {
            k_index: 0,
            left: 1,
            right: 2,
        },
        auxiliary.mt_dimension(),
        auxiliary.interstitial_dimension(),
        vec![Complex64::new(1.0, 0.0); 3],
        Provenance::default(),
    )
    .unwrap();
    assert_eq!(vertex.mt().len(), 1);
    assert_eq!(vertex.interstitial().len(), 2);
    assert!(matches!(
        vertex.pair(),
        OrbitalPair::Bloch {
            k_index: 0,
            left: 1,
            right: 2
        }
    ));
    let mixed = CompiledAuxiliaryBasis {
        partition,
        q,
        representation: AuxiliaryRepresentation::MixedProduct(MixedProductAuxiliary {
            sites: Vec::new(),
            interstitial: libmuffintin_product::AuxiliaryInterstitialSupport {
                q,
                g_cut: InverseBohr(0.0),
                waves: Vec::new(),
            },
            cutoff: None,
        }),
        provenance: Provenance::default(),
    };
    assert!(matches!(
        mixed.require_interpolation_points(),
        Err(ProductError::ExpectedInterpolationPoints)
    ));
}

#[test]
fn interpolation_points_reject_negative_and_all_zero_weights() {
    use libmuffintin_core::{Bohr, VolumeBohr3};
    use libmuffintin_product::{
        AuxiliaryRepresentation, CompiledAuxiliaryBasis, InterpolationAuxiliaryPoint,
        InterpolationPointAuxiliary, InterpolationRegion,
    };

    let partition = ProductPartition::from_interstitial(geometry());
    let q = q_gamma();
    let negative = CompiledAuxiliaryBasis {
        partition: partition.clone(),
        q,
        representation: AuxiliaryRepresentation::InterpolationPoints(InterpolationPointAuxiliary {
            points: vec![InterpolationAuxiliaryPoint {
                id: 0,
                coordinate: [Bohr(0.1), Bohr(0.0), Bohr(0.0)],
                weight: VolumeBohr3(-0.01),
                region: InterpolationRegion::Uniform,
            }],
        }),
        provenance: Provenance::default(),
    };
    assert!(matches!(
        negative.validate(),
        Err(ProductError::NegativeInterpolationWeight(0))
    ));
    let zeros = CompiledAuxiliaryBasis {
        partition,
        q,
        representation: AuxiliaryRepresentation::InterpolationPoints(InterpolationPointAuxiliary {
            points: vec![InterpolationAuxiliaryPoint {
                id: 0,
                coordinate: [Bohr(0.1), Bohr(0.0), Bohr(0.0)],
                weight: VolumeBohr3(0.0),
                region: InterpolationRegion::Uniform,
            }],
        }),
        provenance: Provenance::default(),
    };
    assert!(matches!(
        zeros.validate(),
        Err(ProductError::NoPositiveInterpolationWeight)
    ));
}
