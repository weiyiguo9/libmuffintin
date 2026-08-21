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
