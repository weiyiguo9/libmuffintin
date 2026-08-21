//! Pair-vertex application and exact layout mismatch rejection.

mod common;

use libmuffintin_core::InverseBohr;
use libmuffintin_coulomb::{CoulombError, CoulombRequest, assemble_coulomb};
use libmuffintin_product::{AuxiliaryLayout, AuxiliaryRegion, OrbitalPair, PairVertex, TransferQ};

#[test]
fn apply_matches_quadratic_form_on_a_unit_vector() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let vertex = common::unit_vertex(&auxiliary, 0);
    let applied = operator.apply(&vertex).unwrap();
    assert_eq!(applied.len(), operator.dimension());
    let quadratic = operator.quadratic_form(&vertex, &vertex).unwrap();
    assert!((quadratic - operator.element(0, 0).unwrap()).norm() < 1.0e-12);
}

#[test]
fn q_mismatch_is_rejected() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let other_q =
        TransferQ::from_cartesian([InverseBohr(0.1), InverseBohr(0.0), InverseBohr(0.0)]).unwrap();
    let mut coefficients = vec![num_complex::Complex64::default(); auxiliary.dimension()];
    coefficients[0] = num_complex::Complex64::new(1.0, 0.0);
    let layout = AuxiliaryLayout::from_regions(other_q, auxiliary.regions());
    let vertex = PairVertex::new(
        layout,
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0,
        },
        coefficients,
        auxiliary.provenance.clone(),
    )
    .unwrap();
    let error = operator.apply(&vertex).unwrap_err();
    assert!(matches!(error, CoulombError::VertexTransferQ));
}

#[test]
fn dimension_mismatch_is_rejected() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let short = auxiliary.regions()[..1].to_vec();
    let layout = AuxiliaryLayout::from_regions(q, short);
    let vertex = PairVertex::new(
        layout,
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0,
        },
        vec![num_complex::Complex64::new(1.0, 0.0)],
        auxiliary.provenance.clone(),
    )
    .unwrap();
    let error = operator.apply(&vertex).unwrap_err();
    assert!(matches!(error, CoulombError::VertexDimension { .. }));
}

#[test]
fn permuted_regions_with_same_counts_are_rejected() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let mut regions = auxiliary.regions();
    assert!(regions.len() >= 2);
    regions.swap(0, 1);
    assert_ne!(regions[0], operator.regions()[0]);
    let mt = regions.iter().filter(|region| region.is_mt_block()).count();
    assert_eq!(mt, operator.mt_dimension());
    assert_eq!(regions.len() - mt, operator.interstitial_dimension());
    let layout = AuxiliaryLayout::from_regions(q, regions);
    let mut coefficients = vec![num_complex::Complex64::default(); layout.dimension()];
    coefficients[0] = num_complex::Complex64::new(1.0, 0.0);
    let vertex = PairVertex::new(
        layout,
        OrbitalPair::Bloch {
            k_index: 0,
            left: 0,
            right: 0,
        },
        coefficients,
        auxiliary.provenance.clone(),
    )
    .unwrap();
    let error = operator.apply(&vertex).unwrap_err();
    assert!(matches!(error, CoulombError::VertexLayout));
    let _ = AuxiliaryRegion::Interstitial {
        g: libmuffintin_core::GVector {
            index: [0; 3],
            cartesian: [InverseBohr(0.0); 3],
            norm: InverseBohr(0.0),
        },
    };
}
