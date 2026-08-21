//! Hermiticity and PSD/action checks on deterministic toy lattices.

mod common;

use faer::{Mat, Side};
use libmuffintin_core::InverseBohr;
use libmuffintin_coulomb::{
    CoulombRequest, InterpolationProjection, assemble_coulomb, assemble_sampled_coulomb,
};
use libmuffintin_product::TransferQ;
use num_complex::Complex64;

fn hermiticity_residual(matrix: &[Complex64], n: usize) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let residual = (matrix[i * n + j] - matrix[j * n + i].conj()).norm();
            worst = worst.max(residual);
        }
    }
    worst
}

fn min_eigenvalue(matrix: &[Complex64], n: usize) -> f64 {
    let mat = Mat::from_fn(n, n, |row, column| matrix[row * n + column]);
    let eigen = mat.self_adjoint_eigen(Side::Lower).expect("Hermitian EVD");
    (0..n)
        .map(|index| eigen.S()[index].re)
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn finite_q_mixed_product_is_hermitian_and_nearly_psd() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let n = operator.dimension();
    let residual = hermiticity_residual(operator.matrix(), n);
    assert!(residual < 1.0e-10, "Hermitian residual {residual}");
    let min_eig = min_eigenvalue(operator.matrix(), n);
    assert!(
        min_eig > -1.0e-6,
        "finite-q mixed-product min eigenvalue {min_eig}"
    );
    assert!(operator.gamma().is_none());
}

#[test]
fn gamma_body_is_finite_records_head_and_stays_hermitian() {
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    assert!(operator.gamma().is_some());
    let head = operator.gamma().unwrap();
    assert!(head.spherical_average_subtracted);
    assert!((head.head_prefactor - 4.0 * std::f64::consts::PI).abs() < 1.0e-12);
    assert_eq!(head.constant_coefficients.len(), operator.dimension());
    assert!(
        head.constant_coefficients
            .iter()
            .any(|value| value.norm() > 1.0e-12),
        "Gamma head ω must not be silently zero"
    );
    assert!(
        operator
            .matrix()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let residual = hermiticity_residual(operator.matrix(), operator.dimension());
    assert!(residual < 1.0e-8, "Gamma Hermitian residual {residual}");
    let reconstructed_head = head.head_prefactor;
    assert!(reconstructed_head.is_finite() && reconstructed_head > 0.0);
    let omega = &head.constant_coefficients;
    let mut head_norm: f64 = 0.0;
    for value in omega {
        head_norm += value.norm_sqr();
    }
    assert!(head_norm > 0.0, "Gamma |ω⟩ must have positive norm");
    let n = operator.dimension();
    for i in 0..n {
        for j in 0..n {
            let rank_one = reconstructed_head * omega[i].conj() * omega[j];
            assert!(rank_one.re.is_finite() && rank_one.im.is_finite());
            assert!(operator.element(i, j).unwrap().re.is_finite());
        }
    }
}

#[test]
fn sampled_interpolation_is_hermitian_and_nearly_psd() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2)
        .unwrap()
        .with_interpolation(InterpolationProjection::new(InverseBohr(1.6), 2).unwrap())
        .unwrap();
    let auxiliary = common::interpolation_auxiliary(q);
    let sampled = common::identity_zeta(&auxiliary);
    let operator = assemble_sampled_coulomb(&auxiliary, &request, &sampled).unwrap();
    assert_eq!(
        operator.kind(),
        libmuffintin_coulomb::AuxiliaryKind::InterpolationPoints
    );
    let residual = hermiticity_residual(operator.matrix(), operator.dimension());
    assert!(residual < 1.0e-8, "sampled Hermitian residual {residual}");
    let min_eig = min_eigenvalue(operator.matrix(), operator.dimension());
    assert!(min_eig > -1.0e-5, "sampled-zeta min eigenvalue {min_eig}");
}
