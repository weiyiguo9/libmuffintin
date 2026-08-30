//! Independent formula oracles for SPEX/Weinert building blocks.

mod common;

use muffintin_prodbasis::TransferQ;
use muffintin_core::{Bohr, ExponentialMesh, InverseBohr, ReciprocalLattice};
use muffintin_coulomb::{
    CoulombError, CoulombRequest, DEFAULT_LEXP, assemble_coulomb, brute_force_structure_constant,
    intra_sphere_poisson, multipole_moment, radial_primitive, second_moment, spex_real_g,
    spherical_bessel_moment, structure_constants, weinert_gmat,
};
use muffintin_core::Cell;
use std::f64::consts::E;
use std::f64::consts::PI;

fn sfac(n: usize) -> Vec<f64> {
    let mut table = vec![1.0; n + 1];
    for i in 1..=n {
        table[i] = table[i - 1] * (i as f64).sqrt();
    }
    table
}

#[test]
fn gmat_l0_l0_is_four_pi_to_the_three_halves() {
    let table = sfac(8);
    let value = weinert_gmat(0, 0, 0, 0, &table).unwrap();
    let expected = (4.0 * PI).powf(1.5);
    assert!((value - expected).abs() < 1.0e-12);
}

#[test]
fn gmat_is_symmetric_and_l1_l0_matches_closed_form() {
    let table = sfac(12);
    let left = weinert_gmat(1, 0, 0, 0, &table).unwrap();
    let right = weinert_gmat(0, 0, 1, 0, &table).unwrap();
    assert!((left - right).abs() < 1.0e-12);
    let expected = (4.0 * PI).powf(1.5) / 3.0;
    assert!((left - expected).abs() < 1.0e-12);
}

fn truncated_exp_inner(a: f64, last: i32) -> f64 {
    let mut acc = 1.0;
    for k in (1..=last).rev() {
        acc = 1.0 + a / f64::from(k) * acc;
    }
    acc
}

#[test]
fn spex_real_g4_skips_a_over_nine() {
    let a: f64 = 2.0;
    let nested = 1.0
        + a * (1.0
            + a / 2.0
                * (1.0
                    + a / 3.0
                        * (1.0
                            + a / 4.0
                                * (1.0
                                    + a / 5.0
                                        * (1.0
                                            + a / 6.0
                                                * (1.0
                                                    + a / 7.0
                                                        * (1.0 + a / 8.0 * (1.0 + a / 10.0))))))));
    let expected = (-a).exp() / a.powi(5) * nested;
    let sequential_inner = truncated_exp_inner(a, 10);
    assert!(
        (nested - sequential_inner).abs() > 1.0e-4,
        "skip-9 inner {nested} sequential {sequential_inner}"
    );
    assert!((spex_real_g(4, a) - expected).abs() < 1.0e-14);
    assert!((spex_real_g(4, a) - (-a).exp() / a.powi(5) * sequential_inner).abs() > 1.0e-6);
}

#[test]
fn spex_real_g_l4_to_l7_match_independent_closed_forms() {
    let a: f64 = 2.0;
    let rexp = (-a).exp();
    let g4 = rexp / a.powi(5)
        * (1.0
            + a * (1.0
                + a / 2.0
                    * (1.0
                        + a / 3.0
                            * (1.0
                                + a / 4.0
                                    * (1.0
                                        + a / 5.0
                                            * (1.0
                                                + a / 6.0
                                                    * (1.0
                                                        + a / 7.0
                                                            * (1.0
                                                                + a / 8.0
                                                                    * (1.0 + a / 10.0)))))))));
    let g5 = rexp / a.powi(6) * truncated_exp_inner(a, 10);
    let g6 = rexp / a.powi(7) * truncated_exp_inner(a, 12);
    let g7 = rexp / a.powi(8) * truncated_exp_inner(a, 13);
    assert!((spex_real_g(4, a) - g4).abs() < 1.0e-14);
    assert!((spex_real_g(5, a) - g5).abs() < 1.0e-14);
    assert!((spex_real_g(6, a) - g6).abs() < 1.0e-14);
    assert!((spex_real_g(7, a) - g7).abs() < 1.0e-14);
    // Recorded regression values at a=2 (SPEX polynomials, not sequential 1..N).
    assert!((spex_real_g(4, a) - 3.124_795_021_873_932e-2).abs() < 1.0e-14);
    assert!((spex_real_g(5, a) - 1.562_487_018_399_424e-2).abs() < 1.0e-14);
    assert!((spex_real_g(6, a) - 7.812_498_380_101_89e-3).abs() < 1.0e-14);
    assert!((spex_real_g(7, a) - 3.906_249_885_524_624e-3).abs() < 1.0e-14);
    let a1: f64 = 1.0;
    assert!(
        (spex_real_g(5, a1) - (-a1).exp() / a1.powi(6) * truncated_exp_inner(a1, 10)).abs()
            < 1.0e-14
    );
    assert!((truncated_exp_inner(a1, 10) - E).abs() < 1.0e-7);
}

#[test]
fn spex_real_g_l8_is_inverse_power() {
    let a: f64 = 1.7;
    assert!((spex_real_g(8, a) - a.powi(-9)).abs() < 1.0e-14);
    assert!((spex_real_g(12, a) - a.powi(-13)).abs() < 1.0e-14);
}

#[test]
fn spherical_bessel_moment_q0_is_sphere_volume_over_four_pi() {
    let radius = 0.8;
    let l0 = spherical_bessel_moment(0, 0.0, radius);
    let l1 = spherical_bessel_moment(1, 0.0, radius);
    assert!((l0 - radius.powi(3) / 3.0).abs() < 1.0e-15);
    assert!(l1.abs() < 1.0e-15);
}

#[test]
fn spherical_bessel_moment_finite_q_matches_numeric_integral() {
    let radius = 0.8;
    let q = 1.7;
    let l = 1u32;
    let analytic = spherical_bessel_moment(l, q, radius);
    let n = 4000;
    let dr = radius / n as f64;
    let mut numeric = 0.0;
    for i in 1..=n {
        let r = dr * i as f64;
        numeric += r.powi(l as i32 + 2) * muffintin_core::spherical_bessel_j(l, q * r) * dr;
    }
    assert!(
        (analytic - numeric).abs() < 5.0e-4,
        "analytic {analytic} numeric {numeric}"
    );
}

#[test]
fn radial_primitive_matches_power_law() {
    let mesh = ExponentialMesh::new(Bohr(1.0e-4), 0.04, 121).unwrap();
    let values: Vec<f64> = mesh
        .radii()
        .iter()
        .map(|radius| radius.get().powi(2))
        .collect();
    let outward = radial_primitive(&mesh, &values, false).unwrap();
    let last = mesh.last().get();
    let expected = last.powi(3) / 3.0;
    assert!(
        (outward.last().copied().unwrap() - expected).abs() < 2.0e-3 * expected,
        "got {} expected {expected}",
        outward.last().unwrap()
    );
}

#[test]
fn l0_constant_self_poisson_is_positive() {
    let mesh = common::mesh();
    let radius = mesh.last().get();
    let constant_norm = (radius.powi(3) / 3.0).sqrt();
    let basm: Vec<f64> = mesh
        .radii()
        .iter()
        .map(|sample| sample.get() / constant_norm)
        .collect();
    let value = intra_sphere_poisson(0, &mesh, &basm, &basm).unwrap();
    assert!(value.is_finite());
    assert!(value > 0.0);
    let charge = (4.0 * PI).sqrt() * multipole_moment(0, &mesh, &basm).unwrap();
    let uniform = 6.0 / 5.0 * charge * charge / radius;
    assert!(
        (value - uniform).abs() / uniform < 0.05,
        "poisson {value} uniform {uniform}"
    );
}

#[test]
fn second_moment_of_l0_constant_is_finite() {
    let mesh = common::mesh();
    let radius = mesh.last().get();
    let constant_norm = (radius.powi(3) / 3.0).sqrt();
    let basm: Vec<f64> = mesh
        .radii()
        .iter()
        .map(|sample| sample.get() / constant_norm)
        .collect();
    let value = second_moment(&mesh, &basm).unwrap();
    assert!(value.is_finite());
    assert!(value > 0.0);
}

#[test]
fn structure_constant_l1_matches_brute_force_at_finite_q() {
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let q = TransferQ::from_cartesian([
        InverseBohr(2.0 * PI / common::LATTICE),
        InverseBohr(0.0),
        InverseBohr(0.0),
    ])
    .unwrap();
    let partition = common::partition();
    let spex = structure_constants(request.cell(), request.reciprocal(), &partition, q, 2).unwrap();
    let assembled = spex.get(0, 0, 1, 0).unwrap();
    let independent = brute_force_structure_constant(request.cell(), q, [0.0; 3], 1, 0, 6).unwrap();
    assert!(
        (assembled - independent).norm() < 5.0e-3 * independent.norm().max(1.0),
        "SPEX {assembled} brute {independent}"
    );
}

#[test]
fn lexp_above_12_is_rejected_and_default_is_four() {
    assert_eq!(DEFAULT_LEXP, 4);
    let error = CoulombRequest::cubic(common::LATTICE, 13).unwrap_err();
    assert!(matches!(error, CoulombError::InvalidLexp(13)));
    let ok = CoulombRequest::cubic(common::LATTICE, 12).unwrap();
    assert_eq!(ok.lexp(), 12);
}

#[test]
fn structure_constant_l8_matches_brute_force() {
    let request = CoulombRequest::cubic(common::LATTICE, 4).unwrap();
    let q = TransferQ::from_cartesian([
        InverseBohr(2.0 * PI / common::LATTICE),
        InverseBohr(0.0),
        InverseBohr(0.0),
    ])
    .unwrap();
    let partition = common::partition();
    let spex = structure_constants(request.cell(), request.reciprocal(), &partition, q, 4).unwrap();
    let assembled = spex.get(0, 0, 8, 0).unwrap();
    let independent =
        brute_force_structure_constant(request.cell(), q, [0.0; 3], 8, 0, 12).unwrap();
    assert!(
        independent.norm() > 1.0e-8,
        "L=8 brute-force structure constant vanished"
    );
    assert!(
        (assembled - independent).norm() < 5.0e-2 * independent.norm(),
        "SPEX {assembled} brute {independent}"
    );
}

#[test]
fn structure_constant_l12_matches_brute_force_at_lexp_six() {
    let request = CoulombRequest::cubic(common::LATTICE, 6).unwrap();
    let q = TransferQ::from_cartesian([
        InverseBohr(2.0 * PI / common::LATTICE),
        InverseBohr(0.0),
        InverseBohr(0.0),
    ])
    .unwrap();
    let partition = common::partition();
    let spex = structure_constants(request.cell(), request.reciprocal(), &partition, q, 6).unwrap();
    let assembled = spex.get(0, 0, 12, 0).unwrap();
    let independent =
        brute_force_structure_constant(request.cell(), q, [0.0; 3], 12, 0, 10).unwrap();
    assert!(
        independent.norm() > 1.0e-12,
        "L=12 brute-force structure constant vanished"
    );
    assert!(
        (assembled - independent).norm() < 5.0e-2 * independent.norm(),
        "SPEX L=12 {assembled} brute {independent}"
    );
}

#[test]
fn same_volume_skew_cell_is_rejected() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let skew = Cell::new([
        [Bohr(16.0), Bohr(0.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(4.0), Bohr(0.0)],
        [Bohr(0.0), Bohr(0.0), Bohr(8.0)],
    ])
    .unwrap();
    assert!((skew.volume().get() - 512.0).abs() < 1.0e-12);
    let request = CoulombRequest::new(skew, 2).unwrap();
    let error = assemble_coulomb(&auxiliary, &request).unwrap_err();
    assert!(matches!(
        error,
        CoulombError::WaveLatticeMismatch { .. } | CoulombError::ReciprocalMismatch
    ));
}

#[test]
fn finite_q_pw_diagonal_contains_four_pi_over_q_squared() {
    let q = common::transfer_q([0.5, 0.0, 0.0]);
    let request = CoulombRequest::cubic(common::LATTICE, 2).unwrap();
    let (_source, auxiliary) = common::mixed_product_auxiliary(q);
    let operator = assemble_coulomb(&auxiliary, &request).unwrap();
    let qnorm = q.norm().get();
    let expected = 4.0 * PI / (qnorm * qnorm);
    let mut found = false;
    for (index, region) in operator.regions().iter().enumerate() {
        if matches!(
            region,
            muffintin_prodbasis::AuxiliaryRegion::Interstitial { .. }
        ) {
            let diagonal = operator.element(index, index).unwrap();
            assert!(diagonal.re.is_finite());
            assert!(diagonal.re > 0.0);
            assert!(
                (diagonal.re - expected).abs() / expected < 0.6,
                "PW diagonal {} vs 4π/q² {expected}",
                diagonal.re
            );
            found = true;
            break;
        }
    }
    assert!(
        found,
        "mixed-product auxiliary must retain an interstitial PW"
    );
    let _ = ReciprocalLattice::from_direct(*request.cell().basis());
}
