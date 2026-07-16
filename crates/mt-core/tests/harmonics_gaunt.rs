use mt_core::{
    complex_spherical_harmonic, complex_spherical_harmonics, gaunt, lm_from_index, lm_index,
    real_gaunt, real_spherical_harmonics, wigner_3j,
};
use proptest::prelude::*;
use std::f64::consts::{PI, TAU};

proptest! {
    #[test]
    fn lm_index_round_trips(index in 0usize..20_000) {
        let lm = lm_from_index(index);
        prop_assert_eq!(lm_index(lm.l, lm.m).unwrap(), index);
    }

    #[test]
    fn condon_shortley_conjugation(
        l in 0u32..12,
        raw_m in -20i32..20,
        theta in 0.0f64..PI,
        phi in -4.0f64*PI..4.0*PI,
    ) {
        let m = raw_m.clamp(-(l as i32), l as i32);
        let positive_m = m.unsigned_abs() as i32;
        let positive = complex_spherical_harmonic(l, positive_m, theta, phi).unwrap();
        let negative = complex_spherical_harmonic(l, -positive_m, theta, phi).unwrap();
        let expected = if positive_m % 2 == 0 { positive.conj() } else { -positive.conj() };
        prop_assert!((negative - expected).norm() < 5e-14);
    }

    #[test]
    fn complex_and_real_addition_theorems(
        l in 0u32..12,
        x in -10.0f64..10.0,
        y in -10.0f64..10.0,
        z in -10.0f64..10.0,
    ) {
        prop_assume!(x*x + y*y + z*z > 1e-20);
        let complex = complex_spherical_harmonics(l, [x,y,z]);
        let real = real_spherical_harmonics(l, [x,y,z]);
        let start = (l*l) as usize;
        let end = ((l+1)*(l+1)) as usize;
        let expected = f64::from(2*l+1) / (4.0*PI);
        let complex_sum: f64 = complex[start..end].iter().map(|v| v.norm_sqr()).sum();
        let real_sum: f64 = real[start..end].iter().map(|v| v*v).sum();
        prop_assert!((complex_sum-expected).abs() < 2e-13);
        prop_assert!((real_sum-expected).abs() < 2e-13);
    }

    #[test]
    fn gaunt_selection_rules_are_exact_zero(
        l1 in 0u32..8,
        l2 in 0u32..8,
        l3 in 0u32..8,
        m1 in -12i32..12,
        m2 in -12i32..12,
        m3 in -12i32..12,
    ) {
        let forbidden = m1.unsigned_abs() > l1
            || m2.unsigned_abs() > l2
            || m3.unsigned_abs() > l3
            || m3 != m2-m1
            || (l1+l2+l3)%2 != 0
            || l3 < l1.abs_diff(l2)
            || l3 > l1+l2;
        if forbidden {
            prop_assert_eq!(gaunt(l1,l2,l3,m1,m2,m3), 0.0);
        }
    }

    #[test]
    fn gaunt_is_symmetric_between_conjugated_factors(
        l1 in 0u32..7,
        l2 in 0u32..7,
        l3 in 0u32..7,
        raw_m1 in -10i32..10,
        raw_m2 in -10i32..10,
    ) {
        let m1 = raw_m1.clamp(-(l1 as i32), l1 as i32);
        let m2 = raw_m2.clamp(-(l2 as i32), l2 as i32);
        let m3 = m2-m1;
        prop_assume!(m3.unsigned_abs() <= l3);
        let left = gaunt(l1,l2,l3,m1,m2,m3);
        let right = gaunt(l3,l2,l1,m3,m2,m1);
        prop_assert!((left-right).abs() <= 1e-13);
    }

    #[test]
    fn wigner_three_j_sign_reversal(
        l1 in 0u32..7,
        l2 in 0u32..7,
        l3 in 0u32..7,
        raw_m1 in -10i32..10,
        raw_m2 in -10i32..10,
    ) {
        let m1 = raw_m1.clamp(-(l1 as i32), l1 as i32);
        let m2 = raw_m2.clamp(-(l2 as i32), l2 as i32);
        let m3 = -m1-m2;
        prop_assume!(m3.unsigned_abs() <= l3);
        let left = wigner_3j(l1,l2,l3,m1,m2,m3);
        let right = wigner_3j(l1,l2,l3,-m1,-m2,-m3);
        let phase = if (l1+l2+l3)%2 == 0 { 1.0 } else { -1.0 };
        prop_assert!((left-phase*right).abs() <= 1e-13);
    }
}

#[test]
fn gaunt_matches_direct_sphere_quadrature() {
    // A nontrivial SPEX-convention coefficient with m3=m2-m1.
    let quantum_numbers = (2, 3, 3, 1, 2, 1);
    let reference = gaunt(
        quantum_numbers.0,
        quantum_numbers.1,
        quantum_numbers.2,
        quantum_numbers.3,
        quantum_numbers.4,
        quantum_numbers.5,
    );
    let (nodes, weights) = gauss_legendre(48);
    let n_phi = 32;
    let mut numerical = num_complex::Complex64::new(0.0, 0.0);
    for (&cos_theta, &weight) in nodes.iter().zip(&weights) {
        let theta = cos_theta.acos();
        for p in 0..n_phi {
            let phi = TAU * (p as f64 + 0.25) / n_phi as f64;
            let y1 = complex_spherical_harmonic(2, 1, theta, phi).unwrap();
            let y2 = complex_spherical_harmonic(3, 2, theta, phi).unwrap();
            let y3 = complex_spherical_harmonic(3, 1, theta, phi).unwrap();
            numerical += y1.conj() * y2 * y3.conj() * (weight * TAU / n_phi as f64);
        }
    }
    assert!(
        (numerical.re - reference).abs() <= 1e-13,
        "analytic={reference:e}, numerical={numerical:e}"
    );
    assert!(numerical.im.abs() <= 1e-14);
}

#[test]
fn real_gaunt_matches_symbolic_cartesian_harmonic() {
    // R_11 = sqrt(3/4pi) x/r and R_20 = sqrt(5/16pi)(3z^2/r^2-1).
    let expected = -5.0_f64.sqrt() / (10.0 * PI.sqrt());
    assert!((real_gaunt(1, 1, 2, 1, 1, 0) - expected).abs() <= 2e-15);

    // Multiplication by R_00 must reproduce orthonormality in every real channel.
    let y00 = 1.0 / (4.0 * PI).sqrt();
    for m in -3..=3 {
        assert!((real_gaunt(3, 3, 0, m, m, 0) - y00).abs() <= 2e-14);
    }
}

fn gauss_legendre(number: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; number];
    let mut weights = vec![0.0; number];
    for i in 0..number.div_ceil(2) {
        let mut x = (PI * (i as f64 + 0.75) / (number as f64 + 0.5)).cos();
        let derivative = loop {
            let (pn, pnm1) = legendre_pair(number, x);
            let derivative = number as f64 * (x * pn - pnm1) / (x * x - 1.0);
            let next = x - pn / derivative;
            if (next - x).abs() < 4.0 * f64::EPSILON {
                x = next;
                break derivative;
            }
            x = next;
        };
        let (_, pnm1) = legendre_pair(number, x);
        let pn_derivative = number as f64 * (x * 0.0 - pnm1) / (x * x - 1.0);
        // Re-evaluate derivative at the final root; pn is within roundoff of 0.
        let weight = 2.0 / ((1.0 - x * x) * pn_derivative.powi(2));
        let mirror = number - 1 - i;
        nodes[i] = x;
        nodes[mirror] = -x;
        weights[i] = weight;
        weights[mirror] = weight;
        let _ = derivative;
    }
    (nodes, weights)
}

fn legendre_pair(number: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    if number == 0 {
        return (previous, 0.0);
    }
    let mut current = x;
    for n in 2..=number {
        let next = ((2 * n - 1) as f64 * x * current - (n - 1) as f64 * previous) / n as f64;
        previous = current;
        current = next;
    }
    (current, previous)
}
