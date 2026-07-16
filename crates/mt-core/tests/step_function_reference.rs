use mt_core::{
    Bohr, InterstitialGeometry, InverseBohr, ReciprocalLattice, Sphere, VolumeBohr3,
    sphere_form_factor, step_function_coefficient_cutoff,
};
use proptest::prelude::*;
use std::collections::BTreeSet;
use std::f64::consts::PI;

#[test]
fn analytic_sphere_transform_matches_numerical_radial_integral() {
    let radius = Bohr(1.37);
    for q in [0.0, 0.02, 0.7, 2.1, 7.3] {
        let analytic =
            4.0 * PI * radius.0.powi(3) / 3.0 * sphere_form_factor(InverseBohr(q), radius);
        let numerical = radial_transform_simpson(q, radius.0, 40_000);
        assert!(
            (analytic - numerical).abs() <= 1e-12,
            "q={q}, analytic={analytic:e}, numerical={numerical:e}"
        );
    }
}

#[test]
fn coefficient_matches_spex_sign_normalization_and_phase() {
    let sphere = Sphere {
        center: [Bohr(0.3), Bohr(-0.4), Bohr(1.1)],
        radius: Bohr(0.8),
    };
    let geometry = InterstitialGeometry::new(VolumeBohr3(200.0), vec![sphere]).unwrap();
    let g = [InverseBohr(0.7), InverseBohr(-0.2), InverseBohr(1.3)];
    let q = g.iter().map(|x| x.0 * x.0).sum::<f64>().sqrt();
    let radial = radial_transform_simpson(q, sphere.radius.0, 40_000) / 200.0;
    let phase = -g
        .iter()
        .zip(sphere.center)
        .map(|(x, r)| x.0 * r.0)
        .sum::<f64>();
    let expected = -num_complex::Complex64::from_polar(radial, phase);
    let actual = geometry.coefficient(g).unwrap();
    assert!(
        (actual - expected).norm() <= 1e-12,
        "actual={actual:e}, expected={expected:e}"
    );
}

#[test]
fn double_cutoff_contains_every_basis_difference() {
    let lattice = ReciprocalLattice::new([
        [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
        [InverseBohr(0.3), InverseBohr(1.1), InverseBohr(0.0)],
        [InverseBohr(-0.2), InverseBohr(0.1), InverseBohr(0.9)],
    ])
    .unwrap();
    let cutoff = InverseBohr(2.25);
    let basis = lattice.enumerate(cutoff).unwrap();
    let coefficient_vectors = lattice
        .enumerate(step_function_coefficient_cutoff(cutoff))
        .unwrap();
    let coefficient_indices: BTreeSet<_> = coefficient_vectors.iter().map(|g| g.index).collect();
    for left in &basis {
        for right in &basis {
            let difference = std::array::from_fn(|i| left.index[i] - right.index[i]);
            assert!(
                coefficient_indices.contains(&difference),
                "missing difference {difference:?}"
            );
        }
    }
}

proptest! {
    #[test]
    fn safe_step_cutoff_obeys_triangle_inequality(
        left in prop::array::uniform3(-100.0f64..100.0),
        right in prop::array::uniform3(-100.0f64..100.0),
    ) {
        let left_norm = left.iter().map(|x| x*x).sum::<f64>().sqrt();
        let right_norm = right.iter().map(|x| x*x).sum::<f64>().sqrt();
        let cutoff = left_norm.max(right_norm);
        let difference_norm = left.iter().zip(right).map(|(x,y)| (x-y)*(x-y)).sum::<f64>().sqrt();
        prop_assert!(difference_norm <= 2.0*cutoff*(1.0+4.0*f64::EPSILON));
    }
}

fn radial_transform_simpson(q: f64, radius: f64, intervals: usize) -> f64 {
    assert_eq!(intervals % 2, 0);
    let h = radius / intervals as f64;
    let integrand = |r: f64| {
        let sinc = if q * r == 0.0 {
            1.0
        } else {
            (q * r).sin() / (q * r)
        };
        4.0 * PI * r * r * sinc
    };
    let mut sum = integrand(0.0) + integrand(radius);
    for i in 1..intervals {
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * integrand(i as f64 * h);
    }
    h * sum / 3.0
}
