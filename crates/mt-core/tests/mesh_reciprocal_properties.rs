use mt_core::{Bohr, ExponentialMesh, InverseBohr, ReciprocalLattice};
use proptest::prelude::*;
use std::collections::BTreeSet;

proptest! {
    #[test]
    fn exponential_mesh_has_declared_ratio(
        first in 1e-8f64..1.0,
        h in 1e-4f64..0.05,
        extra in 0usize..100,
    ) {
        let mesh = ExponentialMesh::new(Bohr(first), h, 7+extra).unwrap();
        for pair in mesh.radii().windows(2) {
            prop_assert!(((pair[1].0/pair[0].0).ln()-h).abs() < 2e-15);
        }
    }

    #[test]
    fn spex_quadrature_integrates_smooth_powers(
        power in 0u32..6,
        h in 0.003f64..0.012,
    ) {
        // Keep f(r0)f(r1) above SPEX's 1e-28 origin-extrapolation threshold
        // even for the highest generated power.
        let mesh = ExponentialMesh::new(Bohr(1e-2), h, 601).unwrap();
        let values: Vec<_> = mesh.radii().iter().map(|r| r.0.powi(power as i32)).collect();
        let expected = mesh.last().0.powi(power as i32+1)/f64::from(power+1);
        let actual = mesh.integrate(&values).unwrap();
        prop_assert!((actual-expected).abs() <= 3e-12*expected.max(1e-200));
    }
}

#[test]
fn skew_lattice_enumeration_has_inversion_and_cutoff_closure() {
    let lattice = ReciprocalLattice::new([
        [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
        [InverseBohr(0.95), InverseBohr(0.20), InverseBohr(0.0)],
        [InverseBohr(0.1), InverseBohr(-0.1), InverseBohr(0.7)],
    ])
    .unwrap();
    let cutoff = 1.25;
    let vectors = lattice.enumerate(InverseBohr(cutoff)).unwrap();
    let indices: BTreeSet<_> = vectors.iter().map(|g| g.index).collect();
    assert_eq!(vectors.first().unwrap().index, [0, 0, 0]);
    for g in &vectors {
        assert!(g.norm.0 <= cutoff + 2e-14);
        assert!(indices.contains(&g.index.map(|n| -n)));
    }
    // This skew cell contains short vectors with coefficients outside the
    // naive component bound floor(Gmax/|b_i|)=1.
    assert!(vectors.iter().any(|g| g.index.iter().any(|n| n.abs() > 1)));
}

#[test]
fn spex_origin_and_end_rule_branches_are_explicit() {
    let outward = ExponentialMesh::new(Bohr(0.2), 0.1, 13).unwrap();
    let sign_changing = (0..outward.len())
        .map(|index| if index == 0 { -1.0 } else { 1.0 })
        .collect::<Vec<_>>();
    assert_eq!(
        outward.origin_contribution(&sign_changing).unwrap().value(),
        0.0
    );

    let inverse_r = outward
        .radii()
        .iter()
        .map(|radius| 1.0 / radius.0)
        .collect::<Vec<_>>();
    assert_eq!(
        outward.origin_contribution(&inverse_r).unwrap().value(),
        0.5
    );

    let inward = ExponentialMesh::new(Bohr(2.0), -0.1, 13).unwrap();
    let positive = vec![1.0; inward.len()];
    assert_eq!(inward.origin_contribution(&positive).unwrap().value(), 0.0);

    // For f=1/r the transformed log-grid integrand is constant. Every
    // possible leading end-rule length must integrate the sampled interval
    // exactly, not only a complete six-interval block.
    for number in 7..=12 {
        let mesh = ExponentialMesh::new(Bohr(0.2), 0.1, number).unwrap();
        let values = mesh
            .radii()
            .iter()
            .map(|radius| 1.0 / radius.0)
            .collect::<Vec<_>>();
        let actual = mesh.integrate_without_origin(&values).unwrap();
        let expected = 0.1 * (number - 1) as f64;
        assert!((actual - expected).abs() <= 4e-15);
    }
}
