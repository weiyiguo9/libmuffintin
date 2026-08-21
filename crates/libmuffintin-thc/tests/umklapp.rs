//! Canonical-q / Umklapp pair-matrix regressions from the scratch scripts.

use libmuffintin_thc::{
    BlochOrbitals, KMesh, PairColumnLayout, UmklappGauge, evaluate_pair_block, pair_density_oracle,
    umklapp_phase,
};
use num_complex::Complex64;

fn constant_orbitals(n_points: usize, n_k: usize, n_orb: usize) -> BlochOrbitals {
    BlochOrbitals::new(
        n_points,
        n_k,
        n_orb,
        vec![Complex64::new(1.0, 0.0); n_points * n_k * n_orb],
    )
    .unwrap()
}

#[test]
fn mt_like_iq2_constant_orbitals_are_minus_i() {
    let mesh = KMesh::gamma_centred([2, 2, 2], 6.0).unwrap();
    let probe = [0.0, 6.0 / 4.0, 0.0];
    let orbitals = constant_orbitals(1, mesh.len(), 6);
    let block = evaluate_pair_block(
        &orbitals,
        &[probe],
        &mesh,
        2,
        Some(0),
        UmklappGauge::Canonical,
    )
    .unwrap();
    let got = block.at(0, 0);
    assert!(
        (got + Complex64::i()).norm() < 2.0e-14,
        "canonical Umklapp: got {got}"
    );
    let omit = evaluate_pair_block(&orbitals, &[probe], &mesh, 2, Some(0), UmklappGauge::Omit)
        .unwrap()
        .at(0, 0);
    let flipped = evaluate_pair_block(
        &orbitals,
        &[probe],
        &mesh,
        2,
        Some(0),
        UmklappGauge::SignFlip,
    )
    .unwrap()
    .at(0, 0);
    let doubled = evaluate_pair_block(
        &orbitals,
        &[probe],
        &mesh,
        2,
        Some(0),
        UmklappGauge::DoubleCount,
    )
    .unwrap()
    .at(0, 0);
    assert!((omit - Complex64::new(1.0, 0.0)).norm() < 2.0e-14);
    assert!((flipped - Complex64::i()).norm() < 2.0e-14);
    assert!((doubled + Complex64::new(1.0, 0.0)).norm() < 2.0e-14);
    assert!((got - omit).norm() > 0.5);
    assert!((got - flipped).norm() > 0.5);
    assert!((got - doubled).norm() > 0.5);
}

#[test]
fn lapw_iq1_constant_orbitals_are_minus_i() {
    let mesh = KMesh::gamma_centred([2, 2, 1], 5.0).unwrap();
    let probe = [0.0, 5.0 / 4.0, 0.0];
    let (_, shift) = mesh.kminus(0, 1).unwrap();
    assert_eq!(shift, [0, -1, 0]);
    let phase = umklapp_phase(probe, shift, mesh.lattice_constant());
    assert!((phase + Complex64::i()).norm() < 2.0e-14);
}

#[test]
fn pair_block_matches_independent_oracle_for_every_column() {
    let mesh = KMesh::gamma_centred([2, 2, 1], 6.0).unwrap();
    let points = [
        [0.1, 0.2, 0.3],
        [1.4, -0.7, 2.1],
        [0.0, 1.5, 0.0],
        [-2.2, 0.4, 1.1],
    ];
    let mut values = Vec::new();
    for (p, point) in points.iter().enumerate() {
        for k in 0..mesh.len() {
            for orb in 0..3 {
                values.push(Complex64::new(
                    (p + 1) as f64 * 0.1 + orb as f64 * 0.01,
                    k as f64 * 0.03 - point[0] * 0.02,
                ));
            }
        }
    }
    let orbitals = BlochOrbitals::new(points.len(), mesh.len(), 3, values).unwrap();
    let layout = PairColumnLayout::new(mesh.len(), 3, None);
    for iq in 0..mesh.len() {
        let block =
            evaluate_pair_block(&orbitals, &points, &mesh, iq, None, UmklappGauge::Canonical)
                .unwrap();
        for (p, point) in points.iter().enumerate() {
            for column in 0..layout.n_columns().unwrap() {
                let (ik, i, j) = layout.decode(column);
                let want = pair_density_oracle(
                    &orbitals,
                    *point,
                    p,
                    &mesh,
                    iq,
                    ik,
                    i,
                    j,
                    UmklappGauge::Canonical,
                )
                .unwrap();
                let got = block.at(p, column);
                assert!(
                    (got - want).norm() < 1.0e-12,
                    "q={iq} p={p} column={column}: {got} vs {want}"
                );
            }
        }
        let omitted =
            evaluate_pair_block(&orbitals, &points, &mesh, iq, None, UmklappGauge::Omit).unwrap();
        if iq != 0 {
            assert!(
                (0..block.n_columns()).any(|column| {
                    (0..points.len())
                        .any(|p| (block.at(p, column) - omitted.at(p, column)).norm() > 1.0e-8)
                }),
                "finite-q block must change if the wrap is omitted"
            );
        }
    }
}
