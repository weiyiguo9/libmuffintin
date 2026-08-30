//! Injected Coulomb Gram contract and allq_coulomb_pool rerank.

mod toy_kit;
use crate::toy_kit::{BlochOrbitals, HEADLINE_SEED, KMesh, SelectionRequest, run_thc};
use muffintin_core::{Bohr, InterstitialGeometry, InverseBohr, Sphere, VolumeBohr3};
use muffintin_prodbasis::thc::{
    CoulombGramSet, GridPath, InjectedCoulombGram, L2Engine, RankPolicy, SelectorStrategy, ThcError,
};
use muffintin_prodbasis::{AuxiliaryPartition, InterpolationRegion, PairColumnLayout, TransferQ};
use num_complex::Complex64;

fn tiny_mesh() -> KMesh {
    KMesh::gamma_centred([2, 1, 1], 6.0).unwrap()
}

fn tiny_partition() -> AuxiliaryPartition {
    AuxiliaryPartition::from_interstitial(
        InterstitialGeometry::new(
            VolumeBohr3(216.0),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: Bohr(1.0),
            }],
        )
        .unwrap(),
    )
}

fn tiny_grid() -> crate::toy_kit::ToyGrid {
    let points = vec![
        [0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.2, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ];
    let n = points.len();
    crate::toy_kit::ToyGrid {
        name: "tiny".to_owned(),
        points,
        weights: vec![1.0; n],
        regions: vec![InterpolationRegion::Uniform; n],
    }
}

fn tiny_orbitals(grid: &crate::toy_kit::ToyGrid, mesh: &KMesh) -> BlochOrbitals {
    let n_k = mesh.len();
    let n_orb = 2;
    let mut values = Vec::with_capacity(grid.len() * n_k * n_orb);
    for (index, _point) in grid.points.iter().enumerate() {
        for _k in 0..n_k {
            let orb0 = if index < 2 { 1.0 } else { 0.0 };
            let orb1 = if index >= 4 { 1.0 } else { 0.0 };
            values.push(Complex64::new(orb0, 0.0));
            values.push(Complex64::new(orb1, 0.0));
        }
    }
    BlochOrbitals::new(grid.len(), n_k, n_orb, values).unwrap()
}

fn stretched_grams(mesh: &KMesh, layout: PairColumnLayout) -> CoulombGramSet {
    let n = layout.n_columns().unwrap();
    let mut grams = Vec::new();
    for iq in 0..mesh.len() {
        let mut data = vec![Complex64::default(); n * n];
        for i in 0..n {
            let (_, left, right) = layout.decode(i);
            let scale = if left == 1 && right == 1 {
                1.0e4
            } else {
                1.0e-8
            };
            data[i * n + i] = Complex64::new(scale, 0.0);
        }
        grams.push(
            InjectedCoulombGram::from_dense(iq, mesh.transfer_q(iq).unwrap(), layout, data)
                .unwrap(),
        );
    }
    CoulombGramSet::new(grams, mesh.len(), layout).unwrap()
}

#[test]
fn injected_gram_rejects_non_hermitian_entries() {
    let layout = PairColumnLayout::new(1, 2, None);
    let q = TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap();
    let mut data = vec![Complex64::default(); 16];
    data[0] = Complex64::new(1.0, 0.0);
    data[5] = Complex64::new(1.0, 0.0);
    data[10] = Complex64::new(1.0, 0.0);
    data[15] = Complex64::new(1.0, 0.0);
    data[1] = Complex64::new(0.0, 1.0);
    data[4] = Complex64::new(0.0, 1.0);
    let error = InjectedCoulombGram::from_dense(0, q, layout, data).unwrap_err();
    assert!(matches!(error, ThcError::GramNotHermitian { index: 0, .. }));
}

#[test]
fn coulomb_pool_changes_selection_and_protects_worst_finite_q() {
    let mesh = tiny_mesh();
    let partition = tiny_partition();
    let grid = tiny_grid();
    let orbitals = tiny_orbitals(&grid, &mesh);
    let layout = PairColumnLayout::new(mesh.len(), 2, None);
    let grams = stretched_grams(&mesh, layout);
    let n_mu = 2;
    let request_l2 = SelectionRequest {
        strategy: SelectorStrategy::AllQL2,
        rank: RankPolicy::Exact { n_mu },
        seed: HEADLINE_SEED,
        pool_factor: 3,
        engine: L2Engine::FullColumnPivotedQr,
        grid_path: GridPath::Uniform {
            divisions: 6,
            shift: muffintin_prodbasis::thc::UniformShift::Origin,
        },
    };
    let mut request_pool = request_l2.clone();
    request_pool.strategy = SelectorStrategy::AllQCoulombPool;
    let l2 = run_thc(
        &orbitals,
        &grid,
        &mesh,
        &partition,
        &request_l2,
        Some(&grams),
        None,
        None,
    )
    .unwrap();
    let pool = run_thc(
        &orbitals,
        &grid,
        &mesh,
        &partition,
        &request_pool,
        Some(&grams),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        pool.selection.provenance.strategy.as_str(),
        "allq_coulomb_pool"
    );
    assert_eq!(pool.selection.provenance.pool_factor, Some(3));
    assert_eq!(pool.selection.provenance.n_pool, Some(6));
    assert_eq!(pool.diagnostics.n_mu, n_mu);
    assert_ne!(
        l2.selection.pivots, pool.selection.pivots,
        "Coulomb-pool rerank must change the selected points"
    );
    let l2_coulomb = l2
        .diagnostics
        .worst_finite_q_coulomb
        .expect("injected grams yield Coulomb residuals")
        .frobenius;
    let pool_coulomb = pool
        .diagnostics
        .worst_finite_q_coulomb
        .expect("injected grams yield Coulomb residuals")
        .frobenius;
    assert!(
        pool_coulomb <= l2_coulomb * 1.05 + 1.0e-12,
        "pool rerank must improve or protect worst finite-q Coulomb residual: {pool_coulomb} vs {l2_coulomb}"
    );
}

#[test]
fn coulomb_pool_without_grams_is_missing_injection() {
    let mesh = tiny_mesh();
    let error = run_thc(
        &tiny_orbitals(&tiny_grid(), &mesh),
        &tiny_grid(),
        &mesh,
        &tiny_partition(),
        &SelectionRequest {
            strategy: SelectorStrategy::AllQCoulombPool,
            rank: RankPolicy::Exact { n_mu: 2 },
            seed: HEADLINE_SEED,
            pool_factor: 3,
            engine: L2Engine::FullColumnPivotedQr,
            grid_path: GridPath::Uniform {
                divisions: 6,
                shift: muffintin_prodbasis::thc::UniformShift::Origin,
            },
        },
        None,
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, ThcError::MissingCoulombGrams));
}

#[test]
fn l2_paths_reject_grams_at_the_wrong_q_or_layout() {
    let mesh = tiny_mesh();
    let grid = tiny_grid();
    let orbitals = tiny_orbitals(&grid, &mesh);
    let layout = PairColumnLayout::new(mesh.len(), 2, None);
    let n = layout.n_columns().unwrap();
    let wrong_q =
        TransferQ::from_cartesian([InverseBohr(0.4), InverseBohr(0.0), InverseBohr(0.0)]).unwrap();
    let mut wrong_q_grams = Vec::new();
    for iq in 0..mesh.len() {
        let mut data = vec![Complex64::default(); n * n];
        for i in 0..n {
            data[i * n + i] = Complex64::new(1.0, 0.0);
        }
        wrong_q_grams.push(InjectedCoulombGram::from_dense(iq, wrong_q, layout, data).unwrap());
    }
    let wrong_q_set = CoulombGramSet::new(wrong_q_grams, mesh.len(), layout).unwrap();
    let request = SelectionRequest {
        strategy: SelectorStrategy::AllQL2,
        rank: RankPolicy::Exact { n_mu: 2 },
        seed: HEADLINE_SEED,
        pool_factor: 2,
        engine: L2Engine::FullColumnPivotedQr,
        grid_path: GridPath::Uniform {
            divisions: 6,
            shift: muffintin_prodbasis::thc::UniformShift::Origin,
        },
    };
    let error = run_thc(
        &orbitals,
        &grid,
        &mesh,
        &tiny_partition(),
        &request,
        Some(&wrong_q_set),
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, ThcError::GramTransferQ(0)));

    let other_layout = PairColumnLayout::new(mesh.len(), 1, None);
    let n_other = other_layout.n_columns().unwrap();
    let mut wrong_layout_grams = Vec::new();
    for iq in 0..mesh.len() {
        let mut data = vec![Complex64::default(); n_other * n_other];
        for i in 0..n_other {
            data[i * n_other + i] = Complex64::new(1.0, 0.0);
        }
        wrong_layout_grams.push(
            InjectedCoulombGram::from_dense(iq, mesh.transfer_q(iq).unwrap(), other_layout, data)
                .unwrap(),
        );
    }
    let wrong_layout_set =
        CoulombGramSet::new(wrong_layout_grams, mesh.len(), other_layout).unwrap();
    let mut q0 = request.clone();
    q0.strategy = SelectorStrategy::Q0L2;
    let error = run_thc(
        &orbitals,
        &grid,
        &mesh,
        &tiny_partition(),
        &q0,
        Some(&wrong_layout_set),
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, ThcError::GramColumnOrder(0)));
}

#[test]
fn coulomb_pool_rejects_grams_at_the_wrong_transfer_q() {
    let mesh = tiny_mesh();
    let grid = tiny_grid();
    let orbitals = tiny_orbitals(&grid, &mesh);
    let layout = PairColumnLayout::new(mesh.len(), 2, None);
    let n = layout.n_columns().unwrap();
    let wrong =
        TransferQ::from_cartesian([InverseBohr(0.4), InverseBohr(0.0), InverseBohr(0.0)]).unwrap();
    let mut grams = Vec::new();
    for iq in 0..mesh.len() {
        let mut data = vec![Complex64::default(); n * n];
        for i in 0..n {
            data[i * n + i] = Complex64::new(1.0, 0.0);
        }
        grams.push(InjectedCoulombGram::from_dense(iq, wrong, layout, data).unwrap());
    }
    let set = CoulombGramSet::new(grams, mesh.len(), layout).unwrap();
    let error = run_thc(
        &orbitals,
        &grid,
        &mesh,
        &tiny_partition(),
        &SelectionRequest {
            strategy: SelectorStrategy::AllQCoulombPool,
            rank: RankPolicy::Exact { n_mu: 2 },
            seed: HEADLINE_SEED,
            pool_factor: 3,
            engine: L2Engine::FullColumnPivotedQr,
            grid_path: GridPath::Uniform {
                divisions: 6,
                shift: muffintin_prodbasis::thc::UniformShift::Origin,
            },
        },
        Some(&set),
        None,
        None,
    )
    .unwrap_err();
    assert!(matches!(error, ThcError::GramTransferQ(0)));
}
