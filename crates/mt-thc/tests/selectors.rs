//! q0_l2 / allq_l2 / allq_coulomb_pool comparison at identical Nμ.

use muffintin_auxiliary_ir::{InterpolationRegion, PairColumnLayout};
use muffintin_thc::toy::{
    MT_NORB, mt_adaptive_grid, mt_bloch_orbitals, mt_kmesh, mt_orbital_norms, mt_partition,
    mt_reference_grid, mt_uniform_grid,
};
use muffintin_thc::{
    DEFAULT_SELECTOR, DEFAULT_SKETCH_ROWS, GridPath, HEADLINE_SEED, L2Engine, PairBlock,
    RankPolicy, SelectionRequest, SelectorStrategy, ThcError, UniformShift, compare_strategies,
    pivots_from_pair_blocks, run_thc,
};
use num_complex::Complex64;

#[test]
fn production_default_is_allq_l2() {
    assert_eq!(DEFAULT_SELECTOR, SelectorStrategy::AllQL2);
    assert_ne!(DEFAULT_SELECTOR, SelectorStrategy::Q0L2);
}

#[test]
fn q0_l2_hides_a_finite_q_channel_and_cannot_be_the_default() {
    let layout = PairColumnLayout::new(2, 1, None);
    let n_points = 4;
    let n_col = layout.n_columns().unwrap();
    let mut q0 = vec![Complex64::default(); n_points * n_col];
    let mut q1 = vec![Complex64::default(); n_points * n_col];
    for p in 0..n_points {
        for col in 0..n_col {
            let value = Complex64::new(1.0 + 0.1 * col as f64 + 0.02 * p as f64, 0.0);
            if p < 2 {
                q0[p * n_col + col] = value;
            } else {
                q1[p * n_col + col] = value;
            }
        }
    }
    let blocks = [
        PairBlock::new(0, n_points, layout, q0).unwrap(),
        PairBlock::new(1, n_points, layout, q1).unwrap(),
    ];
    let weights = vec![1.0; n_points];
    let q0_pivots = pivots_from_pair_blocks(&blocks[..1], &weights, 2)
        .unwrap()
        .0;
    let allq_pivots = pivots_from_pair_blocks(&blocks, &weights, 2).unwrap().0;
    assert!(
        q0_pivots.iter().all(|&p| p < 2),
        "q0-only QRCP should stay on the q=0 support, got {q0_pivots:?}"
    );
    assert!(
        allq_pivots.iter().any(|&p| p >= 2),
        "all-q QRCP must sample the finite-q support, got {allq_pivots:?}"
    );
    let hidden: f64 = blocks[1]
        .selected_rows(&q0_pivots)
        .unwrap()
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    assert!(
        hidden < 1.0e-20,
        "q0 points carry no finite-q pair density: {hidden}"
    );
    let captured: f64 = blocks[1]
        .selected_rows(&allq_pivots)
        .unwrap()
        .iter()
        .map(|value| value.norm_sqr())
        .sum();
    assert!(
        captured > 0.5,
        "all-q points must capture finite-q pair density: {captured}"
    );
    assert_eq!(DEFAULT_SELECTOR, SelectorStrategy::AllQL2);
}

#[test]
fn selection_is_deterministic_under_fixed_seed_and_column_order() {
    let mesh = mt_kmesh();
    let grid = mt_adaptive_grid(8, 12, 6);
    let reference = mt_reference_grid();
    let norms = mt_orbital_norms(&reference);
    let orbitals = mt_bloch_orbitals(&grid, &norms, &mesh).unwrap();
    let request = SelectionRequest {
        strategy: SelectorStrategy::AllQL2,
        rank: RankPolicy::Exact { n_mu: 12 },
        seed: HEADLINE_SEED,
        pool_factor: 2,
        engine: L2Engine::StructuredSketch {
            rows: DEFAULT_SKETCH_ROWS,
        },
        grid_path: GridPath::Adaptive {
            nrad: 8,
            nang: 12,
            ninter: 6,
        },
    };
    let a = muffintin_thc::select_points(
        &orbitals,
        &grid.points,
        &grid.weights,
        &grid.regions,
        &mesh,
        &request,
        None,
        Some(0),
    )
    .unwrap();
    let b = muffintin_thc::select_points(
        &orbitals,
        &grid.points,
        &grid.weights,
        &grid.regions,
        &mesh,
        &request,
        None,
        Some(0),
    )
    .unwrap();
    assert_eq!(a.pivots, b.pivots);
    assert_eq!(a.points, b.points);
    assert_eq!(a.provenance.seed, HEADLINE_SEED);
    assert_eq!(a.provenance.q_set, "allq");
    assert_eq!(a.provenance.strategy.as_str(), "allq_l2");
}

#[test]
fn full_qrcp_and_pivoted_cholesky_select_the_same_points() {
    let mesh = mt_kmesh();
    let grid = mt_adaptive_grid(8, 12, 6);
    let reference = mt_reference_grid();
    let norms = mt_orbital_norms(&reference);
    let orbitals = mt_bloch_orbitals(&grid, &norms, &mesh).unwrap();
    for strategy in [SelectorStrategy::Q0L2, SelectorStrategy::AllQL2] {
        let mut selections = Vec::new();
        for engine in [L2Engine::FullColumnPivotedQr, L2Engine::FullPivotedCholesky] {
            let request = SelectionRequest {
                strategy,
                rank: RankPolicy::Exact { n_mu: 12 },
                seed: HEADLINE_SEED,
                pool_factor: 2,
                engine,
                grid_path: GridPath::Adaptive {
                    nrad: 8,
                    nang: 12,
                    ninter: 6,
                },
            };
            let selection = muffintin_thc::select_points(
                &orbitals,
                &grid.points,
                &grid.weights,
                &grid.regions,
                &mesh,
                &request,
                None,
                Some(0),
            )
            .unwrap();
            assert_eq!(selection.provenance.n_mu, 12);
            assert_eq!(selection.provenance.engine, engine);
            selections.push(selection);
        }
        assert_eq!(selections[0].pivots, selections[1].pivots);
        assert_eq!(selections[0].points, selections[1].points);
    }
}

#[test]
fn exact_structured_sketch_rejects_more_points_than_ranked_rows() {
    let mesh = mt_kmesh();
    let grid = mt_adaptive_grid(4, 6, 3);
    let reference = mt_reference_grid();
    let norms = mt_orbital_norms(&reference);
    let orbitals = mt_bloch_orbitals(&grid, &norms, &mesh).unwrap();
    let request = SelectionRequest {
        strategy: SelectorStrategy::AllQL2,
        rank: RankPolicy::Exact { n_mu: 3 },
        seed: HEADLINE_SEED,
        pool_factor: 2,
        engine: L2Engine::StructuredSketch { rows: 1 },
        grid_path: GridPath::Adaptive {
            nrad: 4,
            nang: 6,
            ninter: 3,
        },
    };
    assert_eq!(
        muffintin_thc::select_points(
            &orbitals,
            &grid.points,
            &grid.weights,
            &grid.regions,
            &mesh,
            &request,
            None,
            Some(0),
        )
        .unwrap_err(),
        ThcError::SketchRankExceedsRows {
            rows: 1,
            required: 3,
        }
    );
}

#[test]
fn strategies_compare_at_identical_nmu_on_adaptive_and_uniform() {
    let mesh = mt_kmesh();
    let partition = mt_partition();
    let reference = mt_reference_grid();
    let norms = mt_orbital_norms(&reference);
    let n_mu = 12;
    let adaptive = mt_adaptive_grid(8, 12, 6);
    let uniform = mt_uniform_grid(8, UniformShift::Half);
    let adaptive_orbs = mt_bloch_orbitals(&adaptive, &norms, &mesh).unwrap();
    let uniform_orbs = mt_bloch_orbitals(&uniform, &norms, &mesh).unwrap();
    for (grid, orbitals, path) in [
        (
            &adaptive,
            &adaptive_orbs,
            GridPath::Adaptive {
                nrad: 8,
                nang: 12,
                ninter: 6,
            },
        ),
        (
            &uniform,
            &uniform_orbs,
            GridPath::Uniform {
                divisions: 8,
                shift: UniformShift::Half,
            },
        ),
    ] {
        let results = compare_strategies(
            orbitals,
            grid,
            &mesh,
            &partition,
            n_mu,
            HEADLINE_SEED,
            L2Engine::StructuredSketch { rows: 48 },
            path,
            None,
            Some(0),
            None,
        )
        .unwrap();
        assert_eq!(results.len(), 2, "coulomb-pool skipped without grams");
        for result in &results {
            assert_eq!(result.diagnostics.n_mu, n_mu);
            assert!(result.diagnostics.q0_l2.is_some());
            assert!(result.diagnostics.worst_finite_q_l2.is_some());
            assert_ne!(
                result.diagnostics.worst_finite_q_index,
                Some(0),
                "worst finite q must not be reported as Gamma"
            );
            assert!(result.diagnostics.q0_core.is_some());
            assert!(result.diagnostics.q0_valence.is_some());
            assert_eq!(result.vertices[0].len(), mesh.len() * MT_NORB * MT_NORB);
            assert_eq!(
                result.auxiliaries[0].dimension(),
                result.selection.points.len()
            );
            assert!(result.auxiliaries[0].mixed_product().is_none());
        }
    }
}

#[test]
fn interpolation_points_carry_region_tags() {
    let grid = mt_adaptive_grid(8, 12, 6);
    assert!(
        grid.regions
            .iter()
            .any(|region| matches!(region, InterpolationRegion::MuffinTin { site: 0 }))
    );
    assert!(
        grid.regions
            .iter()
            .any(|region| matches!(region, InterpolationRegion::Interstitial))
    );
    let uniform = mt_uniform_grid(4, UniformShift::Origin);
    assert!(
        uniform
            .regions
            .iter()
            .all(|region| matches!(region, InterpolationRegion::Uniform))
    );
}

#[test]
fn q0_and_allq_share_the_same_nmu_and_report_q0_separately() {
    let mesh = mt_kmesh();
    let partition = mt_partition();
    let grid = mt_adaptive_grid(8, 12, 6);
    let reference = mt_reference_grid();
    let norms = mt_orbital_norms(&reference);
    let orbitals = mt_bloch_orbitals(&grid, &norms, &mesh).unwrap();
    let n_mu = 12;
    let mut reports = Vec::new();
    for strategy in [SelectorStrategy::Q0L2, SelectorStrategy::AllQL2] {
        let request = SelectionRequest {
            strategy,
            rank: RankPolicy::Exact { n_mu },
            seed: HEADLINE_SEED,
            pool_factor: 2,
            engine: L2Engine::StructuredSketch { rows: 48 },
            grid_path: GridPath::Adaptive {
                nrad: 8,
                nang: 12,
                ninter: 6,
            },
        };
        reports.push(
            run_thc(
                &orbitals,
                &grid,
                &mesh,
                &partition,
                &request,
                None,
                Some(0),
                None,
            )
            .unwrap(),
        );
    }
    assert_eq!(reports[0].diagnostics.n_mu, reports[1].diagnostics.n_mu);
    for report in &reports {
        let q0 = report.diagnostics.q0_l2.unwrap().frobenius;
        let worst = report.diagnostics.worst_finite_q_l2.unwrap().frobenius;
        assert!(q0.is_finite() && worst.is_finite());
    }
}
