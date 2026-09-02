//! AllQL2 full L2 engines on already-evaluated pair blocks.

mod toy_kit;
use crate::toy_kit::mt_partition;
use muffintin_core::InverseBohr;
use muffintin_envelope::Provenance;
use muffintin_prodbasis::thc::{
    ExchangePairBlock, ExchangeThcResult, GridPath, L2Engine, PairBlock, RankPolicy,
    SelectorStrategy, ThcError, ThcResult, fit_allq_l2_exchange_pair_blocks,
    fit_allq_l2_pair_blocks,
};
use muffintin_prodbasis::{
    ExchangePairLayout, ExchangeSpace, InterpolationRegion, OrbitalPair, PairColumnLayout,
    TransferQ,
};
use num_complex::Complex64;

#[test]
fn allq_l2_pair_block_fit_uses_true_weights_and_full_engines() {
    let layout = PairColumnLayout::new(1, 2, None);
    let n_points = 5;
    let n_col = layout.n_columns().unwrap();
    let mut q0 = vec![Complex64::default(); n_points * n_col];
    let mut q1 = vec![Complex64::default(); n_points * n_col];
    for p in 0..n_points {
        for col in 0..n_col {
            q0[p * n_col + col] = Complex64::new((p + 1) as f64, 0.1 * col as f64);
            q1[p * n_col + col] = Complex64::new(0.2 * p as f64, (col + 1) as f64);
        }
    }
    q0[3 * n_col] = Complex64::new(40.0, 0.0);
    q1[4 * n_col + 1] = Complex64::new(0.0, 35.0);
    let blocks = [
        PairBlock::new(0, n_points, layout, q0).unwrap(),
        PairBlock::new(1, n_points, layout, q1).unwrap(),
    ];
    let points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0],
    ];
    let weights = [0.4, 0.0, 0.25, 1.1, 0.3];
    let regions = [
        InterpolationRegion::MuffinTin { site: 0 },
        InterpolationRegion::MuffinTin { site: 0 },
        InterpolationRegion::Interstitial,
        InterpolationRegion::Interstitial,
        InterpolationRegion::Interstitial,
    ];
    let transfers = [
        TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap(),
        TransferQ::from_cartesian([InverseBohr(0.2), InverseBohr(0.0), InverseBohr(0.0)]).unwrap(),
    ];
    let fit = |engine: L2Engine, rank: RankPolicy, candidates: Option<&[usize]>| {
        fit_allq_l2_pair_blocks(
            &blocks,
            &points,
            &weights,
            &regions,
            mt_partition(),
            &transfers,
            rank,
            engine,
            candidates,
            default_fit_provenance(),
        )
    };
    let qr = fit(
        L2Engine::FullColumnPivotedQr,
        RankPolicy::Exact { n_mu: 2 },
        None,
    )
    .unwrap();
    let chol = fit(
        L2Engine::FullPivotedCholesky,
        RankPolicy::Exact { n_mu: 2 },
        None,
    )
    .unwrap();
    assert_engine_contract(
        &qr,
        L2Engine::FullColumnPivotedQr,
        n_points,
        n_col,
        &transfers,
    );
    assert_engine_contract(
        &chol,
        L2Engine::FullPivotedCholesky,
        n_points,
        n_col,
        &transfers,
    );
    assert_eq!(qr.selection.pivots, chol.selection.pivots);
    assert_eq!(qr.selection.points, chol.selection.points);

    let zero_weight = fit(
        L2Engine::FullPivotedCholesky,
        RankPolicy::Exact { n_mu: 1 },
        Some(&[0, 1]),
    )
    .unwrap_err();
    assert_eq!(zero_weight, ThcError::ZeroWeightCandidate(1));
    let sketch = fit(
        L2Engine::StructuredSketch { rows: 8 },
        RankPolicy::Exact { n_mu: 2 },
        None,
    )
    .unwrap_err();
    assert_eq!(sketch, ThcError::PairBlockRequiresFullEngine);
}

fn default_fit_provenance() -> Provenance {
    Provenance {
        recipe: Some("pair-block-l2".to_owned()),
        reference: Some("mt-thc-test".to_owned()),
    }
}

#[test]
fn auxiliaries_and_vertices_bind_provenance_at_construction() {
    let layout = PairColumnLayout::new(1, 1, None);
    let n_points = 3;
    let n_col = layout.n_columns().unwrap();
    let mut values = vec![Complex64::default(); n_points * n_col];
    values[0] = Complex64::new(2.0, 0.0);
    values[n_col] = Complex64::new(0.5, 0.1);
    values[2 * n_col] = Complex64::new(1.5, -0.2);
    let blocks = [PairBlock::new(0, n_points, layout, values).unwrap()];
    let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let weights = [0.4, 0.3, 0.3];
    let regions = [
        InterpolationRegion::MuffinTin { site: 0 },
        InterpolationRegion::Interstitial,
        InterpolationRegion::Interstitial,
    ];
    let transfers = [TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap()];
    let provenance = Provenance {
        recipe: Some("intended-before-vertex".to_owned()),
        reference: Some("fails-if-mutated-after-vertices".to_owned()),
    };
    let result = fit_allq_l2_pair_blocks(
        &blocks,
        &points,
        &weights,
        &regions,
        mt_partition(),
        &transfers,
        RankPolicy::Exact { n_mu: 1 },
        L2Engine::FullColumnPivotedQr,
        None,
        provenance.clone(),
    )
    .unwrap();
    assert_eq!(result.auxiliaries[0].provenance, provenance);
    assert_eq!(result.vertices[0][0].provenance(), &provenance);
    assert_eq!(
        result.auxiliaries[0].provenance,
        *result.vertices[0][0].provenance()
    );
    assert_ne!(
        result.vertices[0][0].provenance().recipe.as_deref(),
        Some("thc-isdf")
    );
}

fn assert_engine_contract(
    result: &ThcResult,
    engine: L2Engine,
    n_points: usize,
    n_col: usize,
    transfers: &[TransferQ],
) {
    assert_eq!(
        result.selection.provenance.strategy,
        SelectorStrategy::AllQL2
    );
    assert_eq!(result.selection.provenance.engine, engine);
    assert_eq!(result.selection.provenance.n_mu, 2);
    assert_eq!(result.selection.points.len(), 2);
    assert_eq!(result.fits.len(), 2);
    assert_eq!(result.fits[0].q_index, 0);
    assert_eq!(result.fits[1].q_index, 1);
    assert_eq!(result.fits[0].n_points, n_points);
    assert_eq!(result.fits[0].n_mu, 2);
    assert_eq!(result.fits[0].zeta.len(), n_points * 2);
    assert_eq!(
        result.selection.provenance.grid_path,
        GridPath::External {
            n_points,
            n_candidates: 4,
        }
    );
    assert!(!result.selection.points.iter().any(|point| point.id == 1));
    assert!(!result.selection.pivots.contains(&1));
    assert_eq!(result.auxiliaries[0].q, transfers[0]);
    assert_eq!(result.vertices[0].len(), n_col);
    assert!(result.fits.iter().all(|fit| fit.l2_core.is_none()));
    assert!(
        result
            .fits
            .iter()
            .any(|fit| fit.l2_all.frobenius.is_finite() && fit.l2_all.frobenius >= 0.0)
    );
}

#[test]
fn exchange_rank_two_reconstructs_all_rectangular_sectors() {
    let fixture = ExchangeFixture::new();
    let qr = fixture.fit(
        &fixture.blocks,
        RankPolicy::Exact { n_mu: 2 },
        L2Engine::FullColumnPivotedQr,
    );
    let cholesky = fixture.fit(
        &fixture.blocks,
        RankPolicy::Exact { n_mu: 2 },
        L2Engine::FullPivotedCholesky,
    );
    for result in [&qr, &cholesky] {
        assert_eq!(result.selection.pivots, vec![0, 1]);
        assert_eq!(result.fits.len(), 2);
        for fit in &result.fits {
            for residual in [
                fit.residuals.vv,
                fit.residuals.cv,
                fit.residuals.vc,
                fit.residuals.cc,
            ] {
                assert!(residual.frobenius < 1.0e-12);
                assert!(residual.column_max < 1.0e-12);
            }
        }
        assert_eq!(result.rank_scaling.n_k, 2);
        assert_eq!(result.rank_scaling.n_valence, 2);
        assert_eq!(result.rank_scaling.n_core, 1);
        assert_eq!(result.rank_scaling.n_candidates, 3);
        assert_eq!(result.rank_scaling.effective_rank, 2);
        assert_eq!(result.rank_scaling.vv_columns, 8);
        assert_eq!(result.rank_scaling.cv_columns, 4);
        assert_eq!(result.rank_scaling.vc_columns, 4);
        assert_eq!(result.rank_scaling.cc_columns, 2);
        assert_eq!(result.rank_scaling.pooled_columns_per_q, 18);
        assert_eq!(result.rank_scaling.selector_rows, 36);
        assert_eq!(
            result.selection.provenance.row_order,
            "q-major->VV/CV/VC/CC->column"
        );
        assert_eq!(result.vertices[0].cv.layout, fixture.layouts[1]);
        assert_eq!(result.vertices[0].cv.vertices.len(), 4);
        assert!(matches!(
            result.vertices[0].cv.vertices[0].pair(),
            OrbitalPair::Exchange {
                k_index: 0,
                occupied_space: ExchangeSpace::Core,
                occupied: 0,
                target_space: ExchangeSpace::Valence,
                target: 0,
            }
        ));
    }
}

#[test]
fn exchange_rank_one_exposes_core_sector_residuals() {
    let fixture = ExchangeFixture::new();
    let result = fixture.fit(
        &fixture.blocks,
        RankPolicy::Exact { n_mu: 1 },
        L2Engine::FullColumnPivotedQr,
    );
    assert_eq!(result.selection.pivots, vec![0]);
    assert!(result.fits[0].residuals.vv.frobenius < 0.25);
    assert!(result.fits[0].residuals.cv.frobenius > 0.9);
    assert!(result.fits[0].residuals.vc.column_max > 0.9);
    assert!(result.fits[0].residuals.cc.frobenius > 0.9);
}

#[test]
fn exchange_blocks_reject_missing_duplicate_and_wrong_layouts() {
    let fixture = ExchangeFixture::new();

    let mut missing = fixture.blocks.clone();
    missing.pop();
    assert!(matches!(
        fixture.try_fit(&missing, RankPolicy::Exact { n_mu: 2 }),
        Err(ThcError::ExchangePairBlockCount {
            expected: 8,
            actual: 7
        })
    ));

    let mut duplicate = fixture.blocks.clone();
    duplicate[1] = duplicate[0].clone();
    assert!(matches!(
        fixture.try_fit(&duplicate, RankPolicy::Exact { n_mu: 2 }),
        Err(ThcError::ExchangePairBlockSector { index: 1, .. })
    ));

    let mut wrong = fixture.blocks.clone();
    let wrong_cv = ExchangePairLayout::new(ExchangeSpace::Core, ExchangeSpace::Valence, 2, 1, 3);
    wrong[1] = exchange_block(0, wrong_cv, [0.0, 1.0, 0.2], 1.0);
    assert!(matches!(
        fixture.try_fit(&wrong, RankPolicy::Exact { n_mu: 2 }),
        Err(ThcError::ExchangePairBlockLayout { index: 1, .. })
    ));
}

struct ExchangeFixture {
    blocks: Vec<ExchangePairBlock>,
    layouts: [ExchangePairLayout; 4],
    points: [[f64; 3]; 3],
    weights: [f64; 3],
    regions: [InterpolationRegion; 3],
    transfers: [TransferQ; 2],
}

impl ExchangeFixture {
    fn new() -> Self {
        let layouts = [
            ExchangePairLayout::new(ExchangeSpace::Valence, ExchangeSpace::Valence, 2, 2, 2),
            ExchangePairLayout::new(ExchangeSpace::Core, ExchangeSpace::Valence, 2, 1, 2),
            ExchangePairLayout::new(ExchangeSpace::Valence, ExchangeSpace::Core, 2, 2, 1),
            ExchangePairLayout::new(ExchangeSpace::Core, ExchangeSpace::Core, 2, 1, 1),
        ];
        let mut blocks = Vec::new();
        for q_index in 0..2 {
            let q_scale = if q_index == 0 { 1.0 } else { 1.25 };
            blocks.push(exchange_block(
                q_index,
                layouts[0],
                [1.0, 0.0, 0.2],
                5.0 * q_scale,
            ));
            for layout in &layouts[1..] {
                blocks.push(exchange_block(q_index, *layout, [0.0, 1.0, 0.2], q_scale));
            }
        }
        Self {
            blocks,
            layouts,
            points: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            weights: [1.0, 1.0, 1.0],
            regions: [InterpolationRegion::Interstitial; 3],
            transfers: [
                TransferQ::from_cartesian([InverseBohr(0.0); 3]).unwrap(),
                TransferQ::from_cartesian([InverseBohr(0.2), InverseBohr(0.0), InverseBohr(0.0)])
                    .unwrap(),
            ],
        }
    }

    fn fit(
        &self,
        blocks: &[ExchangePairBlock],
        rank: RankPolicy,
        engine: L2Engine,
    ) -> ExchangeThcResult {
        fit_allq_l2_exchange_pair_blocks(
            blocks,
            &self.points,
            &self.weights,
            &self.regions,
            mt_partition(),
            &self.transfers,
            rank,
            engine,
            None,
            default_fit_provenance(),
        )
        .unwrap()
    }

    fn try_fit(
        &self,
        blocks: &[ExchangePairBlock],
        rank: RankPolicy,
    ) -> Result<ExchangeThcResult, ThcError> {
        fit_allq_l2_exchange_pair_blocks(
            blocks,
            &self.points,
            &self.weights,
            &self.regions,
            mt_partition(),
            &self.transfers,
            rank,
            L2Engine::FullColumnPivotedQr,
            None,
            default_fit_provenance(),
        )
    }
}

fn exchange_block(
    q_index: usize,
    layout: ExchangePairLayout,
    spatial: [f64; 3],
    scale: f64,
) -> ExchangePairBlock {
    let n_columns = layout.n_columns().unwrap();
    let mut values = Vec::with_capacity(3 * n_columns);
    for value in spatial {
        for column in 0..n_columns {
            values.push(Complex64::new(
                value * scale * (1.0 + 0.05 * column as f64),
                0.0,
            ));
        }
    }
    ExchangePairBlock::new(q_index, 3, layout, values).unwrap()
}
