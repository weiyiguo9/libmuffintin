//! AllQL2 full L2 engines on already-evaluated pair blocks.

mod toy_kit;
use crate::toy_kit::mt_partition;
use muffintin_core::InverseBohr;
use muffintin_envelope::Provenance;
use muffintin_prodbasis::thc::{
    GridPath, L2Engine, PairBlock, RankPolicy, SelectorStrategy, ThcError, ThcResult,
    fit_allq_l2_pair_blocks,
};
use muffintin_prodbasis::{InterpolationRegion, PairColumnLayout, TransferQ};
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
