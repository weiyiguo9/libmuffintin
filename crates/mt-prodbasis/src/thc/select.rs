//! Weighted ISDF point selection: `q0_l2`, `allq_l2`, `allq_coulomb_pool`.

use crate::thc::ThcError;
use crate::thc::linalg::{column_pivots, pivoted_cholesky_pivots};
use crate::thc::pair::PairBlock;
use crate::{
    InterpolationAuxiliaryPoint, InterpolationRegion, PairColumnLayout, sort_interpolation_points,
};
use muffintin_core::{Bohr, VolumeBohr3};
use num_complex::Complex64;


/// Named ISDF/THC selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectorStrategy {
    /// Weighted QRCP/randomized pair-density sketch using only $q=0$.
    Q0L2,
    /// The same weighted selection using every canonical $q$ block.
    AllQL2,
    /// Oversampled all-q L2 pool, then Coulomb-metric rerank to exact $N_\mu$.
    AllQCoulombPool,
}

impl SelectorStrategy {
    /// Stable strategy name used in provenance and documentation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Q0L2 => "q0_l2",
            Self::AllQL2 => "allq_l2",
            Self::AllQCoulombPool => "allq_coulomb_pool",
        }
    }
}

/// Finite-q / Umklapp-safe L2 default supported by the scratch headline
/// (`all-q` in `thc_mt_kpoint_test.py`) and the q=0 blind-spot regression.
/// `allq_coulomb_pool` is implemented but was plan-only in the Python evidence.
pub const DEFAULT_SELECTOR: SelectorStrategy = SelectorStrategy::AllQL2;

/// Rank termination.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RankPolicy {
    /// Keep exactly `n_mu` interpolation points.
    Exact { n_mu: usize },
    /// L2-only: stop when the residual amplitude falls below
    /// `thresh` times its leading value, capped at `n_max`.
    ///
    /// The amplitude is $|R_{kk}|$ for QRCP and the square root of the
    /// residual Gram diagonal for pivoted Cholesky.
    Threshold { thresh: f64, n_max: usize },
}

/// Linear-algebra engine for the L2 pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum L2Engine {
    /// Structured random sketch from `thc_mt_kpoint_test.py:164-199`.
    StructuredSketch { rows: usize },
    /// Full weighted QRCP from `thc_lapw_end_to_end_test.py:311-323`.
    FullColumnPivotedQr,
    /// Pivoted Cholesky of the weighted point Gram.
    ///
    /// The dense point Gram is not formed. The stacked weighted pair matrix
    /// is still materialized; this is not a matrix-free orbital evaluator.
    FullPivotedCholesky,
}


/// Parent-grid path recorded on the selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GridPath {
    /// MT-adaptive exponential × angular plus interstitial shell.
    Adaptive {
        nrad: usize,
        nang: usize,
        ninter: usize,
    },
    /// Uniform $N^3$ grid.
    Uniform {
        divisions: usize,
        shift: UniformShift,
    },
    /// Synthetic two-region composite grid.
    Composite { name: String },
    /// Externally supplied parent support (production scalar THC).
    External {
        n_points: usize,
        n_candidates: usize,
    },
}

/// Uniform-grid origin convention from `thc_mt_kpoint_test.py:103-119`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UniformShift {
    Origin,
    Half,
    Random { seed: u64 },
}

impl UniformShift {
    /// Provenance label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Origin => "origin",
            Self::Half => "half",
            Self::Random { .. } => "random",
        }
    }
}



/// Recorded selector provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionProvenance {
    pub strategy: SelectorStrategy,
    pub seed: u64,
    pub shift: Option<UniformShift>,
    pub pool_factor: Option<usize>,
    pub n_mu: usize,
    pub n_pool: Option<usize>,
    pub q_set: &'static str,
    pub grid_path: GridPath,
    pub engine: L2Engine,
    pub weights: &'static str,
    pub pair_column_window: PairColumnLayout,
}

/// Shared interpolation-point set.
#[derive(Clone, Debug, PartialEq)]
pub struct Selection {
    /// Point indices in selector-engine pivot order. This order is diagnostic
    /// and does not define the emitted auxiliary layout.
    pub pivots: Vec<usize>,
    /// Selected points in canonical muffin-tin-then-interstitial layout order.
    /// THC fitting and emitted auxiliary objects use this ordering.
    pub points: Vec<InterpolationAuxiliaryPoint>,
    pub provenance: SelectionProvenance,
}





pub fn truncate_rank(
    mut pivots: (Vec<usize>, Vec<f64>),
    rank: RankPolicy,
) -> Result<Vec<usize>, ThcError> {
    match rank {
        RankPolicy::Exact { n_mu } => {
            pivots.0.truncate(n_mu);
            Ok(pivots.0)
        }
        RankPolicy::Threshold { thresh, n_max } => {
            let r0 = match pivots.1.first() {
                Some(&value) if value.is_finite() && value > 0.0 => value,
                _ => return Err(ThcError::DegenerateRank),
            };
            let mut kept = 0;
            for (index, &diag) in pivots.1.iter().enumerate() {
                if index >= n_max {
                    break;
                }
                if diag < thresh * r0 {
                    break;
                }
                kept = index + 1;
            }
            if kept == 0 {
                return Err(ThcError::DegenerateRank);
            }
            pivots.0.truncate(kept);
            Ok(pivots.0)
        }
    }
}





/// Full weighted QRCP on already-evaluated pair blocks (test and pool helpers).
pub fn pivots_from_pair_blocks(
    blocks: &[PairBlock],
    weights: &[f64],
    n_keep: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    let (stacked, nrows, n_pts) = stacked_weighted_pair_blocks(blocks, weights)?;
    let (mut pivots, diag) = column_pivots(&stacked, nrows, n_pts)?;
    pivots.truncate(n_keep.min(pivots.len()));
    Ok((pivots, diag))
}

pub fn cholesky_pivots_from_pair_blocks(
    blocks: &[PairBlock],
    weights: &[f64],
    n_keep: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    let (stacked, nrows, n_pts) = stacked_weighted_pair_blocks(blocks, weights)?;
    pivoted_cholesky_pivots(&stacked, nrows, n_pts, n_keep)
}

/// Weighted AllQL2 pivots on already-evaluated pair blocks for a full L2 engine.
pub(crate) fn pair_block_l2_pivots(
    engine: L2Engine,
    blocks: &[PairBlock],
    weights: &[f64],
    n_keep: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    match engine {
        L2Engine::FullColumnPivotedQr => pivots_from_pair_blocks(blocks, weights, n_keep),
        L2Engine::FullPivotedCholesky => cholesky_pivots_from_pair_blocks(blocks, weights, n_keep),
        L2Engine::StructuredSketch { .. } => Err(ThcError::PairBlockRequiresFullEngine),
    }
}

fn stacked_weighted_pair_blocks(
    blocks: &[PairBlock],
    weights: &[f64],
) -> Result<(Vec<Complex64>, usize, usize), ThcError> {
    if blocks.is_empty() {
        return Err(ThcError::EmptyRank);
    }
    let n_pts = blocks[0].n_points;
    let layout = blocks[0].layout;
    if weights.len() != n_pts {
        return Err(ThcError::GridWeightCount {
            points: n_pts,
            weights: weights.len(),
        });
    }
    crate::thc::error::validate_quadrature_weights(weights)?;
    for (index, block) in blocks.iter().enumerate().skip(1) {
        if block.n_points != n_pts {
            return Err(ThcError::PairBlockPointCount {
                index,
                expected: n_pts,
                actual: block.n_points,
            });
        }
        if block.layout != layout {
            return Err(ThcError::PairBlockLayout { index });
        }
    }
    let n_col = blocks[0].n_columns();
    let nrows = crate::thc::error::checked_storage_len(&[blocks.len(), n_col])?;
    let mut stacked =
        vec![Complex64::default(); crate::thc::error::checked_storage_len(&[nrows, n_pts])?];
    for (q_pos, block) in blocks.iter().enumerate() {
        for p in 0..n_pts {
            let scale = weights[p].sqrt();
            for col in 0..n_col {
                stacked[(q_pos * n_col + col) * n_pts + p] = block.at(p, col) * scale;
            }
        }
    }
    Ok((stacked, nrows, n_pts))
}


pub fn interpolation_points(
    pivots: &[usize],
    points: &[[f64; 3]],
    weights: &[f64],
    regions: &[InterpolationRegion],
) -> Result<Vec<InterpolationAuxiliaryPoint>, ThcError> {
    let mut selected = Vec::with_capacity(pivots.len());
    for &id in pivots {
        if id >= points.len() {
            return Err(ThcError::PointIndex(id));
        }
        selected.push(InterpolationAuxiliaryPoint {
            id,
            coordinate: [
                Bohr(points[id][0]),
                Bohr(points[id][1]),
                Bohr(points[id][2]),
            ],
            weight: VolumeBohr3(weights[id]),
            region: regions[id],
        });
    }
    sort_interpolation_points(&mut selected);
    Ok(selected)
}

/// Multiply a row-major `rows × k` matrix by a `k × cols` matrix.
pub fn matmul(
    left: &[Complex64],
    rows: usize,
    k: usize,
    right: &[Complex64],
    cols: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::default(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut acc = Complex64::default();
            for t in 0..k {
                acc += left[i * k + t] * right[t * cols + j];
            }
            out[i * cols + j] = acc;
        }
    }
    out
}

/// Weighted L2 residual of a pair-block reconstruction.
pub fn weighted_residual(
    exact: &PairBlock,
    reconstructed: &[Complex64],
    weights: &[f64],
    mask: impl Fn(usize) -> bool,
) -> Result<(f64, f64), ThcError> {
    let n_pts = exact.n_points;
    let n_col = exact.n_columns();
    if reconstructed.len() != exact.values().len() {
        return Err(ThcError::PairBlockLength {
            expected: exact.values().len(),
            actual: reconstructed.len(),
        });
    }
    let mut num = 0.0;
    let mut den = 0.0;
    let mut col_num = vec![0.0; n_col];
    let mut col_den = vec![0.0; n_col];
    for p in 0..n_pts {
        let scale = weights[p];
        for col in 0..n_col {
            if !mask(col) {
                continue;
            }
            let e = exact.at(p, col);
            let d = e - reconstructed[p * n_col + col];
            num += scale * d.norm_sqr();
            den += scale * e.norm_sqr();
            col_num[col] += scale * d.norm_sqr();
            col_den[col] += scale * e.norm_sqr();
        }
    }
    let frobenius = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let mut column_max = 0.0_f64;
    let scale = col_den.iter().copied().fold(0.0_f64, f64::max);
    let floor = f64::EPSILON * scale.max(1.0);
    for col in 0..n_col {
        if !mask(col) || col_den[col] <= floor {
            continue;
        }
        column_max = column_max.max((col_num[col] / col_den[col]).sqrt());
    }
    Ok((frobenius, column_max))
}

/// Reconstruct pair densities on a grid from selected rows and a zeta fit.
pub fn reconstruct_pairs(
    selected_rows: &[Complex64],
    n_mu: usize,
    n_col: usize,
    zeta: &[Complex64],
    n_points: usize,
) -> Vec<Complex64> {
    matmul(zeta, n_points, n_mu, selected_rows, n_col)
}
