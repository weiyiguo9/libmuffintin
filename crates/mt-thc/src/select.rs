//! Weighted ISDF point selection: `q0_l2`, `allq_l2`, `allq_coulomb_pool`.

use crate::ThcError;
use crate::gram::CoulombGramSet;
use crate::kmesh::KMesh;
use crate::linalg::{column_pivots, hermitian_sqrt, pivoted_cholesky_pivots};
use crate::pair::{BlochOrbitals, PairBlock, UmklappGauge, evaluate_pair_block};
use muffintin_auxiliary_ir::{
    InterpolationAuxiliaryPoint, InterpolationRegion, PairColumnLayout, sort_interpolation_points,
};
use muffintin_core::{Bohr, VolumeBohr3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Structured-sketch row count from `thc_mt_kpoint_test.py:53`.
pub const DEFAULT_SKETCH_ROWS: usize = 160;
/// Initial Coulomb-pool oversampling from the v0.2 plan.
pub const DEFAULT_POOL_FACTOR: usize = 2;
/// Headline sketch seed from `thc_mt_kpoint_test.py:56`.
pub const HEADLINE_SEED: u64 = 7;
/// Strategy-comparison seeds from `thc_mt_kpoint_test.py:57`.
pub const STRATEGY_SEEDS: [u64; 3] = [7, 19, 43];
/// Uniform-grid random-shift seed from `thc_mt_kpoint_test.py:58`.
pub const RANDOM_SHIFT_SEED: u64 = 29;

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

impl Default for L2Engine {
    fn default() -> Self {
        Self::StructuredSketch {
            rows: DEFAULT_SKETCH_ROWS,
        }
    }
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

/// Selection request compared across strategies at identical $N_\mu$.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionRequest {
    pub strategy: SelectorStrategy,
    pub rank: RankPolicy,
    pub seed: u64,
    pub pool_factor: usize,
    pub engine: L2Engine,
    pub grid_path: GridPath,
}

impl SelectionRequest {
    /// Exact-rank L2 request with the scratch headline seed and sketch.
    pub fn l2(strategy: SelectorStrategy, n_mu: usize, grid_path: GridPath) -> Self {
        Self {
            strategy,
            rank: RankPolicy::Exact { n_mu },
            seed: HEADLINE_SEED,
            pool_factor: DEFAULT_POOL_FACTOR,
            engine: L2Engine::default(),
            grid_path,
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

/// SplitMix64 used for the structured sketch. Not bit-identical to NumPy PCG64;
/// seeds `7/19/43/29` are the scratch integers.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1_u64 << 53) as f64)
    }

    fn next_normal_pair(&mut self) -> (f64, f64) {
        let u = self.next_unit().max(f64::MIN_POSITIVE);
        let v = self.next_unit();
        let radius = (-2.0 * u.ln()).sqrt();
        let theta = 2.0 * PI * v;
        (radius * theta.cos(), radius * theta.sin())
    }

    fn next_complex_normal(&mut self) -> Complex64 {
        let (re, im) = self.next_normal_pair();
        Complex64::new(re, im)
    }
}

fn sketch_factors(
    seed: u64,
    sketch: usize,
    n_k: usize,
    n_orb: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut rng = SplitMix64(seed);
    let len = sketch * n_k * n_orb;
    let mut g1 = Vec::with_capacity(len);
    let mut g2 = Vec::with_capacity(len);
    for _ in 0..len {
        g1.push(rng.next_complex_normal());
    }
    for _ in 0..len {
        g2.push(rng.next_complex_normal());
    }
    (g1, g2)
}

fn factor_at(
    table: &[Complex64],
    n_k: usize,
    n_orb: usize,
    s: usize,
    k: usize,
    orb: usize,
) -> Complex64 {
    table[(s * n_k + k) * n_orb + orb]
}

/// Select interpolation points on `grid` from Bloch orbitals.
#[allow(clippy::too_many_arguments)]
pub fn select_points(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    weights: &[f64],
    regions: &[InterpolationRegion],
    mesh: &KMesh,
    request: &SelectionRequest,
    grams: Option<&CoulombGramSet>,
    core_orbital: Option<usize>,
) -> Result<Selection, ThcError> {
    if orbitals.n_points != points.len() {
        return Err(ThcError::OrbitalPointCount {
            orbitals: orbitals.n_points,
            points: points.len(),
        });
    }
    if orbitals.n_k != mesh.len() {
        return Err(ThcError::OrbitalKCount {
            orbitals: orbitals.n_k,
            mesh: mesh.len(),
        });
    }
    if points.len() != weights.len() {
        return Err(ThcError::GridWeightCount {
            points: points.len(),
            weights: weights.len(),
        });
    }
    if points.len() != regions.len() {
        return Err(ThcError::GridRegionCount {
            points: points.len(),
            regions: regions.len(),
        });
    }
    if points.is_empty() {
        return Err(ThcError::EmptyGrid);
    }
    crate::error::validate_quadrature_weights(weights)?;
    let layout = PairColumnLayout::new(orbitals.n_k, orbitals.n_orb, core_orbital);
    layout.require_core_orbital()?;
    let n_mu_cap = match request.rank {
        RankPolicy::Exact { n_mu } => n_mu,
        RankPolicy::Threshold { thresh, n_max } => {
            if request.strategy == SelectorStrategy::AllQCoulombPool {
                return Err(ThcError::CoulombPoolRequiresExactRank);
            }
            if !thresh.is_finite() || thresh <= 0.0 {
                return Err(ThcError::InvalidThreshold(thresh));
            }
            n_max
        }
    };
    if n_mu_cap == 0 {
        return Err(ThcError::EmptyRank);
    }
    if n_mu_cap > points.len() {
        return Err(ThcError::RankExceedsGrid {
            n_mu: n_mu_cap,
            n_points: points.len(),
        });
    }
    if request.pool_factor == 0 {
        return Err(ThcError::InvalidPoolFactor(request.pool_factor));
    }
    if let (
        RankPolicy::Exact { .. },
        L2Engine::StructuredSketch { rows },
        SelectorStrategy::Q0L2 | SelectorStrategy::AllQL2,
    ) = (request.rank, request.engine, request.strategy)
        && rows < n_mu_cap
    {
        return Err(ThcError::SketchRankExceedsRows {
            rows,
            required: n_mu_cap,
        });
    }
    let (pivots, n_pool) = match request.strategy {
        SelectorStrategy::Q0L2 => {
            let pivots = l2_pivots(
                orbitals,
                points,
                weights,
                mesh,
                request,
                true,
                n_mu_cap,
                core_orbital,
            )?;
            (truncate_rank(pivots, request.rank)?, None)
        }
        SelectorStrategy::AllQL2 => {
            let pivots = l2_pivots(
                orbitals,
                points,
                weights,
                mesh,
                request,
                false,
                n_mu_cap,
                core_orbital,
            )?;
            (truncate_rank(pivots, request.rank)?, None)
        }
        SelectorStrategy::AllQCoulombPool => {
            let grams = grams.ok_or(ThcError::MissingCoulombGrams)?;
            let n_pool = (request.pool_factor.saturating_mul(n_mu_cap)).min(points.len());
            if let L2Engine::StructuredSketch { rows } = request.engine
                && rows < n_pool
            {
                return Err(ThcError::SketchRankExceedsRows {
                    rows,
                    required: n_pool,
                });
            }
            let pool = l2_pivots(
                orbitals,
                points,
                weights,
                mesh,
                request,
                false,
                n_pool,
                core_orbital,
            )?;
            let pool = truncate_rank(pool, RankPolicy::Exact { n_mu: n_pool })?;
            let reranked = coulomb_rerank(
                orbitals,
                points,
                weights,
                mesh,
                &pool,
                n_mu_cap,
                grams,
                core_orbital,
            )?;
            (reranked, Some(n_pool))
        }
    };
    let interpolation = interpolation_points(&pivots, points, weights, regions)?;
    let shift = match request.grid_path {
        GridPath::Uniform { shift, .. } => Some(shift),
        _ => None,
    };
    Ok(Selection {
        pivots: pivots.clone(),
        points: interpolation,
        provenance: SelectionProvenance {
            strategy: request.strategy,
            seed: request.seed,
            shift,
            pool_factor: n_pool.map(|_| request.pool_factor),
            n_mu: pivots.len(),
            n_pool,
            q_set: match request.strategy {
                SelectorStrategy::Q0L2 => "q0",
                SelectorStrategy::AllQL2 | SelectorStrategy::AllQCoulombPool => "allq",
            },
            grid_path: request.grid_path.clone(),
            engine: request.engine,
            weights: "sqrt(quadrature)",
            pair_column_window: layout,
        },
    })
}

pub(crate) fn truncate_rank(
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

#[allow(clippy::too_many_arguments)]
fn l2_pivots(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    weights: &[f64],
    mesh: &KMesh,
    request: &SelectionRequest,
    q0_only: bool,
    n_keep: usize,
    core_orbital: Option<usize>,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    match request.engine {
        L2Engine::StructuredSketch { rows } => {
            if rows == 0 {
                return Err(ThcError::EmptySketch);
            }
            structured_sketch_pivots(
                orbitals,
                points,
                weights,
                mesh,
                request.seed,
                rows,
                q0_only,
                n_keep,
            )
        }
        L2Engine::FullColumnPivotedQr => full_qr_pivots(
            orbitals,
            points,
            weights,
            mesh,
            q0_only,
            core_orbital,
            n_keep,
        ),
        L2Engine::FullPivotedCholesky => full_cholesky_pivots(
            orbitals,
            points,
            weights,
            mesh,
            q0_only,
            core_orbital,
            n_keep,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn structured_sketch_pivots(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    weights: &[f64],
    mesh: &KMesh,
    seed: u64,
    sketch: usize,
    q0_only: bool,
    n_keep: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    let n_k = orbitals.n_k;
    let n_orb = orbitals.n_orb;
    let n_pts = points.len();
    let (g1, g2) = sketch_factors(seed, sketch, n_k, n_orb);
    let mut block = vec![Complex64::default(); sketch * n_pts];
    let q_values: Vec<usize> = if q0_only { vec![0] } else { (0..n_k).collect() };
    for iq in q_values {
        for ik in 0..n_k {
            let (left, shift) = if q0_only {
                (ik, [0, 0, 0])
            } else {
                mesh.kminus(ik, iq)?
            };
            for (p, point) in points.iter().enumerate() {
                let phase = if q0_only {
                    Complex64::new(1.0, 0.0)
                } else {
                    crate::kmesh::umklapp_phase(*point, shift, mesh.lattice_constant())
                };
                for s in 0..sketch {
                    let mut left_proj = Complex64::default();
                    let mut right_proj = Complex64::default();
                    for orb in 0..n_orb {
                        left_proj += orbitals.at(p, left, orb).conj()
                            * factor_at(&g1, n_k, n_orb, s, left, orb);
                        right_proj +=
                            orbitals.at(p, ik, orb) * factor_at(&g2, n_k, n_orb, s, ik, orb);
                    }
                    block[s * n_pts + p] += phase * left_proj * right_proj;
                }
            }
        }
    }
    for p in 0..n_pts {
        let scale = Complex64::new(weights[p].sqrt(), 0.0);
        for s in 0..sketch {
            block[s * n_pts + p] *= scale;
        }
    }
    let (mut pivots, diag) = column_pivots(&block, sketch, n_pts)?;
    let keep = n_keep.min(pivots.len());
    pivots.truncate(keep);
    Ok((pivots, diag))
}

#[allow(clippy::too_many_arguments)]
fn full_qr_pivots(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    weights: &[f64],
    mesh: &KMesh,
    q0_only: bool,
    core_orbital: Option<usize>,
    n_keep: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    let q_indices: Vec<usize> = if q0_only {
        vec![0]
    } else {
        (0..mesh.len()).collect()
    };
    let mut blocks = Vec::with_capacity(q_indices.len());
    for &iq in &q_indices {
        blocks.push(evaluate_pair_block(
            orbitals,
            points,
            mesh,
            iq,
            core_orbital,
            UmklappGauge::Canonical,
        )?);
    }
    pivots_from_pair_blocks(&blocks, weights, n_keep)
}

#[allow(clippy::too_many_arguments)]
fn full_cholesky_pivots(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    weights: &[f64],
    mesh: &KMesh,
    q0_only: bool,
    core_orbital: Option<usize>,
    n_keep: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    let q_indices: Vec<usize> = if q0_only {
        vec![0]
    } else {
        (0..mesh.len()).collect()
    };
    let mut blocks = Vec::with_capacity(q_indices.len());
    for &iq in &q_indices {
        blocks.push(evaluate_pair_block(
            orbitals,
            points,
            mesh,
            iq,
            core_orbital,
            UmklappGauge::Canonical,
        )?);
    }
    cholesky_pivots_from_pair_blocks(&blocks, weights, n_keep)
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

fn cholesky_pivots_from_pair_blocks(
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
    crate::error::validate_quadrature_weights(weights)?;
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
    let nrows = crate::error::checked_storage_len(&[blocks.len(), n_col])?;
    let mut stacked =
        vec![Complex64::default(); crate::error::checked_storage_len(&[nrows, n_pts])?];
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

#[allow(clippy::too_many_arguments)]
fn coulomb_rerank(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    weights: &[f64],
    mesh: &KMesh,
    pool: &[usize],
    n_mu: usize,
    grams: &CoulombGramSet,
    core_orbital: Option<usize>,
) -> Result<Vec<usize>, ThcError> {
    let n_pool = pool.len();
    if n_pool == 0 {
        return Err(ThcError::EmptyRank);
    }
    let mut stacked_rows = Vec::new();
    let mut n_row = 0;
    for iq in 0..mesh.len() {
        let block = evaluate_pair_block(
            orbitals,
            points,
            mesh,
            iq,
            core_orbital,
            UmklappGauge::Canonical,
        )?;
        let gram = grams.get(iq)?;
        gram.require_context(iq, mesh.transfer_q(iq)?, block.layout)?;
        let n = block.n_columns();
        let sqrt_g = hermitian_sqrt(gram.data(), n)?;
        // Y = Z_pool @ sqrt(G), then weight by sqrt(w). Stack as (ncols × n_pool).
        let y_len = crate::error::checked_storage_len(&[n, n_pool])?;
        let mut y = vec![Complex64::default(); y_len];
        for (local, &point) in pool.iter().enumerate() {
            let scale = weights[point].sqrt();
            for col in 0..n {
                let mut acc = Complex64::default();
                for j in 0..n {
                    acc += block.at(point, j) * sqrt_g[j * n + col];
                }
                y[col * n_pool + local] = acc * scale;
            }
        }
        n_row += n;
        stacked_rows.push(y);
    }
    let stacked_len = crate::error::checked_storage_len(&[n_row, n_pool])?;
    let mut stacked = vec![Complex64::default(); stacked_len];
    let mut offset = 0;
    let n_col = PairColumnLayout::new(orbitals.n_k, orbitals.n_orb, None).n_columns()?;
    for block in stacked_rows {
        stacked[offset * n_pool..(offset + n_col) * n_pool].copy_from_slice(&block);
        offset += n_col;
    }
    let (local_pivots, _) = column_pivots(&stacked, n_row, n_pool)?;
    Ok(local_pivots
        .into_iter()
        .take(n_mu)
        .map(|local| pool[local])
        .collect())
}

pub(crate) fn interpolation_points(
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
