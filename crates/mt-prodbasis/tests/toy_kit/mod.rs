//! Shared toy fixture: finite deterministic toy bases, the toy k-mesh,
//! toy Bloch-orbital pair evaluation, selector-strategy sweeps, and the
//! end-to-end toy THC harness. Production code does not link this module;
//! prodbasis and coulomb tests include it by path.
#![allow(dead_code)]

mod kmesh;
mod toy;

pub use kmesh::{KMesh, umklapp_phase};
pub use toy::*;

use muffintin_envelope::Provenance;
use muffintin_prodbasis::{
    AuxiliaryPartition, InterpolationRegion, PairColumnLayout,
};
use muffintin_prodbasis::thc::linalg::{column_pivots, hermitian_sqrt};
use muffintin_prodbasis::thc::{
    CoulombGramSet, GridPath, L2Engine, PairBlock, PerQFit, RankPolicy, Selection,
    SelectionProvenance, SelectorStrategy, StrategyDiagnostics, ThcError, ThcResult,
    bloch_pair_vertices, checked_storage_len,
    cholesky_pivots_from_pair_blocks, fit_per_q, gamma_report, interpolation_auxiliary,
    interpolation_points, pivots_from_pair_blocks, truncate_rank,
    worst_finite_q, worst_finite_q_coulomb,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// How the Umklapp phase is applied to a pair column.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UmklappGauge {
    /// $\exp(+i G_{\mathrm{wrap}}\cdot r)$, the scratch convention.
    Canonical,
    /// Drop the phase (regression: omitted wrap).
    Omit,
    /// $\exp(-i G_{\mathrm{wrap}}\cdot r)$ (regression: sign flip).
    SignFlip,
    /// $\exp(+2i G_{\mathrm{wrap}}\cdot r)$ (regression: double count).
    DoubleCount,
}

/// Cell-periodic Bloch orbitals $u_{ik}(r)$ on a grid: `(point, k, orb)`.
#[derive(Clone, Debug, PartialEq)]
pub struct BlochOrbitals {
    pub n_points: usize,
    pub n_k: usize,
    pub n_orb: usize,
    values: Vec<Complex64>,
}

impl BlochOrbitals {
    /// Construct after checking `values` length.
    pub fn new(
        n_points: usize,
        n_k: usize,
        n_orb: usize,
        values: Vec<Complex64>,
    ) -> Result<Self, ThcError> {
        let expected = checked_storage_len(&[n_points, n_k, n_orb])?;
        if values.len() != expected {
            return Err(ThcError::OrbitalCount {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            n_points,
            n_k,
            n_orb,
            values,
        })
    }

    /// Value $u_{ik}(r_p)$.
    pub fn at(&self, point: usize, k: usize, orb: usize) -> Complex64 {
        self.values[(point * self.n_k + k) * self.n_orb + orb]
    }

    /// Layout implied by these orbitals.
    pub fn layout(&self, core_orbital: Option<usize>) -> PairColumnLayout {
        PairColumnLayout::new(self.n_k, self.n_orb, core_orbital)
    }
}

/// Evaluate $\rho^q_{k,ij}(r)=\mathrm{e}^{+i G_{\mathrm{wrap}}\cdot r}
/// u_{i,k-q}^*(r)\,u_{j,k}(r)$ for every grid point and pair column.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_pair_block(
    orbitals: &BlochOrbitals,
    points: &[[f64; 3]],
    mesh: &KMesh,
    iq: usize,
    core_orbital: Option<usize>,
    gauge: UmklappGauge,
) -> Result<PairBlock, ThcError> {
    if points.len() != orbitals.n_points {
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
    let layout = PairColumnLayout::new(orbitals.n_k, orbitals.n_orb, core_orbital);
    layout.require_core_orbital()?;
    let n_col = layout.n_columns()?;
    let expected = checked_storage_len(&[orbitals.n_points, n_col])?;
    let mut values = vec![Complex64::default(); expected];
    for ik in 0..orbitals.n_k {
        let (left, shift) = mesh.kminus(ik, iq)?;
        for (point_index, point) in points.iter().enumerate() {
            let phase = match gauge {
                UmklappGauge::Canonical => umklapp_phase(*point, shift, mesh.lattice_constant()),
                UmklappGauge::Omit => Complex64::new(1.0, 0.0),
                UmklappGauge::SignFlip => {
                    umklapp_phase(*point, shift, mesh.lattice_constant()).conj()
                }
                UmklappGauge::DoubleCount => {
                    let once = umklapp_phase(*point, shift, mesh.lattice_constant());
                    once * once
                }
            };
            for i in 0..orbitals.n_orb {
                let left_value = orbitals.at(point_index, left, i).conj();
                for j in 0..orbitals.n_orb {
                    let column = layout.encode(ik, i, j);
                    values[point_index * n_col + column] =
                        phase * left_value * orbitals.at(point_index, ik, j);
                }
            }
        }
    }
    PairBlock::new(iq, orbitals.n_points, layout, values)
}

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
    muffintin_prodbasis::thc::validate_quadrature_weights(weights)?;
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
                    kmesh::umklapp_phase(*point, shift, mesh.lattice_constant())
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
        let y_len = muffintin_prodbasis::thc::checked_storage_len(&[n, n_pool])?;
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
    let stacked_len = muffintin_prodbasis::thc::checked_storage_len(&[n_row, n_pool])?;
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
            engine: L2Engine::StructuredSketch { rows: DEFAULT_SKETCH_ROWS },
            grid_path,
        }
    }
}


/// Select, fit, and emit interpolation-point auxiliaries plus Bloch pair vertices.
#[allow(clippy::too_many_arguments)]
pub fn run_thc(
    orbitals: &BlochOrbitals,
    grid: &ToyGrid,
    mesh: &KMesh,
    partition: &AuxiliaryPartition,
    request: &SelectionRequest,
    grams: Option<&CoulombGramSet>,
    core_orbital: Option<usize>,
    reference: Option<(&ToyGrid, &BlochOrbitals)>,
) -> Result<ThcResult, ThcError> {
    let selection = select_points(
        orbitals,
        &grid.points,
        &grid.weights,
        &grid.regions,
        mesh,
        request,
        grams,
        core_orbital,
    )?;
    let ids: Vec<usize> = selection.points.iter().map(|point| point.id).collect();
    let n_mu = ids.len();
    let mut fits = Vec::with_capacity(mesh.len());
    let mut auxiliaries = Vec::with_capacity(mesh.len());
    let mut vertices = Vec::with_capacity(mesh.len());
    for iq in 0..mesh.len() {
        let q = mesh.transfer_q(iq)?;
        let candidate_block = evaluate_pair_block(
            orbitals,
            &grid.points,
            mesh,
            iq,
            core_orbital,
            UmklappGauge::Canonical,
        )?;
        let selected_rows = candidate_block.selected_rows(&ids)?;
        let layout = candidate_block.layout;
        let (target, weights, weight_target) = if let Some((ref_grid, ref_orbitals)) = reference {
            let block = evaluate_pair_block(
                ref_orbitals,
                &ref_grid.points,
                mesh,
                iq,
                core_orbital,
                UmklappGauge::Canonical,
            )?;
            (block, ref_grid.weights.clone(), true)
        } else {
            (candidate_block, grid.weights.clone(), false)
        };
        let gram = grams.map(|set| set.get(iq)).transpose()?;
        let fit = fit_per_q(
            &selected_rows,
            n_mu,
            &target,
            &weights,
            q,
            gram,
            weight_target,
        )?;
        let auxiliary = interpolation_auxiliary(
            partition.clone(),
            q,
            selection.points.clone(),
            Provenance {
                recipe: Some("thc-isdf".to_owned()),
                reference: Some("scratch/thc_mt_kpoint_test.py".to_owned()),
            },
        )?;
        let q_vertices = bloch_pair_vertices(q, &selected_rows, n_mu, layout, &auxiliary)?;
        fits.push(fit);
        auxiliaries.push(auxiliary);
        vertices.push(q_vertices);
    }
    let diagnostics = strategy_diagnostics(request.strategy, n_mu, &fits, mesh);
    Ok(ThcResult {
        selection,
        fits,
        auxiliaries,
        vertices,
        diagnostics,
    })
}

/// Compare `q0_l2`, `allq_l2`, and `allq_coulomb_pool` at identical $N_\mu$.
#[allow(clippy::too_many_arguments)]
pub fn compare_strategies(
    orbitals: &BlochOrbitals,
    grid: &ToyGrid,
    mesh: &KMesh,
    partition: &AuxiliaryPartition,
    n_mu: usize,
    seed: u64,
    engine: muffintin_prodbasis::thc::L2Engine,
    grid_path: GridPath,
    grams: Option<&CoulombGramSet>,
    core_orbital: Option<usize>,
    reference: Option<(&ToyGrid, &BlochOrbitals)>,
) -> Result<Vec<ThcResult>, ThcError> {
    let strategies = [
        SelectorStrategy::Q0L2,
        SelectorStrategy::AllQL2,
        SelectorStrategy::AllQCoulombPool,
    ];
    let mut results = Vec::with_capacity(strategies.len());
    for strategy in strategies {
        if strategy == SelectorStrategy::AllQCoulombPool && grams.is_none() {
            continue;
        }
        let request = SelectionRequest {
            strategy,
            rank: muffintin_prodbasis::thc::RankPolicy::Exact { n_mu },
            seed,
            pool_factor: DEFAULT_POOL_FACTOR,
            engine,
            grid_path: grid_path.clone(),
        };
        results.push(run_thc(
            orbitals,
            grid,
            mesh,
            partition,
            &request,
            grams,
            core_orbital,
            reference,
        )?);
    }
    Ok(results)
}

fn strategy_diagnostics(
    strategy: SelectorStrategy,
    n_mu: usize,
    fits: &[PerQFit],
    mesh: &KMesh,
) -> StrategyDiagnostics {
    let gamma = gamma_report(fits, |index| mesh.is_gamma(index));
    let worst_l2 = worst_finite_q(fits, |index| mesh.is_gamma(index));
    let worst_coulomb = worst_finite_q_coulomb(fits, |index| mesh.is_gamma(index));
    StrategyDiagnostics {
        strategy,
        n_mu,
        q0_l2: gamma.map(|fit| fit.l2_all),
        worst_finite_q_l2: worst_l2.map(|fit| fit.l2_all),
        worst_finite_q_index: worst_l2.map(|fit| fit.q_index),
        q0_coulomb: gamma.and_then(|fit| fit.coulomb),
        worst_finite_q_coulomb: worst_coulomb.and_then(|fit| fit.coulomb),
        worst_finite_q_coulomb_index: worst_coulomb.map(|fit| fit.q_index),
        q0_core: gamma.and_then(|fit| fit.l2_core),
        q0_valence: gamma.and_then(|fit| fit.l2_valence),
        finite_q_core: worst_l2.and_then(|fit| fit.l2_core),
        finite_q_valence: worst_l2.and_then(|fit| fit.l2_valence),
    }
}
