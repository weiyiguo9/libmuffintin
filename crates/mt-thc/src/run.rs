//! End-to-end toy ISDF/THC run over the product-space IR.

use crate::ThcError;
use crate::fit::{
    PerQFit, WeightedResidual, fit_per_q, gamma_report, worst_finite_q, worst_finite_q_coulomb,
};
use crate::gram::CoulombGramSet;
use crate::kmesh::KMesh;
use crate::pair::{BlochOrbitals, PairBlock, UmklappGauge, evaluate_pair_block};
use crate::select::{
    GridPath, L2Engine, RankPolicy, Selection, SelectionProvenance, SelectionRequest,
    SelectorStrategy, interpolation_points, pair_block_l2_pivots, select_points, truncate_rank,
};
use crate::toy::ToyGrid;
use muffintin_auxiliary_ir::{
    AuxiliaryPartition, AuxiliaryRepresentation, CompiledAuxiliaryBasis,
    InterpolationPointAuxiliary, OrbitalPair, PairColumnLayout, PairVertex, TransferQ,
};
use muffintin_basis::Provenance;

/// q=0 versus worst finite-q diagnostics for one strategy.
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyDiagnostics {
    pub strategy: SelectorStrategy,
    pub n_mu: usize,
    pub q0_l2: Option<WeightedResidual>,
    pub worst_finite_q_l2: Option<WeightedResidual>,
    pub worst_finite_q_index: Option<usize>,
    pub q0_coulomb: Option<WeightedResidual>,
    pub worst_finite_q_coulomb: Option<WeightedResidual>,
    /// Mesh index of the worst finite-$q$ Coulomb residual, independent of L2.
    pub worst_finite_q_coulomb_index: Option<usize>,
    pub q0_core: Option<WeightedResidual>,
    pub q0_valence: Option<WeightedResidual>,
    pub finite_q_core: Option<WeightedResidual>,
    pub finite_q_valence: Option<WeightedResidual>,
}

/// One completed THC evaluation at a single strategy.
#[derive(Clone, Debug, PartialEq)]
pub struct ThcResult {
    pub selection: Selection,
    pub fits: Vec<PerQFit>,
    pub auxiliaries: Vec<CompiledAuxiliaryBasis>,
    pub vertices: Vec<Vec<PairVertex>>,
    pub diagnostics: StrategyDiagnostics,
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
    engine: crate::select::L2Engine,
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
            rank: crate::select::RankPolicy::Exact { n_mu },
            seed,
            pool_factor: crate::select::DEFAULT_POOL_FACTOR,
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

/// AllQL2 selection and per-$q$ $\zeta$ fit on evaluated pair blocks.
///
/// Pair blocks are already collocated on the parent grid. Quadrature weights are
/// the true supplied values; zeros are allowed if at least one weight is
/// positive. Zero-weight parent rows may remain on the fit grid and in $\zeta$.
/// `candidates` is `None` for every strictly positive-weight parent point in
/// parent order, or explicit parent indices. Explicit zero-weight indices are
/// rejected rather than dropped. `engine` must be
/// [`L2Engine::FullColumnPivotedQr`] or [`L2Engine::FullPivotedCholesky`]; a
/// structured sketch is rejected. Both full engines consume the same ordered
/// pair blocks, positive-weight candidates, true weights, and [`RankPolicy`].
/// Pivoted Cholesky does not form the dense point Gram; it still materializes
/// the stacked weighted pair matrix. Selected interpolation points are
/// returned in canonical muffin-tin-then-interstitial layout order. This is
/// not Coulomb ranking and does not use [`ToyGrid`].
///
/// `provenance` is stored on each interpolation-point auxiliary **before**
/// [`bloch_pair_vertices`] copies it onto the generated [`PairVertex`] records.
/// Callers must not mutate auxiliary provenance after vertices are built.
#[allow(clippy::too_many_arguments)]
pub fn fit_allq_l2_pair_blocks(
    blocks: &[PairBlock],
    points: &[[f64; 3]],
    weights: &[f64],
    regions: &[muffintin_auxiliary_ir::InterpolationRegion],
    partition: AuxiliaryPartition,
    transfers: &[TransferQ],
    rank: RankPolicy,
    engine: L2Engine,
    candidates: Option<&[usize]>,
    provenance: Provenance,
) -> Result<ThcResult, ThcError> {
    if blocks.is_empty() {
        return Err(ThcError::EmptyRank);
    }
    if points.is_empty() {
        return Err(ThcError::EmptyGrid);
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
    crate::error::validate_quadrature_weights(weights)?;
    if transfers.len() != blocks.len() {
        return Err(ThcError::TransferQCount {
            expected: blocks.len(),
            actual: transfers.len(),
        });
    }
    for (index, block) in blocks.iter().enumerate() {
        if block.q_index != index {
            return Err(ThcError::PairBlockQIndex {
                index,
                expected: index,
                actual: block.q_index,
            });
        }
        if block.n_points != points.len() {
            return Err(ThcError::PairBlockPointCount {
                index,
                expected: points.len(),
                actual: block.n_points,
            });
        }
        if block.layout != blocks[0].layout {
            return Err(ThcError::PairBlockLayout { index });
        }
    }
    let candidate_ids = match candidates {
        None => (0..points.len())
            .filter(|&index| weights[index] > 0.0)
            .collect::<Vec<_>>(),
        Some(indices) => {
            let mut seen = vec![false; points.len()];
            let mut ids = Vec::with_capacity(indices.len());
            for &index in indices {
                if index >= points.len() {
                    return Err(ThcError::PointIndex(index));
                }
                if seen[index] {
                    return Err(ThcError::DuplicateCandidate(index));
                }
                if weights[index] <= 0.0 {
                    return Err(ThcError::ZeroWeightCandidate(index));
                }
                seen[index] = true;
                ids.push(index);
            }
            ids
        }
    };
    if candidate_ids.is_empty() {
        return Err(ThcError::EmptyGrid);
    }
    let n_mu_cap = match rank {
        RankPolicy::Exact { n_mu } => n_mu,
        RankPolicy::Threshold { thresh, n_max } => {
            if !thresh.is_finite() || thresh <= 0.0 {
                return Err(ThcError::InvalidThreshold(thresh));
            }
            n_max
        }
    };
    if n_mu_cap == 0 {
        return Err(ThcError::EmptyRank);
    }
    if n_mu_cap > candidate_ids.len() {
        return Err(ThcError::RankExceedsGrid {
            n_mu: n_mu_cap,
            n_points: candidate_ids.len(),
        });
    }
    let layout = blocks[0].layout;
    layout.require_core_orbital()?;
    let restricted = restrict_blocks(blocks, &candidate_ids)?;
    let local = pair_block_l2_pivots(
        engine,
        &restricted,
        &candidate_weights(weights, &candidate_ids),
        n_mu_cap,
    )?;
    let local_pivots = truncate_rank(local, rank)?;
    let pivots = local_pivots
        .into_iter()
        .map(|local| candidate_ids[local])
        .collect::<Vec<_>>();
    let interpolation = interpolation_points(&pivots, points, weights, regions)?;
    let selection = Selection {
        pivots: pivots.clone(),
        points: interpolation,
        provenance: SelectionProvenance {
            strategy: SelectorStrategy::AllQL2,
            seed: 0,
            shift: None,
            pool_factor: None,
            n_mu: pivots.len(),
            n_pool: None,
            q_set: "allq",
            grid_path: GridPath::External {
                n_points: points.len(),
                n_candidates: candidate_ids.len(),
            },
            engine,
            weights: "sqrt(quadrature)",
            pair_column_window: layout,
        },
    };
    let ids = selection
        .points
        .iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();
    let n_mu = ids.len();
    let mut fits = Vec::with_capacity(blocks.len());
    let mut auxiliaries = Vec::with_capacity(blocks.len());
    let mut vertices = Vec::with_capacity(blocks.len());
    for (block, &q) in blocks.iter().zip(transfers) {
        let selected_rows = block.selected_rows(&ids)?;
        let fit = fit_per_q(&selected_rows, n_mu, block, weights, q, None, false)?;
        let auxiliary = interpolation_auxiliary(
            partition.clone(),
            q,
            selection.points.clone(),
            provenance.clone(),
        )?;
        let q_vertices = bloch_pair_vertices(q, &selected_rows, n_mu, layout, &auxiliary)?;
        fits.push(fit);
        auxiliaries.push(auxiliary);
        vertices.push(q_vertices);
    }
    let diagnostics = StrategyDiagnostics {
        strategy: SelectorStrategy::AllQL2,
        n_mu,
        q0_l2: None,
        worst_finite_q_l2: None,
        worst_finite_q_index: None,
        q0_coulomb: None,
        worst_finite_q_coulomb: None,
        worst_finite_q_coulomb_index: None,
        q0_core: None,
        q0_valence: None,
        finite_q_core: None,
        finite_q_valence: None,
    };
    Ok(ThcResult {
        selection,
        fits,
        auxiliaries,
        vertices,
        diagnostics,
    })
}

fn restrict_blocks(blocks: &[PairBlock], candidates: &[usize]) -> Result<Vec<PairBlock>, ThcError> {
    let mut restricted = Vec::with_capacity(blocks.len());
    for block in blocks {
        let n_col = block.n_columns();
        let mut values = Vec::with_capacity(candidates.len() * n_col);
        for &point in candidates {
            if point >= block.n_points {
                return Err(ThcError::PointIndex(point));
            }
            let start = point * n_col;
            values.extend_from_slice(&block.values()[start..start + n_col]);
        }
        restricted.push(PairBlock::new(
            block.q_index,
            candidates.len(),
            block.layout,
            values,
        )?);
    }
    Ok(restricted)
}

fn candidate_weights(weights: &[f64], candidates: &[usize]) -> Vec<f64> {
    candidates.iter().map(|&index| weights[index]).collect()
}

/// Interpolation-point auxiliary at one $q$.
///
/// `provenance` is stored on the compiled auxiliary and later copied onto
/// Bloch pair vertices by [`bloch_pair_vertices`].
pub fn interpolation_auxiliary(
    partition: AuxiliaryPartition,
    q: TransferQ,
    points: Vec<muffintin_auxiliary_ir::InterpolationAuxiliaryPoint>,
    provenance: Provenance,
) -> Result<CompiledAuxiliaryBasis, ThcError> {
    let auxiliary = CompiledAuxiliaryBasis {
        partition,
        q,
        representation: AuxiliaryRepresentation::InterpolationPoints(InterpolationPointAuxiliary {
            points,
        }),
        provenance,
    };
    auxiliary.validate()?;
    Ok(auxiliary)
}

/// Bloch pair vertices: coefficients are pair densities at interpolation points.
pub fn bloch_pair_vertices(
    q: TransferQ,
    selected_rows: &[num_complex::Complex64],
    n_mu: usize,
    layout: PairColumnLayout,
    auxiliary: &CompiledAuxiliaryBasis,
) -> Result<Vec<PairVertex>, ThcError> {
    let n_col = layout.n_columns()?;
    let expected = crate::error::checked_storage_len(&[n_mu, n_col])?;
    if selected_rows.len() != expected {
        return Err(ThcError::PairBlockLength {
            expected,
            actual: selected_rows.len(),
        });
    }
    if q != auxiliary.q {
        return Err(ThcError::Product(
            muffintin_auxiliary_ir::AuxiliaryIrError::AuxiliarySupportTransferQ,
        ));
    }
    let mt = auxiliary.mt_dimension();
    let interstitial = auxiliary.interstitial_dimension();
    if mt + interstitial != n_mu {
        return Err(ThcError::Product(
            muffintin_auxiliary_ir::AuxiliaryIrError::PairVertexDimension {
                actual: n_mu,
                mt,
                interstitial,
            },
        ));
    }
    let mut vertices = Vec::with_capacity(n_col);
    for column in 0..n_col {
        let (k_index, left, right) = layout.decode(column);
        let mut coefficients = Vec::with_capacity(n_mu);
        for mu in 0..n_mu {
            coefficients.push(selected_rows[mu * n_col + column]);
        }
        vertices.push(PairVertex::from_auxiliary(
            auxiliary,
            OrbitalPair::Bloch {
                k_index,
                left,
                right,
            },
            coefficients,
        )?);
    }
    Ok(vertices)
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
