//! End-to-end toy ISDF/THC run over the product-space IR.

use crate::thc::ThcError;
use crate::thc::fit::{PerQFit, WeightedResidual, fit_per_q};
use crate::thc::pair::PairBlock;
use crate::thc::select::{
    GridPath, L2Engine, RankPolicy, Selection, SelectionProvenance, SelectorStrategy,
    interpolation_points, pair_block_l2_pivots, truncate_rank,
};
use crate::{
    AuxiliaryPartition, AuxiliaryRepresentation, CompiledAuxiliaryBasis,
    InterpolationPointAuxiliary, OrbitalPair, PairColumnLayout, PairVertex, TransferQ,
};
use muffintin_envelope::Provenance;

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
    regions: &[crate::InterpolationRegion],
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
    crate::thc::error::validate_quadrature_weights(weights)?;
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
    points: Vec<crate::InterpolationAuxiliaryPoint>,
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
    let expected = crate::thc::error::checked_storage_len(&[n_mu, n_col])?;
    if selected_rows.len() != expected {
        return Err(ThcError::PairBlockLength {
            expected,
            actual: selected_rows.len(),
        });
    }
    if q != auxiliary.q {
        return Err(ThcError::Product(
            crate::AuxiliaryIrError::AuxiliarySupportTransferQ,
        ));
    }
    let mt = auxiliary.mt_dimension();
    let interstitial = auxiliary.interstitial_dimension();
    if mt + interstitial != n_mu {
        return Err(ThcError::Product(
            crate::AuxiliaryIrError::PairVertexDimension {
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

