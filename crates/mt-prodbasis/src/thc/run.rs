//! End-to-end toy ISDF/THC run over the product-space IR.

use crate::thc::ThcError;
use crate::thc::fit::{ExchangePerQFit, PerQFit, WeightedResidual, fit_exchange_per_q, fit_per_q};
use crate::thc::pair::{ExchangePairBlock, PairBlock};
use crate::thc::select::{
    ExchangeSelection, ExchangeSelectionProvenance, GridPath, L2Engine, RankPolicy, Selection,
    SelectionProvenance, SelectorStrategy, exchange_pair_block_l2_pivots, interpolation_points,
    pair_block_l2_pivots, truncate_rank,
};
use crate::{
    AuxiliaryPartition, AuxiliaryRepresentation, CompiledAuxiliaryBasis, ExchangePairLayout,
    ExchangeSpace, InterpolationPointAuxiliary, OrbitalPair, PairColumnLayout, PairVertex,
    TransferQ,
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

/// Vertices for one exact rectangular exchange layout.
#[derive(Clone, Debug, PartialEq)]
pub struct ExchangeThcSectorVertices {
    pub layout: ExchangePairLayout,
    /// Vertices in the exact layout's column order.
    pub vertices: Vec<PairVertex>,
}

/// Rectangular VV/CV/VC/CC vertices at one canonical q.
#[derive(Clone, Debug, PartialEq)]
pub struct ExchangeThcQVertices {
    pub q_index: usize,
    pub q: TransferQ,
    pub vv: ExchangeThcSectorVertices,
    pub cv: ExchangeThcSectorVertices,
    pub vc: ExchangeThcSectorVertices,
    pub cc: ExchangeThcSectorVertices,
}

/// Explicit rank and selector-matrix accounting for core-aware THC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExchangeThcRankScaling {
    pub n_k: usize,
    pub n_valence: usize,
    pub n_core: usize,
    pub n_candidates: usize,
    pub effective_rank: usize,
    pub vv_columns: usize,
    pub cv_columns: usize,
    pub vc_columns: usize,
    pub cc_columns: usize,
    pub pooled_columns_per_q: usize,
    pub selector_rows: usize,
}

/// Core-aware THC selection, shared per-q fits, and rectangular vertices.
#[derive(Clone, Debug, PartialEq)]
pub struct ExchangeThcResult {
    pub selection: ExchangeSelection,
    pub fits: Vec<ExchangePerQFit>,
    pub auxiliaries: Vec<CompiledAuxiliaryBasis>,
    pub vertices: Vec<ExchangeThcQVertices>,
    pub rank_scaling: ExchangeThcRankScaling,
}

/// Shared AllQL2 selection and per-q zeta fits for VV/CV/VC/CC exchange.
///
/// `blocks` is strictly q-major, with exactly four blocks per q in VV, CV,
/// VC, CC order. The four rectangular layouts must agree on `n_k`,
/// `n_valence`, and `n_core`. Selection stacks every column in the fixed order
/// q-major → VV/CV/VC/CC → column. QRCP or pivoted Cholesky selects only
/// parent-grid points: no pair column is sampled, dropped, balanced, or given
/// a sector quota. Every q then receives one zeta fit shared by all four
/// sectors, with a separate weighted residual reported for each sector.
#[allow(clippy::too_many_arguments)]
pub fn fit_allq_l2_exchange_pair_blocks(
    blocks: &[ExchangePairBlock],
    points: &[[f64; 3]],
    weights: &[f64],
    regions: &[crate::InterpolationRegion],
    partition: AuxiliaryPartition,
    transfers: &[TransferQ],
    rank: RankPolicy,
    engine: L2Engine,
    candidates: Option<&[usize]>,
    provenance: Provenance,
) -> Result<ExchangeThcResult, ThcError> {
    if transfers.is_empty() {
        return Err(ThcError::EmptyRank);
    }
    let expected_blocks = crate::thc::error::checked_storage_len(&[transfers.len(), 4])?;
    if blocks.len() != expected_blocks {
        return Err(ThcError::ExchangePairBlockCount {
            expected: expected_blocks,
            actual: blocks.len(),
        });
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
    let (layouts, n_k, n_valence, n_core) =
        validate_exchange_blocks(blocks, transfers.len(), points.len())?;
    if transfers.len() != n_k {
        return Err(ThcError::TransferQCount {
            expected: n_k,
            actual: transfers.len(),
        });
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
    let restricted = restrict_exchange_blocks(blocks, &candidate_ids)?;
    let local = exchange_pair_block_l2_pivots(
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
    let selection = ExchangeSelection {
        points: interpolation_points(&pivots, points, weights, regions)?,
        pivots: pivots.clone(),
        provenance: ExchangeSelectionProvenance {
            strategy: SelectorStrategy::AllQL2,
            n_mu: pivots.len(),
            q_set: "allq",
            grid_path: GridPath::External {
                n_points: points.len(),
                n_candidates: candidate_ids.len(),
            },
            engine,
            weights: "sqrt(quadrature)",
            pair_column_windows: layouts,
            row_order: "q-major->VV/CV/VC/CC->column",
        },
    };
    let selected_ids = selection
        .points
        .iter()
        .map(|point| point.id)
        .collect::<Vec<_>>();
    let n_mu = selected_ids.len();
    let mut fits = Vec::with_capacity(transfers.len());
    let mut auxiliaries = Vec::with_capacity(transfers.len());
    let mut vertices = Vec::with_capacity(transfers.len());
    for (q_index, &q) in transfers.iter().enumerate() {
        let offset = 4 * q_index;
        let targets = [
            &blocks[offset],
            &blocks[offset + 1],
            &blocks[offset + 2],
            &blocks[offset + 3],
        ];
        let selected = [
            targets[0].selected_rows(&selected_ids)?,
            targets[1].selected_rows(&selected_ids)?,
            targets[2].selected_rows(&selected_ids)?,
            targets[3].selected_rows(&selected_ids)?,
        ];
        let fit = fit_exchange_per_q(
            [&selected[0], &selected[1], &selected[2], &selected[3]],
            n_mu,
            targets,
            weights,
            q,
        )?;
        let auxiliary = interpolation_auxiliary(
            partition.clone(),
            q,
            selection.points.clone(),
            provenance.clone(),
        )?;
        let q_vertices = ExchangeThcQVertices {
            q_index,
            q,
            vv: ExchangeThcSectorVertices {
                layout: layouts[0],
                vertices: exchange_pair_vertices(q, &selected[0], n_mu, layouts[0], &auxiliary)?,
            },
            cv: ExchangeThcSectorVertices {
                layout: layouts[1],
                vertices: exchange_pair_vertices(q, &selected[1], n_mu, layouts[1], &auxiliary)?,
            },
            vc: ExchangeThcSectorVertices {
                layout: layouts[2],
                vertices: exchange_pair_vertices(q, &selected[2], n_mu, layouts[2], &auxiliary)?,
            },
            cc: ExchangeThcSectorVertices {
                layout: layouts[3],
                vertices: exchange_pair_vertices(q, &selected[3], n_mu, layouts[3], &auxiliary)?,
            },
        };
        fits.push(fit);
        auxiliaries.push(auxiliary);
        vertices.push(q_vertices);
    }
    let vv_columns = layouts[0].n_columns()?;
    let cv_columns = layouts[1].n_columns()?;
    let vc_columns = layouts[2].n_columns()?;
    let cc_columns = layouts[3].n_columns()?;
    let pooled_columns_per_q = [vv_columns, cv_columns, vc_columns, cc_columns]
        .into_iter()
        .try_fold(0_usize, |total, count| total.checked_add(count))
        .ok_or_else(|| ThcError::DimensionOverflow {
            dimensions: vec![vv_columns, cv_columns, vc_columns, cc_columns],
        })?;
    let selector_rows =
        crate::thc::error::checked_storage_len(&[transfers.len(), pooled_columns_per_q])?;
    Ok(ExchangeThcResult {
        selection,
        fits,
        auxiliaries,
        vertices,
        rank_scaling: ExchangeThcRankScaling {
            n_k,
            n_valence,
            n_core,
            n_candidates: candidate_ids.len(),
            effective_rank: n_mu,
            vv_columns,
            cv_columns,
            vc_columns,
            cc_columns,
            pooled_columns_per_q,
            selector_rows,
        },
    })
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

fn validate_exchange_blocks(
    blocks: &[ExchangePairBlock],
    n_q: usize,
    n_points: usize,
) -> Result<([ExchangePairLayout; 4], usize, usize, usize), ThcError> {
    let expected_spaces = [
        (ExchangeSpace::Valence, ExchangeSpace::Valence),
        (ExchangeSpace::Core, ExchangeSpace::Valence),
        (ExchangeSpace::Valence, ExchangeSpace::Core),
        (ExchangeSpace::Core, ExchangeSpace::Core),
    ];
    for q_index in 0..n_q {
        for (sector, &(expected_occupied, expected_target)) in expected_spaces.iter().enumerate() {
            let index = 4 * q_index + sector;
            let block = &blocks[index];
            if block.layout.occupied_space != expected_occupied
                || block.layout.target_space != expected_target
            {
                return Err(ThcError::ExchangePairBlockSector {
                    index,
                    expected_occupied,
                    expected_target,
                    actual_occupied: block.layout.occupied_space,
                    actual_target: block.layout.target_space,
                });
            }
        }
    }
    let n_k = blocks[0].layout.n_k;
    let n_valence = blocks[0].layout.n_occupied;
    let n_core = blocks[1].layout.n_occupied;
    if n_valence == 0 {
        return Err(ThcError::EmptyExchangeSpace(ExchangeSpace::Valence));
    }
    if n_core == 0 {
        return Err(ThcError::EmptyExchangeSpace(ExchangeSpace::Core));
    }
    let layouts = [
        ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Valence,
            n_k,
            n_valence,
            n_valence,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Valence,
            n_k,
            n_core,
            n_valence,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Core,
            n_k,
            n_valence,
            n_core,
        ),
        ExchangePairLayout::new(
            ExchangeSpace::Core,
            ExchangeSpace::Core,
            n_k,
            n_core,
            n_core,
        ),
    ];
    for q_index in 0..n_q {
        for (sector, &expected) in layouts.iter().enumerate() {
            let index = 4 * q_index + sector;
            let block = &blocks[index];
            if block.q_index != q_index {
                return Err(ThcError::ExchangePairBlockQIndex {
                    index,
                    expected: q_index,
                    actual: block.q_index,
                });
            }
            if block.n_points != n_points {
                return Err(ThcError::PairBlockPointCount {
                    index,
                    expected: n_points,
                    actual: block.n_points,
                });
            }
            if block.layout != expected {
                return Err(ThcError::ExchangePairBlockLayout {
                    index,
                    expected,
                    actual: block.layout,
                });
            }
        }
    }
    Ok((layouts, n_k, n_valence, n_core))
}

fn restrict_exchange_blocks(
    blocks: &[ExchangePairBlock],
    candidates: &[usize],
) -> Result<Vec<ExchangePairBlock>, ThcError> {
    let mut restricted = Vec::with_capacity(blocks.len());
    for block in blocks {
        let mut values = Vec::with_capacity(crate::thc::error::checked_storage_len(&[
            candidates.len(),
            block.n_columns(),
        ])?);
        for &point in candidates {
            if point >= block.n_points {
                return Err(ThcError::PointIndex(point));
            }
            let start = point * block.n_columns();
            values.extend_from_slice(&block.values()[start..start + block.n_columns()]);
        }
        restricted.push(ExchangePairBlock::new(
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

/// Rectangular exchange vertices in [`ExchangePairLayout`] column order.
pub fn exchange_pair_vertices(
    q: TransferQ,
    selected_rows: &[num_complex::Complex64],
    n_mu: usize,
    layout: ExchangePairLayout,
    auxiliary: &CompiledAuxiliaryBasis,
) -> Result<Vec<PairVertex>, ThcError> {
    let n_columns = layout.n_columns()?;
    let expected = crate::thc::error::checked_storage_len(&[n_mu, n_columns])?;
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
    let mut vertices = Vec::with_capacity(n_columns);
    for column in 0..n_columns {
        let (k_index, occupied, target) = layout.decode(column)?;
        let mut coefficients = Vec::with_capacity(n_mu);
        for mu in 0..n_mu {
            coefficients.push(selected_rows[mu * n_columns + column]);
        }
        vertices.push(PairVertex::from_auxiliary(
            auxiliary,
            OrbitalPair::Exchange {
                k_index,
                occupied_space: layout.occupied_space,
                occupied,
                target_space: layout.target_space,
                target,
            },
            coefficients,
        )?);
    }
    Ok(vertices)
}
