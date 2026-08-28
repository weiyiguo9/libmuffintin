//! End-to-end toy ISDF/THC run over the product-space IR.

use crate::ThcError;
use crate::fit::{
    PerQFit, WeightedResidual, fit_per_q, gamma_report, worst_finite_q, worst_finite_q_coulomb,
};
use crate::gram::CoulombGramSet;
use crate::kmesh::KMesh;
use crate::pair::{BlochOrbitals, UmklappGauge, evaluate_pair_block};
use crate::select::{GridPath, Selection, SelectionRequest, SelectorStrategy, select_points};
use crate::toy::ToyGrid;
use muffintin_auxiliary_ir::{
    AuxiliaryRepresentation, CompiledAuxiliaryBasis, InterpolationPointAuxiliary, OrbitalPair,
    PairColumnLayout, PairVertex, ProductPartition, TransferQ,
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
    partition: &ProductPartition,
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
        let auxiliary = interpolation_auxiliary(partition.clone(), q, selection.points.clone())?;
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
    partition: &ProductPartition,
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

/// Interpolation-point auxiliary at one $q$.
pub fn interpolation_auxiliary(
    partition: ProductPartition,
    q: TransferQ,
    points: Vec<muffintin_auxiliary_ir::InterpolationAuxiliaryPoint>,
) -> Result<CompiledAuxiliaryBasis, ThcError> {
    let auxiliary = CompiledAuxiliaryBasis {
        partition,
        q,
        representation: AuxiliaryRepresentation::InterpolationPoints(InterpolationPointAuxiliary {
            points,
        }),
        provenance: Provenance {
            recipe: Some("thc-isdf".to_owned()),
            reference: Some("scratch/thc_mt_kpoint_test.py".to_owned()),
        },
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
            muffintin_auxiliary_ir::ProductError::AuxiliarySupportTransferQ,
        ));
    }
    let mt = auxiliary.mt_dimension();
    let interstitial = auxiliary.interstitial_dimension();
    if mt + interstitial != n_mu {
        return Err(ThcError::Product(
            muffintin_auxiliary_ir::ProductError::PairVertexDimension {
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
