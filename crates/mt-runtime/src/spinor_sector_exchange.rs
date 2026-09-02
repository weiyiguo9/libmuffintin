//! One-shot frozen-orbital exact-MPB exchange over VV, CV, VC, and CC.

use crate::isdf_exchange::{
    GammaExchangeTreatment, IsdfExchangeBandMatrix, IsdfExchangeError, IsdfExchangeSpec,
    RectangularExchangeRecord, build_spinor_mpb_exchange, contract_rectangular_exchange,
    contract_selected_spinor_mpb_exchange_with_operators, target_trace, validate_k_weights,
    validate_occupations,
};
use crate::spinor_exchange_mpb::{
    SpinorExchangeMpbFeedbackResult, SpinorExchangeMpbResult, SpinorExchangeMpbSector,
};
use crate::spinor_mpb::SpinorMpbResult;
use crate::spinor_product::{SpinorProductInput, SpinorQSliceError, require_spinor_q_slice};
use muffintin_core::{Hartree, Kappa};
use muffintin_coulomb::{CoulombError, CoulombRequest, RadialSlaterTraces, assemble_coulomb};
use muffintin_dft::CoreFixedPotentialResult;
use muffintin_prodbasis::{ExchangePairLayout, ExchangeSpace, OrbitalPair, PairVertex};
use thiserror::Error;

/// One immutable occupation snapshot shared by every exchange sector.
#[derive(Clone, Debug, PartialEq)]
pub struct SectorOccupations {
    /// Positive k weights in the frozen canonical order; they sum to one.
    pub k_weights: Vec<f64>,
    /// Fractional valence occupations `[k][band]`.
    pub valence: Vec<Vec<f64>>,
    /// Fractional occupations in the flat core order sealed by `SpinorCoreTable`.
    pub core: Vec<f64>,
    pub gamma: GammaExchangeTreatment,
}

/// Exchange matrix and occupied-target trace for one rectangular sector.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenExchangeSector {
    pub layout: ExchangePairLayout,
    pub trace: Hartree,
    pub target_matrices: Vec<IsdfExchangeBandMatrix>,
    pub maximum_antihermitian_residual: f64,
}

/// VV and CV matrices needed to update valence orbitals inside a relaxed-core
/// Fock fixed-point loop. VC and CC are deliberately deferred until the
/// fixed point is accepted.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrozenSpinorValenceFeedbackExchange {
    pub(crate) vv: FrozenExchangeSector,
    pub(crate) cv: FrozenExchangeSector,
}

/// Exact-MPB core-valence contractions without VV or CC work.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenCoreValenceExchange {
    pub cv: FrozenExchangeSector,
    pub vc: FrozenExchangeSector,
    /// Exact VC diagonal weighted over k, per flat core orbital and before core occupation.
    pub exact_vc_diagonal: Vec<Hartree>,
    sealed: FrozenSectorContext,
}

/// Complete one-shot exact-MPB sector accounting on one frozen DFT snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenSpinorSectorExchange {
    pub vv: FrozenExchangeSector,
    pub cv: FrozenExchangeSector,
    pub vc: FrozenExchangeSector,
    pub cc: FrozenExchangeSector,
    pub cross_trace_average: Hartree,
    pub cross_trace_mismatch: Hartree,
    pub exchange_vv: Hartree,
    pub exchange_cv: Hartree,
    pub exchange_cc: Hartree,
    pub exchange_total: Hartree,
    sealed: FrozenSectorContext,
}

/// Separate numerical and physical-spill gates for the independent radial oracle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SectorRadialComparisonSpec {
    pub numerical_tolerance: Hartree,
    /// Dimensionless maximum permitted outside norm `norm_total - norm_mt` per core shell.
    pub maximum_shell_spill: f64,
}

/// Exact-MPB versus MT-radial residuals, with spill reported independently.
#[derive(Clone, Debug, PartialEq)]
pub struct SectorRadialComparison {
    pub radial: RadialSlaterTraces,
    pub cc_mpb_mt_difference: Hartree,
    pub cv_mpb_mt_difference: Hartree,
    pub vc_mpb_mt_difference: Hartree,
    pub shell_spill: Vec<CoreShellSpillDiagnostic>,
    pub maximum_measured_shell_spill: f64,
    pub shell_spill_threshold: f64,
}

/// M3a CC-only MPB/action/radial comparison without CV or VC evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreCoreRadialComparison {
    pub radial: RadialSlaterTraces,
    pub mpb_cc_trace: Hartree,
    pub action_cc_trace: Hartree,
    /// Signed finite-body MPB CC trace minus the isolated onsite radial trace.
    pub cc_mpb_mt_difference: Hartree,
    /// Numerical final-action-versus-extended-radial identity residual.
    pub cc_action_extended_residual: Hartree,
    /// Measured extended-minus-MT radial trace difference, reported separately.
    pub extended_spill_allowance: Hartree,
    pub shell_spill: Vec<CoreShellSpillDiagnostic>,
    pub maximum_measured_shell_spill: f64,
    pub shell_spill_threshold: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreShellSpillDiagnostic {
    pub site_index: usize,
    pub n: u32,
    pub kappa: Kappa,
    pub spill: f64,
    pub threshold: f64,
}

impl FrozenSpinorSectorExchange {
    /// Exact freshness check over orbitals, core sidecars, q maps/sources, request, and occupations.
    pub fn frozen_context_matches(
        &self,
        inputs: &[SpinorProductInput],
        request: &CoulombRequest,
        occupations: &SectorOccupations,
    ) -> bool {
        self.sealed.inputs == inputs
            && &self.sealed.request == request
            && &self.sealed.occupations == occupations
    }
}

impl FrozenCoreValenceExchange {
    /// Exact freshness check over orbitals, core sidecars, q maps/sources, request, and occupations.
    pub fn frozen_context_matches(
        &self,
        inputs: &[SpinorProductInput],
        request: &CoulombRequest,
        occupations: &SectorOccupations,
    ) -> bool {
        self.sealed.inputs == inputs
            && &self.sealed.request == request
            && &self.sealed.occupations == occupations
    }

    pub(crate) fn frozen_inputs_occupations_match(
        &self,
        inputs: &[SpinorProductInput],
        occupations: &SectorOccupations,
    ) -> bool {
        self.sealed.inputs == inputs && &self.sealed.occupations == occupations
    }
}

#[derive(Clone, Debug, PartialEq)]
struct FrozenSectorContext {
    inputs: Vec<SpinorProductInput>,
    request: CoulombRequest,
    occupations: SectorOccupations,
}

#[derive(Debug, Error)]
pub enum FrozenSpinorSectorExchangeError {
    #[error(transparent)]
    Exchange(#[from] IsdfExchangeError),
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error("frozen sector evaluator requires a complete compatible spinor q slice")]
    FrozenContext,
    #[error("frozen sector evaluator received {actual} core MPB records for {expected} q points")]
    CoreMpbCount { actual: usize, expected: usize },
    #[error("core MPB result at q index {index} does not match the frozen input")]
    CoreMpbContext { index: usize },
    #[error(
        "core sector {occupied_space:?}->{target_space:?} at q index {q_index} has an incomplete or duplicate column set"
    )]
    CoreMpbColumns {
        q_index: usize,
        occupied_space: ExchangeSpace,
        target_space: ExchangeSpace,
    },
    #[error("sector core occupations do not match the flat frozen core table")]
    CoreOccupations,
    #[error("sector radial comparison tolerance is invalid")]
    RadialTolerance,
    #[error("sector radial comparison {sector} numerical residual {residual} exceeds {tolerance}")]
    RadialNumerical {
        sector: &'static str,
        residual: f64,
        tolerance: f64,
    },
    #[error(
        "core shell site={site} n={n} kappa={kappa} spill {spill} exceeds the dimensionless threshold {threshold}"
    )]
    CoreSpill {
        site: usize,
        n: u32,
        kappa: i32,
        spill: f64,
        threshold: f64,
    },
}

struct PreparedCoreSectors {
    operators: Vec<muffintin_coulomb::CoulombOperator>,
    cv_vertices: Vec<Vec<PairVertex>>,
    vc_vertices: Vec<Vec<PairVertex>>,
    cc_vertices: Vec<Vec<PairVertex>>,
    maps: Vec<Vec<usize>>,
    cv_layout: ExchangePairLayout,
    vc_layout: ExchangePairLayout,
    cc_layout: ExchangePairLayout,
}

/// Contract only the exact-MPB CV and VC sectors on one frozen q slice.
///
/// The result exposes the VC diagonal weighted over k per flat core target without
/// applying a target-core occupation. No VV or CC contraction is performed.
pub fn build_frozen_core_valence_exchange(
    inputs: &[SpinorProductInput],
    core_mpb: &[SpinorExchangeMpbResult],
    request: &CoulombRequest,
    occupations: &SectorOccupations,
) -> Result<FrozenCoreValenceExchange, FrozenSpinorSectorExchangeError> {
    let first = validate_frozen_core_context(inputs, request, occupations)?;
    let prepared = prepare_core_sectors(inputs, core_mpb, request, first)?;
    let core_rows = vec![occupations.core.clone(); prepared.cv_layout.n_k];
    let (cv, vc, exact_vc_diagonal) = contract_core_valence(
        &prepared,
        &occupations.k_weights,
        &occupations.valence,
        &core_rows,
        occupations.gamma,
    )?;
    Ok(FrozenCoreValenceExchange {
        cv,
        vc,
        exact_vc_diagonal,
        sealed: FrozenSectorContext {
            inputs: inputs.to_vec(),
            request: request.clone(),
            occupations: occupations.clone(),
        },
    })
}

pub(crate) fn build_cached_spinor_valence_feedback_exchange(
    inputs: &[SpinorProductInput],
    vv_mpb: &[SpinorMpbResult],
    vv_operators: &[muffintin_coulomb::CoulombOperator],
    occupied_bands: &[usize],
    core_mpb: &[SpinorExchangeMpbFeedbackResult],
    core_operators: &[muffintin_coulomb::CoulombOperator],
    request: &CoulombRequest,
    occupations: &SectorOccupations,
) -> Result<FrozenSpinorValenceFeedbackExchange, FrozenSpinorSectorExchangeError> {
    let first = validate_frozen_core_context(inputs, request, occupations)?;
    let vv_result = contract_selected_spinor_mpb_exchange_with_operators(
        inputs,
        vv_mpb,
        vv_operators,
        occupied_bands,
        &IsdfExchangeSpec {
            k_weights: occupations.k_weights.clone(),
            occupations: occupations.valence.clone(),
            gamma: occupations.gamma,
        },
    )?;
    let vv = FrozenExchangeSector {
        layout: ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Valence,
            first.pair_columns.n_k,
            occupied_bands.len(),
            first.pair_columns.n_orb,
        ),
        trace: Hartree(2.0 * vv_result.exchange_energy.get()),
        target_matrices: vv_result.band_matrices,
        maximum_antihermitian_residual: vv_result.maximum_antihermitian_residual,
    };

    let n_k = first.pair_columns.n_k;
    let cv_layout = ExchangePairLayout::new(
        ExchangeSpace::Core,
        ExchangeSpace::Valence,
        n_k,
        first.core.orbitals.len(),
        first.pair_columns.n_orb,
    );
    if core_mpb.len() != n_k || core_operators.len() != n_k {
        return Err(FrozenSpinorSectorExchangeError::CoreMpbCount {
            actual: core_mpb.len().min(core_operators.len()),
            expected: n_k,
        });
    }
    let mut cv_vertices = Vec::with_capacity(n_k);
    for (q_index, (input, result)) in inputs.iter().zip(core_mpb).enumerate() {
        if !result.frozen_input_identity().matches(input) || result.cv.layout != cv_layout {
            return Err(FrozenSpinorSectorExchangeError::CoreMpbContext { index: q_index });
        }
        cv_vertices.push(order_sector(q_index, &result.cv)?);
    }
    let maps = inputs
        .iter()
        .map(|input| {
            input
                .k_minus_q
                .iter()
                .map(|mapped| mapped.kq_index)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let core_rows = vec![occupations.core.clone(); n_k];
    let cv = contract_sector(
        cv_layout,
        &maps,
        &cv_vertices,
        core_operators,
        &occupations.k_weights,
        &core_rows,
        &occupations.valence,
        occupations.gamma,
    )?;
    Ok(FrozenSpinorValenceFeedbackExchange { vv, cv })
}

/// Evaluate VV, CV, VC, and CC independently on one converged frozen DFT snapshot.
///
/// This function performs no orbital update, density feedback, core solve, or SCF
/// iteration. Each rectangular kernel applies `w_(k-q) f_A` on its occupied side;
/// the trace applies `w_k f_B` once on its target side.
pub fn build_frozen_spinor_sector_exchange(
    inputs: &[SpinorProductInput],
    vv_mpb: &[SpinorMpbResult],
    core_mpb: &[SpinorExchangeMpbResult],
    request: &CoulombRequest,
    occupations: &SectorOccupations,
) -> Result<FrozenSpinorSectorExchange, FrozenSpinorSectorExchangeError> {
    let first = validate_frozen_core_context(inputs, request, occupations)?;
    let vv = contract_vv(inputs, vv_mpb, request, occupations, first)?;
    complete_frozen_spinor_sector_exchange(inputs, core_mpb, request, occupations, vv)
}

pub(crate) fn complete_frozen_spinor_sector_exchange(
    inputs: &[SpinorProductInput],
    core_mpb: &[SpinorExchangeMpbResult],
    request: &CoulombRequest,
    occupations: &SectorOccupations,
    vv: FrozenExchangeSector,
) -> Result<FrozenSpinorSectorExchange, FrozenSpinorSectorExchangeError> {
    let first = validate_frozen_core_context(inputs, request, occupations)?;
    let n_k = first.pair_columns.n_k;
    if vv.layout.occupied_space != ExchangeSpace::Valence
        || vv.layout.target_space != ExchangeSpace::Valence
        || vv.layout.n_k != n_k
        || vv.layout.n_target != first.pair_columns.n_orb
        || vv.target_matrices.len() != n_k
    {
        return Err(FrozenSpinorSectorExchangeError::FrozenContext);
    }
    let core_rows = vec![occupations.core.clone(); n_k];
    let prepared = prepare_core_sectors(inputs, core_mpb, request, first)?;
    let (cv, vc, _) = contract_core_valence(
        &prepared,
        &occupations.k_weights,
        &occupations.valence,
        &core_rows,
        occupations.gamma,
    )?;
    let cc = contract_sector(
        prepared.cc_layout,
        &prepared.maps,
        &prepared.cc_vertices,
        &prepared.operators,
        &occupations.k_weights,
        &core_rows,
        &core_rows,
        occupations.gamma,
    )?;

    let cross_trace_average = Hartree(0.5 * (cv.trace.get() + vc.trace.get()));
    let cross_trace_mismatch = Hartree((cv.trace.get() - vc.trace.get()).abs());
    let exchange_vv = Hartree(0.5 * vv.trace.get());
    let exchange_cv = cross_trace_average;
    let exchange_cc = Hartree(0.5 * cc.trace.get());
    let exchange_total = Hartree(exchange_vv.get() + exchange_cv.get() + exchange_cc.get());
    Ok(FrozenSpinorSectorExchange {
        vv,
        cv,
        vc,
        cc,
        cross_trace_average,
        cross_trace_mismatch,
        exchange_vv,
        exchange_cv,
        exchange_cc,
        exchange_total,
        sealed: FrozenSectorContext {
            inputs: inputs.to_vec(),
            request: request.clone(),
            occupations: occupations.clone(),
        },
    })
}

/// Compare an independent radial trace with all three core-member MPB traces.
///
/// MPB-minus-MT values are signed physical diagnostics because finite-body MPB
/// and isolated onsite radial contractions do not use the same Coulomb body.
/// The radial imaginary residual and dimensionless shell spill remain gated;
/// no conversion from a norm spill to Hartree is made.
pub fn compare_frozen_sector_radial(
    exchange: &FrozenSpinorSectorExchange,
    radial: &RadialSlaterTraces,
    spec: SectorRadialComparisonSpec,
) -> Result<SectorRadialComparison, FrozenSpinorSectorExchangeError> {
    if !spec.numerical_tolerance.get().is_finite()
        || spec.numerical_tolerance.get() < 0.0
        || !spec.maximum_shell_spill.is_finite()
        || spec.maximum_shell_spill < 0.0
    {
        return Err(FrozenSpinorSectorExchangeError::RadialTolerance);
    }
    let cc = exchange.cc.trace.get() - radial.cc_mt.total.get();
    let cv = exchange.cv.trace.get() - radial.cv_mt.total.get();
    let vc = exchange.vc.trace.get() - radial.cv_mt.total.get();
    if radial.cv_imaginary_residual > spec.numerical_tolerance.get() {
        return Err(FrozenSpinorSectorExchangeError::RadialNumerical {
            sector: "CV imaginary",
            residual: radial.cv_imaginary_residual,
            tolerance: spec.numerical_tolerance.get(),
        });
    }
    let mut shell_spill = Vec::new();
    for orbital in exchange
        .sealed
        .inputs
        .first()
        .into_iter()
        .flat_map(|input| &input.core.orbitals)
    {
        if shell_spill.iter().any(|item: &CoreShellSpillDiagnostic| {
            item.site_index == orbital.site_index
                && item.n == orbital.n
                && item.kappa == orbital.kappa
        }) {
            continue;
        }
        if orbital.spill > spec.maximum_shell_spill {
            return Err(FrozenSpinorSectorExchangeError::CoreSpill {
                site: orbital.site_index,
                n: orbital.n,
                kappa: orbital.kappa.get(),
                spill: orbital.spill,
                threshold: spec.maximum_shell_spill,
            });
        }
        shell_spill.push(CoreShellSpillDiagnostic {
            site_index: orbital.site_index,
            n: orbital.n,
            kappa: orbital.kappa,
            spill: orbital.spill,
            threshold: spec.maximum_shell_spill,
        });
    }
    let maximum_measured_shell_spill = shell_spill
        .iter()
        .map(|item| item.spill)
        .fold(0.0_f64, f64::max);
    Ok(SectorRadialComparison {
        radial: radial.clone(),
        cc_mpb_mt_difference: Hartree(cc),
        cv_mpb_mt_difference: Hartree(cv),
        vc_mpb_mt_difference: Hartree(vc),
        shell_spill,
        maximum_measured_shell_spill,
        shell_spill_threshold: spec.maximum_shell_spill,
    })
}

/// Compare a fresh M3a relaxed-core result against CC-only MPB and radial traces.
///
/// The full finite-body MPB CC trace and the isolated onsite radial trace are
/// different physical Coulomb bodies when the occupied CC pair has a nonzero
/// monopole, so their signed difference is reported rather than gated. The
/// final sampled radial action is independently checked against the extended
/// radial trace. `cc_spill_allowance` and dimensionless shell spill are never
/// folded into the action numerical tolerance.
pub fn compare_relaxed_core_core_radial(
    relaxed: &CoreFixedPotentialResult,
    mpb_cc_trace: Hartree,
    radial: &RadialSlaterTraces,
    spec: SectorRadialComparisonSpec,
) -> Result<CoreCoreRadialComparison, FrozenSpinorSectorExchangeError> {
    if !spec.numerical_tolerance.get().is_finite()
        || spec.numerical_tolerance.get() < 0.0
        || !spec.maximum_shell_spill.is_finite()
        || spec.maximum_shell_spill < 0.0
    {
        return Err(FrozenSpinorSectorExchangeError::RadialTolerance);
    }
    let mpb_mt = mpb_cc_trace.get() - radial.cc_mt.total.get();
    let action_extended =
        (relaxed.final_cc_trace.total.get() - radial.cc_extended.total.get()).abs();
    if action_extended > spec.numerical_tolerance.get() {
        return Err(FrozenSpinorSectorExchangeError::RadialNumerical {
            sector: "CC action",
            residual: action_extended,
            tolerance: spec.numerical_tolerance.get(),
        });
    }

    let mut shell_spill = Vec::with_capacity(relaxed.orbitals.shells.len());
    for shell in &relaxed.orbitals.shells {
        if shell.spill > spec.maximum_shell_spill {
            return Err(FrozenSpinorSectorExchangeError::CoreSpill {
                site: relaxed.orbitals.site_index,
                n: shell.state.n,
                kappa: shell.state.kappa.get(),
                spill: shell.spill,
                threshold: spec.maximum_shell_spill,
            });
        }
        shell_spill.push(CoreShellSpillDiagnostic {
            site_index: relaxed.orbitals.site_index,
            n: shell.state.n,
            kappa: shell.state.kappa,
            spill: shell.spill,
            threshold: spec.maximum_shell_spill,
        });
    }
    let maximum_measured_shell_spill = shell_spill
        .iter()
        .map(|item| item.spill)
        .fold(0.0_f64, f64::max);
    Ok(CoreCoreRadialComparison {
        radial: radial.clone(),
        mpb_cc_trace,
        action_cc_trace: relaxed.final_cc_trace.total,
        cc_mpb_mt_difference: Hartree(mpb_mt),
        cc_action_extended_residual: Hartree(action_extended),
        extended_spill_allowance: radial.cc_spill_allowance,
        shell_spill,
        maximum_measured_shell_spill,
        shell_spill_threshold: spec.maximum_shell_spill,
    })
}

fn validate_frozen_core_context<'a>(
    inputs: &'a [SpinorProductInput],
    request: &CoulombRequest,
    occupations: &SectorOccupations,
) -> Result<&'a SpinorProductInput, FrozenSpinorSectorExchangeError> {
    let first = require_spinor_q_slice(inputs).map_err(q_slice_error)?;
    let n_k = first.pair_columns.n_k;
    let n_valence = first.pair_columns.n_orb;
    let n_core = first.core.orbitals.len();
    validate_k_weights(&occupations.k_weights, n_k)?;
    validate_occupations(&occupations.valence, n_k, n_valence)?;
    let core_rows = vec![occupations.core.clone(); n_k];
    validate_occupations(&core_rows, n_k, n_core)?;
    if occupations.core
        != first
            .core
            .orbitals
            .iter()
            .map(|orbital| orbital.occupation)
            .collect::<Vec<_>>()
    {
        return Err(FrozenSpinorSectorExchangeError::CoreOccupations);
    }
    if request.reciprocal() != &first.reciprocal {
        return Err(FrozenSpinorSectorExchangeError::FrozenContext);
    }
    Ok(first)
}

fn contract_vv(
    inputs: &[SpinorProductInput],
    vv_mpb: &[SpinorMpbResult],
    request: &CoulombRequest,
    occupations: &SectorOccupations,
    first: &SpinorProductInput,
) -> Result<FrozenExchangeSector, FrozenSpinorSectorExchangeError> {
    let result = build_spinor_mpb_exchange(
        inputs,
        vv_mpb,
        request,
        &IsdfExchangeSpec {
            k_weights: occupations.k_weights.clone(),
            occupations: occupations.valence.clone(),
            gamma: occupations.gamma,
        },
    )?;
    Ok(FrozenExchangeSector {
        layout: ExchangePairLayout::new(
            ExchangeSpace::Valence,
            ExchangeSpace::Valence,
            first.pair_columns.n_k,
            first.pair_columns.n_orb,
            first.pair_columns.n_orb,
        ),
        trace: Hartree(2.0 * result.exchange_energy.get()),
        target_matrices: result.band_matrices,
        maximum_antihermitian_residual: result.maximum_antihermitian_residual,
    })
}

fn prepare_core_sectors(
    inputs: &[SpinorProductInput],
    core_mpb: &[SpinorExchangeMpbResult],
    request: &CoulombRequest,
    first: &SpinorProductInput,
) -> Result<PreparedCoreSectors, FrozenSpinorSectorExchangeError> {
    let n_k = first.pair_columns.n_k;
    let n_valence = first.pair_columns.n_orb;
    let n_core = first.core.orbitals.len();
    if core_mpb.len() != n_k {
        return Err(FrozenSpinorSectorExchangeError::CoreMpbCount {
            actual: core_mpb.len(),
            expected: n_k,
        });
    }
    let cv_layout = ExchangePairLayout::new(
        ExchangeSpace::Core,
        ExchangeSpace::Valence,
        n_k,
        n_core,
        n_valence,
    );
    let vc_layout = ExchangePairLayout::new(
        ExchangeSpace::Valence,
        ExchangeSpace::Core,
        n_k,
        n_valence,
        n_core,
    );
    let cc_layout = ExchangePairLayout::new(
        ExchangeSpace::Core,
        ExchangeSpace::Core,
        n_k,
        n_core,
        n_core,
    );
    let mut operators = Vec::with_capacity(n_k);
    let mut cv_vertices = Vec::with_capacity(n_k);
    let mut vc_vertices = Vec::with_capacity(n_k);
    let mut cc_vertices = Vec::with_capacity(n_k);
    for (q_index, (input, result)) in inputs.iter().zip(core_mpb).enumerate() {
        if !result.frozen_input_identity().matches(input)
            || result.auxiliary.q != input.source.q
            || result.auxiliary.partition != input.source.partition
            || result.cv.layout != cv_layout
            || result.vc.layout != vc_layout
            || result.cc.layout != cc_layout
        {
            return Err(FrozenSpinorSectorExchangeError::CoreMpbContext { index: q_index });
        }
        cv_vertices.push(order_sector(q_index, &result.cv)?);
        vc_vertices.push(order_sector(q_index, &result.vc)?);
        cc_vertices.push(order_sector(q_index, &result.cc)?);
        operators.push(assemble_coulomb(&result.auxiliary, request)?);
    }
    let maps = inputs
        .iter()
        .map(|input| {
            input
                .k_minus_q
                .iter()
                .map(|mapped| mapped.kq_index)
                .collect::<Vec<_>>()
        })
        .collect();
    Ok(PreparedCoreSectors {
        operators,
        cv_vertices,
        vc_vertices,
        cc_vertices,
        maps,
        cv_layout,
        vc_layout,
        cc_layout,
    })
}

fn contract_core_valence(
    prepared: &PreparedCoreSectors,
    k_weights: &[f64],
    valence_occupations: &[Vec<f64>],
    core_rows: &[Vec<f64>],
    gamma: GammaExchangeTreatment,
) -> Result<
    (FrozenExchangeSector, FrozenExchangeSector, Vec<Hartree>),
    FrozenSpinorSectorExchangeError,
> {
    let cv = contract_sector(
        prepared.cv_layout,
        &prepared.maps,
        &prepared.cv_vertices,
        &prepared.operators,
        k_weights,
        core_rows,
        valence_occupations,
        gamma,
    )?;
    let vc = contract_sector(
        prepared.vc_layout,
        &prepared.maps,
        &prepared.vc_vertices,
        &prepared.operators,
        k_weights,
        valence_occupations,
        core_rows,
        gamma,
    )?;
    let exact_vc_diagonal = (0..prepared.vc_layout.n_target)
        .map(|core| {
            Hartree(
                vc.target_matrices
                    .iter()
                    .enumerate()
                    .map(|(k, matrix)| {
                        k_weights[k]
                            * matrix
                                .element(core, core)
                                .expect("VC matrix dimension follows its checked layout")
                                .re
                    })
                    .sum(),
            )
        })
        .collect();
    Ok((cv, vc, exact_vc_diagonal))
}

#[allow(clippy::too_many_arguments)]
fn contract_sector(
    layout: ExchangePairLayout,
    maps: &[Vec<usize>],
    vertices: &[Vec<PairVertex>],
    operators: &[muffintin_coulomb::CoulombOperator],
    k_weights: &[f64],
    occupied_occupations: &[Vec<f64>],
    target_occupations: &[Vec<f64>],
    gamma: GammaExchangeTreatment,
) -> Result<FrozenExchangeSector, FrozenSpinorSectorExchangeError> {
    validate_occupations(target_occupations, layout.n_k, layout.n_target)?;
    let records = vertices
        .iter()
        .zip(operators)
        .map(|(vertices, operator)| RectangularExchangeRecord {
            layout,
            vertices,
            operator,
        })
        .collect::<Vec<_>>();
    let contracted = contract_rectangular_exchange(
        layout,
        maps,
        &records,
        k_weights,
        occupied_occupations,
        gamma,
    )?;
    let trace = target_trace(
        layout,
        &contracted.target_matrices,
        k_weights,
        target_occupations,
    );
    Ok(FrozenExchangeSector {
        layout,
        trace: Hartree(trace),
        target_matrices: contracted.target_matrices,
        maximum_antihermitian_residual: contracted.maximum_antihermitian_residual,
    })
}

fn order_sector(
    q_index: usize,
    sector: &SpinorExchangeMpbSector,
) -> Result<Vec<PairVertex>, FrozenSpinorSectorExchangeError> {
    let expected =
        sector
            .layout
            .n_columns()
            .map_err(|_| FrozenSpinorSectorExchangeError::CoreMpbColumns {
                q_index,
                occupied_space: sector.layout.occupied_space,
                target_space: sector.layout.target_space,
            })?;
    let mut ordered = vec![None; expected];
    for selected in &sector.vertices {
        let pair = OrbitalPair::Exchange {
            k_index: selected.k,
            occupied_space: sector.layout.occupied_space,
            occupied: selected.occupied,
            target_space: sector.layout.target_space,
            target: selected.target,
        };
        let valid = selected.column < expected
            && sector
                .layout
                .decode(selected.column)
                .is_ok_and(|decoded| decoded == (selected.k, selected.occupied, selected.target))
            && selected.vertex.pair() == pair;
        if !valid
            || ordered[selected.column]
                .replace(selected.vertex.clone())
                .is_some()
        {
            return Err(FrozenSpinorSectorExchangeError::CoreMpbColumns {
                q_index,
                occupied_space: sector.layout.occupied_space,
                target_space: sector.layout.target_space,
            });
        }
    }
    ordered.into_iter().collect::<Option<Vec<_>>>().ok_or(
        FrozenSpinorSectorExchangeError::CoreMpbColumns {
            q_index,
            occupied_space: sector.layout.occupied_space,
            target_space: sector.layout.target_space,
        },
    )
}

fn q_slice_error(_: SpinorQSliceError) -> FrozenSpinorSectorExchangeError {
    FrozenSpinorSectorExchangeError::FrozenContext
}
