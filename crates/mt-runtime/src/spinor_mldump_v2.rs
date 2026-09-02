//! MLDUMP v2 exchange summary over the final rebuilt relaxed-core frame.

use std::path::Path;

use muffintin_io::{
    IoError, MLDUMP_EXCHANGE_BACKEND_V2, MLDUMP_EXCHANGE_SOURCE_FRAME_V2, MldumpCoreOccupationV2,
    MldumpExchangeFitResidualV2, MldumpExchangeLayoutV2, MldumpExchangeMpbQuadraticV2,
    MldumpExchangeProvenanceV2, MldumpExchangeRankScalingV2, MldumpExchangeSectorV2,
    MldumpExchangeSpaceV2, MldumpExchangeV2, MldumpGammaPolicyV2, MldumpHeaderV1,
    MldumpRequestedRankV2, MldumpSelectorEngineV2, MldumpSelectorStrategyV2,
    upgrade_mldump_v1_with_exchange_v2,
};
use muffintin_prodbasis::thc::{
    GridPath, L2Engine, RankPolicy, SelectorStrategy, WeightedResidual,
};
use muffintin_prodbasis::{ExchangePairLayout, ExchangeSpace, OrbitalPair};
use thiserror::Error;

use crate::hf_scf::{RelaxedCoreHfResult, RelaxedCoreHfSpec};
use crate::isdf_exchange::GammaExchangeTreatment;
use crate::spinor_coulomb::{
    SpinorCoulombError, SpinorCoulombResult, require_spinor_coulomb_export_context,
};
use crate::spinor_mldump::{SpinorMldumpError, write_spinor_mldump};
use crate::spinor_product::SpinorProductInput;
use crate::spinor_sector_exchange::{FrozenExchangeSector, SectorOccupations};
use crate::spinor_sector_thc::{
    SpinorSectorThcDiagnostics, SpinorSectorThcMpbComparison, SpinorSectorThcMpbSectorComparison,
    SpinorSectorThcResult,
};
use crate::spinor_thc::SpinorThcResult;

const CONSISTENCY_TOLERANCE: f64 = 1.0e-12;

/// Failure while binding the MLDUMP v2 exchange summary to one final HF frame.
#[derive(Debug, Error)]
pub enum SpinorMldumpV2Error {
    #[error(transparent)]
    V1(#[from] SpinorMldumpError),
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Coulomb(#[from] SpinorCoulombError),
    #[error("spinor MLDUMP v2 frozen-frame mismatch at {0}")]
    FrozenFrame(&'static str),
    #[error("spinor MLDUMP v2 inconsistent exchange summary at {0}")]
    ExchangeSummary(&'static str),
    #[error("spinor MLDUMP v2 core twice-mu value {0} does not fit the interchange type")]
    TwiceMu(i64),
}

/// Write the existing spinor MLDUMP payload and add the M5 exchange summary.
///
/// The common orbitals/products/THC/Coulomb payload is first emitted through
/// [`write_spinor_mldump`], so the intermediate file is a strict valid v1
/// file. Only after every v2 frame and summary invariant passes preflight is
/// that file upgraded with the typed exchange record.
pub fn write_spinor_mldump_v2(
    path: impl AsRef<Path>,
    header: &MldumpHeaderV1,
    result: &RelaxedCoreHfResult,
    spec: &RelaxedCoreHfSpec,
    vv_thc: &SpinorThcResult,
    vv_coulomb: &SpinorCoulombResult,
    sector_thc: &SpinorSectorThcResult,
    sector_comparison: &SpinorSectorThcMpbComparison,
) -> Result<(), SpinorMldumpV2Error> {
    let exchange = preflight_exchange_v2(
        header,
        result,
        spec,
        vv_thc,
        vv_coulomb,
        sector_thc,
        sector_comparison,
    )?;
    let path = path.as_ref();
    write_spinor_mldump(
        path,
        header,
        &result.final_exchange_inputs,
        vv_thc,
        vv_coulomb,
        vv_coulomb.sealed_spec(),
    )?;
    upgrade_mldump_v1_with_exchange_v2(path, &exchange)?;
    Ok(())
}

fn preflight_exchange_v2(
    header: &MldumpHeaderV1,
    result: &RelaxedCoreHfResult,
    spec: &RelaxedCoreHfSpec,
    vv_thc: &SpinorThcResult,
    vv_coulomb: &SpinorCoulombResult,
    sector_thc: &SpinorSectorThcResult,
    sector_comparison: &SpinorSectorThcMpbComparison,
) -> Result<MldumpExchangeV2, SpinorMldumpV2Error> {
    let inputs = &result.final_exchange_inputs;
    let first = inputs
        .first()
        .ok_or(SpinorMldumpV2Error::FrozenFrame("final_exchange_inputs"))?;
    let core_occupations = first
        .core
        .orbitals
        .iter()
        .map(|orbital| orbital.occupation)
        .collect::<Vec<_>>();
    let occupations = SectorOccupations {
        k_weights: result.k_weights.clone(),
        valence: result.occupations.clone(),
        core: core_occupations,
        gamma: spec.gamma,
    };

    if !result
        .sector_exchange
        .frozen_context_matches(inputs, &spec.coulomb, &occupations)
    {
        return Err(SpinorMldumpV2Error::FrozenFrame("sector_exchange"));
    }
    if !vv_coulomb.frozen_inputs_match(inputs) || vv_coulomb.sealed_spec().request != spec.coulomb {
        return Err(SpinorMldumpV2Error::FrozenFrame("vv_coulomb"));
    }
    require_spinor_coulomb_export_context(inputs, vv_thc, vv_coulomb, vv_coulomb.sealed_spec())?;
    if !sector_thc.frozen_context_matches(inputs) {
        return Err(SpinorMldumpV2Error::FrozenFrame("sector_thc"));
    }
    if !sector_comparison.frozen_context_matches(inputs) {
        return Err(SpinorMldumpV2Error::FrozenFrame("sector_comparison"));
    }
    require_result_frame(result, first)?;
    require_header_frame(header, result)?;

    let n_k = first.orbitals.k_fractional.len();
    let n_valence = first.orbitals.band_window.count;
    let n_core = first.core.orbitals.len();
    let layouts = exchange_layouts(n_k, n_valence, n_core);
    require_exchange_energy_relations(&result.sector_exchange)?;
    require_exchange_layouts(&result.sector_exchange, layouts)?;
    require_sector_thc(sector_thc, inputs, layouts)?;
    require_sector_comparison(sector_comparison, inputs.len(), layouts)?;

    let diagnostics = sector_thc.diagnostics;
    Ok(MldumpExchangeV2 {
        vv: exchange_sector(
            &result.sector_exchange.vv,
            diagnostics.vv,
            &sector_comparison.vv,
        ),
        cv: exchange_sector(
            &result.sector_exchange.cv,
            diagnostics.cv,
            &sector_comparison.cv,
        ),
        vc: exchange_sector(
            &result.sector_exchange.vc,
            diagnostics.vc,
            &sector_comparison.vc,
        ),
        cc: exchange_sector(
            &result.sector_exchange.cc,
            diagnostics.cc,
            &sector_comparison.cc,
        ),
        exchange_vv_hartree: result.sector_exchange.exchange_vv.get(),
        exchange_cv_hartree: result.sector_exchange.exchange_cv.get(),
        exchange_cc_hartree: result.sector_exchange.exchange_cc.get(),
        exchange_total_hartree: result.sector_exchange.exchange_total.get(),
        cross_trace_average_hartree: result.sector_exchange.cross_trace_average.get(),
        cross_trace_mismatch_hartree: result.sector_exchange.cross_trace_mismatch.get(),
        provenance: exchange_provenance(result, spec, vv_coulomb, sector_thc)?,
    })
}

fn require_result_frame(
    result: &RelaxedCoreHfResult,
    first: &SpinorProductInput,
) -> Result<(), SpinorMldumpV2Error> {
    if result.k_fractional != first.orbitals.k_fractional {
        return Err(SpinorMldumpV2Error::FrozenFrame("result.k_fractional"));
    }
    if result.orbital_energies != first.orbitals.energies {
        return Err(SpinorMldumpV2Error::FrozenFrame("result.orbital_energies"));
    }
    if result.core_orbitals != first.core.sidecars {
        return Err(SpinorMldumpV2Error::FrozenFrame("result.core_orbitals"));
    }
    if result.q_fractional.len() != result.final_exchange_inputs.len()
        || result.k_weights.len() != result.k_fractional.len()
        || result.occupations.len() != result.k_fractional.len()
    {
        return Err(SpinorMldumpV2Error::FrozenFrame("result.mesh_shape"));
    }
    Ok(())
}

fn require_header_frame(
    header: &MldumpHeaderV1,
    result: &RelaxedCoreHfResult,
) -> Result<(), SpinorMldumpV2Error> {
    if header.mesh.k_points.len() != result.k_fractional.len()
        || header.mesh.q_entries.len() != result.q_fractional.len()
    {
        return Err(SpinorMldumpV2Error::FrozenFrame("header.mesh_shape"));
    }
    for ((stored, fractional), weight) in header
        .mesh
        .k_points
        .iter()
        .zip(&result.k_fractional)
        .zip(&result.k_weights)
    {
        if !same_vector(stored.fractional, *fractional) || !same_float(stored.weight, *weight) {
            return Err(SpinorMldumpV2Error::FrozenFrame("header.k_points"));
        }
    }
    for (stored, fractional) in header.mesh.q_entries.iter().zip(&result.q_fractional) {
        if !same_vector(stored.canonical_fractional, *fractional) {
            return Err(SpinorMldumpV2Error::FrozenFrame("header.q_entries"));
        }
    }
    Ok(())
}

fn exchange_layouts(n_k: usize, n_valence: usize, n_core: usize) -> [ExchangePairLayout; 4] {
    [
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
    ]
}

fn require_exchange_layouts(
    exchange: &crate::FrozenSpinorSectorExchange,
    expected: [ExchangePairLayout; 4],
) -> Result<(), SpinorMldumpV2Error> {
    let actual = [
        exchange.vv.layout,
        exchange.cv.layout,
        exchange.vc.layout,
        exchange.cc.layout,
    ];
    if actual != expected {
        return Err(SpinorMldumpV2Error::ExchangeSummary("exchange.layouts"));
    }
    Ok(())
}

fn require_sector_thc(
    thc: &SpinorSectorThcResult,
    inputs: &[SpinorProductInput],
    layouts: [ExchangePairLayout; 4],
) -> Result<(), SpinorMldumpV2Error> {
    let n_q = inputs.len();
    if thc.records.len() != n_q || !thc.records_match_parent_grid() {
        return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.records"));
    }
    if thc.selection.provenance.strategy != SelectorStrategy::AllQL2
        || thc.selection.provenance.q_set != "allq"
        || thc.selection.provenance.weights != "sqrt(quadrature)"
        || thc.selection.provenance.row_order != "q-major->VV/CV/VC/CC->column"
        || thc.selection.provenance.pair_column_windows != layouts
        || thc.selection.provenance.n_mu != thc.effective_rank
        || thc.selection.points.len() != thc.effective_rank
        || thc.selection.pivots.len() != thc.effective_rank
    {
        return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.selection"));
    }
    if !matches!(
        thc.selection.provenance.engine,
        L2Engine::FullColumnPivotedQr | L2Engine::FullPivotedCholesky
    ) {
        return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.engine"));
    }

    let scaling = thc.rank_scaling;
    if !matches!(
        &thc.selection.provenance.grid_path,
        GridPath::External {
            n_points,
            n_candidates,
        } if *n_points == thc.grid.points().len() && *n_candidates == scaling.n_candidates
    ) {
        return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.grid_path"));
    }
    let columns = layouts
        .map(|layout| layout.n_columns())
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| SpinorMldumpV2Error::ExchangeSummary("sector_thc.layouts"))?;
    let pooled = columns.iter().sum::<usize>();
    if scaling.n_k != layouts[0].n_k
        || scaling.n_valence != layouts[0].n_occupied
        || scaling.n_core != layouts[3].n_occupied
        || scaling.effective_rank != thc.effective_rank
        || scaling.n_candidates < scaling.effective_rank
        || scaling.vv_columns_per_q != columns[0]
        || scaling.cv_columns_per_q != columns[1]
        || scaling.vc_columns_per_q != columns[2]
        || scaling.cc_columns_per_q != columns[3]
        || scaling.pooled_columns_per_q != pooled
        || scaling.selector_rows != n_q * pooled
    {
        return Err(SpinorMldumpV2Error::ExchangeSummary(
            "sector_thc.rank_scaling",
        ));
    }
    require_residuals(thc.diagnostics)?;
    for (q_index, (record, input)) in thc.records.iter().zip(inputs).enumerate() {
        let sectors = [&record.vv, &record.cv, &record.vc, &record.cc];
        let points = record
            .auxiliary
            .require_interpolation_points()
            .map_err(|_| SpinorMldumpV2Error::ExchangeSummary("sector_thc.auxiliary"))?;
        if record.q_index != q_index
            || record.q != input.source.q
            || record.auxiliary.q != record.q
            || record.auxiliary.partition != input.source.partition
            || record.n_mu != thc.effective_rank
            || record.n_mu != record.auxiliary.dimension()
            || record.n_points != thc.grid.points().len()
            || record.zeta.len() != record.n_points * record.n_mu
            || [
                record.vv.layout,
                record.cv.layout,
                record.vc.layout,
                record.cc.layout,
            ] != layouts
            || [
                record.vv.vertices.len(),
                record.cv.vertices.len(),
                record.vc.vertices.len(),
                record.cc.vertices.len(),
            ] != [columns[0], columns[1], columns[2], columns[3]]
        {
            return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.q_record"));
        }
        if points != thc.selection.points {
            return Err(SpinorMldumpV2Error::ExchangeSummary(
                "sector_thc.interpolation_points",
            ));
        }
        for (sector, layout) in sectors.into_iter().zip(layouts) {
            for (column, vertex) in sector.vertices.iter().enumerate() {
                if vertex.pair() != expected_pair(layout, column)?
                    || vertex.layout() != &record.auxiliary.layout()
                    || vertex.coefficients().len() != record.n_mu
                {
                    return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.vertex"));
                }
            }
        }
        require_residuals(record.residuals)?;
    }
    match thc.requested_rank {
        RankPolicy::Exact { n_mu } if n_mu != thc.effective_rank => Err(
            SpinorMldumpV2Error::ExchangeSummary("sector_thc.requested_rank"),
        ),
        RankPolicy::Threshold { thresh, n_max }
            if !thresh.is_finite()
                || thresh <= 0.0
                || thc.effective_rank == 0
                || thc.effective_rank > n_max =>
        {
            Err(SpinorMldumpV2Error::ExchangeSummary(
                "sector_thc.requested_rank",
            ))
        }
        _ => Ok(()),
    }
}

fn require_residuals(residuals: SpinorSectorThcDiagnostics) -> Result<(), SpinorMldumpV2Error> {
    for residual in [residuals.vv, residuals.cv, residuals.vc, residuals.cc] {
        if !residual.frobenius.is_finite()
            || residual.frobenius < 0.0
            || !residual.column_max.is_finite()
            || residual.column_max < 0.0
        {
            return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.residual"));
        }
    }
    Ok(())
}

fn require_sector_comparison(
    comparison: &SpinorSectorThcMpbComparison,
    n_q: usize,
    layouts: [ExchangePairLayout; 4],
) -> Result<(), SpinorMldumpV2Error> {
    for (sector, layout) in [
        (&comparison.vv, layouts[0]),
        (&comparison.cv, layouts[1]),
        (&comparison.vc, layouts[2]),
        (&comparison.cc, layouts[3]),
    ] {
        let n_columns = layout
            .n_columns()
            .map_err(|_| SpinorMldumpV2Error::ExchangeSummary("comparison.layout"))?;
        if sector.pairs.len() != n_q * n_columns {
            return Err(SpinorMldumpV2Error::ExchangeSummary("comparison.coverage"));
        }
        for q_index in 0..n_q {
            for column in 0..n_columns {
                let pair = sector
                    .pairs
                    .iter()
                    .filter(|pair| pair.q_index == q_index && pair.column == column)
                    .collect::<Vec<_>>();
                if pair.len() != 1 || pair[0].pair != expected_pair(layout, column)? {
                    return Err(SpinorMldumpV2Error::ExchangeSummary("comparison.pair"));
                }
            }
        }
        if sector.pairs.iter().any(|pair| {
            !pair.mpb_quadratic.re.is_finite()
                || !pair.mpb_quadratic.im.is_finite()
                || !pair.thc_quadratic.re.is_finite()
                || !pair.thc_quadratic.im.is_finite()
                || !pair.absolute.is_finite()
                || pair.absolute < 0.0
                || !pair.relative.is_finite()
                || pair.relative < 0.0
        }) {
            return Err(SpinorMldumpV2Error::ExchangeSummary("comparison.values"));
        }
        let maximum_absolute = sector
            .pairs
            .iter()
            .map(|pair| pair.absolute)
            .fold(0.0_f64, f64::max);
        let maximum_relative = sector
            .pairs
            .iter()
            .map(|pair| pair.relative)
            .fold(0.0_f64, f64::max);
        let worst_absolute_matches = sector.pairs.iter().any(|pair| {
            pair.q_index == sector.worst_absolute_q_index
                && pair.column == sector.worst_absolute_column
                && pair.absolute == maximum_absolute
        });
        let worst_relative_matches = sector.pairs.iter().any(|pair| {
            pair.q_index == sector.worst_relative_q_index
                && pair.column == sector.worst_relative_column
                && pair.relative == maximum_relative
        });
        if sector.maximum_absolute != maximum_absolute
            || sector.maximum_relative != maximum_relative
            || !worst_absolute_matches
            || !worst_relative_matches
        {
            return Err(SpinorMldumpV2Error::ExchangeSummary("comparison.maximum"));
        }
    }
    Ok(())
}

fn expected_pair(
    layout: ExchangePairLayout,
    column: usize,
) -> Result<OrbitalPair, SpinorMldumpV2Error> {
    let (k_index, occupied, target) = layout
        .decode(column)
        .map_err(|_| SpinorMldumpV2Error::ExchangeSummary("comparison.column"))?;
    Ok(OrbitalPair::Exchange {
        k_index,
        occupied_space: layout.occupied_space,
        occupied,
        target_space: layout.target_space,
        target,
    })
}

fn require_exchange_energy_relations(
    exchange: &crate::FrozenSpinorSectorExchange,
) -> Result<(), SpinorMldumpV2Error> {
    let cross_average = 0.5 * (exchange.cv.trace.get() + exchange.vc.trace.get());
    let cross_mismatch = (exchange.cv.trace.get() - exchange.vc.trace.get()).abs();
    let vv = 0.5 * exchange.vv.trace.get();
    let cc = 0.5 * exchange.cc.trace.get();
    let total = vv + cross_average + cc;
    if !same_float(exchange.cross_trace_average.get(), cross_average)
        || !same_float(exchange.cross_trace_mismatch.get(), cross_mismatch)
        || !same_float(exchange.exchange_vv.get(), vv)
        || !same_float(exchange.exchange_cv.get(), cross_average)
        || !same_float(exchange.exchange_cc.get(), cc)
        || !same_float(exchange.exchange_total.get(), total)
    {
        return Err(SpinorMldumpV2Error::ExchangeSummary(
            "exchange.energy_relations",
        ));
    }
    Ok(())
}

fn exchange_sector(
    exchange: &FrozenExchangeSector,
    residual: WeightedResidual,
    comparison: &SpinorSectorThcMpbSectorComparison,
) -> MldumpExchangeSectorV2 {
    MldumpExchangeSectorV2 {
        layout: exchange_layout(exchange.layout),
        trace_hartree: exchange.trace.get(),
        maximum_antihermitian: exchange.maximum_antihermitian_residual,
        fit_residual: MldumpExchangeFitResidualV2 {
            frobenius: residual.frobenius,
            column_max: residual.column_max,
        },
        mpb_quadratic: MldumpExchangeMpbQuadraticV2 {
            maximum_absolute: comparison.maximum_absolute,
            maximum_relative: comparison.maximum_relative,
            worst_absolute_q_index: comparison.worst_absolute_q_index,
            worst_absolute_column: comparison.worst_absolute_column,
            worst_relative_q_index: comparison.worst_relative_q_index,
            worst_relative_column: comparison.worst_relative_column,
        },
    }
}

fn exchange_layout(layout: ExchangePairLayout) -> MldumpExchangeLayoutV2 {
    MldumpExchangeLayoutV2 {
        occupied_space: exchange_space(layout.occupied_space),
        target_space: exchange_space(layout.target_space),
        n_k: layout.n_k,
        n_occupied: layout.n_occupied,
        n_target: layout.n_target,
    }
}

fn exchange_space(space: ExchangeSpace) -> MldumpExchangeSpaceV2 {
    match space {
        ExchangeSpace::Valence => MldumpExchangeSpaceV2::Valence,
        ExchangeSpace::Core => MldumpExchangeSpaceV2::Core,
    }
}

fn exchange_provenance(
    result: &RelaxedCoreHfResult,
    spec: &RelaxedCoreHfSpec,
    vv_coulomb: &SpinorCoulombResult,
    thc: &SpinorSectorThcResult,
) -> Result<MldumpExchangeProvenanceV2, SpinorMldumpV2Error> {
    let first = &result.final_exchange_inputs[0];
    let vv_spec = vv_coulomb.sealed_spec();
    let projection = vv_spec.projection;
    let core_occupations = first
        .core
        .orbitals
        .iter()
        .map(|orbital| {
            Ok(MldumpCoreOccupationV2 {
                site_index: orbital.site_index,
                n: orbital.n,
                signed_kappa: orbital.kappa.get(),
                twice_mu: i32::try_from(orbital.twice_mu.get())
                    .map_err(|_| SpinorMldumpV2Error::TwiceMu(orbital.twice_mu.get()))?,
                occupation: orbital.occupation,
            })
        })
        .collect::<Result<Vec<_>, SpinorMldumpV2Error>>()?;
    Ok(MldumpExchangeProvenanceV2 {
        source_frame: MLDUMP_EXCHANGE_SOURCE_FRAME_V2.to_owned(),
        backend: MLDUMP_EXCHANGE_BACKEND_V2.to_owned(),
        gamma_policy: match spec.gamma {
            GammaExchangeTreatment::FiniteBody => MldumpGammaPolicyV2::FiniteBody,
            GammaExchangeTreatment::Reject => MldumpGammaPolicyV2::Reject,
        },
        product_l_max: spec.product_l_max,
        product_g_max_inv_bohr: spec.product_g_max.get(),
        overlap_tolerance: spec.overlap_tolerance,
        coulomb_lexp: vv_spec.request.lexp(),
        interpolation_l_max: projection.l_max,
        interpolation_pw_cutoff_inv_bohr: projection.pw_cutoff.get(),
        selector_strategy: MldumpSelectorStrategyV2::AllQL2,
        selector_engine: match thc.selection.provenance.engine {
            L2Engine::StructuredSketch { .. } => {
                return Err(SpinorMldumpV2Error::ExchangeSummary("sector_thc.engine"));
            }
            L2Engine::FullColumnPivotedQr => MldumpSelectorEngineV2::FullColumnPivotedQr,
            L2Engine::FullPivotedCholesky => MldumpSelectorEngineV2::FullPivotedCholesky,
        },
        requested_rank: match thc.requested_rank {
            RankPolicy::Exact { n_mu } => MldumpRequestedRankV2::Exact { n_mu },
            RankPolicy::Threshold { thresh, n_max } => MldumpRequestedRankV2::Threshold {
                threshold: thresh,
                n_max,
            },
        },
        rank_scaling: MldumpExchangeRankScalingV2 {
            n_k: thc.rank_scaling.n_k,
            n_valence: thc.rank_scaling.n_valence,
            n_core: thc.rank_scaling.n_core,
            n_candidates: thc.rank_scaling.n_candidates,
            effective_rank: thc.rank_scaling.effective_rank,
            vv_columns_per_q: thc.rank_scaling.vv_columns_per_q,
            cv_columns_per_q: thc.rank_scaling.cv_columns_per_q,
            vc_columns_per_q: thc.rank_scaling.vc_columns_per_q,
            cc_columns_per_q: thc.rank_scaling.cc_columns_per_q,
            pooled_columns_per_q: thc.rank_scaling.pooled_columns_per_q,
            selector_rows: thc.rank_scaling.selector_rows,
        },
        k_weights: result.k_weights.clone(),
        valence_occupations: result.occupations.clone(),
        core_occupations,
    })
}

fn same_vector(left: [f64; 3], right: [f64; 3]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| same_float(left, right))
}

fn same_float(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= CONSISTENCY_TOLERANCE * left.abs().max(right.abs()).max(1.0)
}
