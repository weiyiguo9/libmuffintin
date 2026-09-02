use std::path::PathBuf;

use muffintin_io::{
    MLDUMP_EXCHANGE_TOTAL_RELATION_V2, MldumpExchangeLayoutV2, MldumpExchangeProvenanceV2,
    MldumpExchangeSectorV2, MldumpExchangeSpaceV2, MldumpGammaPolicyV2, MldumpRequestedRankV2,
    MldumpSelectorEngineV2, MldumpSelectorStrategyV2,
};
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::export::{array2, export_exchange_dict_v2};

fn exchange_space(space: &MldumpExchangeSpaceV2) -> &'static str {
    match space {
        MldumpExchangeSpaceV2::Valence => "valence",
        MldumpExchangeSpaceV2::Core => "core",
    }
}

fn export_layout<'py>(
    py: Python<'py>,
    layout: &MldumpExchangeLayoutV2,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("occupied_space", exchange_space(&layout.occupied_space))?;
    dict.set_item("target_space", exchange_space(&layout.target_space))?;
    dict.set_item("n_k", layout.n_k)?;
    dict.set_item("n_occupied", layout.n_occupied)?;
    dict.set_item("n_target", layout.n_target)?;
    Ok(dict)
}

fn export_sector<'py>(
    py: Python<'py>,
    sector: &MldumpExchangeSectorV2,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("layout", export_layout(py, &sector.layout)?)?;
    dict.set_item("trace_hartree", sector.trace_hartree)?;
    dict.set_item(
        "maximum_antihermitian_residual",
        sector.maximum_antihermitian,
    )?;
    dict.set_item("fit_frobenius", sector.fit_residual.frobenius)?;
    dict.set_item("fit_column_max", sector.fit_residual.column_max)?;
    dict.set_item(
        "mpb_quadratic_maximum_absolute",
        sector.mpb_quadratic.maximum_absolute,
    )?;
    dict.set_item(
        "mpb_quadratic_maximum_relative",
        sector.mpb_quadratic.maximum_relative,
    )?;
    dict.set_item(
        "mpb_quadratic_worst_absolute_q_index",
        sector.mpb_quadratic.worst_absolute_q_index,
    )?;
    dict.set_item(
        "mpb_quadratic_worst_absolute_column",
        sector.mpb_quadratic.worst_absolute_column,
    )?;
    dict.set_item(
        "mpb_quadratic_worst_relative_q_index",
        sector.mpb_quadratic.worst_relative_q_index,
    )?;
    dict.set_item(
        "mpb_quadratic_worst_relative_column",
        sector.mpb_quadratic.worst_relative_column,
    )?;
    Ok(dict)
}

fn export_provenance<'py>(
    py: Python<'py>,
    provenance: &MldumpExchangeProvenanceV2,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("source_frame", &provenance.source_frame)?;
    dict.set_item("backend", &provenance.backend)?;
    dict.set_item(
        "gamma_policy",
        match &provenance.gamma_policy {
            MldumpGammaPolicyV2::FiniteBody => "finite_body",
            MldumpGammaPolicyV2::Reject => "reject",
        },
    )?;
    dict.set_item("product_l_max", provenance.product_l_max)?;
    dict.set_item("product_g_max_inv_bohr", provenance.product_g_max_inv_bohr)?;
    dict.set_item("overlap_tolerance", provenance.overlap_tolerance)?;
    dict.set_item("coulomb_lexp", provenance.coulomb_lexp)?;
    dict.set_item("interpolation_l_max", provenance.interpolation_l_max)?;
    dict.set_item(
        "interpolation_pw_cutoff_inv_bohr",
        provenance.interpolation_pw_cutoff_inv_bohr,
    )?;
    dict.set_item(
        "selector_strategy",
        match &provenance.selector_strategy {
            MldumpSelectorStrategyV2::AllQL2 => "allq_l2",
        },
    )?;
    let selector_engine = match &provenance.selector_engine {
        MldumpSelectorEngineV2::FullColumnPivotedQr => "full_column_pivoted_qr",
        MldumpSelectorEngineV2::FullPivotedCholesky => "full_pivoted_cholesky",
    };
    dict.set_item("selector_engine", selector_engine)?;
    let (rank_policy, rank_n_mu, rank_threshold, rank_n_max) = match &provenance.requested_rank {
        MldumpRequestedRankV2::Exact { n_mu } => ("exact", Some(*n_mu), None, None),
        MldumpRequestedRankV2::Threshold { threshold, n_max } => {
            ("threshold", None, Some(*threshold), Some(*n_max))
        }
    };
    dict.set_item("requested_rank_policy", rank_policy)?;
    dict.set_item("requested_rank_n_mu", rank_n_mu)?;
    dict.set_item("requested_rank_threshold", rank_threshold)?;
    dict.set_item("requested_rank_n_max", rank_n_max)?;

    let scaling = &provenance.rank_scaling;
    let rank_scaling = PyDict::new(py);
    rank_scaling.set_item("n_k", scaling.n_k)?;
    rank_scaling.set_item("n_valence", scaling.n_valence)?;
    rank_scaling.set_item("n_core", scaling.n_core)?;
    rank_scaling.set_item("n_candidates", scaling.n_candidates)?;
    rank_scaling.set_item("effective_rank", scaling.effective_rank)?;
    rank_scaling.set_item("vv_columns_per_q", scaling.vv_columns_per_q)?;
    rank_scaling.set_item("cv_columns_per_q", scaling.cv_columns_per_q)?;
    rank_scaling.set_item("vc_columns_per_q", scaling.vc_columns_per_q)?;
    rank_scaling.set_item("cc_columns_per_q", scaling.cc_columns_per_q)?;
    rank_scaling.set_item("pooled_columns_per_q", scaling.pooled_columns_per_q)?;
    rank_scaling.set_item("selector_rows", scaling.selector_rows)?;
    dict.set_item("rank_scaling", rank_scaling)?;

    dict.set_item(
        "k_weights",
        PyArray1::from_vec(py, provenance.k_weights.clone()),
    )?;
    let valence_occupations = provenance
        .valence_occupations
        .iter()
        .flatten()
        .copied()
        .collect();
    dict.set_item(
        "valence_occupations",
        array2(py, scaling.n_k, scaling.n_valence, valence_occupations),
    )?;

    let core_identity = provenance
        .core_occupations
        .iter()
        .flat_map(|core| {
            [
                core.site_index as i64,
                i64::from(core.n),
                i64::from(core.signed_kappa),
                i64::from(core.twice_mu),
            ]
        })
        .collect();
    dict.set_item(
        "core_identity",
        array2(py, provenance.core_occupations.len(), 4, core_identity),
    )?;
    dict.set_item(
        "core_occupations",
        PyArray1::from_vec(
            py,
            provenance
                .core_occupations
                .iter()
                .map(|core| core.occupation)
                .collect(),
        ),
    )?;
    Ok(dict)
}

#[pyfunction]
pub(crate) fn read_mldump_v2(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyDict>> {
    let file = muffintin_io::read_mldump_v2(path)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let dict = export_exchange_dict_v2(py)?;
    dict.set_item("producer_name", &file.header.meta.producer_name)?;
    dict.set_item("producer_version", &file.header.meta.producer_version)?;
    dict.set_item("source_revision", &file.header.meta.source_revision)?;
    dict.set_item(
        "feature_representation",
        &file.header.meta.feature_representation,
    )?;
    dict.set_item("exchange_vv_hartree", file.exchange.exchange_vv_hartree)?;
    dict.set_item("exchange_cv_hartree", file.exchange.exchange_cv_hartree)?;
    dict.set_item("exchange_cc_hartree", file.exchange.exchange_cc_hartree)?;
    dict.set_item(
        "exchange_total_hartree",
        file.exchange.exchange_total_hartree,
    )?;
    dict.set_item("exchange_total_relation", MLDUMP_EXCHANGE_TOTAL_RELATION_V2)?;
    dict.set_item(
        "cross_trace_average_hartree",
        file.exchange.cross_trace_average_hartree,
    )?;
    dict.set_item(
        "cross_trace_mismatch_hartree",
        file.exchange.cross_trace_mismatch_hartree,
    )?;
    let sectors = PyDict::new(py);
    sectors.set_item("vv", export_sector(py, &file.exchange.vv)?)?;
    sectors.set_item("cv", export_sector(py, &file.exchange.cv)?)?;
    sectors.set_item("vc", export_sector(py, &file.exchange.vc)?)?;
    sectors.set_item("cc", export_sector(py, &file.exchange.cc)?)?;
    dict.set_item("sectors", sectors)?;
    dict.set_item(
        "provenance",
        export_provenance(py, &file.exchange.provenance)?,
    )?;
    Ok(dict.unbind())
}
