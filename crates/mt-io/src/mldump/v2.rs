//! MLDUMP v2 exchange summary layered onto the complete v1 spinor payload.

use std::collections::BTreeSet;
use std::path::Path;

use hdf5_metno::{File, Group, Location};

use super::{
    ATTR_SCHEMA_NAME, ATTR_SCHEMA_VERSION, ATTR_STATUS, GROUP_COULOMB, GROUP_EXCHANGE, GROUP_MPB,
    GROUP_ORBITALS, GROUP_PRODUCTS, GROUP_THC, MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
    MLDUMP_SCHEMA_NAME, MLDUMP_SCHEMA_VERSION_V2, MLDUMP_STATUS_PRESENT, MldumpFileV1,
    MldumpHeaderV1, MldumpPayloadV1, MldumpStatus, SpinorMldumpV1, approx_eq, read_absent_child,
    read_absent_group, read_f64_attr, read_f64_dataset, read_geometry_group, read_i64_dataset,
    read_mesh_group, read_meta_group, read_mldump_v1, read_numeric_attr, read_payload_status,
    read_present_payload, read_str_attr, read_units_group, require_exact_members,
    require_finite_f64s, require_no_payload, require_nonnegative_index, require_status_present,
    require_top_level_groups, usize_as_i64, write_f64_attr, write_f64_dataset, write_i64_attr,
    write_i64_dataset, write_str_attr,
};
use crate::error::{IoError, ValidationError, finite, nonempty, positive};

/// Frozen source frame required by the v2 exchange provenance.
pub const MLDUMP_EXCHANGE_SOURCE_FRAME_V2: &str = "relaxed_core_hf_final_rebuilt_frame";
/// Core-aware backend required by the v2 exchange provenance.
pub const MLDUMP_EXCHANGE_BACKEND_V2: &str = "core_aware_thc_with_exact_mpb_oracle";
/// Fixed relation between sector traces and the stored exchange energies.
pub const MLDUMP_EXCHANGE_TOTAL_RELATION_V2: &str =
    "exchange_total=exchange_vv+exchange_cv+exchange_cc;exchange_cv=(trace_cv+trace_vc)/2";

const GROUP_SECTORS: &str = "sectors";
const GROUP_PROVENANCE: &str = "provenance";
const GROUP_FIT_RESIDUAL: &str = "fit_residual";
const GROUP_MPB_QUADRATIC: &str = "mpb_quadratic";
const GROUP_RANK_SCALING: &str = "rank_scaling";
const SECTOR_NAMES: [&str; 4] = ["vv", "cv", "vc", "cc"];

/// One side of an exact rectangular exchange-pair layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MldumpExchangeSpaceV2 {
    Valence,
    Core,
}

impl MldumpExchangeSpaceV2 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Valence => "valence",
            Self::Core => "core",
        }
    }

    fn parse(path: &str, value: &str) -> Result<Self, IoError> {
        match value {
            "valence" => Ok(Self::Valence),
            "core" => Ok(Self::Core),
            actual => Err(invalid(path, "valence or core", actual)),
        }
    }
}

/// Exact rectangular `(k, occupied, target)` sector layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MldumpExchangeLayoutV2 {
    pub occupied_space: MldumpExchangeSpaceV2,
    pub target_space: MldumpExchangeSpaceV2,
    pub n_k: usize,
    pub n_occupied: usize,
    pub n_target: usize,
}

impl MldumpExchangeLayoutV2 {
    fn n_columns(self, path: &str) -> Result<usize, IoError> {
        self.n_k
            .checked_mul(self.n_occupied)
            .and_then(|value| value.checked_mul(self.n_target))
            .ok_or_else(|| invalid(path, "layout size fitting usize", "overflow"))
    }
}

/// Weighted fit residuals for one exact exchange sector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpExchangeFitResidualV2 {
    pub frobenius: f64,
    pub column_max: f64,
}

/// MPB-versus-THC quadratic maxima and their independent worst locations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpExchangeMpbQuadraticV2 {
    pub maximum_absolute: f64,
    pub maximum_relative: f64,
    pub worst_absolute_q_index: usize,
    pub worst_absolute_column: usize,
    pub worst_relative_q_index: usize,
    pub worst_relative_column: usize,
}

/// Trace, Hermiticity, fit, and MPB-oracle summary for one sector.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpExchangeSectorV2 {
    pub layout: MldumpExchangeLayoutV2,
    pub trace_hartree: f64,
    pub maximum_antihermitian: f64,
    pub fit_residual: MldumpExchangeFitResidualV2,
    pub mpb_quadratic: MldumpExchangeMpbQuadraticV2,
}

/// Periodic Gamma-head policy used by the exchange contraction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MldumpGammaPolicyV2 {
    FiniteBody,
    Reject,
}

impl MldumpGammaPolicyV2 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::FiniteBody => "finite_body",
            Self::Reject => "reject",
        }
    }

    fn parse(path: &str, value: &str) -> Result<Self, IoError> {
        match value {
            "finite_body" => Ok(Self::FiniteBody),
            "reject" => Ok(Self::Reject),
            actual => Err(invalid(path, "finite_body or reject", actual)),
        }
    }
}

/// Core-aware selector strategy. MLDUMP v2 admits only the implemented all-q path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MldumpSelectorStrategyV2 {
    AllQL2,
}

impl MldumpSelectorStrategyV2 {
    const fn as_str(self) -> &'static str {
        "allq_l2"
    }

    fn parse(path: &str, value: &str) -> Result<Self, IoError> {
        match value {
            "allq_l2" => Ok(Self::AllQL2),
            actual => Err(invalid(path, "allq_l2", actual)),
        }
    }
}

/// Linear-algebra engine used for the core-aware selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MldumpSelectorEngineV2 {
    FullColumnPivotedQr,
    FullPivotedCholesky,
}

impl MldumpSelectorEngineV2 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::FullColumnPivotedQr => "full_column_pivoted_qr",
            Self::FullPivotedCholesky => "full_pivoted_cholesky",
        }
    }
}

/// Requested interpolation rank before selector truncation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MldumpRequestedRankV2 {
    Exact { n_mu: usize },
    Threshold { threshold: f64, n_max: usize },
}

impl MldumpRequestedRankV2 {
    const fn policy(self) -> &'static str {
        match self {
            Self::Exact { .. } => "exact",
            Self::Threshold { .. } => "threshold",
        }
    }
}

/// Complete selector column and row accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MldumpExchangeRankScalingV2 {
    pub n_k: usize,
    pub n_valence: usize,
    pub n_core: usize,
    pub n_candidates: usize,
    pub effective_rank: usize,
    pub vv_columns_per_q: usize,
    pub cv_columns_per_q: usize,
    pub vc_columns_per_q: usize,
    pub cc_columns_per_q: usize,
    pub pooled_columns_per_q: usize,
    pub selector_rows: usize,
}

/// One flat core spin-orbital identity and its frozen occupation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpCoreOccupationV2 {
    pub site_index: usize,
    pub n: u32,
    pub signed_kappa: i32,
    pub twice_mu: i32,
    pub occupation: f64,
}

/// Provenance sufficient to reproduce the exported exchange summary.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpExchangeProvenanceV2 {
    pub source_frame: String,
    pub backend: String,
    pub gamma_policy: MldumpGammaPolicyV2,
    pub product_l_max: u32,
    pub product_g_max_inv_bohr: f64,
    pub overlap_tolerance: f64,
    pub coulomb_lexp: u32,
    pub interpolation_l_max: u32,
    pub interpolation_pw_cutoff_inv_bohr: f64,
    pub selector_strategy: MldumpSelectorStrategyV2,
    pub selector_engine: MldumpSelectorEngineV2,
    pub requested_rank: MldumpRequestedRankV2,
    pub rank_scaling: MldumpExchangeRankScalingV2,
    pub k_weights: Vec<f64>,
    pub valence_occupations: Vec<Vec<f64>>,
    pub core_occupations: Vec<MldumpCoreOccupationV2>,
}

/// Complete MLDUMP v2 exchange summary.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpExchangeV2 {
    pub vv: MldumpExchangeSectorV2,
    pub cv: MldumpExchangeSectorV2,
    pub vc: MldumpExchangeSectorV2,
    pub cc: MldumpExchangeSectorV2,
    pub exchange_vv_hartree: f64,
    pub exchange_cv_hartree: f64,
    pub exchange_cc_hartree: f64,
    pub exchange_total_hartree: f64,
    pub cross_trace_average_hartree: f64,
    pub cross_trace_mismatch_hartree: f64,
    pub provenance: MldumpExchangeProvenanceV2,
}

/// Owned v2 file: the unchanged v1 header and spinor common payload plus exchange.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpFileV2 {
    pub header: MldumpHeaderV1,
    pub payload: SpinorMldumpV1,
    pub exchange: MldumpExchangeV2,
}

impl MldumpExchangeV2 {
    fn validate(&self, header: &MldumpHeaderV1, payload: &SpinorMldumpV1) -> Result<(), IoError> {
        let n_k = header.mesh.k_points.len();
        let n_q = header.mesh.q_entries.len();
        let n_valence = payload.orbitals.band_window_count;
        if n_q != n_k {
            return Err(length_mismatch("mesh.q_entries", n_k, n_q));
        }

        validate_layout(
            "exchange.vv.layout",
            self.vv.layout,
            MldumpExchangeSpaceV2::Valence,
            MldumpExchangeSpaceV2::Valence,
            n_k,
            n_valence,
            n_valence,
        )?;
        let n_core = self.cc.layout.n_occupied;
        validate_layout(
            "exchange.cv.layout",
            self.cv.layout,
            MldumpExchangeSpaceV2::Core,
            MldumpExchangeSpaceV2::Valence,
            n_k,
            n_core,
            n_valence,
        )?;
        validate_layout(
            "exchange.vc.layout",
            self.vc.layout,
            MldumpExchangeSpaceV2::Valence,
            MldumpExchangeSpaceV2::Core,
            n_k,
            n_valence,
            n_core,
        )?;
        validate_layout(
            "exchange.cc.layout",
            self.cc.layout,
            MldumpExchangeSpaceV2::Core,
            MldumpExchangeSpaceV2::Core,
            n_k,
            n_core,
            n_core,
        )?;
        for (name, sector) in [
            ("vv", &self.vv),
            ("cv", &self.cv),
            ("vc", &self.vc),
            ("cc", &self.cc),
        ] {
            validate_sector(name, sector, n_q)?;
        }

        for (path, value) in [
            ("exchange.exchange_vv_hartree", self.exchange_vv_hartree),
            ("exchange.exchange_cv_hartree", self.exchange_cv_hartree),
            ("exchange.exchange_cc_hartree", self.exchange_cc_hartree),
            (
                "exchange.exchange_total_hartree",
                self.exchange_total_hartree,
            ),
            (
                "exchange.cross_trace_average_hartree",
                self.cross_trace_average_hartree,
            ),
            (
                "exchange.cross_trace_mismatch_hartree",
                self.cross_trace_mismatch_hartree,
            ),
        ] {
            finite(path, value)?;
        }
        nonnegative(
            "exchange.cross_trace_mismatch_hartree",
            self.cross_trace_mismatch_hartree,
        )?;
        require_relation(
            "exchange.exchange_vv_hartree",
            self.exchange_vv_hartree,
            0.5 * self.vv.trace_hartree,
        )?;
        require_relation(
            "exchange.cross_trace_average_hartree",
            self.cross_trace_average_hartree,
            0.5 * (self.cv.trace_hartree + self.vc.trace_hartree),
        )?;
        require_relation(
            "exchange.cross_trace_mismatch_hartree",
            self.cross_trace_mismatch_hartree,
            (self.cv.trace_hartree - self.vc.trace_hartree).abs(),
        )?;
        require_relation(
            "exchange.exchange_cv_hartree",
            self.exchange_cv_hartree,
            self.cross_trace_average_hartree,
        )?;
        require_relation(
            "exchange.exchange_cc_hartree",
            self.exchange_cc_hartree,
            0.5 * self.cc.trace_hartree,
        )?;
        require_relation(
            "exchange.exchange_total_hartree",
            self.exchange_total_hartree,
            self.exchange_vv_hartree + self.exchange_cv_hartree + self.exchange_cc_hartree,
        )?;
        validate_provenance(&self.provenance, header, n_valence, n_core, &self.sectors())
    }

    fn sectors(&self) -> [&MldumpExchangeSectorV2; 4] {
        [&self.vv, &self.cv, &self.vc, &self.cc]
    }
}

/// Upgrade one complete valid v1 spinor file by attaching the v2 exchange summary.
///
/// The exchange payload is written and read back under schema version 1. The root
/// version is changed to 2 only after that validation succeeds.
pub fn upgrade_mldump_v1_with_exchange_v2(
    path: impl AsRef<Path>,
    exchange: &MldumpExchangeV2,
) -> Result<(), IoError> {
    let path = path.as_ref();
    let MldumpFileV1 {
        header,
        payload,
        exchange: _,
    } = read_mldump_v1(path)?;
    let spinor = match payload {
        MldumpPayloadV1::Spinor(spinor) => spinor,
        MldumpPayloadV1::HeaderOnly => {
            return Err(invalid(
                "payload",
                "complete spinor v1 payload",
                "header-only v1 payload",
            ));
        }
        MldumpPayloadV1::Scalar(_) => {
            return Err(invalid(
                "payload",
                "complete spinor v1 payload",
                "scalar v1 payload",
            ));
        }
    };
    exchange.validate(&header, &spinor)?;

    let file = File::open_rw(path)?;
    write_exchange_v2(&file, exchange)?;
    let stored = read_exchange_v2(&file, &header, &spinor)?;
    if stored != *exchange {
        return Err(invalid(
            "/exchange",
            "roundtrip-equal v2 exchange payload",
            "stored payload differs",
        ));
    }
    file.attr(ATTR_SCHEMA_VERSION)?
        .write_scalar(&MLDUMP_SCHEMA_VERSION_V2)?;
    Ok(())
}

/// Read a strict MLDUMP v2 file.
pub fn read_mldump_v2(path: impl AsRef<Path>) -> Result<MldumpFileV2, IoError> {
    let file = open_version(path, MLDUMP_SCHEMA_VERSION_V2)?;
    require_top_level_groups(&file)?;
    let header = MldumpHeaderV1 {
        meta: read_meta_group(&file)?,
        geometry: read_geometry_group(&file)?,
        mesh: {
            read_units_group(&file)?;
            read_mesh_group(&file)?
        },
    };
    header.validate()?;
    for group in [GROUP_ORBITALS, GROUP_PRODUCTS, GROUP_THC, GROUP_COULOMB] {
        let status = read_payload_status(&file, group)?;
        if status != MldumpStatus::Present {
            return Err(invalid(
                &format!("/{group}/@{ATTR_STATUS}"),
                MLDUMP_STATUS_PRESENT,
                status.as_str(),
            ));
        }
    }
    read_absent_group(&file, GROUP_MPB)?;
    let payload = match read_present_payload(&file, &header)? {
        MldumpPayloadV1::Spinor(spinor) => spinor,
        MldumpPayloadV1::Scalar(_) => {
            return Err(invalid(
                "/orbitals/@representation",
                MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
                "scalar_koelling_harmon",
            ));
        }
        MldumpPayloadV1::HeaderOnly => unreachable!("all common payload groups are present"),
    };
    let exchange = read_exchange_v2(&file, &header, &payload)?;
    Ok(MldumpFileV2 {
        header,
        payload,
        exchange,
    })
}

fn open_version(path: impl AsRef<Path>, expected_version: u32) -> Result<File, IoError> {
    let file = File::open(path)?;
    let schema_name = read_str_attr(&file, ATTR_SCHEMA_NAME)?;
    if schema_name != MLDUMP_SCHEMA_NAME {
        return Err(IoError::InvalidFormat {
            expected: MLDUMP_SCHEMA_NAME,
            found: schema_name,
        });
    }
    let schema_version: u32 =
        read_numeric_attr::<u32>(&file, ATTR_SCHEMA_VERSION, "/@schema_version/dtype")?;
    if schema_version != expected_version {
        return Err(IoError::UnsupportedVersion {
            format: MLDUMP_SCHEMA_NAME,
            supported: expected_version,
            found: schema_version,
        });
    }
    Ok(file)
}

fn validate_layout(
    path: &str,
    actual: MldumpExchangeLayoutV2,
    occupied_space: MldumpExchangeSpaceV2,
    target_space: MldumpExchangeSpaceV2,
    n_k: usize,
    n_occupied: usize,
    n_target: usize,
) -> Result<(), IoError> {
    let expected = MldumpExchangeLayoutV2 {
        occupied_space,
        target_space,
        n_k,
        n_occupied,
        n_target,
    };
    if actual == expected && n_k > 0 && n_occupied > 0 && n_target > 0 {
        actual.n_columns(path)?;
        Ok(())
    } else {
        Err(invalid(
            path,
            &format!("{expected:?}"),
            &format!("{actual:?}"),
        ))
    }
}

fn validate_sector(name: &str, sector: &MldumpExchangeSectorV2, n_q: usize) -> Result<(), IoError> {
    let path = format!("exchange.{name}");
    finite(format!("{path}.trace_hartree"), sector.trace_hartree)?;
    for (field, value) in [
        ("maximum_antihermitian", sector.maximum_antihermitian),
        ("fit_residual.frobenius", sector.fit_residual.frobenius),
        ("fit_residual.column_max", sector.fit_residual.column_max),
        (
            "mpb_quadratic.maximum_absolute",
            sector.mpb_quadratic.maximum_absolute,
        ),
        (
            "mpb_quadratic.maximum_relative",
            sector.mpb_quadratic.maximum_relative,
        ),
    ] {
        nonnegative(format!("{path}.{field}"), value)?;
    }
    let n_columns = sector.layout.n_columns(&format!("{path}.layout"))?;
    for (field, value, upper) in [
        (
            "mpb_quadratic.worst_absolute_q_index",
            sector.mpb_quadratic.worst_absolute_q_index,
            n_q,
        ),
        (
            "mpb_quadratic.worst_relative_q_index",
            sector.mpb_quadratic.worst_relative_q_index,
            n_q,
        ),
        (
            "mpb_quadratic.worst_absolute_column",
            sector.mpb_quadratic.worst_absolute_column,
            n_columns,
        ),
        (
            "mpb_quadratic.worst_relative_column",
            sector.mpb_quadratic.worst_relative_column,
            n_columns,
        ),
    ] {
        if value >= upper {
            return Err(invalid(
                &format!("{path}.{field}"),
                &format!("index < {upper}"),
                &value.to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_provenance(
    provenance: &MldumpExchangeProvenanceV2,
    header: &MldumpHeaderV1,
    n_valence: usize,
    n_core: usize,
    sectors: &[&MldumpExchangeSectorV2; 4],
) -> Result<(), IoError> {
    nonempty("exchange.provenance.source_frame", &provenance.source_frame)?;
    nonempty("exchange.provenance.backend", &provenance.backend)?;
    if provenance.source_frame != MLDUMP_EXCHANGE_SOURCE_FRAME_V2 {
        return Err(invalid(
            "exchange.provenance.source_frame",
            MLDUMP_EXCHANGE_SOURCE_FRAME_V2,
            &provenance.source_frame,
        ));
    }
    if provenance.backend != MLDUMP_EXCHANGE_BACKEND_V2 {
        return Err(invalid(
            "exchange.provenance.backend",
            MLDUMP_EXCHANGE_BACKEND_V2,
            &provenance.backend,
        ));
    }
    positive(
        "exchange.provenance.product_g_max_inv_bohr",
        provenance.product_g_max_inv_bohr,
    )?;
    positive(
        "exchange.provenance.overlap_tolerance",
        provenance.overlap_tolerance,
    )?;
    nonnegative(
        "exchange.provenance.interpolation_pw_cutoff_inv_bohr",
        provenance.interpolation_pw_cutoff_inv_bohr,
    )?;
    if provenance.interpolation_l_max > provenance.coulomb_lexp {
        return Err(invalid(
            "exchange.provenance.interpolation_l_max",
            "<= coulomb_lexp",
            &provenance.interpolation_l_max.to_string(),
        ));
    }
    let scaling = provenance.rank_scaling;
    let n_k = header.mesh.k_points.len();
    if scaling.n_k != n_k
        || scaling.n_valence != n_valence
        || scaling.n_core != n_core
        || scaling.n_candidates == 0
        || scaling.effective_rank == 0
        || scaling.effective_rank > scaling.n_candidates
    {
        return Err(invalid(
            "exchange.provenance.rank_scaling",
            "common n_k/n_valence/n_core and 0 < effective_rank <= n_candidates",
            &format!("{scaling:?}"),
        ));
    }
    match provenance.requested_rank {
        MldumpRequestedRankV2::Exact { n_mu } => {
            if n_mu == 0 || scaling.effective_rank != n_mu {
                return Err(invalid(
                    "exchange.provenance.requested_rank",
                    "positive exact n_mu equal to effective_rank",
                    &format!("{n_mu}"),
                ));
            }
        }
        MldumpRequestedRankV2::Threshold { threshold, n_max } => {
            positive("exchange.provenance.requested_rank.threshold", threshold)?;
            if n_max == 0 || scaling.effective_rank > n_max {
                return Err(invalid(
                    "exchange.provenance.requested_rank.n_max",
                    "positive and >= effective_rank",
                    &n_max.to_string(),
                ));
            }
        }
    }
    let sector_columns = [
        sectors[0].layout.n_columns("exchange.vv.layout")?,
        sectors[1].layout.n_columns("exchange.cv.layout")?,
        sectors[2].layout.n_columns("exchange.vc.layout")?,
        sectors[3].layout.n_columns("exchange.cc.layout")?,
    ];
    if [
        scaling.vv_columns_per_q,
        scaling.cv_columns_per_q,
        scaling.vc_columns_per_q,
        scaling.cc_columns_per_q,
    ] != sector_columns
    {
        return Err(invalid(
            "exchange.provenance.rank_scaling.*_columns_per_q",
            &format!("{sector_columns:?}"),
            &format!(
                "{:?}",
                [
                    scaling.vv_columns_per_q,
                    scaling.cv_columns_per_q,
                    scaling.vc_columns_per_q,
                    scaling.cc_columns_per_q,
                ]
            ),
        ));
    }
    let pooled = sector_columns
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| {
            invalid(
                "exchange.provenance.rank_scaling",
                "size fitting usize",
                "overflow",
            )
        })?;
    let selector_rows = n_k.checked_mul(pooled).ok_or_else(|| {
        invalid(
            "exchange.provenance.rank_scaling.selector_rows",
            "size fitting usize",
            "overflow",
        )
    })?;
    if scaling.pooled_columns_per_q != pooled || scaling.selector_rows != selector_rows {
        return Err(invalid(
            "exchange.provenance.rank_scaling",
            &format!("pooled_columns_per_q={pooled}, selector_rows={selector_rows}"),
            &format!(
                "pooled_columns_per_q={}, selector_rows={}",
                scaling.pooled_columns_per_q, scaling.selector_rows
            ),
        ));
    }

    if provenance.k_weights.len() != n_k {
        return Err(length_mismatch(
            "exchange.provenance.k_weights",
            n_k,
            provenance.k_weights.len(),
        ));
    }
    let mut weight_sum = 0.0;
    for (k, (&stored, point)) in provenance
        .k_weights
        .iter()
        .zip(&header.mesh.k_points)
        .enumerate()
    {
        positive(format!("exchange.provenance.k_weights[{k}]"), stored)?;
        if stored != point.weight {
            return Err(invalid(
                &format!("exchange.provenance.k_weights[{k}]"),
                &format!("exact common-mesh weight {}", point.weight),
                &stored.to_string(),
            ));
        }
        weight_sum += stored;
    }
    require_relation("exchange.provenance.k_weights.sum", weight_sum, 1.0)?;

    if provenance.valence_occupations.len() != n_k {
        return Err(length_mismatch(
            "exchange.provenance.valence_occupations",
            n_k,
            provenance.valence_occupations.len(),
        ));
    }
    for (k, row) in provenance.valence_occupations.iter().enumerate() {
        if row.len() != n_valence {
            return Err(length_mismatch(
                &format!("exchange.provenance.valence_occupations[{k}]"),
                n_valence,
                row.len(),
            ));
        }
        for (band, value) in row.iter().copied().enumerate() {
            fraction(
                &format!("exchange.provenance.valence_occupations[{k}][{band}]"),
                value,
            )?;
        }
    }
    if provenance.core_occupations.len() != n_core {
        return Err(length_mismatch(
            "exchange.provenance.core_occupations",
            n_core,
            provenance.core_occupations.len(),
        ));
    }
    let mut identities = BTreeSet::new();
    for (core, record) in provenance.core_occupations.iter().enumerate() {
        if record.site_index >= header.geometry.sites.len() {
            return Err(invalid(
                &format!("exchange.provenance.core_occupations[{core}].site_index"),
                &format!("index < {}", header.geometry.sites.len()),
                &record.site_index.to_string(),
            ));
        }
        if record.n == 0 {
            return Err(invalid(
                &format!("exchange.provenance.core_occupations[{core}].n"),
                "positive",
                "0",
            ));
        }
        if record.signed_kappa == 0 {
            return Err(invalid(
                &format!("exchange.provenance.core_occupations[{core}].signed_kappa"),
                "nonzero",
                "0",
            ));
        }
        let twice_j = record.signed_kappa.unsigned_abs().saturating_mul(2) - 1;
        if record.twice_mu.unsigned_abs() > twice_j
            || record.twice_mu.rem_euclid(2) != i32::try_from(twice_j % 2).unwrap_or(0)
        {
            return Err(invalid(
                &format!("exchange.provenance.core_occupations[{core}].twice_mu"),
                &format!(
                    "a magnetic projection for signed_kappa={}",
                    record.signed_kappa
                ),
                &record.twice_mu.to_string(),
            ));
        }
        fraction(
            &format!("exchange.provenance.core_occupations[{core}].occupation"),
            record.occupation,
        )?;
        if !identities.insert((
            record.site_index,
            record.n,
            record.signed_kappa,
            record.twice_mu,
        )) {
            return Err(ValidationError::Duplicate {
                path: "exchange.provenance.core_occupations".to_owned(),
                key: format!(
                    "{},{},{},{}",
                    record.site_index, record.n, record.signed_kappa, record.twice_mu
                ),
            }
            .into());
        }
    }
    Ok(())
}

fn write_exchange_v2(file: &File, exchange: &MldumpExchangeV2) -> Result<(), IoError> {
    let root = file.group(GROUP_EXCHANGE)?;
    require_status_present(&root)?;
    for name in ["valence", "core", "total"] {
        read_absent_child(&root, name)?;
    }
    write_str_attr(&root, "total_relation", MLDUMP_EXCHANGE_TOTAL_RELATION_V2)?;
    for (name, value) in [
        ("exchange_vv_hartree", exchange.exchange_vv_hartree),
        ("exchange_cv_hartree", exchange.exchange_cv_hartree),
        ("exchange_cc_hartree", exchange.exchange_cc_hartree),
        ("exchange_total_hartree", exchange.exchange_total_hartree),
        (
            "cross_trace_average_hartree",
            exchange.cross_trace_average_hartree,
        ),
        (
            "cross_trace_mismatch_hartree",
            exchange.cross_trace_mismatch_hartree,
        ),
    ] {
        write_f64_attr(&root, name, value)?;
    }
    let sectors = root.create_group(GROUP_SECTORS)?;
    for (name, sector) in SECTOR_NAMES.into_iter().zip(exchange.sectors()) {
        write_sector(&sectors.create_group(name)?, sector)?;
    }
    write_provenance(&root.create_group(GROUP_PROVENANCE)?, &exchange.provenance)
}

fn write_sector(group: &Group, sector: &MldumpExchangeSectorV2) -> Result<(), IoError> {
    write_str_attr(
        group,
        "occupied_space",
        sector.layout.occupied_space.as_str(),
    )?;
    write_str_attr(group, "target_space", sector.layout.target_space.as_str())?;
    write_usize_attr(group, "n_k", sector.layout.n_k)?;
    write_usize_attr(group, "n_occupied", sector.layout.n_occupied)?;
    write_usize_attr(group, "n_target", sector.layout.n_target)?;
    write_f64_attr(group, "trace_hartree", sector.trace_hartree)?;
    write_f64_attr(group, "maximum_antihermitian", sector.maximum_antihermitian)?;
    let fit = group.create_group(GROUP_FIT_RESIDUAL)?;
    write_f64_attr(&fit, "frobenius", sector.fit_residual.frobenius)?;
    write_f64_attr(&fit, "column_max", sector.fit_residual.column_max)?;
    let quadratic = group.create_group(GROUP_MPB_QUADRATIC)?;
    write_f64_attr(
        &quadratic,
        "maximum_absolute",
        sector.mpb_quadratic.maximum_absolute,
    )?;
    write_f64_attr(
        &quadratic,
        "maximum_relative",
        sector.mpb_quadratic.maximum_relative,
    )?;
    write_usize_attr(
        &quadratic,
        "worst_absolute_q_index",
        sector.mpb_quadratic.worst_absolute_q_index,
    )?;
    write_usize_attr(
        &quadratic,
        "worst_absolute_column",
        sector.mpb_quadratic.worst_absolute_column,
    )?;
    write_usize_attr(
        &quadratic,
        "worst_relative_q_index",
        sector.mpb_quadratic.worst_relative_q_index,
    )?;
    write_usize_attr(
        &quadratic,
        "worst_relative_column",
        sector.mpb_quadratic.worst_relative_column,
    )
}

fn write_provenance(group: &Group, provenance: &MldumpExchangeProvenanceV2) -> Result<(), IoError> {
    write_str_attr(group, "source_frame", &provenance.source_frame)?;
    write_str_attr(group, "backend", &provenance.backend)?;
    write_str_attr(group, "gamma_policy", provenance.gamma_policy.as_str())?;
    write_u32_attr(group, "product_l_max", provenance.product_l_max)?;
    write_f64_attr(
        group,
        "product_g_max_inv_bohr",
        provenance.product_g_max_inv_bohr,
    )?;
    write_f64_attr(group, "overlap_tolerance", provenance.overlap_tolerance)?;
    write_u32_attr(group, "coulomb_lexp", provenance.coulomb_lexp)?;
    write_u32_attr(group, "interpolation_l_max", provenance.interpolation_l_max)?;
    write_f64_attr(
        group,
        "interpolation_pw_cutoff_inv_bohr",
        provenance.interpolation_pw_cutoff_inv_bohr,
    )?;
    write_str_attr(
        group,
        "selector_strategy",
        provenance.selector_strategy.as_str(),
    )?;
    write_str_attr(
        group,
        "selector_engine",
        provenance.selector_engine.as_str(),
    )?;
    write_str_attr(
        group,
        "requested_rank_policy",
        provenance.requested_rank.policy(),
    )?;
    match provenance.requested_rank {
        MldumpRequestedRankV2::Exact { n_mu } => {
            write_usize_attr(group, "requested_rank_n_mu", n_mu)?;
        }
        MldumpRequestedRankV2::Threshold { threshold, n_max } => {
            write_f64_attr(group, "requested_rank_threshold", threshold)?;
            write_usize_attr(group, "requested_rank_n_max", n_max)?;
        }
    }
    write_rank_scaling(
        &group.create_group(GROUP_RANK_SCALING)?,
        provenance.rank_scaling,
    )?;
    write_f64_dataset(
        group,
        "k_weights",
        &[provenance.k_weights.len()],
        &provenance.k_weights,
        &["k"],
    )?;
    let n_k = provenance.valence_occupations.len();
    let n_valence = provenance
        .valence_occupations
        .first()
        .map(Vec::len)
        .unwrap_or(0);
    let valence = provenance
        .valence_occupations
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    write_f64_dataset(
        group,
        "valence_occupations",
        &[n_k, n_valence],
        &valence,
        &["k", "valence"],
    )?;
    let n_core = provenance.core_occupations.len();
    let site = provenance
        .core_occupations
        .iter()
        .map(|record| usize_as_i64("core_site_index", record.site_index))
        .collect::<Result<Vec<_>, _>>()?;
    let n = provenance
        .core_occupations
        .iter()
        .map(|record| i64::from(record.n))
        .collect::<Vec<_>>();
    let signed_kappa = provenance
        .core_occupations
        .iter()
        .map(|record| i64::from(record.signed_kappa))
        .collect::<Vec<_>>();
    let twice_mu = provenance
        .core_occupations
        .iter()
        .map(|record| i64::from(record.twice_mu))
        .collect::<Vec<_>>();
    let occupations = provenance
        .core_occupations
        .iter()
        .map(|record| record.occupation)
        .collect::<Vec<_>>();
    write_i64_dataset(group, "core_site_index", &[n_core], &site, &["core"])?;
    write_i64_dataset(group, "core_n", &[n_core], &n, &["core"])?;
    write_i64_dataset(
        group,
        "core_signed_kappa",
        &[n_core],
        &signed_kappa,
        &["core"],
    )?;
    write_i64_dataset(group, "core_twice_mu", &[n_core], &twice_mu, &["core"])?;
    write_f64_dataset(
        group,
        "core_occupations",
        &[n_core],
        &occupations,
        &["core"],
    )
}

fn write_rank_scaling(group: &Group, scaling: MldumpExchangeRankScalingV2) -> Result<(), IoError> {
    for (name, value) in [
        ("n_k", scaling.n_k),
        ("n_valence", scaling.n_valence),
        ("n_core", scaling.n_core),
        ("n_candidates", scaling.n_candidates),
        ("effective_rank", scaling.effective_rank),
        ("vv_columns_per_q", scaling.vv_columns_per_q),
        ("cv_columns_per_q", scaling.cv_columns_per_q),
        ("vc_columns_per_q", scaling.vc_columns_per_q),
        ("cc_columns_per_q", scaling.cc_columns_per_q),
        ("pooled_columns_per_q", scaling.pooled_columns_per_q),
        ("selector_rows", scaling.selector_rows),
    ] {
        write_usize_attr(group, name, value)?;
    }
    Ok(())
}

fn read_exchange_v2(
    file: &File,
    header: &MldumpHeaderV1,
    payload: &SpinorMldumpV1,
) -> Result<MldumpExchangeV2, IoError> {
    let root = file.group(GROUP_EXCHANGE)?;
    require_exact_attributes(
        &root,
        &[
            ATTR_STATUS,
            "total_relation",
            "exchange_vv_hartree",
            "exchange_cv_hartree",
            "exchange_cc_hartree",
            "exchange_total_hartree",
            "cross_trace_average_hartree",
            "cross_trace_mismatch_hartree",
        ],
    )?;
    require_exact_members(
        &root,
        ["valence", "core", "total", GROUP_SECTORS, GROUP_PROVENANCE],
    )?;
    require_status_present(&root)?;
    require_str_value(&root, "total_relation", MLDUMP_EXCHANGE_TOTAL_RELATION_V2)?;
    for name in ["valence", "core", "total"] {
        read_absent_child(&root, name)?;
    }
    let sectors = root.group(GROUP_SECTORS)?;
    require_exact_members(&sectors, SECTOR_NAMES)?;
    require_exact_attributes(&sectors, &[])?;
    let exchange = MldumpExchangeV2 {
        vv: read_sector(&sectors.group("vv")?)?,
        cv: read_sector(&sectors.group("cv")?)?,
        vc: read_sector(&sectors.group("vc")?)?,
        cc: read_sector(&sectors.group("cc")?)?,
        exchange_vv_hartree: read_f64_attr(
            &root,
            "exchange_vv_hartree",
            "/exchange/@exchange_vv_hartree",
        )?,
        exchange_cv_hartree: read_f64_attr(
            &root,
            "exchange_cv_hartree",
            "/exchange/@exchange_cv_hartree",
        )?,
        exchange_cc_hartree: read_f64_attr(
            &root,
            "exchange_cc_hartree",
            "/exchange/@exchange_cc_hartree",
        )?,
        exchange_total_hartree: read_f64_attr(
            &root,
            "exchange_total_hartree",
            "/exchange/@exchange_total_hartree",
        )?,
        cross_trace_average_hartree: read_f64_attr(
            &root,
            "cross_trace_average_hartree",
            "/exchange/@cross_trace_average_hartree",
        )?,
        cross_trace_mismatch_hartree: read_f64_attr(
            &root,
            "cross_trace_mismatch_hartree",
            "/exchange/@cross_trace_mismatch_hartree",
        )?,
        provenance: read_provenance(&root.group(GROUP_PROVENANCE)?)?,
    };
    exchange.validate(header, payload)?;
    Ok(exchange)
}

fn read_sector(group: &Group) -> Result<MldumpExchangeSectorV2, IoError> {
    require_exact_attributes(
        group,
        &[
            "occupied_space",
            "target_space",
            "n_k",
            "n_occupied",
            "n_target",
            "trace_hartree",
            "maximum_antihermitian",
        ],
    )?;
    require_exact_members(group, [GROUP_FIT_RESIDUAL, GROUP_MPB_QUADRATIC])?;
    let fit = group.group(GROUP_FIT_RESIDUAL)?;
    require_exact_attributes(&fit, &["frobenius", "column_max"])?;
    require_no_payload(&fit)?;
    let quadratic = group.group(GROUP_MPB_QUADRATIC)?;
    require_exact_attributes(
        &quadratic,
        &[
            "maximum_absolute",
            "maximum_relative",
            "worst_absolute_q_index",
            "worst_absolute_column",
            "worst_relative_q_index",
            "worst_relative_column",
        ],
    )?;
    require_no_payload(&quadratic)?;
    Ok(MldumpExchangeSectorV2 {
        layout: MldumpExchangeLayoutV2 {
            occupied_space: MldumpExchangeSpaceV2::parse(
                &format!("{}/@occupied_space", group.name()),
                &read_str_attr(group, "occupied_space")?,
            )?,
            target_space: MldumpExchangeSpaceV2::parse(
                &format!("{}/@target_space", group.name()),
                &read_str_attr(group, "target_space")?,
            )?,
            n_k: read_usize_attr(group, "n_k")?,
            n_occupied: read_usize_attr(group, "n_occupied")?,
            n_target: read_usize_attr(group, "n_target")?,
        },
        trace_hartree: read_f64_attr(
            group,
            "trace_hartree",
            &format!("{}/@trace_hartree", group.name()),
        )?,
        maximum_antihermitian: read_f64_attr(
            group,
            "maximum_antihermitian",
            &format!("{}/@maximum_antihermitian", group.name()),
        )?,
        fit_residual: MldumpExchangeFitResidualV2 {
            frobenius: read_f64_attr(&fit, "frobenius", &format!("{}/@frobenius", fit.name()))?,
            column_max: read_f64_attr(&fit, "column_max", &format!("{}/@column_max", fit.name()))?,
        },
        mpb_quadratic: MldumpExchangeMpbQuadraticV2 {
            maximum_absolute: read_f64_attr(
                &quadratic,
                "maximum_absolute",
                &format!("{}/@maximum_absolute", quadratic.name()),
            )?,
            maximum_relative: read_f64_attr(
                &quadratic,
                "maximum_relative",
                &format!("{}/@maximum_relative", quadratic.name()),
            )?,
            worst_absolute_q_index: read_usize_attr(&quadratic, "worst_absolute_q_index")?,
            worst_absolute_column: read_usize_attr(&quadratic, "worst_absolute_column")?,
            worst_relative_q_index: read_usize_attr(&quadratic, "worst_relative_q_index")?,
            worst_relative_column: read_usize_attr(&quadratic, "worst_relative_column")?,
        },
    })
}

fn read_provenance(group: &Group) -> Result<MldumpExchangeProvenanceV2, IoError> {
    require_exact_members(
        group,
        [
            GROUP_RANK_SCALING,
            "k_weights",
            "valence_occupations",
            "core_site_index",
            "core_n",
            "core_signed_kappa",
            "core_twice_mu",
            "core_occupations",
        ],
    )?;
    let engine_name = read_str_attr(group, "selector_engine")?;
    let rank_policy = read_str_attr(group, "requested_rank_policy")?;
    let mut expected_attributes = vec![
        "source_frame",
        "backend",
        "gamma_policy",
        "product_l_max",
        "product_g_max_inv_bohr",
        "overlap_tolerance",
        "coulomb_lexp",
        "interpolation_l_max",
        "interpolation_pw_cutoff_inv_bohr",
        "selector_strategy",
        "selector_engine",
        "requested_rank_policy",
    ];
    match rank_policy.as_str() {
        "exact" => expected_attributes.push("requested_rank_n_mu"),
        "threshold" => {
            expected_attributes.push("requested_rank_threshold");
            expected_attributes.push("requested_rank_n_max");
        }
        actual => {
            return Err(invalid(
                &format!("{}/@requested_rank_policy", group.name()),
                "exact or threshold",
                actual,
            ));
        }
    }
    require_exact_attributes(group, &expected_attributes)?;
    let selector_engine = match engine_name.as_str() {
        "full_column_pivoted_qr" => MldumpSelectorEngineV2::FullColumnPivotedQr,
        "full_pivoted_cholesky" => MldumpSelectorEngineV2::FullPivotedCholesky,
        actual => {
            return Err(invalid(
                &format!("{}/@selector_engine", group.name()),
                "full_column_pivoted_qr or full_pivoted_cholesky",
                actual,
            ));
        }
    };
    let requested_rank = match rank_policy.as_str() {
        "exact" => MldumpRequestedRankV2::Exact {
            n_mu: read_usize_attr(group, "requested_rank_n_mu")?,
        },
        "threshold" => MldumpRequestedRankV2::Threshold {
            threshold: read_f64_attr(
                group,
                "requested_rank_threshold",
                &format!("{}/@requested_rank_threshold", group.name()),
            )?,
            n_max: read_usize_attr(group, "requested_rank_n_max")?,
        },
        _ => unreachable!("rank policy checked above"),
    };
    let rank_scaling = read_rank_scaling(&group.group(GROUP_RANK_SCALING)?)?;
    let n_k = rank_scaling.n_k;
    let n_valence = rank_scaling.n_valence;
    let n_core = rank_scaling.n_core;
    let k_weights = read_f64_dataset(group, "k_weights", &[n_k], &["k"])?;
    let valence_flat = read_f64_dataset(
        group,
        "valence_occupations",
        &[n_k, n_valence],
        &["k", "valence"],
    )?;
    let valence_occupations = valence_flat
        .chunks_exact(n_valence)
        .map(<[f64]>::to_vec)
        .collect::<Vec<_>>();
    let site = read_i64_dataset(group, "core_site_index", &[n_core], &["core"])?;
    let n = read_i64_dataset(group, "core_n", &[n_core], &["core"])?;
    let signed_kappa = read_i64_dataset(group, "core_signed_kappa", &[n_core], &["core"])?;
    let twice_mu = read_i64_dataset(group, "core_twice_mu", &[n_core], &["core"])?;
    let occupations = read_f64_dataset(group, "core_occupations", &[n_core], &["core"])?;
    require_finite_f64s("/exchange/provenance/k_weights", &k_weights)?;
    require_finite_f64s("/exchange/provenance/valence_occupations", &valence_flat)?;
    require_finite_f64s("/exchange/provenance/core_occupations", &occupations)?;
    let core_occupations = (0..n_core)
        .map(|index| {
            Ok(MldumpCoreOccupationV2 {
                site_index: require_nonnegative_index(
                    &format!("/exchange/provenance/core_site_index[{index}]"),
                    site[index],
                )?,
                n: u32::try_from(require_nonnegative_index(
                    &format!("/exchange/provenance/core_n[{index}]"),
                    n[index],
                )?)
                .map_err(|_| {
                    invalid(
                        &format!("/exchange/provenance/core_n[{index}]"),
                        "u32",
                        &n[index].to_string(),
                    )
                })?,
                signed_kappa: i32::try_from(signed_kappa[index]).map_err(|_| {
                    invalid(
                        &format!("/exchange/provenance/core_signed_kappa[{index}]"),
                        "i32",
                        &signed_kappa[index].to_string(),
                    )
                })?,
                twice_mu: i32::try_from(twice_mu[index]).map_err(|_| {
                    invalid(
                        &format!("/exchange/provenance/core_twice_mu[{index}]"),
                        "i32",
                        &twice_mu[index].to_string(),
                    )
                })?,
                occupation: occupations[index],
            })
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    Ok(MldumpExchangeProvenanceV2 {
        source_frame: read_str_attr(group, "source_frame")?,
        backend: read_str_attr(group, "backend")?,
        gamma_policy: MldumpGammaPolicyV2::parse(
            &format!("{}/@gamma_policy", group.name()),
            &read_str_attr(group, "gamma_policy")?,
        )?,
        product_l_max: read_u32_attr(group, "product_l_max")?,
        product_g_max_inv_bohr: read_f64_attr(
            group,
            "product_g_max_inv_bohr",
            &format!("{}/@product_g_max_inv_bohr", group.name()),
        )?,
        overlap_tolerance: read_f64_attr(
            group,
            "overlap_tolerance",
            &format!("{}/@overlap_tolerance", group.name()),
        )?,
        coulomb_lexp: read_u32_attr(group, "coulomb_lexp")?,
        interpolation_l_max: read_u32_attr(group, "interpolation_l_max")?,
        interpolation_pw_cutoff_inv_bohr: read_f64_attr(
            group,
            "interpolation_pw_cutoff_inv_bohr",
            &format!("{}/@interpolation_pw_cutoff_inv_bohr", group.name()),
        )?,
        selector_strategy: MldumpSelectorStrategyV2::parse(
            &format!("{}/@selector_strategy", group.name()),
            &read_str_attr(group, "selector_strategy")?,
        )?,
        selector_engine,
        requested_rank,
        rank_scaling,
        k_weights,
        valence_occupations,
        core_occupations,
    })
}

fn read_rank_scaling(group: &Group) -> Result<MldumpExchangeRankScalingV2, IoError> {
    let names = [
        "n_k",
        "n_valence",
        "n_core",
        "n_candidates",
        "effective_rank",
        "vv_columns_per_q",
        "cv_columns_per_q",
        "vc_columns_per_q",
        "cc_columns_per_q",
        "pooled_columns_per_q",
        "selector_rows",
    ];
    require_exact_attributes(group, &names)?;
    require_no_payload(group)?;
    Ok(MldumpExchangeRankScalingV2 {
        n_k: read_usize_attr(group, "n_k")?,
        n_valence: read_usize_attr(group, "n_valence")?,
        n_core: read_usize_attr(group, "n_core")?,
        n_candidates: read_usize_attr(group, "n_candidates")?,
        effective_rank: read_usize_attr(group, "effective_rank")?,
        vv_columns_per_q: read_usize_attr(group, "vv_columns_per_q")?,
        cv_columns_per_q: read_usize_attr(group, "cv_columns_per_q")?,
        vc_columns_per_q: read_usize_attr(group, "vc_columns_per_q")?,
        cc_columns_per_q: read_usize_attr(group, "cc_columns_per_q")?,
        pooled_columns_per_q: read_usize_attr(group, "pooled_columns_per_q")?,
        selector_rows: read_usize_attr(group, "selector_rows")?,
    })
}

fn require_exact_attributes(object: &Location, expected: &[&str]) -> Result<(), IoError> {
    let observed = object.attr_names()?.into_iter().collect::<BTreeSet<_>>();
    let expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    if observed == expected {
        Ok(())
    } else {
        Err(invalid(
            &format!("{}/attributes", object.name()),
            &expected.into_iter().collect::<Vec<_>>().join(","),
            &observed.into_iter().collect::<Vec<_>>().join(","),
        ))
    }
}

fn require_str_value(group: &Group, name: &str, expected: &str) -> Result<(), IoError> {
    let actual = read_str_attr(group, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(invalid(
            &format!("{}/@{name}", group.name()),
            expected,
            &actual,
        ))
    }
}

fn write_usize_attr(object: &Location, name: &str, value: usize) -> Result<(), IoError> {
    write_i64_attr(object, name, usize_as_i64(name, value)?)
}

fn read_usize_attr(object: &Location, name: &str) -> Result<usize, IoError> {
    let path = format!("{}/@{name}", object.name());
    let value = super::read_i64_attr(object, name, &path)?;
    require_nonnegative_index(&path, value)
}

fn write_u32_attr(object: &Location, name: &str, value: u32) -> Result<(), IoError> {
    write_i64_attr(object, name, i64::from(value))
}

fn read_u32_attr(object: &Location, name: &str) -> Result<u32, IoError> {
    let value = read_usize_attr(object, name)?;
    u32::try_from(value).map_err(|_| invalid(name, "u32", &value.to_string()))
}

fn require_relation(path: &str, actual: f64, expected: f64) -> Result<(), IoError> {
    if approx_eq(actual, expected) {
        Ok(())
    } else {
        Err(invalid(path, &expected.to_string(), &actual.to_string()))
    }
}

fn nonnegative(path: impl Into<String>, value: f64) -> Result<(), IoError> {
    let path = path.into();
    finite(path.clone(), value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(invalid(&path, "nonnegative", &value.to_string()))
    }
}

fn fraction(path: &str, value: f64) -> Result<(), IoError> {
    finite(path, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid(path, "[0, 1]", &value.to_string()))
    }
}

fn length_mismatch(path: &str, expected: usize, actual: usize) -> IoError {
    ValidationError::LengthMismatch {
        path: path.to_owned(),
        expected,
        actual,
    }
    .into()
}

fn invalid(path: &str, expected: &str, actual: &str) -> IoError {
    ValidationError::InvalidValue {
        path: path.to_owned(),
        expected: expected.to_owned(),
        actual: actual.to_owned(),
    }
    .into()
}
