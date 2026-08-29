//! Representation-neutral THC interpolation and finite Coulomb body payload.

use std::collections::BTreeSet;

use hdf5_metno::Group;

use super::scalar_orbitals::ScalarOrbitalsV1;
use super::scalar_products::{MLDUMP_PAIR_ORDER_K_LEFT_RIGHT, ScalarProductsV1};
use super::spinor_orbitals::SpinorOrbitalsV1;
use super::spinor_products::SpinorProductsV1;
use super::{
    GROUP_COULOMB, GROUP_THC, MldumpHeaderV1, MldumpStatus, PREFIX_GAMMA, PREFIX_PARENT_GRID,
    PREFIX_Q, approx_eq, collect_padded_groups, complex_len, create_padded_group,
    fractional_to_cartesian, padded_child, read_f64_attr, read_f64_dataset, read_i64_attr,
    read_i64_dataset, read_str_attr, read_usize_attr, reopen_present_group, require_dataset_names,
    require_exact_members, require_finite_f64s, require_flat_len, require_group_names, require_len,
    require_no_payload, require_nonnegative_index, require_status_present, require_str_array_attr,
    require_str_attr, require_str_attr_if_present, usize_as_i64, write_absent_group,
    write_f64_attr, write_f64_dataset, write_i64_attr, write_i64_dataset, write_status,
    write_str_array_attr, write_str_attr,
};
use crate::error::{IoError, ValidationError, nonempty};

/// Complex pair stored as a final length-2 `re_im` axis: `[re, im]`.
pub type ComplexF64V1 = [f64; 2];

/// AllQL2 selection strategy tag.
pub const MLDUMP_THC_STRATEGY_ALL_QL2: &str = "AllQL2";
/// Full column-pivoted QR engine string.
pub const MLDUMP_THC_ENGINE_QRCP: &str = "full_column_pivoted_qr";
/// Full pivoted Cholesky engine string.
pub const MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY: &str = "full_pivoted_cholesky";
/// Parent-grid muffin-tin region kind.
pub const MLDUMP_PARENT_REGION_MUFFIN_TIN: i64 = 0;
/// Parent-grid interstitial region kind.
pub const MLDUMP_PARENT_REGION_INTERSTITIAL: i64 = 1;
/// Sentinel stored in site/radial tables for interstitial points.
pub const MLDUMP_INTERSTITIAL_SENTINEL: i64 = -1;

const RESIDUAL_LABELS: [&str; 2] = ["frobenius", "column_max"];

/// Owned scalar payload: orbitals, products, THC, and Coulomb. MPB is not a field.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarMldumpV1 {
    pub orbitals: ScalarOrbitalsV1,
    pub products: ScalarProductsV1,
    pub thc: MldumpThcV1,
    pub coulomb: MldumpCoulombV1,
}

/// Owned spinor payload: orbitals, products, THC, and Coulomb. MPB is not a field.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorMldumpV1 {
    pub orbitals: SpinorOrbitalsV1,
    pub products: SpinorProductsV1,
    pub thc: MldumpThcV1,
    pub coulomb: MldumpCoulombV1,
}

/// Shared `/thc` parent grid, engine, and selection for a streaming session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpThcBeginV1<'a> {
    pub parent_grid: MldumpThcParentGridRefV1<'a>,
    pub strategy: &'a str,
    pub engine: &'a str,
    pub requested_rank: usize,
    pub effective_rank: usize,
    pub n_candidates: usize,
    pub selection: MldumpThcSelectionRefV1<'a>,
}

/// Shared parent grid stored once, including zero-weight rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpThcParentGridRefV1<'a> {
    pub n_points: usize,
    pub coordinates: &'a [f64],
    pub weights: &'a [f64],
    pub region_kind: &'a [i64],
    pub site_index: &'a [i64],
    pub radial_index: &'a [i64],
    pub provenance: &'a str,
}

/// Distinct QRCP/Cholesky pivot order and sorted auxiliary-layout points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpThcSelectionRefV1<'a> {
    pub pivots: &'a [i64],
    pub points: &'a [i64],
}

/// One positional $q$ interpolation record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpThcQRecordRefV1<'a> {
    pub q_index: usize,
    pub aux_dimension: usize,
    pub layout_provenance: &'a str,
    pub zeta: &'a [f64],
    pub residual_l2_all_frobenius: f64,
    pub residual_l2_all_column_max: f64,
    pub vertices: MldumpThcVertexTableRefV1<'a>,
}

/// Semantic vertex tables in pair-column layout order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpThcVertexTableRefV1<'a> {
    pub n_vertex: usize,
    pub column: &'a [i64],
    pub k_left_right: &'a [i64],
    pub coefficients: &'a [f64],
}

/// Owned THC section.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpThcV1 {
    pub parent_grid: MldumpThcParentGridV1,
    pub strategy: String,
    pub engine: String,
    pub requested_rank: usize,
    pub effective_rank: usize,
    pub n_candidates: usize,
    pub selection: MldumpThcSelectionV1,
    pub q_records: Vec<MldumpThcQRecordV1>,
}

/// Owned parent grid.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpThcParentGridV1 {
    pub coordinates: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
    pub region_kind: Vec<i64>,
    pub site_index: Vec<i64>,
    pub radial_index: Vec<i64>,
    pub provenance: String,
}

/// Owned selection: `pivots` is engine rank order; `points` is layout order.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpThcSelectionV1 {
    pub pivots: Vec<i64>,
    pub points: Vec<i64>,
}

/// Owned per-$q$ $\zeta$, residuals, and vertices.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpThcQRecordV1 {
    pub q_index: usize,
    pub aux_dimension: usize,
    pub layout_provenance: String,
    pub zeta: Vec<f64>,
    pub residual: MldumpThcResidualV1,
    pub vertices: Vec<MldumpThcVertexV1>,
}

/// Weighted L2 residual pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpThcResidualV1 {
    pub frobenius: f64,
    pub column_max: f64,
}

/// One semantic pair vertex.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpThcVertexV1 {
    pub column: i64,
    pub k: i64,
    pub left: i64,
    pub right: i64,
    pub coefficients: Vec<f64>,
}

/// Shared `/coulomb` request attributes for a streaming session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpCoulombBeginV1 {
    pub lexp: u32,
    pub interpolation_l_max: u32,
    pub interpolation_pw_cutoff: f64,
}

/// Finite Hermitian body at one $q$, with optional Gamma metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpCoulombQRecordRefV1<'a> {
    pub q_index: usize,
    pub aux_dimension: usize,
    pub layout_provenance: &'a str,
    pub body: &'a [f64],
    pub gamma: Option<MldumpCoulombGammaRefV1<'a>>,
}

/// Finite Gamma-head metadata. The singular $4\pi/|q|^2$ head is never stored in $V$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpCoulombGammaRefV1<'a> {
    pub spherical_average_subtracted: bool,
    pub head_prefactor: f64,
    pub constant_coefficients: &'a [f64],
}

/// Owned Coulomb section.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpCoulombV1 {
    pub lexp: u32,
    pub interpolation_l_max: u32,
    pub interpolation_pw_cutoff: f64,
    pub q_records: Vec<MldumpCoulombQRecordV1>,
}

/// Owned per-$q$ Coulomb body.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpCoulombQRecordV1 {
    pub q_index: usize,
    pub aux_dimension: usize,
    pub layout_provenance: String,
    pub body: Vec<f64>,
    pub gamma: Option<MldumpCoulombGammaV1>,
}

/// Owned finite Gamma-head metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpCoulombGammaV1 {
    pub spherical_average_subtracted: bool,
    pub head_prefactor: f64,
    pub constant_coefficients: Vec<f64>,
}

/// Small orbital counts retained for writer `finish` and the owned reader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OrbitalAlignmentSummary {
    pub spin_count: usize,
    pub n_k: usize,
    pub band_window_count: usize,
}

/// Small product $q$ bindings. Cartesian and global labels only; no pair arrays.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ProductAlignmentSummary {
    pub n_k: usize,
    pub n_orb: usize,
    pub pair_order: &'static str,
    pub q_records: Vec<ProductQAlignment>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ProductQAlignment {
    pub q_index: usize,
    pub transfer_cartesian: [f64; 3],
    pub global_transfer: [i32; 3],
}

/// THC $q$ identity, auxiliary dimension, and provenance. No vertex tables.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ThcAlignmentSummary {
    pub q_records: Vec<ThcQAlignment>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ThcQAlignment {
    pub q_index: usize,
    pub aux_dimension: usize,
    pub layout_provenance: String,
    pub n_vertex: usize,
}

/// Coulomb $q$ identity, auxiliary provenance, and Gamma presence. No $V$ body.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoulombAlignmentSummary {
    pub q_records: Vec<CoulombQAlignment>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoulombQAlignment {
    pub q_index: usize,
    pub aux_dimension: usize,
    pub layout_provenance: String,
    pub gamma_present: bool,
}

impl OrbitalAlignmentSummary {
    pub(crate) fn new(spin_count: usize, n_k: usize, band_window_count: usize) -> Self {
        Self {
            spin_count,
            n_k,
            band_window_count,
        }
    }

    pub(crate) fn from_owned(orbitals: &ScalarOrbitalsV1) -> Self {
        Self {
            spin_count: orbitals.spin_count,
            n_k: orbitals
                .spins
                .first()
                .expect("scalar orbitals spin_count is 2")
                .k_points
                .len(),
            band_window_count: orbitals.band_window_count,
        }
    }
}

impl ProductAlignmentSummary {
    pub(crate) fn new(n_k: usize, n_orb: usize) -> Self {
        Self {
            n_k,
            n_orb,
            pair_order: MLDUMP_PAIR_ORDER_K_LEFT_RIGHT,
            q_records: Vec::new(),
        }
    }

    pub(crate) fn push_q_binding(
        &mut self,
        q_index: usize,
        transfer_cartesian: [f64; 3],
        global_transfer: [i32; 3],
    ) {
        self.q_records.push(ProductQAlignment {
            q_index,
            transfer_cartesian,
            global_transfer,
        });
    }

    pub(crate) fn from_owned(products: &ScalarProductsV1) -> Self {
        Self::from_q_bindings(
            products.n_k,
            products.n_orb,
            products
                .q_records
                .iter()
                .map(|record| ProductQAlignment {
                    q_index: record.q_index,
                    transfer_cartesian: record.transfer_cartesian,
                    global_transfer: record.global_transfer,
                })
                .collect(),
        )
    }

    pub(crate) fn from_q_bindings(
        n_k: usize,
        n_orb: usize,
        q_records: Vec<ProductQAlignment>,
    ) -> Self {
        Self {
            n_k,
            n_orb,
            pair_order: MLDUMP_PAIR_ORDER_K_LEFT_RIGHT,
            q_records,
        }
    }
}

impl ThcAlignmentSummary {
    pub(crate) fn new() -> Self {
        Self {
            q_records: Vec::new(),
        }
    }

    pub(crate) fn push_q(&mut self, record: &MldumpThcQRecordRefV1<'_>) {
        self.q_records.push(ThcQAlignment {
            q_index: record.q_index,
            aux_dimension: record.aux_dimension,
            layout_provenance: record.layout_provenance.to_owned(),
            n_vertex: record.vertices.n_vertex,
        });
    }

    pub(crate) fn from_owned(thc: &MldumpThcV1) -> Self {
        Self {
            q_records: thc
                .q_records
                .iter()
                .map(|record| ThcQAlignment {
                    q_index: record.q_index,
                    aux_dimension: record.aux_dimension,
                    layout_provenance: record.layout_provenance.clone(),
                    n_vertex: record.vertices.len(),
                })
                .collect(),
        }
    }
}

impl CoulombAlignmentSummary {
    pub(crate) fn new() -> Self {
        Self {
            q_records: Vec::new(),
        }
    }

    pub(crate) fn push_q(&mut self, record: &MldumpCoulombQRecordRefV1<'_>) {
        self.q_records.push(CoulombQAlignment {
            q_index: record.q_index,
            aux_dimension: record.aux_dimension,
            layout_provenance: record.layout_provenance.to_owned(),
            gamma_present: record.gamma.is_some(),
        });
    }

    pub(crate) fn from_owned(coulomb: &MldumpCoulombV1) -> Self {
        Self {
            q_records: coulomb
                .q_records
                .iter()
                .map(|record| CoulombQAlignment {
                    q_index: record.q_index,
                    aux_dimension: record.aux_dimension,
                    layout_provenance: record.layout_provenance.clone(),
                    gamma_present: record.gamma.is_some(),
                })
                .collect(),
        }
    }
}

pub(crate) fn begin_mldump_thc(
    file: &Group,
    header: &MldumpHeaderV1,
    thc: &MldumpThcBeginV1<'_>,
    representation: &str,
) -> Result<(), IoError> {
    validate_thc_begin(header, thc)?;
    let group = reopen_present_group(file, GROUP_THC)?;
    write_str_attr(&group, "representation", representation)?;
    write_str_attr(&group, "strategy", thc.strategy)?;
    write_str_attr(&group, "engine", thc.engine)?;
    write_i64_attr(
        &group,
        "requested_rank",
        usize_as_i64("/thc/@requested_rank", thc.requested_rank)?,
    )?;
    write_i64_attr(
        &group,
        "effective_rank",
        usize_as_i64("/thc/@effective_rank", thc.effective_rank)?,
    )?;
    write_i64_attr(
        &group,
        "n_candidates",
        usize_as_i64("/thc/@n_candidates", thc.n_candidates)?,
    )?;
    write_parent_grid(&group, &thc.parent_grid)?;
    write_i64_dataset(
        &group,
        "pivots",
        &[thc.effective_rank],
        thc.selection.pivots,
        &["rank_order"],
    )?;
    write_i64_dataset(
        &group,
        "points",
        &[thc.effective_rank],
        thc.selection.points,
        &["aux"],
    )?;
    Ok(())
}

pub(crate) fn write_mldump_thc_q(
    file: &Group,
    q: usize,
    n_parent: usize,
    effective_rank: usize,
    n_k: usize,
    n_orb: usize,
    record: &MldumpThcQRecordRefV1<'_>,
) -> Result<(), IoError> {
    validate_thc_q_ref(q, n_parent, effective_rank, n_k, n_orb, record)?;
    let group = file.group(GROUP_THC)?;
    write_thc_q(&group, n_parent, record)
}

pub(crate) fn begin_mldump_coulomb(
    file: &Group,
    coulomb: &MldumpCoulombBeginV1,
    representation: &str,
) -> Result<(), IoError> {
    validate_coulomb_begin(coulomb)?;
    let group = reopen_present_group(file, GROUP_COULOMB)?;
    write_str_attr(&group, "representation", representation)?;
    write_i64_attr(&group, "lexp", i64::from(coulomb.lexp))?;
    write_i64_attr(
        &group,
        "interpolation_l_max",
        i64::from(coulomb.interpolation_l_max),
    )?;
    write_f64_attr(
        &group,
        "interpolation_pw_cutoff",
        coulomb.interpolation_pw_cutoff,
    )?;
    Ok(())
}

pub(crate) fn write_mldump_coulomb_q(
    file: &Group,
    q: usize,
    record: &MldumpCoulombQRecordRefV1<'_>,
) -> Result<(), IoError> {
    validate_coulomb_q_payload(
        q,
        record.q_index,
        record.aux_dimension,
        record.layout_provenance,
        record.body,
        record
            .gamma
            .map(|gamma| (gamma.head_prefactor, gamma.constant_coefficients)),
    )?;
    let group = file.group(GROUP_COULOMB)?;
    write_coulomb_q(&group, record)
}

pub(crate) fn read_mldump_thc(
    file: &Group,
    header: &MldumpHeaderV1,
    representation: &str,
) -> Result<MldumpThcV1, IoError> {
    let group = file.group(GROUP_THC)?;
    require_status_present(&group)?;
    require_str_attr_if_present(&group, "representation", representation)?;
    require_str_attr(&group, "strategy", MLDUMP_THC_STRATEGY_ALL_QL2)?;
    let engine = read_str_attr(&group, "engine")?;
    require_known_engine("/thc/@engine", &engine)?;
    let requested_rank = read_usize_attr(&group, "requested_rank", "/thc/@requested_rank")?;
    let effective_rank = read_usize_attr(&group, "effective_rank", "/thc/@effective_rank")?;
    let n_candidates = read_usize_attr(&group, "n_candidates", "/thc/@n_candidates")?;
    if effective_rank == 0 || effective_rank > requested_rank {
        return Err(ValidationError::InvalidValue {
            path: "/thc/@effective_rank".to_owned(),
            expected: format!("1..={requested_rank}"),
            actual: effective_rank.to_string(),
        }
        .into());
    }
    require_dataset_names(&group, &["pivots", "points"])?;
    let parent_grid = read_parent_grid(&group.group(PREFIX_PARENT_GRID)?)?;
    let pivots = read_i64_dataset(&group, "pivots", &[effective_rank], &["rank_order"])?;
    let points = read_i64_dataset(&group, "points", &[effective_rank], &["aux"])?;
    let q_groups = collect_padded_groups(&group, PREFIX_Q)?;
    require_len("/thc/q_*", header.mesh.q_entries.len(), q_groups.len())?;
    let mut members = vec![
        PREFIX_PARENT_GRID.to_owned(),
        "pivots".to_owned(),
        "points".to_owned(),
    ];
    members.extend((0..header.mesh.q_entries.len()).map(|q| padded_child(PREFIX_Q, q)));
    require_exact_members(&group, members)?;
    let mut q_records = Vec::with_capacity(q_groups.len());
    for (q, q_group) in q_groups.iter().enumerate() {
        q_records.push(read_thc_q(
            q_group,
            q,
            parent_grid.weights.len(),
            effective_rank,
        )?);
    }
    let thc = MldumpThcV1 {
        parent_grid,
        strategy: MLDUMP_THC_STRATEGY_ALL_QL2.to_owned(),
        engine,
        requested_rank,
        effective_rank,
        n_candidates,
        selection: MldumpThcSelectionV1 { pivots, points },
        q_records,
    };
    validate_thc_owned(header, &thc)?;
    Ok(thc)
}

pub(crate) fn read_mldump_coulomb(
    file: &Group,
    header: &MldumpHeaderV1,
    representation: &str,
) -> Result<MldumpCoulombV1, IoError> {
    let group = file.group(GROUP_COULOMB)?;
    require_status_present(&group)?;
    require_str_attr_if_present(&group, "representation", representation)?;
    let lexp = require_u32_attr(&group, "lexp", "/coulomb/@lexp")?;
    let interpolation_l_max = require_u32_attr(
        &group,
        "interpolation_l_max",
        "/coulomb/@interpolation_l_max",
    )?;
    let interpolation_pw_cutoff = read_f64_attr(
        &group,
        "interpolation_pw_cutoff",
        "/coulomb/@interpolation_pw_cutoff",
    )?;
    let q_groups = collect_padded_groups(&group, PREFIX_Q)?;
    require_len("/coulomb/q_*", header.mesh.q_entries.len(), q_groups.len())?;
    require_exact_members(
        &group,
        (0..header.mesh.q_entries.len()).map(|q| padded_child(PREFIX_Q, q)),
    )?;
    let mut q_records = Vec::with_capacity(q_groups.len());
    for (q, q_group) in q_groups.iter().enumerate() {
        q_records.push(read_coulomb_q(q_group, q)?);
    }
    let coulomb = MldumpCoulombV1 {
        lexp,
        interpolation_l_max,
        interpolation_pw_cutoff,
        q_records,
    };
    validate_coulomb_owned(header, &coulomb)?;
    Ok(coulomb)
}

/// Cross-section scalar alignment used by writer `finish` and the owned reader.
///
/// Summaries hold counts, $q$ Cartesian/global labels, and provenance strings.
/// They do not hold eigenvector, $\zeta$, vertex tables, or $V$ arrays.
/// Semantic THC vertex identity is checked $q$-locally on write and from the
/// owned payload on read.
pub(crate) fn validate_scalar_alignment(
    header: &MldumpHeaderV1,
    orbitals: &OrbitalAlignmentSummary,
    products: &ProductAlignmentSummary,
    thc: &ThcAlignmentSummary,
    coulomb: &CoulombAlignmentSummary,
) -> Result<(), IoError> {
    let n_k = header.mesh.k_points.len();
    require_len("orbitals.spin_count", 2, orbitals.spin_count)?;
    require_len("orbitals.n_k", n_k, orbitals.n_k)?;
    validate_payload_alignment(header, orbitals.band_window_count, products, thc, coulomb)
}

pub(crate) fn validate_payload_alignment(
    header: &MldumpHeaderV1,
    band_window_count: usize,
    products: &ProductAlignmentSummary,
    thc: &ThcAlignmentSummary,
    coulomb: &CoulombAlignmentSummary,
) -> Result<(), IoError> {
    let n_k = header.mesh.k_points.len();
    let n_q = header.mesh.q_entries.len();
    require_len("products.n_k", n_k, products.n_k)?;
    require_len("products.n_orb", band_window_count, products.n_orb)?;
    if products.pair_order != MLDUMP_PAIR_ORDER_K_LEFT_RIGHT {
        return Err(ValidationError::InvalidValue {
            path: "products.pair_order".to_owned(),
            expected: MLDUMP_PAIR_ORDER_K_LEFT_RIGHT.to_owned(),
            actual: products.pair_order.to_owned(),
        }
        .into());
    }
    require_len("products.q_records", n_q, products.q_records.len())?;
    require_len("thc.q_records", n_q, thc.q_records.len())?;
    require_len("coulomb.q_records", n_q, coulomb.q_records.len())?;
    for q in 0..n_q {
        let mesh_q = &header.mesh.q_entries[q];
        let product_q = &products.q_records[q];
        let thc_q = &thc.q_records[q];
        let coulomb_q = &coulomb.q_records[q];
        if product_q.q_index != q {
            return Err(ValidationError::InvalidValue {
                path: format!("products.q_records[{q}].q_index"),
                expected: q.to_string(),
                actual: product_q.q_index.to_string(),
            }
            .into());
        }
        if thc_q.q_index != q {
            return Err(ValidationError::InvalidValue {
                path: format!("thc.q_records[{q}].q_index"),
                expected: q.to_string(),
                actual: thc_q.q_index.to_string(),
            }
            .into());
        }
        if coulomb_q.q_index != q {
            return Err(ValidationError::InvalidValue {
                path: format!("coulomb.q_records[{q}].q_index"),
                expected: q.to_string(),
                actual: coulomb_q.q_index.to_string(),
            }
            .into());
        }
        if product_q.global_transfer != mesh_q.global_umklapp {
            return Err(ValidationError::InvalidValue {
                path: format!("products.q_records[{q}].global_transfer"),
                expected: format!("{:?}", mesh_q.global_umklapp),
                actual: format!("{:?}", product_q.global_transfer),
            }
            .into());
        }
        let expected_cart = fractional_to_cartesian(
            header.geometry.reciprocal_basis_inv_bohr,
            mesh_q.canonical_fractional,
        );
        for (axis, (stored, expected)) in product_q
            .transfer_cartesian
            .iter()
            .zip(expected_cart)
            .enumerate()
        {
            if !approx_eq(*stored, expected) {
                return Err(ValidationError::InvalidValue {
                    path: format!("products.q_records[{q}].transfer_cartesian[{axis}]"),
                    expected: format!("canonical q · b = {expected}"),
                    actual: stored.to_string(),
                }
                .into());
            }
        }
        if thc_q.aux_dimension != coulomb_q.aux_dimension {
            return Err(ValidationError::LayoutMismatch {
                path: format!("coulomb.q_records[{q}].aux_dimension"),
                reference: format!("thc.q_records[{q}].aux_dimension={}", thc_q.aux_dimension),
            }
            .into());
        }
        if thc_q.layout_provenance != coulomb_q.layout_provenance {
            return Err(ValidationError::InvalidValue {
                path: format!("coulomb.q_records[{q}].layout_provenance"),
                expected: thc_q.layout_provenance.clone(),
                actual: coulomb_q.layout_provenance.clone(),
            }
            .into());
        }
        if thc_q.n_vertex == 0 {
            return Err(ValidationError::Empty {
                path: format!("thc.q_records[{q}].vertices"),
            }
            .into());
        }
        let gamma_q = mesh_q
            .canonical_fractional
            .iter()
            .all(|component| approx_eq(*component, 0.0));
        if coulomb_q.gamma_present && !gamma_q {
            return Err(ValidationError::InvalidValue {
                path: format!("coulomb.q_records[{q}].gamma"),
                expected: "present only when canonical q is the zero vector".to_owned(),
                actual: format!("canonical={:?}", mesh_q.canonical_fractional),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_thc_begin(header: &MldumpHeaderV1, thc: &MldumpThcBeginV1<'_>) -> Result<(), IoError> {
    validate_thc_shared(
        header,
        ThcSharedView {
            strategy: thc.strategy,
            engine: thc.engine,
            requested_rank: thc.requested_rank,
            effective_rank: thc.effective_rank,
            n_candidates: thc.n_candidates,
            parent: &thc.parent_grid,
            pivots: thc.selection.pivots,
            points: thc.selection.points,
        },
    )
}

fn validate_thc_owned(header: &MldumpHeaderV1, thc: &MldumpThcV1) -> Result<(), IoError> {
    let coordinates: Vec<f64> = thc
        .parent_grid
        .coordinates
        .iter()
        .flat_map(|point| point.iter().copied())
        .collect();
    let parent = MldumpThcParentGridRefV1 {
        n_points: thc.parent_grid.weights.len(),
        coordinates: &coordinates,
        weights: &thc.parent_grid.weights,
        region_kind: &thc.parent_grid.region_kind,
        site_index: &thc.parent_grid.site_index,
        radial_index: &thc.parent_grid.radial_index,
        provenance: &thc.parent_grid.provenance,
    };
    validate_thc_shared(
        header,
        ThcSharedView {
            strategy: &thc.strategy,
            engine: &thc.engine,
            requested_rank: thc.requested_rank,
            effective_rank: thc.effective_rank,
            n_candidates: thc.n_candidates,
            parent: &parent,
            pivots: &thc.selection.pivots,
            points: &thc.selection.points,
        },
    )?;
    require_len(
        "thc.q_records",
        header.mesh.q_entries.len(),
        thc.q_records.len(),
    )?;
    for (q, record) in thc.q_records.iter().enumerate() {
        validate_thc_q_owned(q, parent.n_points, thc.effective_rank, record)?;
    }
    Ok(())
}

struct ThcSharedView<'a> {
    strategy: &'a str,
    engine: &'a str,
    requested_rank: usize,
    effective_rank: usize,
    n_candidates: usize,
    parent: &'a MldumpThcParentGridRefV1<'a>,
    pivots: &'a [i64],
    points: &'a [i64],
}

fn validate_thc_shared(header: &MldumpHeaderV1, thc: ThcSharedView<'_>) -> Result<(), IoError> {
    if thc.strategy != MLDUMP_THC_STRATEGY_ALL_QL2 {
        return Err(ValidationError::InvalidValue {
            path: "thc.strategy".to_owned(),
            expected: MLDUMP_THC_STRATEGY_ALL_QL2.to_owned(),
            actual: thc.strategy.to_owned(),
        }
        .into());
    }
    require_known_engine("thc.engine", thc.engine)?;
    if thc.effective_rank == 0 || thc.effective_rank > thc.requested_rank {
        return Err(ValidationError::InvalidValue {
            path: "thc.effective_rank".to_owned(),
            expected: format!("1..={}", thc.requested_rank),
            actual: thc.effective_rank.to_string(),
        }
        .into());
    }
    if thc.n_candidates == 0 || thc.n_candidates > thc.parent.n_points {
        return Err(ValidationError::InvalidValue {
            path: "thc.n_candidates".to_owned(),
            expected: format!("1..={}", thc.parent.n_points),
            actual: thc.n_candidates.to_string(),
        }
        .into());
    }
    validate_parent_grid_ref(header, thc.parent)?;
    require_len("thc.selection.pivots", thc.effective_rank, thc.pivots.len())?;
    require_len("thc.selection.points", thc.effective_rank, thc.points.len())?;
    validate_selected_parent_indices("thc.selection.pivots", thc.pivots, thc.parent.weights)?;
    validate_selected_parent_indices("thc.selection.points", thc.points, thc.parent.weights)?;
    require_identical_selection_sets(thc.pivots, thc.points)?;
    Ok(())
}

fn validate_parent_grid_ref(
    header: &MldumpHeaderV1,
    grid: &MldumpThcParentGridRefV1<'_>,
) -> Result<(), IoError> {
    if grid.n_points == 0 {
        return Err(ValidationError::Empty {
            path: "thc.parent_grid".to_owned(),
        }
        .into());
    }
    nonempty("thc.parent_grid.provenance", grid.provenance)?;
    require_flat_len(
        "thc.parent_grid.coordinates",
        &[grid.n_points, 3],
        grid.coordinates.len(),
    )?;
    require_len("thc.parent_grid.weights", grid.n_points, grid.weights.len())?;
    require_len(
        "thc.parent_grid.region_kind",
        grid.n_points,
        grid.region_kind.len(),
    )?;
    require_len(
        "thc.parent_grid.site_index",
        grid.n_points,
        grid.site_index.len(),
    )?;
    require_len(
        "thc.parent_grid.radial_index",
        grid.n_points,
        grid.radial_index.len(),
    )?;
    require_finite_f64s("thc.parent_grid.coordinates", grid.coordinates)?;
    require_finite_f64s("thc.parent_grid.weights", grid.weights)?;
    for (point, weight) in grid.weights.iter().enumerate() {
        if *weight < 0.0 {
            return Err(ValidationError::InvalidValue {
                path: format!("thc.parent_grid.weights[{point}]"),
                expected: "nonnegative".to_owned(),
                actual: weight.to_string(),
            }
            .into());
        }
    }
    let n_sites = header.geometry.sites.len();
    for point in 0..grid.n_points {
        match grid.region_kind[point] {
            MLDUMP_PARENT_REGION_MUFFIN_TIN => {
                let site = require_nonnegative_index(
                    &format!("thc.parent_grid.site_index[{point}]"),
                    grid.site_index[point],
                )?;
                let radial = require_nonnegative_index(
                    &format!("thc.parent_grid.radial_index[{point}]"),
                    grid.radial_index[point],
                )?;
                if site >= n_sites {
                    return Err(ValidationError::InvalidValue {
                        path: format!("thc.parent_grid.site_index[{point}]"),
                        expected: format!("index < {n_sites}"),
                        actual: site.to_string(),
                    }
                    .into());
                }
                if radial >= header.geometry.sites[site].radial_mesh.point_count {
                    return Err(ValidationError::InvalidValue {
                        path: format!("thc.parent_grid.radial_index[{point}]"),
                        expected: format!(
                            "index < {}",
                            header.geometry.sites[site].radial_mesh.point_count
                        ),
                        actual: radial.to_string(),
                    }
                    .into());
                }
            }
            MLDUMP_PARENT_REGION_INTERSTITIAL => {
                if grid.site_index[point] != MLDUMP_INTERSTITIAL_SENTINEL
                    || grid.radial_index[point] != MLDUMP_INTERSTITIAL_SENTINEL
                {
                    return Err(ValidationError::InvalidValue {
                        path: format!("thc.parent_grid.site_index[{point}]"),
                        expected: format!("interstitial sentinel {MLDUMP_INTERSTITIAL_SENTINEL}"),
                        actual: format!(
                            "site={} radial={}",
                            grid.site_index[point], grid.radial_index[point]
                        ),
                    }
                    .into());
                }
            }
            other => {
                return Err(ValidationError::InvalidValue {
                    path: format!("thc.parent_grid.region_kind[{point}]"),
                    expected: "0 muffin_tin or 1 interstitial".to_owned(),
                    actual: other.to_string(),
                }
                .into());
            }
        }
    }
    Ok(())
}

fn validate_thc_q_ref(
    q: usize,
    n_parent: usize,
    effective_rank: usize,
    n_k: usize,
    n_orb: usize,
    record: &MldumpThcQRecordRefV1<'_>,
) -> Result<(), IoError> {
    if record.q_index != q {
        return Err(ValidationError::InvalidValue {
            path: format!("thc.q_records[{q}].q_index"),
            expected: q.to_string(),
            actual: record.q_index.to_string(),
        }
        .into());
    }
    nonempty(
        format!("thc.q_records[{q}].layout_provenance"),
        record.layout_provenance,
    )?;
    if record.aux_dimension != effective_rank {
        return Err(ValidationError::InvalidValue {
            path: format!("thc.q_records[{q}].aux_dimension"),
            expected: format!("effective_rank={effective_rank}"),
            actual: record.aux_dimension.to_string(),
        }
        .into());
    }
    require_flat_len(
        &format!("thc.q_records[{q}].zeta"),
        &[n_parent, record.aux_dimension, 2],
        record.zeta.len(),
    )?;
    require_finite_f64s(&format!("thc.q_records[{q}].zeta"), record.zeta)?;
    finite_residual(
        &format!("thc.q_records[{q}].residual_l2_all_frobenius"),
        record.residual_l2_all_frobenius,
    )?;
    finite_residual(
        &format!("thc.q_records[{q}].residual_l2_all_column_max"),
        record.residual_l2_all_column_max,
    )?;
    let vertices = record.vertices;
    if vertices.n_vertex == 0 {
        return Err(ValidationError::Empty {
            path: format!("thc.q_records[{q}].vertices"),
        }
        .into());
    }
    require_len(
        &format!("thc.q_records[{q}].vertices.column"),
        vertices.n_vertex,
        vertices.column.len(),
    )?;
    require_flat_len(
        &format!("thc.q_records[{q}].vertices.k_left_right"),
        &[vertices.n_vertex, 3],
        vertices.k_left_right.len(),
    )?;
    require_flat_len(
        &format!("thc.q_records[{q}].vertices.coefficients"),
        &[vertices.n_vertex, record.aux_dimension, 2],
        vertices.coefficients.len(),
    )?;
    require_finite_f64s(
        &format!("thc.q_records[{q}].vertices.coefficients"),
        vertices.coefficients,
    )?;
    validate_thc_vertex_triples(
        q,
        n_k,
        n_orb,
        vertices.n_vertex,
        vertices.column,
        vertices.k_left_right,
    )
}

fn validate_thc_q_owned(
    q: usize,
    n_parent: usize,
    effective_rank: usize,
    record: &MldumpThcQRecordV1,
) -> Result<(), IoError> {
    if record.q_index != q {
        return Err(ValidationError::InvalidValue {
            path: format!("thc.q_records[{q}].q_index"),
            expected: q.to_string(),
            actual: record.q_index.to_string(),
        }
        .into());
    }
    nonempty(
        format!("thc.q_records[{q}].layout_provenance"),
        &record.layout_provenance,
    )?;
    if record.aux_dimension != effective_rank {
        return Err(ValidationError::InvalidValue {
            path: format!("thc.q_records[{q}].aux_dimension"),
            expected: format!("effective_rank={effective_rank}"),
            actual: record.aux_dimension.to_string(),
        }
        .into());
    }
    require_flat_len(
        &format!("thc.q_records[{q}].zeta"),
        &[n_parent, record.aux_dimension, 2],
        record.zeta.len(),
    )?;
    require_finite_f64s(&format!("thc.q_records[{q}].zeta"), &record.zeta)?;
    finite_residual(
        &format!("thc.q_records[{q}].residual.frobenius"),
        record.residual.frobenius,
    )?;
    finite_residual(
        &format!("thc.q_records[{q}].residual.column_max"),
        record.residual.column_max,
    )?;
    if record.vertices.is_empty() {
        return Err(ValidationError::Empty {
            path: format!("thc.q_records[{q}].vertices"),
        }
        .into());
    }
    for (vertex, item) in record.vertices.iter().enumerate() {
        require_flat_len(
            &format!("thc.q_records[{q}].vertices[{vertex}].coefficients"),
            &[record.aux_dimension, 2],
            item.coefficients.len(),
        )?;
        require_finite_f64s(
            &format!("thc.q_records[{q}].vertices[{vertex}].coefficients"),
            &item.coefficients,
        )?;
    }
    Ok(())
}

pub(crate) fn validate_owned_thc_vertex_identity(
    n_k: usize,
    n_orb: usize,
    thc: &MldumpThcV1,
) -> Result<(), IoError> {
    for (q, record) in thc.q_records.iter().enumerate() {
        for (vertex, item) in record.vertices.iter().enumerate() {
            validate_one_thc_vertex(
                q,
                vertex,
                n_k,
                n_orb,
                item.column,
                [item.k, item.left, item.right],
            )?;
        }
    }
    Ok(())
}

fn validate_thc_vertex_triples(
    q: usize,
    n_k: usize,
    n_orb: usize,
    n_vertex: usize,
    column: &[i64],
    k_left_right: &[i64],
) -> Result<(), IoError> {
    for vertex in 0..n_vertex {
        validate_one_thc_vertex(
            q,
            vertex,
            n_k,
            n_orb,
            column[vertex],
            [
                k_left_right[vertex * 3],
                k_left_right[vertex * 3 + 1],
                k_left_right[vertex * 3 + 2],
            ],
        )?;
    }
    Ok(())
}

fn validate_one_thc_vertex(
    q: usize,
    vertex: usize,
    n_k: usize,
    n_orb: usize,
    column: i64,
    k_left_right: [i64; 3],
) -> Result<(), IoError> {
    let (decoded_k, decoded_left, decoded_right) = decode_pair_column(
        &format!("thc.q_records[{q}].vertices[{vertex}].column"),
        n_k,
        n_orb,
        column,
    )?;
    let expected_k = i64_from_usize(
        &format!("thc.q_records[{q}].vertices[{vertex}].k"),
        decoded_k,
    )?;
    let expected_left = i64_from_usize(
        &format!("thc.q_records[{q}].vertices[{vertex}].left"),
        decoded_left,
    )?;
    let expected_right = i64_from_usize(
        &format!("thc.q_records[{q}].vertices[{vertex}].right"),
        decoded_right,
    )?;
    if k_left_right != [expected_k, expected_left, expected_right] {
        return Err(ValidationError::InvalidValue {
            path: format!("thc.q_records[{q}].vertices[{vertex}].k_left_right"),
            expected: format!(
                "decode(column={column}) = ({decoded_k},{decoded_left},{decoded_right})"
            ),
            actual: format!(
                "({},{},{})",
                k_left_right[0], k_left_right[1], k_left_right[2]
            ),
        }
        .into());
    }
    Ok(())
}

fn validate_coulomb_begin(coulomb: &MldumpCoulombBeginV1) -> Result<(), IoError> {
    if coulomb.lexp > 12 {
        return Err(ValidationError::InvalidValue {
            path: "coulomb.lexp".to_owned(),
            expected: "0..=12".to_owned(),
            actual: coulomb.lexp.to_string(),
        }
        .into());
    }
    Ok(())
}

fn validate_coulomb_owned(
    header: &MldumpHeaderV1,
    coulomb: &MldumpCoulombV1,
) -> Result<(), IoError> {
    if coulomb.lexp > 12 {
        return Err(ValidationError::InvalidValue {
            path: "coulomb.lexp".to_owned(),
            expected: "0..=12".to_owned(),
            actual: coulomb.lexp.to_string(),
        }
        .into());
    }
    require_len(
        "coulomb.q_records",
        header.mesh.q_entries.len(),
        coulomb.q_records.len(),
    )?;
    for (q, record) in coulomb.q_records.iter().enumerate() {
        validate_coulomb_q_payload(
            q,
            record.q_index,
            record.aux_dimension,
            &record.layout_provenance,
            &record.body,
            record
                .gamma
                .as_ref()
                .map(|gamma| (gamma.head_prefactor, gamma.constant_coefficients.as_slice())),
        )?;
    }
    Ok(())
}

fn validate_coulomb_q_payload(
    q: usize,
    q_index: usize,
    aux_dimension: usize,
    layout_provenance: &str,
    body: &[f64],
    gamma: Option<(f64, &[f64])>,
) -> Result<(), IoError> {
    if q_index != q {
        return Err(ValidationError::InvalidValue {
            path: format!("coulomb.q_records[{q}].q_index"),
            expected: q.to_string(),
            actual: q_index.to_string(),
        }
        .into());
    }
    nonempty(
        format!("coulomb.q_records[{q}].layout_provenance"),
        layout_provenance,
    )?;
    if aux_dimension == 0 {
        return Err(ValidationError::NotPositive {
            path: format!("coulomb.q_records[{q}].aux_dimension"),
            value: 0.0,
        }
        .into());
    }
    require_flat_len(
        &format!("coulomb.q_records[{q}].body"),
        &[aux_dimension, aux_dimension, 2],
        body.len(),
    )?;
    require_finite_f64s(&format!("coulomb.q_records[{q}].body"), body)?;
    require_hermitian(&format!("coulomb.q_records[{q}].body"), aux_dimension, body)?;
    if let Some((head_prefactor, constant_coefficients)) = gamma {
        require_flat_len(
            &format!("coulomb.q_records[{q}].gamma.constant_coefficients"),
            &[aux_dimension, 2],
            constant_coefficients.len(),
        )?;
        require_finite_f64s(
            &format!("coulomb.q_records[{q}].gamma.constant_coefficients"),
            constant_coefficients,
        )?;
        finite_residual(
            &format!("coulomb.q_records[{q}].gamma.head_prefactor"),
            head_prefactor,
        )?;
    }
    Ok(())
}

fn write_parent_grid(parent: &Group, grid: &MldumpThcParentGridRefV1<'_>) -> Result<(), IoError> {
    let group = parent.create_group(PREFIX_PARENT_GRID)?;
    write_str_attr(&group, "provenance", grid.provenance)?;
    write_f64_dataset(
        &group,
        "coordinates",
        &[grid.n_points, 3],
        grid.coordinates,
        &["parent_point", "cartesian"],
    )?;
    write_f64_dataset(
        &group,
        "weights",
        &[grid.n_points],
        grid.weights,
        &["parent_point"],
    )?;
    write_i64_dataset(
        &group,
        "region_kind",
        &[grid.n_points],
        grid.region_kind,
        &["parent_point"],
    )?;
    write_i64_dataset(
        &group,
        "site_index",
        &[grid.n_points],
        grid.site_index,
        &["parent_point"],
    )?;
    write_i64_dataset(
        &group,
        "radial_index",
        &[grid.n_points],
        grid.radial_index,
        &["parent_point"],
    )?;
    Ok(())
}

fn write_thc_q(
    parent: &Group,
    n_parent: usize,
    record: &MldumpThcQRecordRefV1<'_>,
) -> Result<(), IoError> {
    let group = create_padded_group(parent, PREFIX_Q, record.q_index)?;
    write_i64_attr(&group, "q_index", usize_as_i64("q_index", record.q_index)?)?;
    write_i64_attr(
        &group,
        "aux_dimension",
        usize_as_i64("aux_dimension", record.aux_dimension)?,
    )?;
    write_str_attr(&group, "layout_provenance", record.layout_provenance)?;
    write_f64_dataset(
        &group,
        "zeta",
        &[n_parent, record.aux_dimension, 2],
        record.zeta,
        &["parent_point", "aux", "re_im"],
    )?;
    write_f64_dataset(
        &group,
        "fit_residual_l2_all",
        &[2],
        &[
            record.residual_l2_all_frobenius,
            record.residual_l2_all_column_max,
        ],
        &["metric"],
    )?;
    let residual = group.dataset("fit_residual_l2_all")?;
    write_str_array_attr(&residual, "metric_labels", &RESIDUAL_LABELS)?;
    write_i64_dataset(
        &group,
        "vertex_column",
        &[record.vertices.n_vertex],
        record.vertices.column,
        &["vertex"],
    )?;
    write_i64_dataset(
        &group,
        "vertex_k_left_right",
        &[record.vertices.n_vertex, 3],
        record.vertices.k_left_right,
        &["vertex", "k_left_right"],
    )?;
    write_f64_dataset(
        &group,
        "vertex_coefficients",
        &[record.vertices.n_vertex, record.aux_dimension, 2],
        record.vertices.coefficients,
        &["vertex", "aux", "re_im"],
    )?;
    Ok(())
}

fn write_coulomb_q(parent: &Group, record: &MldumpCoulombQRecordRefV1<'_>) -> Result<(), IoError> {
    let group = create_padded_group(parent, PREFIX_Q, record.q_index)?;
    write_i64_attr(&group, "q_index", usize_as_i64("q_index", record.q_index)?)?;
    write_i64_attr(
        &group,
        "aux_dimension",
        usize_as_i64("aux_dimension", record.aux_dimension)?,
    )?;
    write_str_attr(&group, "layout_provenance", record.layout_provenance)?;
    write_f64_dataset(
        &group,
        "body",
        &[record.aux_dimension, record.aux_dimension, 2],
        record.body,
        &["aux_row", "aux_col", "re_im"],
    )?;
    if let Some(gamma) = record.gamma {
        let gamma_group = group.create_group(PREFIX_GAMMA)?;
        write_status(&gamma_group, MldumpStatus::Present)?;
        write_i64_attr(
            &gamma_group,
            "spherical_average_subtracted",
            i64::from(gamma.spherical_average_subtracted),
        )?;
        write_f64_attr(&gamma_group, "head_prefactor", gamma.head_prefactor)?;
        write_f64_dataset(
            &gamma_group,
            "constant_coefficients",
            &[record.aux_dimension, 2],
            gamma.constant_coefficients,
            &["aux", "re_im"],
        )?;
    } else {
        write_absent_group(&group, PREFIX_GAMMA)?;
    }
    Ok(())
}

fn read_parent_grid(group: &Group) -> Result<MldumpThcParentGridV1, IoError> {
    require_dataset_names(
        group,
        &[
            "coordinates",
            "weights",
            "region_kind",
            "site_index",
            "radial_index",
        ],
    )?;
    let n_points = group
        .dataset("weights")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: format!("{}/weights/shape", group.name()),
            expected: "[parent_point]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    let coordinates = super::triples_to_owned(
        &read_f64_dataset(
            group,
            "coordinates",
            &[n_points, 3],
            &["parent_point", "cartesian"],
        )?,
        n_points,
        &format!("{}/coordinates", group.name()),
    )?;
    Ok(MldumpThcParentGridV1 {
        coordinates,
        weights: read_f64_dataset(group, "weights", &[n_points], &["parent_point"])?,
        region_kind: read_i64_dataset(group, "region_kind", &[n_points], &["parent_point"])?,
        site_index: read_i64_dataset(group, "site_index", &[n_points], &["parent_point"])?,
        radial_index: read_i64_dataset(group, "radial_index", &[n_points], &["parent_point"])?,
        provenance: read_str_attr(group, "provenance")?,
    })
}

fn read_thc_q(
    group: &Group,
    q: usize,
    n_parent: usize,
    effective_rank: usize,
) -> Result<MldumpThcQRecordV1, IoError> {
    let stored = read_usize_attr(group, "q_index", &format!("{}/@q_index", group.name()))?;
    if stored != q {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@q_index", group.name()),
            expected: q.to_string(),
            actual: stored.to_string(),
        }
        .into());
    }
    let aux_dimension = read_usize_attr(
        group,
        "aux_dimension",
        &format!("{}/@aux_dimension", group.name()),
    )?;
    if aux_dimension != effective_rank {
        return Err(ValidationError::LayoutMismatch {
            path: format!("{}/@aux_dimension", group.name()),
            reference: format!("/thc/@effective_rank={effective_rank}"),
        }
        .into());
    }
    require_dataset_names(
        group,
        &[
            "zeta",
            "fit_residual_l2_all",
            "vertex_column",
            "vertex_k_left_right",
            "vertex_coefficients",
        ],
    )?;
    let zeta = read_f64_dataset(
        group,
        "zeta",
        &[n_parent, aux_dimension, 2],
        &["parent_point", "aux", "re_im"],
    )?;
    let residual = read_f64_dataset(group, "fit_residual_l2_all", &[2], &["metric"])?;
    let residual_ds = group.dataset("fit_residual_l2_all")?;
    require_str_array_attr(&residual_ds, "metric_labels", &RESIDUAL_LABELS)?;
    let n_vertex = group
        .dataset("vertex_column")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: format!("{}/vertex_column/shape", group.name()),
            expected: "[vertex]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    let column = read_i64_dataset(group, "vertex_column", &[n_vertex], &["vertex"])?;
    let k_left_right = read_i64_dataset(
        group,
        "vertex_k_left_right",
        &[n_vertex, 3],
        &["vertex", "k_left_right"],
    )?;
    let coefficients = read_f64_dataset(
        group,
        "vertex_coefficients",
        &[n_vertex, aux_dimension, 2],
        &["vertex", "aux", "re_im"],
    )?;
    let coeff_width = complex_len(aux_dimension)?;
    let mut vertices = Vec::with_capacity(n_vertex);
    for vertex in 0..n_vertex {
        let start = vertex * coeff_width;
        vertices.push(MldumpThcVertexV1 {
            column: column[vertex],
            k: k_left_right[vertex * 3],
            left: k_left_right[vertex * 3 + 1],
            right: k_left_right[vertex * 3 + 2],
            coefficients: coefficients[start..start + coeff_width].to_vec(),
        });
    }
    Ok(MldumpThcQRecordV1 {
        q_index: q,
        aux_dimension,
        layout_provenance: read_str_attr(group, "layout_provenance")?,
        zeta,
        residual: MldumpThcResidualV1 {
            frobenius: residual[0],
            column_max: residual[1],
        },
        vertices,
    })
}

fn read_coulomb_q(group: &Group, q: usize) -> Result<MldumpCoulombQRecordV1, IoError> {
    let stored = read_usize_attr(group, "q_index", &format!("{}/@q_index", group.name()))?;
    if stored != q {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@q_index", group.name()),
            expected: q.to_string(),
            actual: stored.to_string(),
        }
        .into());
    }
    let aux_dimension = read_usize_attr(
        group,
        "aux_dimension",
        &format!("{}/@aux_dimension", group.name()),
    )?;
    require_dataset_names(group, &["body"])?;
    require_group_names(group, &[PREFIX_GAMMA])?;
    let body = read_f64_dataset(
        group,
        "body",
        &[aux_dimension, aux_dimension, 2],
        &["aux_row", "aux_col", "re_im"],
    )?;
    require_hermitian(&format!("{}/body", group.name()), aux_dimension, &body)?;
    let gamma_group = group.group(PREFIX_GAMMA)?;
    let gamma_status = super::read_status(&gamma_group)?;
    let gamma = match gamma_status {
        MldumpStatus::AbsentNotComputed => {
            require_no_payload(&gamma_group)?;
            None
        }
        MldumpStatus::Present => {
            require_dataset_names(&gamma_group, &["constant_coefficients"])?;
            let subtracted = read_i64_attr(
                &gamma_group,
                "spherical_average_subtracted",
                &format!("{}/@spherical_average_subtracted", gamma_group.name()),
            )?;
            if subtracted != 0 && subtracted != 1 {
                return Err(ValidationError::InvalidValue {
                    path: format!("{}/@spherical_average_subtracted", gamma_group.name()),
                    expected: "0 or 1".to_owned(),
                    actual: subtracted.to_string(),
                }
                .into());
            }
            Some(MldumpCoulombGammaV1 {
                spherical_average_subtracted: subtracted == 1,
                head_prefactor: read_f64_attr(
                    &gamma_group,
                    "head_prefactor",
                    &format!("{}/@head_prefactor", gamma_group.name()),
                )?,
                constant_coefficients: read_f64_dataset(
                    &gamma_group,
                    "constant_coefficients",
                    &[aux_dimension, 2],
                    &["aux", "re_im"],
                )?,
            })
        }
    };
    Ok(MldumpCoulombQRecordV1 {
        q_index: q,
        aux_dimension,
        layout_provenance: read_str_attr(group, "layout_provenance")?,
        body,
        gamma,
    })
}

fn require_known_engine(path: &str, engine: &str) -> Result<(), IoError> {
    if engine == MLDUMP_THC_ENGINE_QRCP || engine == MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: format!("{MLDUMP_THC_ENGINE_QRCP} or {MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY}"),
            actual: engine.to_owned(),
        }
        .into())
    }
}

fn validate_selected_parent_indices(
    path: &str,
    values: &[i64],
    weights: &[f64],
) -> Result<(), IoError> {
    let n_parent = weights.len();
    let mut seen = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let point = require_nonnegative_index(&format!("{path}[{index}]"), *value)?;
        if point >= n_parent {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}[{index}]"),
                expected: format!("index < {n_parent}"),
                actual: point.to_string(),
            }
            .into());
        }
        if !seen.insert(point) {
            return Err(ValidationError::Duplicate {
                path: path.to_owned(),
                key: point.to_string(),
            }
            .into());
        }
        if !(weights[point].is_finite() && weights[point] > 0.0) {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}[{index}]"),
                expected: "strictly positive parent weight".to_owned(),
                actual: weights[point].to_string(),
            }
            .into());
        }
    }
    Ok(())
}

fn require_identical_selection_sets(pivots: &[i64], points: &[i64]) -> Result<(), IoError> {
    let pivot_set = pivots.iter().copied().collect::<BTreeSet<_>>();
    let point_set = points.iter().copied().collect::<BTreeSet<_>>();
    if pivot_set == point_set {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: "thc.selection".to_owned(),
            expected: format!(
                "points contain the same parent indices as pivots {pivot_set:?} (layout order may differ)"
            ),
            actual: format!("{point_set:?}"),
        }
        .into())
    }
}

fn decode_pair_column(
    path: &str,
    n_k: usize,
    n_orb: usize,
    column: i64,
) -> Result<(usize, usize, usize), IoError> {
    if n_orb == 0 {
        return Err(ValidationError::NotPositive {
            path: "products.n_orb".to_owned(),
            value: 0.0,
        }
        .into());
    }
    let column_u = u64::try_from(column).map_err(|_| ValidationError::InvalidValue {
        path: path.to_owned(),
        expected: "nonnegative pair column".to_owned(),
        actual: column.to_string(),
    })?;
    let n_orb_u = u64::try_from(n_orb).map_err(|_| ValidationError::InvalidValue {
        path: "products.n_orb".to_owned(),
        expected: "n_orb fitting u64".to_owned(),
        actual: n_orb.to_string(),
    })?;
    let n_k_u = u64::try_from(n_k).map_err(|_| ValidationError::InvalidValue {
        path: "products.n_k".to_owned(),
        expected: "n_k fitting u64".to_owned(),
        actual: n_k.to_string(),
    })?;
    let block = n_orb_u
        .checked_mul(n_orb_u)
        .ok_or_else(|| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "n_orb^2 fitting u64".to_owned(),
            actual: n_orb.to_string(),
        })?;
    let k_u = column_u / block;
    if k_u >= n_k_u {
        return Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: format!("column with k < {n_k}"),
            actual: column.to_string(),
        }
        .into());
    }
    let rem = column_u % block;
    let left_u = rem / n_orb_u;
    let right_u = rem % n_orb_u;
    Ok((
        usize::try_from(k_u).map_err(|_| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "decoded k fitting usize".to_owned(),
            actual: k_u.to_string(),
        })?,
        usize::try_from(left_u).map_err(|_| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "decoded left fitting usize".to_owned(),
            actual: left_u.to_string(),
        })?,
        usize::try_from(right_u).map_err(|_| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "decoded right fitting usize".to_owned(),
            actual: right_u.to_string(),
        })?,
    ))
}

fn i64_from_usize(path: &str, value: usize) -> Result<i64, IoError> {
    i64::try_from(value).map_err(|_| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "i64".to_owned(),
            actual: value.to_string(),
        }
        .into()
    })
}

fn require_u32_attr(group: &Group, name: &str, path: &str) -> Result<u32, IoError> {
    let value = read_i64_attr(group, name, path)?;
    u32::try_from(value).map_err(|_| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "u32".to_owned(),
            actual: value.to_string(),
        }
        .into()
    })
}

fn finite_residual(path: &str, value: f64) -> Result<(), IoError> {
    crate::error::finite(path, value)?;
    if value < 0.0 {
        Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "nonnegative".to_owned(),
            actual: value.to_string(),
        }
        .into())
    } else {
        Ok(())
    }
}

fn require_hermitian(path: &str, n: usize, body: &[f64]) -> Result<(), IoError> {
    for row in 0..n {
        for col in 0..n {
            let re = body[(row * n + col) * 2];
            let im = body[(row * n + col) * 2 + 1];
            let re_t = body[(col * n + row) * 2];
            let im_t = body[(col * n + row) * 2 + 1];
            if !approx_eq(re, re_t) || !approx_eq(im, -im_t) {
                return Err(ValidationError::InvalidValue {
                    path: format!("{path}[{row},{col}]"),
                    expected: "Hermitian body V[j,i] = conj(V[i,j])".to_owned(),
                    actual: format!("({re},{im}) vs ({re_t},{im_t})"),
                }
                .into());
            }
        }
    }
    Ok(())
}
