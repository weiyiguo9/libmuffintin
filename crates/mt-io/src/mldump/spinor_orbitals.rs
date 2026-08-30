//! Full first-variation spinor orbital payload for MLDUMP v1.

use std::collections::BTreeSet;

use hdf5_metno::Group;

use super::scalar_orbitals::MLDUMP_OCCUPATIONS_NOT_EXPORTED;
use super::{
    GROUP_ORBITALS, MldumpHeaderV1, PREFIX_BASIS, PREFIX_K, PREFIX_SITE, collect_padded_groups,
    create_padded_group, dataset_leading_len, i32_triples_to_owned, padded_child, read_f64_dataset,
    read_i32_dataset, read_i64_attr, read_i64_dataset, read_usize_attr, reopen_present_group,
    require_dataset_names, require_exact_members, require_finite_f64s, require_flat_len,
    require_group_names, require_len, require_nonnegative_index, require_status_present,
    require_str_attr, triples_to_owned, usize_as_i64, validate_plane_wave_identity,
    write_f64_dataset, write_i32_dataset, write_i64_attr, write_i64_dataset, write_str_attr,
};
use crate::error::{IoError, ValidationError};

/// Representation tag stored on spinor `/orbitals`.
pub const MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION: &str = "spinor_full_first_variation";

const N_PAULI: usize = 2;

/// Shared `/orbitals` attributes for a streaming spinor session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpinorOrbitalsBeginV1 {
    pub band_window_start: i64,
    /// Common leading exported window. Eigenvalues and eigenvector columns use
    /// this count, not per-$k$ [`SpinorOrbitalKRefV1::available_bands`].
    pub band_window_count: usize,
}

/// One $k$ compiled spinor basis and eigenvectors. No collinear spin groups.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorOrbitalKRefV1<'a> {
    pub k_index: usize,
    /// Per-$k$ available band count. May exceed the exported common window.
    pub available_bands: usize,
    pub basis_dimension: usize,
    pub eigenvalues: &'a [f64],
    /// C-order `[basis_row, band, re_im]`.
    pub eigenvectors: &'a [f64],
    pub n_plane_waves: usize,
    pub plane_wave_g: &'a [i32],
    pub plane_wave_k_cartesian: &'a [f64],
    pub plane_wave_q_cartesian: &'a [f64],
    pub pauli_rows: SpinorPauliRowMapRefV1<'a>,
    pub local_orbitals: SpinorLocalOrbitalTableRefV1<'a>,
    pub site_matches: &'a [SpinorSiteMatchRefV1<'a>],
}

/// Explicit Pauli plane-wave row map `row = pauli_component * n_plane_wave + plane_wave_index`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorPauliRowMapRefV1<'a> {
    pub n_row: usize,
    pub row_index: &'a [i64],
    pub pauli_component: &'a [i64],
    pub plane_wave_index: &'a [i64],
}

/// Confined LO/RLO eigenbasis rows. APW $(P,\dot P)$ are matching columns, not rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorLocalOrbitalTableRefV1<'a> {
    pub n_local_orbitals: usize,
    pub row_index: &'a [i64],
    pub site: &'a [i64],
    pub signed_kappa: &'a [i64],
    pub twice_mu: &'a [i64],
    pub ordinal: &'a [i64],
    pub radial_n: &'a [i64],
}

/// Per-site projection coordinates and APW matching coefficients.
///
/// Coordinates are the live APW-then-LO table: a strict APW prefix of
/// `(signed_kappa, twice_mu, n=0)` then `(same, n=1)` pairs in native
/// strictly increasing `(signed_kappa, twice_mu)` order, followed by the
/// LO/RLO tail. `n_apw_projection` is that prefix length and the matching
/// third axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorSiteMatchRefV1<'a> {
    pub site_index: usize,
    pub n_projection: usize,
    /// APW prefix length; matching third axis maps these coordinates only.
    pub n_apw_projection: usize,
    pub coordinate: &'a [i64],
    pub signed_kappa: &'a [i64],
    pub twice_mu: &'a [i64],
    pub radial_n: &'a [i64],
    /// C-order `[plane_wave, pauli_component, apw_projection_coordinate, re_im]`.
    pub matching_coefficients: &'a [f64],
}

/// Owned spinor orbital section.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorOrbitalsV1 {
    pub band_window_start: i64,
    pub band_window_count: usize,
    pub k_points: Vec<SpinorOrbitalKRecordV1>,
}

/// Owned per-$k$ spinor orbital record.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorOrbitalKRecordV1 {
    pub k_index: usize,
    pub available_bands: usize,
    pub basis_dimension: usize,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<f64>,
    pub plane_wave_g: Vec<[i32; 3]>,
    pub plane_wave_k_cartesian: Vec<[f64; 3]>,
    pub plane_wave_q_cartesian: Vec<[f64; 3]>,
    pub pauli_rows: SpinorPauliRowMapV1,
    pub local_orbitals: Vec<SpinorLocalOrbitalRowV1>,
    pub site_matches: Vec<SpinorSiteMatchV1>,
}

/// Owned Pauli plane-wave row map.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorPauliRowMapV1 {
    pub row_index: Vec<i64>,
    pub pauli_component: Vec<i64>,
    pub plane_wave_index: Vec<i64>,
}

/// One confined LO/RLO eigenbasis row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpinorLocalOrbitalRowV1 {
    pub row_index: usize,
    pub site: usize,
    pub signed_kappa: i64,
    pub twice_mu: i64,
    pub ordinal: usize,
    pub radial_n: usize,
}

/// Owned per-site projection table and matching coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorSiteMatchV1 {
    pub site_index: usize,
    pub coordinates: Vec<SpinorProjectionCoordV1>,
    pub matching_coefficients: Vec<f64>,
}

/// One site-projection coordinate in live APW-then-LO order.
///
/// APW channels are native strictly increasing `(signed_kappa, twice_mu)`,
/// each as `radial_n=0` then `radial_n=1`. The tail is LO/RLO `radial_n>=2`
/// with identities equal to that site's local-row table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpinorProjectionCoordV1 {
    pub coordinate: usize,
    pub signed_kappa: i64,
    pub twice_mu: i64,
    pub radial_n: usize,
}

pub(crate) fn begin_spinor_orbitals(
    file: &Group,
    begin: &SpinorOrbitalsBeginV1,
) -> Result<(), IoError> {
    validate_orbitals_begin(begin)?;
    let group = reopen_present_group(file, GROUP_ORBITALS)?;
    write_str_attr(
        &group,
        "representation",
        MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
    )?;
    write_i64_attr(&group, "band_window_start", begin.band_window_start)?;
    write_i64_attr(
        &group,
        "band_window_count",
        usize_as_i64("/orbitals/@band_window_count", begin.band_window_count)?,
    )?;
    write_str_attr(
        &group,
        "occupations_status",
        MLDUMP_OCCUPATIONS_NOT_EXPORTED,
    )?;
    Ok(())
}

pub(crate) fn write_spinor_orbital_k(
    file: &Group,
    header: &MldumpHeaderV1,
    band_window_count: usize,
    record: &SpinorOrbitalKRefV1<'_>,
) -> Result<(), IoError> {
    validate_k_record(header, band_window_count, record)?;
    let group = file.group(GROUP_ORBITALS)?;
    write_k_group(&group, band_window_count, record)
}

pub(crate) fn read_spinor_orbitals(
    file: &Group,
    header: &MldumpHeaderV1,
) -> Result<SpinorOrbitalsV1, IoError> {
    let group = file.group(GROUP_ORBITALS)?;
    require_status_present(&group)?;
    require_str_attr(
        &group,
        "representation",
        MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
    )?;
    require_str_attr(
        &group,
        "occupations_status",
        MLDUMP_OCCUPATIONS_NOT_EXPORTED,
    )?;
    let band_window_start =
        read_i64_attr(&group, "band_window_start", "/orbitals/@band_window_start")?;
    if band_window_start != 0 {
        return Err(ValidationError::InvalidValue {
            path: "/orbitals/@band_window_start".to_owned(),
            expected: "0".to_owned(),
            actual: band_window_start.to_string(),
        }
        .into());
    }
    let band_window_count =
        read_usize_attr(&group, "band_window_count", "/orbitals/@band_window_count")?;
    if band_window_count == 0 {
        return Err(ValidationError::NotPositive {
            path: "/orbitals/@band_window_count".to_owned(),
            value: 0.0,
        }
        .into());
    }
    let n_k = header.mesh.k_points.len();
    let k_groups = collect_padded_groups(&group, PREFIX_K)?;
    require_len("/orbitals/k_*", n_k, k_groups.len())?;
    require_exact_members(&group, (0..n_k).map(|k| padded_child(PREFIX_K, k)))?;
    let mut k_points = Vec::with_capacity(n_k);
    for (k, k_group) in k_groups.iter().enumerate() {
        k_points.push(read_k_group(k_group, header, k, band_window_count)?);
    }
    let orbitals = SpinorOrbitalsV1 {
        band_window_start,
        band_window_count,
        k_points,
    };
    validate_orbitals_owned(header, &orbitals)?;
    Ok(orbitals)
}

fn validate_orbitals_begin(begin: &SpinorOrbitalsBeginV1) -> Result<(), IoError> {
    if begin.band_window_start != 0 {
        return Err(ValidationError::InvalidValue {
            path: "orbitals.band_window_start".to_owned(),
            expected: "0".to_owned(),
            actual: begin.band_window_start.to_string(),
        }
        .into());
    }
    if begin.band_window_count == 0 {
        return Err(ValidationError::NotPositive {
            path: "orbitals.band_window_count".to_owned(),
            value: 0.0,
        }
        .into());
    }
    Ok(())
}

fn validate_k_record(
    header: &MldumpHeaderV1,
    band_window_count: usize,
    record: &SpinorOrbitalKRefV1<'_>,
) -> Result<(), IoError> {
    let path = format!("orbitals.k_points[{}]", record.k_index);
    if record.k_index >= header.mesh.k_points.len() {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.k_index"),
            expected: format!("index < {}", header.mesh.k_points.len()),
            actual: record.k_index.to_string(),
        }
        .into());
    }
    if record.available_bands == 0 || record.available_bands < band_window_count {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.available_bands"),
            expected: format!(">= band_window_count {band_window_count}"),
            actual: record.available_bands.to_string(),
        }
        .into());
    }
    require_len(
        &format!("{path}.eigenvalues"),
        band_window_count,
        record.eigenvalues.len(),
    )?;
    require_finite_f64s(&format!("{path}.eigenvalues"), record.eigenvalues)?;
    require_flat_len(
        &format!("{path}.eigenvectors"),
        &[record.basis_dimension, band_window_count, 2],
        record.eigenvectors.len(),
    )?;
    require_finite_f64s(&format!("{path}.eigenvectors"), record.eigenvectors)?;
    if record.n_plane_waves == 0 {
        return Err(ValidationError::Empty {
            path: format!("{path}.plane_waves"),
        }
        .into());
    }
    require_flat_len(
        &format!("{path}.plane_wave_g"),
        &[record.n_plane_waves, 3],
        record.plane_wave_g.len(),
    )?;
    require_flat_len(
        &format!("{path}.plane_wave_k_cartesian"),
        &[record.n_plane_waves, 3],
        record.plane_wave_k_cartesian.len(),
    )?;
    require_flat_len(
        &format!("{path}.plane_wave_q_cartesian"),
        &[record.n_plane_waves, 3],
        record.plane_wave_q_cartesian.len(),
    )?;
    require_finite_f64s(
        &format!("{path}.plane_wave_k_cartesian"),
        record.plane_wave_k_cartesian,
    )?;
    require_finite_f64s(
        &format!("{path}.plane_wave_q_cartesian"),
        record.plane_wave_q_cartesian,
    )?;
    validate_plane_wave_identity(
        &header.geometry.reciprocal_basis_inv_bohr,
        record.n_plane_waves,
        record.plane_wave_g,
        record.plane_wave_k_cartesian,
        record.plane_wave_q_cartesian,
        &path,
    )?;
    validate_pauli_rows(record, &path)?;
    validate_local_orbitals(header.geometry.sites.len(), record, &path)?;
    require_len(
        &format!("{path}.site_matches"),
        header.geometry.sites.len(),
        record.site_matches.len(),
    )?;
    for (site, matching) in record.site_matches.iter().enumerate() {
        validate_site_match(
            header.geometry.sites.len(),
            record.n_plane_waves,
            site,
            matching,
            &record.local_orbitals,
            &path,
        )?;
    }
    Ok(())
}

fn validate_pauli_rows(record: &SpinorOrbitalKRefV1<'_>, path: &str) -> Result<(), IoError> {
    let n_row =
        N_PAULI
            .checked_mul(record.n_plane_waves)
            .ok_or_else(|| ValidationError::InvalidValue {
                path: format!("{path}.pauli_rows"),
                expected: "2 * n_plane_waves fitting usize".to_owned(),
                actual: record.n_plane_waves.to_string(),
            })?;
    if record.pauli_rows.n_row != n_row {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.pauli_rows.n_row"),
            expected: n_row.to_string(),
            actual: record.pauli_rows.n_row.to_string(),
        }
        .into());
    }
    for (name, values) in [
        ("row_index", record.pauli_rows.row_index),
        ("pauli_component", record.pauli_rows.pauli_component),
        ("plane_wave_index", record.pauli_rows.plane_wave_index),
    ] {
        require_len(&format!("{path}.pauli_rows.{name}"), n_row, values.len())?;
    }
    let mut seen = BTreeSet::new();
    for row in 0..n_row {
        let stored = require_nonnegative_index(
            &format!("{path}.pauli_rows.row_index[{row}]"),
            record.pauli_rows.row_index[row],
        )?;
        let component = require_nonnegative_index(
            &format!("{path}.pauli_rows.pauli_component[{row}]"),
            record.pauli_rows.pauli_component[row],
        )?;
        let pw = require_nonnegative_index(
            &format!("{path}.pauli_rows.plane_wave_index[{row}]"),
            record.pauli_rows.plane_wave_index[row],
        )?;
        if component >= N_PAULI {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.pauli_rows.pauli_component[{row}]"),
                expected: "0 or 1".to_owned(),
                actual: component.to_string(),
            }
            .into());
        }
        if pw >= record.n_plane_waves {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.pauli_rows.plane_wave_index[{row}]"),
                expected: format!("index < {}", record.n_plane_waves),
                actual: pw.to_string(),
            }
            .into());
        }
        let expected = component * record.n_plane_waves + pw;
        if stored != expected {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.pauli_rows.row_index[{row}]"),
                expected: format!("pauli_component * n_plane_wave + plane_wave_index = {expected}"),
                actual: stored.to_string(),
            }
            .into());
        }
        if !seen.insert(stored) {
            return Err(ValidationError::Duplicate {
                path: format!("{path}.pauli_rows.row_index"),
                key: stored.to_string(),
            }
            .into());
        }
    }
    if seen.len() != n_row {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.pauli_rows"),
            expected: format!("{n_row} unique Pauli rows"),
            actual: seen.len().to_string(),
        }
        .into());
    }
    Ok(())
}

fn validate_local_orbitals(
    n_sites: usize,
    record: &SpinorOrbitalKRefV1<'_>,
    path: &str,
) -> Result<(), IoError> {
    let table = record.local_orbitals;
    let n_lo = table.n_local_orbitals;
    let n_pauli = N_PAULI * record.n_plane_waves;
    let expected_dim = n_pauli + n_lo;
    if record.basis_dimension != expected_dim {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.basis_dimension"),
            expected: format!("2 * n_plane_wave + n_lo = {expected_dim}"),
            actual: record.basis_dimension.to_string(),
        }
        .into());
    }
    for (name, values) in [
        ("row_index", table.row_index),
        ("site", table.site),
        ("signed_kappa", table.signed_kappa),
        ("twice_mu", table.twice_mu),
        ("ordinal", table.ordinal),
        ("radial_n", table.radial_n),
    ] {
        require_len(&format!("{path}.local_orbitals.{name}"), n_lo, values.len())?;
    }
    let mut rows = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for lo in 0..n_lo {
        let row = require_nonnegative_index(
            &format!("{path}.local_orbitals.row_index[{lo}]"),
            table.row_index[lo],
        )?;
        if row < n_pauli || row >= record.basis_dimension {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.local_orbitals.row_index[{lo}]"),
                expected: format!("Pauli-block offset {n_pauli}..{}", record.basis_dimension),
                actual: row.to_string(),
            }
            .into());
        }
        if !rows.insert(row) {
            return Err(ValidationError::Duplicate {
                path: format!("{path}.local_orbitals.row_index"),
                key: row.to_string(),
            }
            .into());
        }
        let site = require_nonnegative_index(
            &format!("{path}.local_orbitals.site[{lo}]"),
            table.site[lo],
        )?;
        if site >= n_sites {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.local_orbitals.site[{lo}]"),
                expected: format!("index < {n_sites}"),
                actual: site.to_string(),
            }
            .into());
        }
        let ordinal = require_nonnegative_index(
            &format!("{path}.local_orbitals.ordinal[{lo}]"),
            table.ordinal[lo],
        )?;
        let radial_n = require_nonnegative_index(
            &format!("{path}.local_orbitals.radial_n[{lo}]"),
            table.radial_n[lo],
        )?;
        if radial_n != 2 + ordinal {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.local_orbitals.radial_n[{lo}]"),
                expected: format!("2 + ordinal = {}", 2 + ordinal),
                actual: radial_n.to_string(),
            }
            .into());
        }
        require_valid_dirac_channel(
            &format!("{path}.local_orbitals[{lo}]"),
            table.signed_kappa[lo],
            table.twice_mu[lo],
        )?;
        let id = (site, table.signed_kappa[lo], table.twice_mu[lo], ordinal);
        if !ids.insert(id) {
            return Err(ValidationError::Duplicate {
                path: format!("{path}.local_orbitals"),
                key: format!(
                    "site={} kappa={} twice_mu={} ordinal={}",
                    id.0, id.1, id.2, id.3
                ),
            }
            .into());
        }
    }
    let expected_rows = (n_pauli..record.basis_dimension).collect::<BTreeSet<_>>();
    if rows != expected_rows {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.local_orbitals.row_index"),
            expected: format!(
                "complete nonoverlapping rows {n_pauli}..{}",
                record.basis_dimension
            ),
            actual: format!("{} distinct in-range rows", rows.len()),
        }
        .into());
    }
    Ok(())
}

fn require_valid_dirac_channel(
    path: &str,
    signed_kappa: i64,
    twice_mu: i64,
) -> Result<(), IoError> {
    if signed_kappa == 0 {
        return Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "nonzero signed kappa".to_owned(),
            actual: "0".to_owned(),
        }
        .into());
    }
    let abs_kappa = signed_kappa
        .checked_abs()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "signed_kappa with |kappa| fitting i64".to_owned(),
            actual: signed_kappa.to_string(),
        })?;
    let twice_abs = abs_kappa
        .checked_mul(2)
        .ok_or_else(|| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "2*|kappa| fitting i64".to_owned(),
            actual: abs_kappa.to_string(),
        })?;
    let twice_j = twice_abs
        .checked_sub(1)
        .ok_or_else(|| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "2*|kappa|-1 fitting i64".to_owned(),
            actual: twice_abs.to_string(),
        })?;
    let abs_mu = twice_mu
        .checked_abs()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "twice_mu with |2mu| fitting i64".to_owned(),
            actual: twice_mu.to_string(),
        })?;
    if abs_mu > twice_j {
        return Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: format!("|twice_mu| <= 2*|kappa|-1 = {twice_j}"),
            actual: twice_mu.to_string(),
        }
        .into());
    }
    let shifted = twice_mu
        .checked_add(twice_j)
        .ok_or_else(|| ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "twice_mu + (2*|kappa|-1) fitting i64".to_owned(),
            actual: format!("twice_mu={twice_mu} twice_j={twice_j}"),
        })?;
    match shifted.checked_rem_euclid(2) {
        Some(0) => Ok(()),
        _ => Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: format!("twice_mu has same parity as 2*|kappa|-1 = {twice_j} (step 2)"),
            actual: twice_mu.to_string(),
        }
        .into()),
    }
}

fn validate_site_match(
    n_sites: usize,
    n_plane_waves: usize,
    site: usize,
    matching: &SpinorSiteMatchRefV1<'_>,
    local_orbitals: &SpinorLocalOrbitalTableRefV1<'_>,
    path: &str,
) -> Result<(), IoError> {
    if matching.site_index != site {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.site_matches[{site}].site_index"),
            expected: site.to_string(),
            actual: matching.site_index.to_string(),
        }
        .into());
    }
    if site >= n_sites {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.site_matches[{site}].site_index"),
            expected: format!("index < {n_sites}"),
            actual: site.to_string(),
        }
        .into());
    }
    if matching.n_projection == 0 {
        return Err(ValidationError::Empty {
            path: format!("{path}.site_matches[{site}].projection"),
        }
        .into());
    }
    for (name, values) in [
        ("coordinate", matching.coordinate),
        ("signed_kappa", matching.signed_kappa),
        ("twice_mu", matching.twice_mu),
        ("radial_n", matching.radial_n),
    ] {
        require_len(
            &format!("{path}.site_matches[{site}].{name}"),
            matching.n_projection,
            values.len(),
        )?;
    }
    let mut ids = BTreeSet::new();
    let mut prefix_len = matching.n_projection;
    for coord in 0..matching.n_projection {
        let stored = require_nonnegative_index(
            &format!("{path}.site_matches[{site}].coordinate[{coord}]"),
            matching.coordinate[coord],
        )?;
        if stored != coord {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.site_matches[{site}].coordinate[{coord}]"),
                expected: coord.to_string(),
                actual: stored.to_string(),
            }
            .into());
        }
        let radial_n = require_nonnegative_index(
            &format!("{path}.site_matches[{site}].radial_n[{coord}]"),
            matching.radial_n[coord],
        )?;
        if radial_n >= 2 && prefix_len == matching.n_projection {
            prefix_len = coord;
        }
        if prefix_len != matching.n_projection && radial_n <= 1 {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.site_matches[{site}].projection"),
                expected: "APW n<=1 prefix then LO n>=2 tail".to_owned(),
                actual: format!("radial_n={radial_n} after APW prefix at {prefix_len}"),
            }
            .into());
        }
        require_valid_dirac_channel(
            &format!("{path}.site_matches[{site}].projection[{coord}]"),
            matching.signed_kappa[coord],
            matching.twice_mu[coord],
        )?;
        let id = (
            matching.signed_kappa[coord],
            matching.twice_mu[coord],
            radial_n,
        );
        if !ids.insert(id) {
            return Err(ValidationError::Duplicate {
                path: format!("{path}.site_matches[{site}].projection"),
                key: format!("kappa={} twice_mu={} n={}", id.0, id.1, id.2),
            }
            .into());
        }
    }
    if prefix_len == 0 || !prefix_len.is_multiple_of(2) {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.site_matches[{site}].apw_projection"),
            expected: "nonempty APW prefix of (n=0,n=1) channel pairs".to_owned(),
            actual: prefix_len.to_string(),
        }
        .into());
    }
    let mut apw_channels = BTreeSet::new();
    let mut previous_channel: Option<(i64, i64)> = None;
    for channel in 0..(prefix_len / 2) {
        let first = 2 * channel;
        let second = first + 1;
        let n0 = matching.radial_n[first];
        let n1 = matching.radial_n[second];
        if n0 != 0 || n1 != 1 {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.site_matches[{site}].projection[{first}..{second}]"),
                expected: "(signed_kappa,twice_mu,radial_n=0) then (same,radial_n=1)".to_owned(),
                actual: format!("n={n0},{n1}"),
            }
            .into());
        }
        if matching.signed_kappa[first] != matching.signed_kappa[second]
            || matching.twice_mu[first] != matching.twice_mu[second]
        {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.site_matches[{site}].projection[{first}..{second}]"),
                expected: "identical (signed_kappa,twice_mu) for n=0 then n=1".to_owned(),
                actual: format!(
                    "kappa={},{} twice_mu={},{}",
                    matching.signed_kappa[first],
                    matching.signed_kappa[second],
                    matching.twice_mu[first],
                    matching.twice_mu[second]
                ),
            }
            .into());
        }
        let identity = (matching.signed_kappa[first], matching.twice_mu[first]);
        if !apw_channels.insert(identity) {
            return Err(ValidationError::Duplicate {
                path: format!("{path}.site_matches[{site}].apw_projection"),
                key: format!("kappa={} twice_mu={}", identity.0, identity.1),
            }
            .into());
        }
        if let Some(previous) = previous_channel
            && identity <= previous
        {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.site_matches[{site}].apw_projection"),
                expected: format!(
                    "strictly increasing (signed_kappa, twice_mu) after ({}, {})",
                    previous.0, previous.1
                ),
                actual: format!("({}, {})", identity.0, identity.1),
            }
            .into());
        }
        previous_channel = Some(identity);
    }
    if matching.n_apw_projection != prefix_len {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.site_matches[{site}].n_apw_projection"),
            expected: prefix_len.to_string(),
            actual: matching.n_apw_projection.to_string(),
        }
        .into());
    }
    let mut projection_lo = Vec::new();
    for coord in prefix_len..matching.n_projection {
        let radial_n = require_nonnegative_index(
            &format!("{path}.site_matches[{site}].radial_n[{coord}]"),
            matching.radial_n[coord],
        )?;
        projection_lo.push((
            site,
            matching.signed_kappa[coord],
            matching.twice_mu[coord],
            radial_n,
        ));
    }
    let mut local_lo = Vec::new();
    for lo in 0..local_orbitals.n_local_orbitals {
        let lo_site = require_nonnegative_index(
            &format!("{path}.local_orbitals.site[{lo}]"),
            local_orbitals.site[lo],
        )?;
        if lo_site != site {
            continue;
        }
        let radial_n = require_nonnegative_index(
            &format!("{path}.local_orbitals.radial_n[{lo}]"),
            local_orbitals.radial_n[lo],
        )?;
        local_lo.push((
            lo_site,
            local_orbitals.signed_kappa[lo],
            local_orbitals.twice_mu[lo],
            radial_n,
        ));
    }
    if projection_lo != local_lo {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.site_matches[{site}].projection"),
            expected: format!(
                "{} local-row (site,signed_kappa,twice_mu,radial_n) identities in table order",
                local_lo.len()
            ),
            actual: format!("{} projection LO identities", projection_lo.len()),
        }
        .into());
    }
    require_flat_len(
        &format!("{path}.site_matches[{site}].matching_coefficients"),
        &[n_plane_waves, N_PAULI, prefix_len, 2],
        matching.matching_coefficients.len(),
    )?;
    require_finite_f64s(
        &format!("{path}.site_matches[{site}].matching_coefficients"),
        matching.matching_coefficients,
    )?;
    Ok(())
}

fn validate_orbitals_owned(
    header: &MldumpHeaderV1,
    orbitals: &SpinorOrbitalsV1,
) -> Result<(), IoError> {
    validate_orbitals_begin(&SpinorOrbitalsBeginV1 {
        band_window_start: orbitals.band_window_start,
        band_window_count: orbitals.band_window_count,
    })?;
    require_len(
        "orbitals.k_points",
        header.mesh.k_points.len(),
        orbitals.k_points.len(),
    )?;
    for (k, record) in orbitals.k_points.iter().enumerate() {
        let pauli_g: Vec<i32> = record
            .plane_wave_g
            .iter()
            .flat_map(|g| g.iter().copied())
            .collect();
        let k_cart: Vec<f64> = record
            .plane_wave_k_cartesian
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let q_cart: Vec<f64> = record
            .plane_wave_q_cartesian
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let lo_row: Vec<i64> = record
            .local_orbitals
            .iter()
            .map(|row| i64::try_from(row.row_index).expect("index fits i64"))
            .collect();
        let lo_site: Vec<i64> = record
            .local_orbitals
            .iter()
            .map(|row| i64::try_from(row.site).expect("index fits i64"))
            .collect();
        let lo_kappa: Vec<i64> = record
            .local_orbitals
            .iter()
            .map(|row| row.signed_kappa)
            .collect();
        let lo_mu: Vec<i64> = record
            .local_orbitals
            .iter()
            .map(|row| row.twice_mu)
            .collect();
        let lo_ord: Vec<i64> = record
            .local_orbitals
            .iter()
            .map(|row| i64::try_from(row.ordinal).expect("index fits i64"))
            .collect();
        let lo_n: Vec<i64> = record
            .local_orbitals
            .iter()
            .map(|row| i64::try_from(row.radial_n).expect("index fits i64"))
            .collect();
        let site_tables = record
            .site_matches
            .iter()
            .map(|site| {
                let mut coordinate = Vec::with_capacity(site.coordinates.len());
                let mut signed_kappa = Vec::with_capacity(site.coordinates.len());
                let mut twice_mu = Vec::with_capacity(site.coordinates.len());
                let mut radial_n = Vec::with_capacity(site.coordinates.len());
                for coord in &site.coordinates {
                    coordinate.push(i64::try_from(coord.coordinate).expect("index fits i64"));
                    signed_kappa.push(coord.signed_kappa);
                    twice_mu.push(coord.twice_mu);
                    radial_n.push(i64::try_from(coord.radial_n).expect("index fits i64"));
                }
                (site, coordinate, signed_kappa, twice_mu, radial_n)
            })
            .collect::<Vec<_>>();
        let n_pw = record.plane_wave_g.len();
        let matches = site_tables
            .iter()
            .map(|(site, coordinate, signed_kappa, twice_mu, radial_n)| {
                let denom = n_pw.saturating_mul(N_PAULI).saturating_mul(2);
                let n_apw = if denom > 0 && site.matching_coefficients.len() % denom == 0 {
                    site.matching_coefficients.len() / denom
                } else {
                    0
                };
                SpinorSiteMatchRefV1 {
                    site_index: site.site_index,
                    n_projection: site.coordinates.len(),
                    n_apw_projection: n_apw,
                    coordinate,
                    signed_kappa,
                    twice_mu,
                    radial_n,
                    matching_coefficients: &site.matching_coefficients,
                }
            })
            .collect::<Vec<_>>();
        validate_k_record(
            header,
            orbitals.band_window_count,
            &SpinorOrbitalKRefV1 {
                k_index: record.k_index,
                available_bands: record.available_bands,
                basis_dimension: record.basis_dimension,
                eigenvalues: &record.eigenvalues,
                eigenvectors: &record.eigenvectors,
                n_plane_waves: record.plane_wave_g.len(),
                plane_wave_g: &pauli_g,
                plane_wave_k_cartesian: &k_cart,
                plane_wave_q_cartesian: &q_cart,
                pauli_rows: SpinorPauliRowMapRefV1 {
                    n_row: record.pauli_rows.row_index.len(),
                    row_index: &record.pauli_rows.row_index,
                    pauli_component: &record.pauli_rows.pauli_component,
                    plane_wave_index: &record.pauli_rows.plane_wave_index,
                },
                local_orbitals: SpinorLocalOrbitalTableRefV1 {
                    n_local_orbitals: record.local_orbitals.len(),
                    row_index: &lo_row,
                    site: &lo_site,
                    signed_kappa: &lo_kappa,
                    twice_mu: &lo_mu,
                    ordinal: &lo_ord,
                    radial_n: &lo_n,
                },
                site_matches: &matches,
            },
        )?;
        if record.k_index != k {
            return Err(ValidationError::InvalidValue {
                path: format!("orbitals.k_points[{k}].k_index"),
                expected: k.to_string(),
                actual: record.k_index.to_string(),
            }
            .into());
        }
    }
    Ok(())
}

fn write_k_group(
    parent: &Group,
    band_window_count: usize,
    record: &SpinorOrbitalKRefV1<'_>,
) -> Result<(), IoError> {
    let group = create_padded_group(parent, PREFIX_K, record.k_index)?;
    write_i64_attr(&group, "k", usize_as_i64("k", record.k_index)?)?;
    write_i64_attr(
        &group,
        "available_bands",
        usize_as_i64("available_bands", record.available_bands)?,
    )?;
    write_i64_attr(
        &group,
        "basis_dimension",
        usize_as_i64("basis_dimension", record.basis_dimension)?,
    )?;
    write_f64_dataset(
        &group,
        "eigenvalues",
        &[band_window_count],
        record.eigenvalues,
        &["band"],
    )?;
    write_f64_dataset(
        &group,
        "eigenvectors",
        &[record.basis_dimension, band_window_count, 2],
        record.eigenvectors,
        &["basis_row", "band", "re_im"],
    )?;
    let basis = group.create_group(PREFIX_BASIS)?;
    write_i32_dataset(
        &basis,
        "plane_wave_g",
        &[record.n_plane_waves, 3],
        record.plane_wave_g,
        &["plane_wave", "reciprocal_axis"],
    )?;
    write_f64_dataset(
        &basis,
        "plane_wave_k_cartesian",
        &[record.n_plane_waves, 3],
        record.plane_wave_k_cartesian,
        &["plane_wave", "cartesian"],
    )?;
    write_f64_dataset(
        &basis,
        "plane_wave_q_cartesian",
        &[record.n_plane_waves, 3],
        record.plane_wave_q_cartesian,
        &["plane_wave", "cartesian"],
    )?;
    write_i64_dataset(
        &basis,
        "pauli_row_index",
        &[record.pauli_rows.n_row],
        record.pauli_rows.row_index,
        &["pauli_row"],
    )?;
    write_i64_dataset(
        &basis,
        "pauli_component",
        &[record.pauli_rows.n_row],
        record.pauli_rows.pauli_component,
        &["pauli_row"],
    )?;
    write_i64_dataset(
        &basis,
        "pauli_plane_wave_index",
        &[record.pauli_rows.n_row],
        record.pauli_rows.plane_wave_index,
        &["pauli_row"],
    )?;
    write_i64_dataset(
        &basis,
        "local_orbital_row_index",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.row_index,
        &["local_orbital"],
    )?;
    write_i64_dataset(
        &basis,
        "local_orbital_site",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.site,
        &["local_orbital"],
    )?;
    write_i64_dataset(
        &basis,
        "local_orbital_signed_kappa",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.signed_kappa,
        &["local_orbital"],
    )?;
    write_i64_dataset(
        &basis,
        "local_orbital_twice_mu",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.twice_mu,
        &["local_orbital"],
    )?;
    write_i64_dataset(
        &basis,
        "local_orbital_ordinal",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.ordinal,
        &["local_orbital"],
    )?;
    write_i64_dataset(
        &basis,
        "local_orbital_radial_n",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.radial_n,
        &["local_orbital"],
    )?;
    for matching in record.site_matches {
        write_site_match(&basis, record.n_plane_waves, matching)?;
    }
    Ok(())
}

fn write_site_match(
    basis: &Group,
    n_plane_waves: usize,
    matching: &SpinorSiteMatchRefV1<'_>,
) -> Result<(), IoError> {
    let group = create_padded_group(basis, PREFIX_SITE, matching.site_index)?;
    write_i64_attr(&group, "site", usize_as_i64("site", matching.site_index)?)?;
    write_i64_dataset(
        &group,
        "projection_coordinate",
        &[matching.n_projection],
        matching.coordinate,
        &["projection_coordinate"],
    )?;
    write_i64_dataset(
        &group,
        "projection_signed_kappa",
        &[matching.n_projection],
        matching.signed_kappa,
        &["projection_coordinate"],
    )?;
    write_i64_dataset(
        &group,
        "projection_twice_mu",
        &[matching.n_projection],
        matching.twice_mu,
        &["projection_coordinate"],
    )?;
    write_i64_dataset(
        &group,
        "projection_radial_n",
        &[matching.n_projection],
        matching.radial_n,
        &["projection_coordinate"],
    )?;
    write_f64_dataset(
        &group,
        "matching_coefficients",
        &[n_plane_waves, N_PAULI, matching.n_apw_projection, 2],
        matching.matching_coefficients,
        &[
            "plane_wave",
            "pauli_component",
            "projection_coordinate",
            "re_im",
        ],
    )?;
    Ok(())
}

fn read_k_group(
    group: &Group,
    header: &MldumpHeaderV1,
    k: usize,
    band_window_count: usize,
) -> Result<SpinorOrbitalKRecordV1, IoError> {
    let stored_k = read_usize_attr(group, "k", &format!("{}/@k", group.name()))?;
    if stored_k != k {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@k", group.name()),
            expected: k.to_string(),
            actual: stored_k.to_string(),
        }
        .into());
    }
    let available_bands = read_usize_attr(
        group,
        "available_bands",
        &format!("{}/@available_bands", group.name()),
    )?;
    let basis_dimension = read_usize_attr(
        group,
        "basis_dimension",
        &format!("{}/@basis_dimension", group.name()),
    )?;
    let eigenvalues = read_f64_dataset(group, "eigenvalues", &[band_window_count], &["band"])?;
    let eigenvectors = read_f64_dataset(
        group,
        "eigenvectors",
        &[basis_dimension, band_window_count, 2],
        &["basis_row", "band", "re_im"],
    )?;
    require_dataset_names(group, &["eigenvalues", "eigenvectors"])?;
    require_group_names(group, &[PREFIX_BASIS])?;
    let basis = group.group(PREFIX_BASIS)?;
    let n_plane_waves = dataset_leading_len(
        &basis.dataset("plane_wave_g")?,
        format!("{}/basis/plane_wave_g/shape", group.name()),
        "[n_plane_wave, 3]",
    )?;
    require_dataset_names(
        &basis,
        &[
            "plane_wave_g",
            "plane_wave_k_cartesian",
            "plane_wave_q_cartesian",
            "pauli_row_index",
            "pauli_component",
            "pauli_plane_wave_index",
            "local_orbital_row_index",
            "local_orbital_site",
            "local_orbital_signed_kappa",
            "local_orbital_twice_mu",
            "local_orbital_ordinal",
            "local_orbital_radial_n",
        ],
    )?;
    let plane_wave_g = i32_triples_to_owned(
        &read_i32_dataset(
            &basis,
            "plane_wave_g",
            &[n_plane_waves, 3],
            &["plane_wave", "reciprocal_axis"],
        )?,
        n_plane_waves,
        &format!("{}/plane_wave_g", basis.name()),
    )?;
    let plane_wave_k_cartesian = triples_to_owned(
        &read_f64_dataset(
            &basis,
            "plane_wave_k_cartesian",
            &[n_plane_waves, 3],
            &["plane_wave", "cartesian"],
        )?,
        n_plane_waves,
        &format!("{}/plane_wave_k_cartesian", basis.name()),
    )?;
    let plane_wave_q_cartesian = triples_to_owned(
        &read_f64_dataset(
            &basis,
            "plane_wave_q_cartesian",
            &[n_plane_waves, 3],
            &["plane_wave", "cartesian"],
        )?,
        n_plane_waves,
        &format!("{}/plane_wave_q_cartesian", basis.name()),
    )?;
    let n_pauli_row = N_PAULI * n_plane_waves;
    let pauli_rows = SpinorPauliRowMapV1 {
        row_index: read_i64_dataset(&basis, "pauli_row_index", &[n_pauli_row], &["pauli_row"])?,
        pauli_component: read_i64_dataset(
            &basis,
            "pauli_component",
            &[n_pauli_row],
            &["pauli_row"],
        )?,
        plane_wave_index: read_i64_dataset(
            &basis,
            "pauli_plane_wave_index",
            &[n_pauli_row],
            &["pauli_row"],
        )?,
    };
    let n_lo = dataset_leading_len(
        &basis.dataset("local_orbital_row_index")?,
        format!("{}/local_orbital_row_index/shape", basis.name()),
        "[n_local_orbital]",
    )?;
    let lo_row = read_i64_dataset(
        &basis,
        "local_orbital_row_index",
        &[n_lo],
        &["local_orbital"],
    )?;
    let lo_site = read_i64_dataset(&basis, "local_orbital_site", &[n_lo], &["local_orbital"])?;
    let lo_kappa = read_i64_dataset(
        &basis,
        "local_orbital_signed_kappa",
        &[n_lo],
        &["local_orbital"],
    )?;
    let lo_mu = read_i64_dataset(
        &basis,
        "local_orbital_twice_mu",
        &[n_lo],
        &["local_orbital"],
    )?;
    let lo_ord = read_i64_dataset(&basis, "local_orbital_ordinal", &[n_lo], &["local_orbital"])?;
    let lo_n = read_i64_dataset(
        &basis,
        "local_orbital_radial_n",
        &[n_lo],
        &["local_orbital"],
    )?;
    let mut local_orbitals = Vec::with_capacity(n_lo);
    for lo in 0..n_lo {
        local_orbitals.push(SpinorLocalOrbitalRowV1 {
            row_index: require_nonnegative_index(
                &format!("{}/local_orbital_row_index[{lo}]", basis.name()),
                lo_row[lo],
            )?,
            site: require_nonnegative_index(
                &format!("{}/local_orbital_site[{lo}]", basis.name()),
                lo_site[lo],
            )?,
            signed_kappa: lo_kappa[lo],
            twice_mu: lo_mu[lo],
            ordinal: require_nonnegative_index(
                &format!("{}/local_orbital_ordinal[{lo}]", basis.name()),
                lo_ord[lo],
            )?,
            radial_n: require_nonnegative_index(
                &format!("{}/local_orbital_radial_n[{lo}]", basis.name()),
                lo_n[lo],
            )?,
        });
    }
    let n_site = header.geometry.sites.len();
    let site_groups = collect_padded_groups(&basis, PREFIX_SITE)?;
    require_len(
        &format!("{}/site_*", basis.name()),
        n_site,
        site_groups.len(),
    )?;
    let mut members = vec![
        "plane_wave_g".to_owned(),
        "plane_wave_k_cartesian".to_owned(),
        "plane_wave_q_cartesian".to_owned(),
        "pauli_row_index".to_owned(),
        "pauli_component".to_owned(),
        "pauli_plane_wave_index".to_owned(),
        "local_orbital_row_index".to_owned(),
        "local_orbital_site".to_owned(),
        "local_orbital_signed_kappa".to_owned(),
        "local_orbital_twice_mu".to_owned(),
        "local_orbital_ordinal".to_owned(),
        "local_orbital_radial_n".to_owned(),
    ];
    members.extend((0..n_site).map(|site| padded_child(PREFIX_SITE, site)));
    require_exact_members(&basis, members)?;
    let mut site_matches = Vec::with_capacity(n_site);
    for (site, site_group) in site_groups.iter().enumerate() {
        site_matches.push(read_site_match(site_group, site, n_plane_waves)?);
    }
    Ok(SpinorOrbitalKRecordV1 {
        k_index: k,
        available_bands,
        basis_dimension,
        eigenvalues,
        eigenvectors,
        plane_wave_g,
        plane_wave_k_cartesian,
        plane_wave_q_cartesian,
        pauli_rows,
        local_orbitals,
        site_matches,
    })
}

fn read_site_match(
    group: &Group,
    site: usize,
    n_plane_waves: usize,
) -> Result<SpinorSiteMatchV1, IoError> {
    let stored = read_usize_attr(group, "site", &format!("{}/@site", group.name()))?;
    if stored != site {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@site", group.name()),
            expected: site.to_string(),
            actual: stored.to_string(),
        }
        .into());
    }
    let n_projection = dataset_leading_len(
        &group.dataset("projection_coordinate")?,
        format!("{}/projection_coordinate/shape", group.name()),
        "[n_projection]",
    )?;
    let coordinate = read_i64_dataset(
        group,
        "projection_coordinate",
        &[n_projection],
        &["projection_coordinate"],
    )?;
    let signed_kappa = read_i64_dataset(
        group,
        "projection_signed_kappa",
        &[n_projection],
        &["projection_coordinate"],
    )?;
    let twice_mu = read_i64_dataset(
        group,
        "projection_twice_mu",
        &[n_projection],
        &["projection_coordinate"],
    )?;
    let radial_n = read_i64_dataset(
        group,
        "projection_radial_n",
        &[n_projection],
        &["projection_coordinate"],
    )?;
    let matching_shape = group.dataset("matching_coefficients")?.shape();
    if matching_shape.len() != 4 {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/matching_coefficients/shape", group.name()),
            expected: "[plane_wave, pauli_component, apw_projection, re_im]".to_owned(),
            actual: format!("{matching_shape:?}"),
        }
        .into());
    }
    let n_apw = matching_shape[2];
    let mut coordinates = Vec::with_capacity(n_projection);
    for index in 0..n_projection {
        let n = require_nonnegative_index(
            &format!("{}/projection_radial_n[{index}]", group.name()),
            radial_n[index],
        )?;
        coordinates.push(SpinorProjectionCoordV1 {
            coordinate: require_nonnegative_index(
                &format!("{}/projection_coordinate[{index}]", group.name()),
                coordinate[index],
            )?,
            signed_kappa: signed_kappa[index],
            twice_mu: twice_mu[index],
            radial_n: n,
        });
    }
    let matching_coefficients = read_f64_dataset(
        group,
        "matching_coefficients",
        &[n_plane_waves, N_PAULI, n_apw, 2],
        &[
            "plane_wave",
            "pauli_component",
            "projection_coordinate",
            "re_im",
        ],
    )?;
    Ok(SpinorSiteMatchV1 {
        site_index: site,
        coordinates,
        matching_coefficients,
    })
}
