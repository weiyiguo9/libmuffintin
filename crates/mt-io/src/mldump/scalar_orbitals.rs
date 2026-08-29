//! Scalar Koelling–Harmon orbital payload for MLDUMP v1.

use hdf5_metno::Group;

use super::{
    GROUP_ORBITALS, MldumpHeaderV1, PREFIX_BASIS, PREFIX_K, PREFIX_SITE, PREFIX_SPIN,
    collect_padded_groups, create_padded_group, dataset_leading_len, i32_triples_to_owned,
    padded_child, read_f64_dataset, read_i32_dataset, read_i64_attr, read_i64_dataset,
    read_usize_attr, reopen_present_group, require_dataset_names, require_exact_members,
    require_finite_f64s, require_flat_len, require_group_names, require_len,
    require_nonnegative_index, require_status_present, require_str_array_attr, require_str_attr,
    triples_to_owned, usize_as_i64, validate_plane_wave_identity, write_f64_dataset,
    write_i32_dataset, write_i64_attr, write_i64_dataset, write_str_array_attr, write_str_attr,
};
use crate::error::{IoError, ValidationError};

/// Representation tag stored on `/orbitals`.
pub const MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON: &str = "scalar_koelling_harmon";
/// Occupations are not serialized in this stage.
pub const MLDUMP_OCCUPATIONS_NOT_EXPORTED: &str = "not_exported_not_available";

const RADIAL_U_UDOT: [&str; 2] = ["u", "udot"];

/// Shared `/orbitals` attributes for a streaming scalar session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarOrbitalsBeginV1 {
    pub spin_count: usize,
    pub band_window_start: i64,
    /// Common leading exported window. Eigenvalues and eigenvector columns use
    /// this count, not per-$k$ [`ScalarOrbitalKRefV1::available_bands`].
    pub band_window_count: usize,
}

/// One spin/$k$ compiled basis and eigenvectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarOrbitalKRefV1<'a> {
    pub k_index: usize,
    /// Per-$k$ available band count. May exceed the exported common window;
    /// it is metadata only and does not size the stored eigendata.
    pub available_bands: usize,
    pub basis_dimension: usize,
    pub eigenvalues: &'a [f64],
    /// C-order `[basis_row, band, re_im]`.
    pub eigenvectors: &'a [f64],
    pub n_plane_waves: usize,
    pub plane_wave_g: &'a [i32],
    pub plane_wave_k_cartesian: &'a [f64],
    pub plane_wave_q_cartesian: &'a [f64],
    pub site_matches: &'a [ScalarApwSiteMatchRefV1<'a>],
    pub local_orbitals: ScalarLocalOrbitalTableRefV1<'a>,
}

/// APW matching coefficients for one muffin-tin site, tied to the PW rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarApwSiteMatchRefV1<'a> {
    pub site_index: usize,
    pub n_lm: usize,
    pub lm_l: &'a [i32],
    pub lm_m: &'a [i32],
    /// C-order `[plane_wave, lm, radial_component, re_im]`.
    pub matching_coefficients: &'a [f64],
}

/// Parallel LO identity tables. Each non-PW basis row appears exactly once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarLocalOrbitalTableRefV1<'a> {
    pub n_local_orbitals: usize,
    pub row_index: &'a [i64],
    pub site: &'a [i64],
    pub l: &'a [i64],
    pub m: &'a [i64],
    pub ordinal: &'a [i64],
    pub radial_n: &'a [i64],
}

/// Owned scalar orbital section returned by the reader.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarOrbitalsV1 {
    pub spin_count: usize,
    pub band_window_start: i64,
    /// Common leading exported window. Eigenvalue/eigenvector lengths equal this
    /// count even when a $k$ point reports a larger `available_bands`.
    pub band_window_count: usize,
    pub spins: Vec<ScalarOrbitalSpinV1>,
}

/// Owned spin channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarOrbitalSpinV1 {
    pub spin: usize,
    pub k_points: Vec<ScalarOrbitalKRecordV1>,
}

/// Owned per-$k$ orbital record with declared PW/LO dimensions.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarOrbitalKRecordV1 {
    pub k_index: usize,
    /// Per-$k$ available band metadata; may exceed [`ScalarOrbitalsV1::band_window_count`].
    pub available_bands: usize,
    pub basis_dimension: usize,
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<f64>,
    pub plane_wave_g: Vec<[i32; 3]>,
    pub plane_wave_k_cartesian: Vec<[f64; 3]>,
    pub plane_wave_q_cartesian: Vec<[f64; 3]>,
    pub site_matches: Vec<ScalarApwSiteMatchV1>,
    pub local_orbitals: Vec<ScalarLocalOrbitalRowV1>,
}

/// Owned APW matching block for one site.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarApwSiteMatchV1 {
    pub site_index: usize,
    pub lm_l: Vec<i32>,
    pub lm_m: Vec<i32>,
    pub matching_coefficients: Vec<f64>,
}

/// One confined local-orbital basis row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScalarLocalOrbitalRowV1 {
    pub row_index: usize,
    pub site: usize,
    pub l: i64,
    pub m: i64,
    pub ordinal: usize,
    pub radial_n: usize,
}

pub(crate) fn begin_scalar_orbitals(
    file: &Group,
    begin: &ScalarOrbitalsBeginV1,
) -> Result<(), IoError> {
    validate_orbitals_begin(begin)?;
    let group = reopen_present_group(file, GROUP_ORBITALS)?;
    write_str_attr(
        &group,
        "representation",
        MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON,
    )?;
    write_i64_attr(
        &group,
        "spin_count",
        usize_as_i64("/orbitals/@spin_count", begin.spin_count)?,
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
    for spin in 0..begin.spin_count {
        let spin_group = create_padded_group(&group, PREFIX_SPIN, spin)?;
        write_i64_attr(
            &spin_group,
            "spin",
            usize_as_i64(&format!("/orbitals/spin_{spin:06}/@spin"), spin)?,
        )?;
    }
    Ok(())
}

pub(crate) fn write_scalar_orbital_k(
    file: &Group,
    header: &MldumpHeaderV1,
    spin: usize,
    band_window_count: usize,
    record: &ScalarOrbitalKRefV1<'_>,
) -> Result<(), IoError> {
    validate_k_record(
        header,
        header.geometry.sites.len(),
        band_window_count,
        spin,
        record.k_index,
        record,
    )?;
    let spin_group = file
        .group(GROUP_ORBITALS)?
        .group(&padded_child(PREFIX_SPIN, spin))?;
    write_k_group(&spin_group, spin, band_window_count, record)
}

pub(crate) fn read_scalar_orbitals(
    file: &Group,
    header: &MldumpHeaderV1,
) -> Result<ScalarOrbitalsV1, IoError> {
    let group = file.group(GROUP_ORBITALS)?;
    require_status_present(&group)?;
    require_str_attr(
        &group,
        "representation",
        MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON,
    )?;
    require_str_attr(
        &group,
        "occupations_status",
        MLDUMP_OCCUPATIONS_NOT_EXPORTED,
    )?;
    let spin_count = read_usize_attr(&group, "spin_count", "/orbitals/@spin_count")?;
    if spin_count != 2 {
        return Err(ValidationError::InvalidValue {
            path: "/orbitals/@spin_count".to_owned(),
            expected: "2".to_owned(),
            actual: spin_count.to_string(),
        }
        .into());
    }
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
    let spin_groups = collect_padded_groups(&group, PREFIX_SPIN)?;
    require_len("/orbitals/spin_*", spin_count, spin_groups.len())?;
    require_exact_members(
        &group,
        (0..spin_count).map(|spin| padded_child(PREFIX_SPIN, spin)),
    )?;
    let mut spins = Vec::with_capacity(spin_count);
    for (spin, spin_group) in spin_groups.iter().enumerate() {
        spins.push(read_spin_group(
            spin_group,
            header,
            spin,
            band_window_count,
        )?);
    }
    let orbitals = ScalarOrbitalsV1 {
        spin_count,
        band_window_start,
        band_window_count,
        spins,
    };
    validate_orbitals_owned(header, &orbitals)?;
    Ok(orbitals)
}

fn validate_orbitals_begin(begin: &ScalarOrbitalsBeginV1) -> Result<(), IoError> {
    if begin.spin_count != 2 {
        return Err(ValidationError::InvalidValue {
            path: "orbitals.spin_count".to_owned(),
            expected: "2".to_owned(),
            actual: begin.spin_count.to_string(),
        }
        .into());
    }
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
    n_sites: usize,
    band_window_count: usize,
    spin: usize,
    k: usize,
    record: &ScalarOrbitalKRefV1<'_>,
) -> Result<(), IoError> {
    let path = format!("orbitals.spins[{spin}].k_points[{k}]");
    if record.k_index != k {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.k_index"),
            expected: k.to_string(),
            actual: record.k_index.to_string(),
        }
        .into());
    }
    require_exported_eigendata(
        &path,
        record.available_bands,
        band_window_count,
        record.eigenvalues,
        record.eigenvectors,
        record.basis_dimension,
    )?;
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
    require_len(
        &format!("{path}.site_matches"),
        n_sites,
        record.site_matches.len(),
    )?;
    for (site, matching) in record.site_matches.iter().enumerate() {
        validate_site_match(record.n_plane_waves, site, matching, &path)?;
    }
    validate_local_orbitals(record, &path)?;
    Ok(())
}

fn validate_site_match(
    n_plane_waves: usize,
    site: usize,
    matching: &ScalarApwSiteMatchRefV1<'_>,
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
    if matching.n_lm == 0 {
        return Err(ValidationError::Empty {
            path: format!("{path}.site_matches[{site}].lm"),
        }
        .into());
    }
    require_len(
        &format!("{path}.site_matches[{site}].lm_l"),
        matching.n_lm,
        matching.lm_l.len(),
    )?;
    require_len(
        &format!("{path}.site_matches[{site}].lm_m"),
        matching.n_lm,
        matching.lm_m.len(),
    )?;
    for lm in 0..matching.n_lm {
        let l = matching.lm_l[lm];
        let m = matching.lm_m[lm];
        if l < 0 || m < -l || m > l {
            return Err(invalid_lm(
                format!("{path}.site_matches[{site}].lm[{lm}]"),
                i64::from(l),
                i64::from(m),
            ));
        }
    }
    require_flat_len(
        &format!("{path}.site_matches[{site}].matching_coefficients"),
        &[n_plane_waves, matching.n_lm, 2, 2],
        matching.matching_coefficients.len(),
    )?;
    require_finite_f64s(
        &format!("{path}.site_matches[{site}].matching_coefficients"),
        matching.matching_coefficients,
    )?;
    Ok(())
}

fn validate_local_orbitals(record: &ScalarOrbitalKRefV1<'_>, path: &str) -> Result<(), IoError> {
    let table = record.local_orbitals;
    let n_lo = table.n_local_orbitals;
    let expected = record
        .basis_dimension
        .checked_sub(record.n_plane_waves)
        .ok_or_else(|| ValidationError::InvalidValue {
            path: format!("{path}.basis_dimension"),
            expected: format!(">= n_plane_waves {}", record.n_plane_waves),
            actual: record.basis_dimension.to_string(),
        })?;
    require_len(&format!("{path}.local_orbitals"), expected, n_lo)?;
    for (name, values) in [
        ("row_index", table.row_index),
        ("site", table.site),
        ("l", table.l),
        ("m", table.m),
        ("ordinal", table.ordinal),
        ("radial_n", table.radial_n),
    ] {
        require_len(&format!("{path}.local_orbitals.{name}"), n_lo, values.len())?;
    }
    let mut seen = vec![false; n_lo];
    for lo in 0..n_lo {
        let row = require_nonnegative_index(
            &format!("{path}.local_orbitals.row_index[{lo}]"),
            table.row_index[lo],
        )?;
        if row < record.n_plane_waves || row >= record.basis_dimension {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}.local_orbitals.row_index[{lo}]"),
                expected: format!("in [{}, {})", record.n_plane_waves, record.basis_dimension),
                actual: row.to_string(),
            }
            .into());
        }
        let slot = row - record.n_plane_waves;
        if seen[slot] {
            return Err(ValidationError::Duplicate {
                path: format!("{path}.local_orbitals.row_index"),
                key: row.to_string(),
            }
            .into());
        }
        seen[slot] = true;
        let l = table.l[lo];
        let m = table.m[lo];
        if l < 0 || m < -l || m > l {
            return Err(invalid_lm(format!("{path}.local_orbitals[{lo}]"), l, m));
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
        require_nonnegative_index(&format!("{path}.local_orbitals.site[{lo}]"), table.site[lo])?;
    }
    Ok(())
}

fn invalid_lm(path: String, l: i64, m: i64) -> IoError {
    match (u32::try_from(l), i32::try_from(m)) {
        (Ok(l), Ok(m)) => ValidationError::InvalidLm { path, l, m }.into(),
        _ => ValidationError::InvalidValue {
            path,
            expected: "l >= 0 and |m| <= l".to_owned(),
            actual: format!("l={l} m={m}"),
        }
        .into(),
    }
}

fn validate_orbitals_owned(
    header: &MldumpHeaderV1,
    orbitals: &ScalarOrbitalsV1,
) -> Result<(), IoError> {
    validate_orbitals_begin(&ScalarOrbitalsBeginV1 {
        spin_count: orbitals.spin_count,
        band_window_start: orbitals.band_window_start,
        band_window_count: orbitals.band_window_count,
    })?;
    require_len("orbitals.spins", orbitals.spin_count, orbitals.spins.len())?;
    let n_sites = header.geometry.sites.len();
    for (spin, channel) in orbitals.spins.iter().enumerate() {
        require_len(
            &format!("orbitals.spins[{spin}].k_points"),
            header.mesh.k_points.len(),
            channel.k_points.len(),
        )?;
        for (k, record) in channel.k_points.iter().enumerate() {
            let g = flatten_i32_triples(&record.plane_wave_g);
            let k_cart = flatten_f64_triples(&record.plane_wave_k_cartesian);
            let q_cart = flatten_f64_triples(&record.plane_wave_q_cartesian);
            let lo_row = record
                .local_orbitals
                .iter()
                .map(|row| i64::try_from(row.row_index).expect("index fits i64"))
                .collect::<Vec<_>>();
            let lo_site = record
                .local_orbitals
                .iter()
                .map(|row| i64::try_from(row.site).expect("index fits i64"))
                .collect::<Vec<_>>();
            let lo_l = record
                .local_orbitals
                .iter()
                .map(|row| row.l)
                .collect::<Vec<_>>();
            let lo_m = record
                .local_orbitals
                .iter()
                .map(|row| row.m)
                .collect::<Vec<_>>();
            let lo_ord = record
                .local_orbitals
                .iter()
                .map(|row| i64::try_from(row.ordinal).expect("index fits i64"))
                .collect::<Vec<_>>();
            let lo_n = record
                .local_orbitals
                .iter()
                .map(|row| i64::try_from(row.radial_n).expect("index fits i64"))
                .collect::<Vec<_>>();
            let matches = record
                .site_matches
                .iter()
                .map(|matching| ScalarApwSiteMatchRefV1 {
                    site_index: matching.site_index,
                    n_lm: matching.lm_l.len(),
                    lm_l: &matching.lm_l,
                    lm_m: &matching.lm_m,
                    matching_coefficients: &matching.matching_coefficients,
                })
                .collect::<Vec<_>>();
            validate_k_record(
                header,
                n_sites,
                orbitals.band_window_count,
                spin,
                k,
                &ScalarOrbitalKRefV1 {
                    k_index: record.k_index,
                    available_bands: record.available_bands,
                    basis_dimension: record.basis_dimension,
                    eigenvalues: &record.eigenvalues,
                    eigenvectors: &record.eigenvectors,
                    n_plane_waves: record.plane_wave_g.len(),
                    plane_wave_g: &g,
                    plane_wave_k_cartesian: &k_cart,
                    plane_wave_q_cartesian: &q_cart,
                    site_matches: &matches,
                    local_orbitals: ScalarLocalOrbitalTableRefV1 {
                        n_local_orbitals: record.local_orbitals.len(),
                        row_index: &lo_row,
                        site: &lo_site,
                        l: &lo_l,
                        m: &lo_m,
                        ordinal: &lo_ord,
                        radial_n: &lo_n,
                    },
                },
            )?;
        }
    }
    Ok(())
}

fn flatten_i32_triples(values: &[[i32; 3]]) -> Vec<i32> {
    values.iter().flat_map(|g| g.iter().copied()).collect()
}

fn flatten_f64_triples(values: &[[f64; 3]]) -> Vec<f64> {
    values.iter().flat_map(|v| v.iter().copied()).collect()
}

fn require_exported_eigendata(
    path: &str,
    available_bands: usize,
    band_window_count: usize,
    eigenvalues: &[f64],
    eigenvectors: &[f64],
    basis_dimension: usize,
) -> Result<(), IoError> {
    if available_bands == 0 || available_bands < band_window_count {
        return Err(ValidationError::InvalidValue {
            path: format!("{path}.available_bands"),
            expected: format!(">= band_window_count {band_window_count}"),
            actual: available_bands.to_string(),
        }
        .into());
    }
    require_len(
        &format!("{path}.eigenvalues"),
        band_window_count,
        eigenvalues.len(),
    )?;
    require_finite_f64s(&format!("{path}.eigenvalues"), eigenvalues)?;
    require_flat_len(
        &format!("{path}.eigenvectors"),
        &[basis_dimension, band_window_count, 2],
        eigenvectors.len(),
    )?;
    require_finite_f64s(&format!("{path}.eigenvectors"), eigenvectors)?;
    Ok(())
}

fn write_k_group(
    parent: &Group,
    spin: usize,
    band_window_count: usize,
    record: &ScalarOrbitalKRefV1<'_>,
) -> Result<(), IoError> {
    let group = create_padded_group(parent, PREFIX_K, record.k_index)?;
    write_i64_attr(&group, "spin", usize_as_i64("spin", spin)?)?;
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
        "local_orbital_l",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.l,
        &["local_orbital"],
    )?;
    write_i64_dataset(
        &basis,
        "local_orbital_m",
        &[record.local_orbitals.n_local_orbitals],
        record.local_orbitals.m,
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
    matching: &ScalarApwSiteMatchRefV1<'_>,
) -> Result<(), IoError> {
    let group = create_padded_group(basis, PREFIX_SITE, matching.site_index)?;
    write_i64_attr(&group, "site", usize_as_i64("site", matching.site_index)?)?;
    write_i32_dataset(&group, "lm_l", &[matching.n_lm], matching.lm_l, &["lm"])?;
    write_i32_dataset(&group, "lm_m", &[matching.n_lm], matching.lm_m, &["lm"])?;
    write_f64_dataset(
        &group,
        "matching_coefficients",
        &[n_plane_waves, matching.n_lm, 2, 2],
        matching.matching_coefficients,
        &["plane_wave", "lm", "radial_component", "re_im"],
    )?;
    let dataset = group.dataset("matching_coefficients")?;
    write_str_array_attr(&dataset, "radial_component_labels", &RADIAL_U_UDOT)?;
    Ok(())
}

fn read_spin_group(
    group: &Group,
    header: &MldumpHeaderV1,
    spin: usize,
    band_window_count: usize,
) -> Result<ScalarOrbitalSpinV1, IoError> {
    let stored_spin = read_usize_attr(group, "spin", &format!("{}/@spin", group.name()))?;
    if stored_spin != spin {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@spin", group.name()),
            expected: spin.to_string(),
            actual: stored_spin.to_string(),
        }
        .into());
    }
    let n_k = header.mesh.k_points.len();
    let k_groups = collect_padded_groups(group, PREFIX_K)?;
    require_len(&format!("{}/k_*", group.name()), n_k, k_groups.len())?;
    require_exact_members(group, (0..n_k).map(|k| padded_child(PREFIX_K, k)))?;
    let mut k_points = Vec::with_capacity(n_k);
    for (k, k_group) in k_groups.iter().enumerate() {
        k_points.push(read_k_group(k_group, header, spin, k, band_window_count)?);
    }
    Ok(ScalarOrbitalSpinV1 { spin, k_points })
}

fn read_k_group(
    group: &Group,
    header: &MldumpHeaderV1,
    spin: usize,
    k: usize,
    band_window_count: usize,
) -> Result<ScalarOrbitalKRecordV1, IoError> {
    let stored_spin = read_usize_attr(group, "spin", &format!("{}/@spin", group.name()))?;
    let stored_k = read_usize_attr(group, "k", &format!("{}/@k", group.name()))?;
    if stored_spin != spin || stored_k != k {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@k", group.name()),
            expected: format!("spin={spin} k={k}"),
            actual: format!("spin={stored_spin} k={stored_k}"),
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
    require_dataset_names(group, &["eigenvalues", "eigenvectors"])?;
    require_group_names(group, &[PREFIX_BASIS])?;
    let eigenvalues = read_f64_dataset(group, "eigenvalues", &[band_window_count], &["band"])?;
    let eigenvectors = read_f64_dataset(
        group,
        "eigenvectors",
        &[basis_dimension, band_window_count, 2],
        &["basis_row", "band", "re_im"],
    )?;
    require_exported_eigendata(
        &group.name(),
        available_bands,
        band_window_count,
        &eigenvalues,
        &eigenvectors,
        basis_dimension,
    )?;
    let basis = group.group(PREFIX_BASIS)?;
    let n_pw = dataset_leading_len(
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
            "local_orbital_row_index",
            "local_orbital_site",
            "local_orbital_l",
            "local_orbital_m",
            "local_orbital_ordinal",
            "local_orbital_radial_n",
        ],
    )?;
    let plane_wave_g = i32_triples_to_owned(
        &read_i32_dataset(
            &basis,
            "plane_wave_g",
            &[n_pw, 3],
            &["plane_wave", "reciprocal_axis"],
        )?,
        n_pw,
        &format!("{}/basis/plane_wave_g", group.name()),
    )?;
    let plane_wave_k_cartesian = triples_to_owned(
        &read_f64_dataset(
            &basis,
            "plane_wave_k_cartesian",
            &[n_pw, 3],
            &["plane_wave", "cartesian"],
        )?,
        n_pw,
        &format!("{}/basis/plane_wave_k_cartesian", group.name()),
    )?;
    let plane_wave_q_cartesian = triples_to_owned(
        &read_f64_dataset(
            &basis,
            "plane_wave_q_cartesian",
            &[n_pw, 3],
            &["plane_wave", "cartesian"],
        )?,
        n_pw,
        &format!("{}/basis/plane_wave_q_cartesian", group.name()),
    )?;
    let n_sites = header.geometry.sites.len();
    let site_groups = collect_padded_groups(&basis, PREFIX_SITE)?;
    require_len(
        &format!("{}/basis/site_*", group.name()),
        n_sites,
        site_groups.len(),
    )?;
    let mut site_names = (0..n_sites)
        .map(|site| padded_child(PREFIX_SITE, site))
        .collect::<Vec<_>>();
    site_names.extend(
        [
            "plane_wave_g",
            "plane_wave_k_cartesian",
            "plane_wave_q_cartesian",
            "local_orbital_row_index",
            "local_orbital_site",
            "local_orbital_l",
            "local_orbital_m",
            "local_orbital_ordinal",
            "local_orbital_radial_n",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    require_exact_members(&basis, site_names)?;
    let mut site_matches = Vec::with_capacity(n_sites);
    for (site, site_group) in site_groups.iter().enumerate() {
        site_matches.push(read_site_match(site_group, n_pw, site)?);
    }
    let n_lo = dataset_leading_len(
        &basis.dataset("local_orbital_row_index")?,
        format!("{}/basis/local_orbital_row_index/shape", group.name()),
        "[n_local_orbital]",
    )?;
    let row_index = read_i64_dataset(
        &basis,
        "local_orbital_row_index",
        &[n_lo],
        &["local_orbital"],
    )?;
    let site = read_i64_dataset(&basis, "local_orbital_site", &[n_lo], &["local_orbital"])?;
    let l = read_i64_dataset(&basis, "local_orbital_l", &[n_lo], &["local_orbital"])?;
    let m = read_i64_dataset(&basis, "local_orbital_m", &[n_lo], &["local_orbital"])?;
    let ordinal = read_i64_dataset(&basis, "local_orbital_ordinal", &[n_lo], &["local_orbital"])?;
    let radial_n = read_i64_dataset(
        &basis,
        "local_orbital_radial_n",
        &[n_lo],
        &["local_orbital"],
    )?;
    let mut local_orbitals = Vec::with_capacity(n_lo);
    for lo in 0..n_lo {
        local_orbitals.push(ScalarLocalOrbitalRowV1 {
            row_index: require_nonnegative_index("local_orbital_row_index", row_index[lo])?,
            site: require_nonnegative_index("local_orbital_site", site[lo])?,
            l: l[lo],
            m: m[lo],
            ordinal: require_nonnegative_index("local_orbital_ordinal", ordinal[lo])?,
            radial_n: require_nonnegative_index("local_orbital_radial_n", radial_n[lo])?,
        });
    }
    Ok(ScalarOrbitalKRecordV1 {
        k_index: k,
        available_bands,
        basis_dimension,
        eigenvalues,
        eigenvectors,
        plane_wave_g,
        plane_wave_k_cartesian,
        plane_wave_q_cartesian,
        site_matches,
        local_orbitals,
    })
}

fn read_site_match(
    group: &Group,
    n_plane_waves: usize,
    site: usize,
) -> Result<ScalarApwSiteMatchV1, IoError> {
    let stored = read_usize_attr(group, "site", &format!("{}/@site", group.name()))?;
    if stored != site {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@site", group.name()),
            expected: site.to_string(),
            actual: stored.to_string(),
        }
        .into());
    }
    require_dataset_names(group, &["lm_l", "lm_m", "matching_coefficients"])?;
    let n_lm = group
        .dataset("lm_l")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: format!("{}/lm_l/shape", group.name()),
            expected: "[lm]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    let lm_l = read_i32_dataset(group, "lm_l", &[n_lm], &["lm"])?;
    let lm_m = read_i32_dataset(group, "lm_m", &[n_lm], &["lm"])?;
    let matching_coefficients = read_f64_dataset(
        group,
        "matching_coefficients",
        &[n_plane_waves, n_lm, 2, 2],
        &["plane_wave", "lm", "radial_component", "re_im"],
    )?;
    let dataset = group.dataset("matching_coefficients")?;
    require_str_array_attr(&dataset, "radial_component_labels", &RADIAL_U_UDOT)?;
    Ok(ScalarApwSiteMatchV1 {
        site_index: site,
        lm_l,
        lm_m,
        matching_coefficients,
    })
}
