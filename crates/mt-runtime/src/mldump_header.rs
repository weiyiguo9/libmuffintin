//! Shared MLDUMP header geometry/$k$/$q$ binding used by scalar and spinor writers.

use muffintin_io::MldumpHeaderV1;

pub(crate) const PREFLIGHT_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HeaderBindError {
    pub path: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HeaderBindSite {
    pub position: [f64; 3],
    pub radius: f64,
    pub mesh_first: f64,
    pub mesh_increment: f64,
    pub mesh_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HeaderBindKMinusQ {
    pub k_index: usize,
    pub mapped_index: usize,
    pub g_wrap: [i32; 3],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HeaderBindQ<'a> {
    pub cartesian: [f64; 3],
    pub umklapp: [i32; 3],
    pub k_minus_q: &'a [HeaderBindKMinusQ],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HeaderBind<'a> {
    pub direct_basis: [[f64; 3]; 3],
    pub reciprocal_basis: [[f64; 3]; 3],
    pub cell_volume: f64,
    pub partition_volume: f64,
    pub sites: &'a [HeaderBindSite],
    pub k_fractional: &'a [[f64; 3]],
    pub q_records: &'a [HeaderBindQ<'a>],
}

pub(crate) fn preflight_mldump_header(
    header: &MldumpHeaderV1,
    bind: &HeaderBind<'_>,
) -> Result<(), HeaderBindError> {
    for (row, (stored, expected)) in header
        .geometry
        .direct_basis_bohr
        .iter()
        .zip(bind.direct_basis.iter())
        .enumerate()
    {
        for (axis, (left, right)) in stored.iter().zip(expected.iter()).enumerate() {
            require_approx(
                &format!("header.geometry.direct_basis_bohr[{row}][{axis}]"),
                *left,
                *right,
            )?;
        }
    }
    for (row, (stored, expected)) in header
        .geometry
        .reciprocal_basis_inv_bohr
        .iter()
        .zip(bind.reciprocal_basis.iter())
        .enumerate()
    {
        for (axis, (left, right)) in stored.iter().zip(expected.iter()).enumerate() {
            require_approx(
                &format!("header.geometry.reciprocal_basis_inv_bohr[{row}][{axis}]"),
                *left,
                *right,
            )?;
        }
    }
    require_approx(
        "header.geometry.cell_volume_bohr3",
        header.geometry.cell_volume_bohr3,
        bind.cell_volume,
    )?;
    require_approx(
        "header.geometry.cell_volume_bohr3",
        header.geometry.cell_volume_bohr3,
        bind.partition_volume,
    )?;
    let n_site = bind.sites.len();
    if header.geometry.sites.len() != n_site {
        return Err(header_mismatch(
            "header.geometry.sites",
            n_site.to_string(),
            header.geometry.sites.len().to_string(),
        ));
    }
    for (site, (header_site, partition_site)) in
        header.geometry.sites.iter().zip(bind.sites).enumerate()
    {
        for (axis, (stored, expected)) in header_site
            .position_bohr
            .iter()
            .zip(partition_site.position.iter())
            .enumerate()
        {
            require_approx(
                &format!("header.geometry.sites[{site}].position_bohr[{axis}]"),
                *stored,
                *expected,
            )?;
        }
        require_approx(
            &format!("header.geometry.sites[{site}].radius_bohr"),
            header_site.radius_bohr,
            partition_site.radius,
        )?;
        require_approx(
            &format!("header.geometry.sites[{site}].radial_mesh.first_bohr"),
            header_site.radial_mesh.first_bohr,
            partition_site.mesh_first,
        )?;
        require_approx(
            &format!("header.geometry.sites[{site}].radial_mesh.log_increment"),
            header_site.radial_mesh.log_increment,
            partition_site.mesh_increment,
        )?;
        if header_site.radial_mesh.point_count != partition_site.mesh_count {
            return Err(header_mismatch(
                &format!("header.geometry.sites[{site}].radial_mesh.point_count"),
                partition_site.mesh_count.to_string(),
                header_site.radial_mesh.point_count.to_string(),
            ));
        }
    }
    let n_k = bind.k_fractional.len();
    if header.mesh.k_points.len() != n_k {
        return Err(header_mismatch(
            "header.mesh.k_points",
            n_k.to_string(),
            header.mesh.k_points.len().to_string(),
        ));
    }
    let weight = 1.0 / n_k as f64;
    for (k, (stored, expected)) in header
        .mesh
        .k_points
        .iter()
        .zip(bind.k_fractional.iter())
        .enumerate()
    {
        for (axis, (left, right)) in stored.fractional.iter().zip(expected.iter()).enumerate() {
            require_approx(
                &format!("header.mesh.k_points[{k}].fractional[{axis}]"),
                *left,
                *right,
            )?;
        }
        require_approx(
            &format!("header.mesh.k_points[{k}].weight"),
            stored.weight,
            weight,
        )?;
    }
    if header.mesh.q_entries.len() != bind.q_records.len() {
        return Err(header_mismatch(
            "header.mesh.q_entries",
            bind.q_records.len().to_string(),
            header.mesh.q_entries.len().to_string(),
        ));
    }
    for (q, (entry, record)) in header.mesh.q_entries.iter().zip(bind.q_records).enumerate() {
        let expected_cart = fractional_to_cartesian(
            header.geometry.reciprocal_basis_inv_bohr,
            entry.canonical_fractional,
        );
        for (axis, (stored, expected)) in record.cartesian.iter().zip(expected_cart).enumerate() {
            require_approx(
                &format!("header.mesh.q_entries[{q}].canonical cartesian[{axis}]"),
                *stored,
                expected,
            )?;
        }
        if entry.global_umklapp != record.umklapp {
            return Err(header_mismatch(
                &format!("header.mesh.q_entries[{q}].global_umklapp"),
                format!("{:?}", record.umklapp),
                format!("{:?}", entry.global_umklapp),
            ));
        }
        if entry.k_minus_q.len() != n_k {
            return Err(header_mismatch(
                &format!("header.mesh.q_entries[{q}].k_minus_q"),
                n_k.to_string(),
                entry.k_minus_q.len().to_string(),
            ));
        }
        if record.k_minus_q.len() != n_k {
            return Err(header_mismatch(
                &format!("header.mesh.q_entries[{q}].k_minus_q"),
                n_k.to_string(),
                record.k_minus_q.len().to_string(),
            ));
        }
        for (k, (stored, mapped)) in entry.k_minus_q.iter().zip(record.k_minus_q).enumerate() {
            if stored.k_index != mapped.k_index || stored.mapped_index != mapped.mapped_index {
                return Err(header_mismatch(
                    &format!("header.mesh.q_entries[{q}].k_minus_q[{k}]"),
                    format!("k={} mapped={}", mapped.k_index, mapped.mapped_index),
                    format!("k={} mapped={}", stored.k_index, stored.mapped_index),
                ));
            }
            if stored.g_wrap != mapped.g_wrap {
                return Err(header_mismatch(
                    &format!("header.mesh.q_entries[{q}].k_minus_q[{k}].g_wrap"),
                    format!("{:?}", mapped.g_wrap),
                    format!("{:?}", stored.g_wrap),
                ));
            }
        }
    }
    Ok(())
}

fn fractional_to_cartesian(reciprocal: [[f64; 3]; 3], fractional: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        fractional[0] * reciprocal[0][axis]
            + fractional[1] * reciprocal[1][axis]
            + fractional[2] * reciprocal[2][axis]
    })
}

fn require_approx(path: &str, stored: f64, expected: f64) -> Result<(), HeaderBindError> {
    let scale = stored.abs().max(expected.abs()).max(1.0);
    if (stored - expected).abs() <= PREFLIGHT_TOLERANCE * scale {
        Ok(())
    } else {
        Err(header_mismatch(
            path,
            expected.to_string(),
            stored.to_string(),
        ))
    }
}

fn header_mismatch(path: &str, expected: String, actual: String) -> HeaderBindError {
    HeaderBindError {
        path: path.to_owned(),
        expected,
        actual,
    }
}
