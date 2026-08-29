//! Full first-variation spinor product-function payload for MLDUMP v1.

use std::collections::BTreeSet;

use hdf5_metno::Group;

use super::scalar_products::{
    MLDUMP_CORE_EMPTY_NOT_FITTED, MLDUMP_PAIR_ORDER_K_LEFT_RIGHT, MLDUMP_RADIAL_KIND_CORE,
    MLDUMP_RADIAL_KIND_VALENCE,
};
use super::spinor_orbitals::MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION;
use super::{
    GROUP_PRODUCTS, MldumpHeaderV1, PREFIX_Q, PREFIX_SITE, approx_eq, collect_padded_groups,
    create_padded_group, i32_triples_to_owned, padded_child, read_f64_attr, read_f64_dataset,
    read_i32_dataset, read_i64_dataset, read_str_attr, read_usize_attr, reopen_present_group,
    require_dataset_names, require_exact_members, require_finite_f64s, require_flat_len,
    require_len, require_nonnegative_index, require_status_present, require_str_attr,
    triples_to_owned, usize_as_i64, write_f64_attr, write_f64_dataset, write_i32_dataset,
    write_i64_attr, write_i64_dataset, write_str_attr,
};
use crate::error::{IoError, ValidationError, nonempty};

/// Shared `/products` attributes and geometry binding for a spinor session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorProductsBeginV1<'a> {
    pub n_k: usize,
    pub n_orb: usize,
    pub provenance_recipe: &'a str,
    pub provenance_reference: &'a str,
    pub site_indices: &'a [i64],
    pub site_positions: &'a [f64],
    pub site_radii: &'a [f64],
    pub interstitial_volume_bohr3: f64,
}

/// One site's valence Dirac radial factors on the `/geometry` mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorProductSiteRefV1<'a> {
    pub site_index: usize,
    pub n_radial: usize,
    pub n_radial_samples: usize,
    pub kind: &'a [i64],
    pub signed_kappa: &'a [i64],
    pub n: &'a [i64],
    pub p: &'a [f64],
    pub q: &'a [f64],
}

/// Positional $q$ record bound by mesh $q$ index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpinorProductQRecordRefV1<'a> {
    pub q_index: usize,
    pub transfer_cartesian: [f64; 3],
    pub global_transfer: [i32; 3],
    pub n_raw_g: usize,
    pub raw_relative_g: &'a [i32],
    pub provenance: &'a str,
}

/// Owned spinor product section.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorProductsV1 {
    pub n_k: usize,
    pub n_orb: usize,
    pub provenance_recipe: String,
    pub provenance_reference: String,
    pub site_indices: Vec<usize>,
    pub site_positions: Vec<[f64; 3]>,
    pub site_radii: Vec<f64>,
    pub interstitial_volume_bohr3: f64,
    pub sites: Vec<SpinorProductSiteV1>,
    pub q_records: Vec<SpinorProductQRecordV1>,
}

/// Owned site Dirac radial set.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorProductSiteV1 {
    pub site_index: usize,
    pub kind: Vec<i64>,
    pub signed_kappa: Vec<i64>,
    pub n: Vec<i64>,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
}

/// Owned raw pair-G record at one mesh $q$.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinorProductQRecordV1 {
    pub q_index: usize,
    pub transfer_cartesian: [f64; 3],
    pub global_transfer: [i32; 3],
    pub raw_relative_g: Vec<[i32; 3]>,
    pub provenance: String,
}

pub(crate) fn begin_spinor_products(
    file: &Group,
    header: &MldumpHeaderV1,
    products: &SpinorProductsBeginV1<'_>,
) -> Result<(), IoError> {
    validate_products_begin(header, products)?;
    let n_site = header.geometry.sites.len();
    let group = reopen_present_group(file, GROUP_PRODUCTS)?;
    write_str_attr(
        &group,
        "representation",
        MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
    )?;
    write_i64_attr(&group, "n_k", usize_as_i64("/products/@n_k", products.n_k)?)?;
    write_i64_attr(
        &group,
        "n_orb",
        usize_as_i64("/products/@n_orb", products.n_orb)?,
    )?;
    write_str_attr(&group, "pair_order", MLDUMP_PAIR_ORDER_K_LEFT_RIGHT)?;
    write_str_attr(&group, "core_status", MLDUMP_CORE_EMPTY_NOT_FITTED)?;
    write_str_attr(&group, "provenance_recipe", products.provenance_recipe)?;
    write_str_attr(
        &group,
        "provenance_reference",
        products.provenance_reference,
    )?;
    write_i64_attr(&group, "n_site", usize_as_i64("/products/@n_site", n_site)?)?;
    write_f64_attr(
        &group,
        "interstitial_volume_bohr3",
        products.interstitial_volume_bohr3,
    )?;
    write_i64_dataset(
        &group,
        "site_indices",
        &[n_site],
        products.site_indices,
        &["site"],
    )?;
    write_f64_dataset(
        &group,
        "site_positions",
        &[n_site, 3],
        products.site_positions,
        &["site", "cartesian"],
    )?;
    write_f64_dataset(
        &group,
        "site_radii",
        &[n_site],
        products.site_radii,
        &["site"],
    )?;
    Ok(())
}

pub(crate) fn write_spinor_product_site(
    file: &Group,
    header: &MldumpHeaderV1,
    site: usize,
    record: &SpinorProductSiteRefV1<'_>,
) -> Result<(), IoError> {
    validate_site_ref(header, site, record)?;
    let group = file.group(GROUP_PRODUCTS)?;
    write_site_group(&group, header, record)
}

pub(crate) fn write_spinor_product_q(
    file: &Group,
    q: usize,
    record: &SpinorProductQRecordRefV1<'_>,
) -> Result<(), IoError> {
    validate_q_ref(q, record)?;
    let group = file.group(GROUP_PRODUCTS)?;
    write_q_group(&group, record)
}

pub(crate) fn read_spinor_products(
    file: &Group,
    header: &MldumpHeaderV1,
) -> Result<SpinorProductsV1, IoError> {
    let group = file.group(GROUP_PRODUCTS)?;
    require_status_present(&group)?;
    require_str_attr(
        &group,
        "representation",
        MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION,
    )?;
    require_str_attr(&group, "pair_order", MLDUMP_PAIR_ORDER_K_LEFT_RIGHT)?;
    require_str_attr(&group, "core_status", MLDUMP_CORE_EMPTY_NOT_FITTED)?;
    let n_k = read_usize_attr(&group, "n_k", "/products/@n_k")?;
    let n_orb = read_usize_attr(&group, "n_orb", "/products/@n_orb")?;
    require_len("/products/@n_k", header.mesh.k_points.len(), n_k)?;
    if n_orb == 0 {
        return Err(ValidationError::NotPositive {
            path: "/products/@n_orb".to_owned(),
            value: 0.0,
        }
        .into());
    }
    let n_site = read_usize_attr(&group, "n_site", "/products/@n_site")?;
    require_len("/products/@n_site", header.geometry.sites.len(), n_site)?;
    let interstitial_volume_bohr3 = read_f64_attr(
        &group,
        "interstitial_volume_bohr3",
        "/products/@interstitial_volume_bohr3",
    )?;
    require_dataset_names(&group, &["site_indices", "site_positions", "site_radii"])?;
    let site_index_i64 = read_i64_dataset(&group, "site_indices", &[n_site], &["site"])?;
    let mut site_indices = Vec::with_capacity(n_site);
    for (index, value) in site_index_i64.iter().enumerate() {
        let site = require_nonnegative_index(&format!("/products/site_indices[{index}]"), *value)?;
        if site != index {
            return Err(ValidationError::InvalidValue {
                path: format!("/products/site_indices[{index}]"),
                expected: index.to_string(),
                actual: site.to_string(),
            }
            .into());
        }
        site_indices.push(site);
    }
    let site_positions = triples_to_owned(
        &read_f64_dataset(
            &group,
            "site_positions",
            &[n_site, 3],
            &["site", "cartesian"],
        )?,
        n_site,
        "/products/site_positions",
    )?;
    let site_radii = read_f64_dataset(&group, "site_radii", &[n_site], &["site"])?;
    bind_partition(header, &site_indices, &site_positions, &site_radii)?;
    let site_groups = collect_padded_groups(&group, PREFIX_SITE)?;
    let q_groups = collect_padded_groups(&group, PREFIX_Q)?;
    require_len("/products/site_*", n_site, site_groups.len())?;
    require_len("/products/q_*", header.mesh.q_entries.len(), q_groups.len())?;
    let mut members = vec![
        "site_indices".to_owned(),
        "site_positions".to_owned(),
        "site_radii".to_owned(),
    ];
    members.extend((0..n_site).map(|site| padded_child(PREFIX_SITE, site)));
    members.extend((0..header.mesh.q_entries.len()).map(|q| padded_child(PREFIX_Q, q)));
    require_exact_members(&group, members)?;
    let mut sites = Vec::with_capacity(n_site);
    for (site, site_group) in site_groups.iter().enumerate() {
        sites.push(read_site_group(site_group, header, site)?);
    }
    let mut q_records = Vec::with_capacity(q_groups.len());
    for (q, q_group) in q_groups.iter().enumerate() {
        q_records.push(read_q_group(q_group, q)?);
    }
    let products = SpinorProductsV1 {
        n_k,
        n_orb,
        provenance_recipe: read_str_attr(&group, "provenance_recipe")?,
        provenance_reference: read_str_attr(&group, "provenance_reference")?,
        site_indices,
        site_positions,
        site_radii,
        interstitial_volume_bohr3,
        sites,
        q_records,
    };
    validate_products_owned(header, &products)?;
    Ok(products)
}

fn validate_products_begin(
    header: &MldumpHeaderV1,
    products: &SpinorProductsBeginV1<'_>,
) -> Result<(), IoError> {
    nonempty("products.provenance_recipe", products.provenance_recipe)?;
    nonempty(
        "products.provenance_reference",
        products.provenance_reference,
    )?;
    require_len("products.n_k", header.mesh.k_points.len(), products.n_k)?;
    if products.n_orb == 0 {
        return Err(ValidationError::NotPositive {
            path: "products.n_orb".to_owned(),
            value: 0.0,
        }
        .into());
    }
    let n_site = header.geometry.sites.len();
    require_len("products.site_indices", n_site, products.site_indices.len())?;
    require_flat_len(
        "products.site_positions",
        &[n_site, 3],
        products.site_positions.len(),
    )?;
    require_len("products.site_radii", n_site, products.site_radii.len())?;
    require_finite_f64s("products.site_positions", products.site_positions)?;
    require_finite_f64s("products.site_radii", products.site_radii)?;
    if !products.interstitial_volume_bohr3.is_finite() || products.interstitial_volume_bohr3 <= 0.0
    {
        return Err(ValidationError::NotPositive {
            path: "products.interstitial_volume_bohr3".to_owned(),
            value: products.interstitial_volume_bohr3,
        }
        .into());
    }
    let mut positions = Vec::with_capacity(n_site);
    for site in 0..n_site {
        let index = require_nonnegative_index(
            &format!("products.site_indices[{site}]"),
            products.site_indices[site],
        )?;
        if index != site {
            return Err(ValidationError::InvalidValue {
                path: format!("products.site_indices[{site}]"),
                expected: site.to_string(),
                actual: index.to_string(),
            }
            .into());
        }
        positions.push([
            products.site_positions[site * 3],
            products.site_positions[site * 3 + 1],
            products.site_positions[site * 3 + 2],
        ]);
    }
    bind_partition(
        header,
        &(0..n_site).collect::<Vec<_>>(),
        &positions,
        products.site_radii,
    )?;
    Ok(())
}

fn bind_partition(
    header: &MldumpHeaderV1,
    site_indices: &[usize],
    positions: &[[f64; 3]],
    radii: &[f64],
) -> Result<(), IoError> {
    for (slot, &site) in site_indices.iter().enumerate() {
        let geometry =
            header
                .geometry
                .sites
                .get(site)
                .ok_or_else(|| ValidationError::InvalidValue {
                    path: format!("products.site_indices[{slot}]"),
                    expected: format!("index < {}", header.geometry.sites.len()),
                    actual: site.to_string(),
                })?;
        for (axis, (stored, expected)) in positions[slot]
            .iter()
            .zip(geometry.position_bohr)
            .enumerate()
        {
            if !approx_eq(*stored, expected) {
                return Err(ValidationError::InvalidValue {
                    path: format!("products.site_positions[{slot}][{axis}]"),
                    expected: format!("geometry.sites[{site}] = {expected}"),
                    actual: stored.to_string(),
                }
                .into());
            }
        }
        if !approx_eq(radii[slot], geometry.radius_bohr) {
            return Err(ValidationError::InvalidValue {
                path: format!("products.site_radii[{slot}]"),
                expected: format!("geometry.sites[{site}] = {}", geometry.radius_bohr),
                actual: radii[slot].to_string(),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_site_ref(
    header: &MldumpHeaderV1,
    site: usize,
    record: &SpinorProductSiteRefV1<'_>,
) -> Result<(), IoError> {
    if record.site_index != site {
        return Err(ValidationError::InvalidValue {
            path: format!("products.sites[{site}].site_index"),
            expected: site.to_string(),
            actual: record.site_index.to_string(),
        }
        .into());
    }
    let mesh = header.geometry.sites[site].radial_mesh;
    require_len(
        &format!("products.sites[{site}].n_radial_samples"),
        mesh.point_count,
        record.n_radial_samples,
    )?;
    if record.n_radial == 0 {
        return Err(ValidationError::Empty {
            path: format!("products.sites[{site}].radial"),
        }
        .into());
    }
    for (name, values) in [
        ("kind", record.kind),
        ("signed_kappa", record.signed_kappa),
        ("n", record.n),
    ] {
        require_len(
            &format!("products.sites[{site}].{name}"),
            record.n_radial,
            values.len(),
        )?;
    }
    require_flat_len(
        &format!("products.sites[{site}].p"),
        &[record.n_radial, record.n_radial_samples],
        record.p.len(),
    )?;
    require_flat_len(
        &format!("products.sites[{site}].q"),
        &[record.n_radial, record.n_radial_samples],
        record.q.len(),
    )?;
    require_finite_f64s(&format!("products.sites[{site}].p"), record.p)?;
    require_finite_f64s(&format!("products.sites[{site}].q"), record.q)?;
    let mut ids = BTreeSet::new();
    for radial in 0..record.n_radial {
        if record.kind[radial] == MLDUMP_RADIAL_KIND_CORE {
            return Err(ValidationError::InvalidValue {
                path: format!("products.sites[{site}].kind[{radial}]"),
                expected: MLDUMP_CORE_EMPTY_NOT_FITTED.to_owned(),
                actual: "core".to_owned(),
            }
            .into());
        }
        if record.kind[radial] != MLDUMP_RADIAL_KIND_VALENCE {
            return Err(ValidationError::InvalidValue {
                path: format!("products.sites[{site}].kind[{radial}]"),
                expected: MLDUMP_RADIAL_KIND_VALENCE.to_string(),
                actual: record.kind[radial].to_string(),
            }
            .into());
        }
        if record.signed_kappa[radial] == 0 {
            return Err(ValidationError::InvalidValue {
                path: format!("products.sites[{site}].signed_kappa[{radial}]"),
                expected: "nonzero signed kappa".to_owned(),
                actual: "0".to_owned(),
            }
            .into());
        }
        require_nonnegative_index(
            &format!("products.sites[{site}].n[{radial}]"),
            record.n[radial],
        )?;
        let id = (
            record.kind[radial],
            record.signed_kappa[radial],
            record.n[radial],
        );
        if !ids.insert(id) {
            return Err(ValidationError::Duplicate {
                path: format!("products.sites[{site}].radial_id"),
                key: format!("kind={} kappa={} n={}", id.0, id.1, id.2),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_products_owned(
    header: &MldumpHeaderV1,
    products: &SpinorProductsV1,
) -> Result<(), IoError> {
    let mut site_indices = Vec::with_capacity(products.site_indices.len());
    for (index, site) in products.site_indices.iter().enumerate() {
        site_indices.push(usize_as_i64(
            &format!("products.site_indices[{index}]"),
            *site,
        )?);
    }
    let mut site_positions = Vec::with_capacity(products.site_positions.len() * 3);
    for position in &products.site_positions {
        site_positions.extend_from_slice(position);
    }
    let mut raw_g = Vec::with_capacity(products.q_records.len());
    for record in &products.q_records {
        raw_g.push(
            record
                .raw_relative_g
                .iter()
                .flat_map(|g| g.iter().copied())
                .collect::<Vec<_>>(),
        );
    }
    validate_products_begin(
        header,
        &SpinorProductsBeginV1 {
            n_k: products.n_k,
            n_orb: products.n_orb,
            provenance_recipe: &products.provenance_recipe,
            provenance_reference: &products.provenance_reference,
            site_indices: &site_indices,
            site_positions: &site_positions,
            site_radii: &products.site_radii,
            interstitial_volume_bohr3: products.interstitial_volume_bohr3,
        },
    )?;
    require_len(
        "products.sites",
        header.geometry.sites.len(),
        products.sites.len(),
    )?;
    require_len(
        "products.q_records",
        header.mesh.q_entries.len(),
        products.q_records.len(),
    )?;
    for (site, record) in products.sites.iter().enumerate() {
        let n_radial_samples = header.geometry.sites[site].radial_mesh.point_count;
        validate_site_ref(
            header,
            site,
            &SpinorProductSiteRefV1 {
                site_index: record.site_index,
                n_radial: record.kind.len(),
                n_radial_samples,
                kind: &record.kind,
                signed_kappa: &record.signed_kappa,
                n: &record.n,
                p: &record.p,
                q: &record.q,
            },
        )?;
    }
    for (q, (record, g)) in products.q_records.iter().zip(raw_g.iter()).enumerate() {
        validate_q_ref(
            q,
            &SpinorProductQRecordRefV1 {
                q_index: record.q_index,
                transfer_cartesian: record.transfer_cartesian,
                global_transfer: record.global_transfer,
                n_raw_g: record.raw_relative_g.len(),
                raw_relative_g: g,
                provenance: &record.provenance,
            },
        )?;
    }
    Ok(())
}

fn validate_q_ref(q: usize, record: &SpinorProductQRecordRefV1<'_>) -> Result<(), IoError> {
    if record.q_index != q {
        return Err(ValidationError::InvalidValue {
            path: format!("products.q_records[{q}].q_index"),
            expected: q.to_string(),
            actual: record.q_index.to_string(),
        }
        .into());
    }
    nonempty(
        format!("products.q_records[{q}].provenance"),
        record.provenance,
    )?;
    require_finite_f64s(
        &format!("products.q_records[{q}].transfer_cartesian"),
        &record.transfer_cartesian,
    )?;
    require_flat_len(
        &format!("products.q_records[{q}].raw_relative_g"),
        &[record.n_raw_g, 3],
        record.raw_relative_g.len(),
    )?;
    Ok(())
}

fn write_site_group(
    parent: &Group,
    header: &MldumpHeaderV1,
    record: &SpinorProductSiteRefV1<'_>,
) -> Result<(), IoError> {
    let group = create_padded_group(parent, PREFIX_SITE, record.site_index)?;
    let mesh = header.geometry.sites[record.site_index].radial_mesh;
    write_i64_attr(&group, "site", usize_as_i64("site", record.site_index)?)?;
    write_f64_attr(&group, "mesh_first_bohr", mesh.first_bohr)?;
    write_f64_attr(&group, "mesh_log_increment", mesh.log_increment)?;
    write_i64_attr(
        &group,
        "mesh_point_count",
        usize_as_i64("mesh_point_count", mesh.point_count)?,
    )?;
    write_i64_dataset(&group, "kind", &[record.n_radial], record.kind, &["radial"])?;
    write_i64_dataset(
        &group,
        "signed_kappa",
        &[record.n_radial],
        record.signed_kappa,
        &["radial"],
    )?;
    write_i64_dataset(&group, "n", &[record.n_radial], record.n, &["radial"])?;
    write_f64_dataset(
        &group,
        "p",
        &[record.n_radial, record.n_radial_samples],
        record.p,
        &["radial", "radial_sample"],
    )?;
    write_f64_dataset(
        &group,
        "q",
        &[record.n_radial, record.n_radial_samples],
        record.q,
        &["radial", "radial_sample"],
    )?;
    Ok(())
}

fn write_q_group(parent: &Group, record: &SpinorProductQRecordRefV1<'_>) -> Result<(), IoError> {
    let group = create_padded_group(parent, PREFIX_Q, record.q_index)?;
    write_i64_attr(&group, "q_index", usize_as_i64("q_index", record.q_index)?)?;
    write_str_attr(&group, "provenance", record.provenance)?;
    write_f64_dataset(
        &group,
        "transfer_cartesian",
        &[3],
        &record.transfer_cartesian,
        &["cartesian"],
    )?;
    write_i32_dataset(
        &group,
        "global_transfer",
        &[3],
        &record.global_transfer,
        &["reciprocal_axis"],
    )?;
    write_i32_dataset(
        &group,
        "raw_relative_g",
        &[record.n_raw_g, 3],
        record.raw_relative_g,
        &["raw_g", "reciprocal_axis"],
    )?;
    Ok(())
}

fn read_site_group(
    group: &Group,
    header: &MldumpHeaderV1,
    site: usize,
) -> Result<SpinorProductSiteV1, IoError> {
    let stored = read_usize_attr(group, "site", &format!("{}/@site", group.name()))?;
    if stored != site {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@site", group.name()),
            expected: site.to_string(),
            actual: stored.to_string(),
        }
        .into());
    }
    let mesh = header.geometry.sites[site].radial_mesh;
    let first = read_f64_attr(
        group,
        "mesh_first_bohr",
        &format!("{}/@mesh_first_bohr", group.name()),
    )?;
    let increment = read_f64_attr(
        group,
        "mesh_log_increment",
        &format!("{}/@mesh_log_increment", group.name()),
    )?;
    let count = read_usize_attr(
        group,
        "mesh_point_count",
        &format!("{}/@mesh_point_count", group.name()),
    )?;
    if !approx_eq(first, mesh.first_bohr)
        || !approx_eq(increment, mesh.log_increment)
        || count != mesh.point_count
    {
        return Err(ValidationError::LayoutMismatch {
            path: format!("{}/mesh", group.name()),
            reference: format!("/geometry site {site} radial mesh"),
        }
        .into());
    }
    let n_radial = group
        .dataset("kind")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: format!("{}/kind/shape", group.name()),
            expected: "[radial]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    require_dataset_names(group, &["kind", "signed_kappa", "n", "p", "q"])?;
    let kind = read_i64_dataset(group, "kind", &[n_radial], &["radial"])?;
    let signed_kappa = read_i64_dataset(group, "signed_kappa", &[n_radial], &["radial"])?;
    let n = read_i64_dataset(group, "n", &[n_radial], &["radial"])?;
    let p = read_f64_dataset(group, "p", &[n_radial, count], &["radial", "radial_sample"])?;
    let q = read_f64_dataset(group, "q", &[n_radial, count], &["radial", "radial_sample"])?;
    Ok(SpinorProductSiteV1 {
        site_index: site,
        kind,
        signed_kappa,
        n,
        p,
        q,
    })
}

fn read_q_group(group: &Group, q: usize) -> Result<SpinorProductQRecordV1, IoError> {
    let stored = read_usize_attr(group, "q_index", &format!("{}/@q_index", group.name()))?;
    if stored != q {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/@q_index", group.name()),
            expected: q.to_string(),
            actual: stored.to_string(),
        }
        .into());
    }
    require_dataset_names(
        group,
        &["transfer_cartesian", "global_transfer", "raw_relative_g"],
    )?;
    let transfer = read_f64_dataset(group, "transfer_cartesian", &[3], &["cartesian"])?;
    let global = read_i32_dataset(group, "global_transfer", &[3], &["reciprocal_axis"])?;
    let n_raw = group
        .dataset("raw_relative_g")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: format!("{}/raw_relative_g/shape", group.name()),
            expected: "[n_raw, 3]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    Ok(SpinorProductQRecordV1 {
        q_index: q,
        transfer_cartesian: [transfer[0], transfer[1], transfer[2]],
        global_transfer: [global[0], global[1], global[2]],
        raw_relative_g: i32_triples_to_owned(
            &read_i32_dataset(
                group,
                "raw_relative_g",
                &[n_raw, 3],
                &["raw_g", "reciprocal_axis"],
            )?,
            n_raw,
            &format!("{}/raw_relative_g", group.name()),
        )?,
        provenance: read_str_attr(group, "provenance")?,
    })
}
