//! Strict `spex.snapshot_hdf` v1 reader for SPEX-owned frozen fields.
//!
//! Signed $\kappa$, rLO, and HDLO are not SPEX-owned. [`read_spex_snapshot_hdf`]
//! returns [`SpexFrozenFieldsV1`]. [`materialize_snapshot_v2`] builds
//! [`SnapshotV2`] only with an explicit caller-owned recipe. Missing groups
//! are typed blockers. All-zero $B_x$/$B_y$ require `@zero_source`. String
//! attributes follow Hwrapper `hdf_rdwr_a_str`: `H5T_NATIVE_CHARACTER` of
//! content length, scalar or length-1, trailing spaces/NULs trimmed. Dataset
//! `@axes` may be that scalar (whitespace-split) or a 1-d VL token array.
//! String datasets are 1-d `H5T_NATIVE_CHARACTER` of fixed element length
//! (space/NUL padded); VL is a reader fallback only (parallel HDF5 cannot
//! write VL).
//! The reader does not invent kappa, $n$, or omitted $B$.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use hdf5_metno::types::{TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5_metno::{Attribute, Dataset, File, Group, Location};

use crate::error::{IoError, ValidationError, nonempty, positive};
use crate::mldump::{
    child_basename, create_dataset, require_finite_f64s, require_flat_len, require_len,
    require_numeric_dataset, require_shape, usize_as_i64, write_f64_attr, write_i64_attr,
};
use crate::snapshot::{
    AngularBasisV1, BasisHintsV1, EnergyParameterV1, ExponentialMeshSpecV1, FourierNormalizationV1,
    FourierPhaseV1, LinearizationV1, MetaV1, PotentialConventionV1, PotentialRadialQuantityV1,
    RadialEquationTagV1, SNAPSHOT_FORMAT, SphericalChannelConventionV1,
};
use crate::snapshot_v2::{
    Complex64V2, DensityV2, FieldRepresentationV2, FieldUnitV2, FourierCoefficientV2, GeometryV2,
    InitialV2, InterstitialFieldV2, MuffinTinFieldV2, PotentialV2, RadialBasisSpinV2,
    RegionalFieldV2, SNAPSHOT_VERSION_V2, SiteRadialBasisV2, SiteV2, SnapshotV2,
    SphericalChannelV2,
};
use crate::units::{EnergyUnitV1, InverseLengthUnitV1, LengthUnitV1};

/// Root `schema_name` for this producer file.
pub const SPEX_SNAPSHOT_HDF_SCHEMA_NAME: &str = "spex.snapshot_hdf";
/// Integer schema version.
pub const SPEX_SNAPSHOT_HDF_SCHEMA_VERSION: i64 = 1;
/// Required `source_kind` token.
pub const SPEX_SNAPSHOT_HDF_SOURCE_KIND: &str = "spex-generic-dft";
/// Scale-aware Hermitian ingest gate used only at SPEX `materialize_snapshot_v2`.
///
/// Live Sm fcc `snapshot.h5` (SHA-256 `9f060f74…`) has
/// \(V(\mathbf G)-V(-\mathbf G)^*\) of order \(10^{-20}\) to \(10^{-18}\).
/// The gate is the project fractional tolerance
/// [`crate::mldump::FRACTIONAL_EQ_TOLERANCE`] times
/// \(\max(|c|,|c'|,1)\), the same scale used by `approx_eq`. Snapshot V2
/// still requires exact equality after ingest.
pub const SPEX_FOURIER_HERMITIAN_TOLERANCE: f64 = crate::mldump::FRACTIONAL_EQ_TOLERANCE;

const ATTR_SCHEMA_NAME: &str = "schema_name";
const ATTR_SCHEMA_VERSION: &str = "schema_version";

const GROUP_META: &str = "meta";
const GROUP_UNITS: &str = "units";
const GROUP_HASHES: &str = "hashes";
const GROUP_GEOMETRY: &str = "geometry";
const GROUP_RADIAL: &str = "radial_basis";
const GROUP_INITIAL: &str = "initial";
const GROUP_POTENTIAL: &str = "potential";
const GROUP_DENSITY: &str = "density";
const GROUP_MT: &str = "muffin_tins";
const GROUP_INTERSTITIAL: &str = "interstitial";
const GROUP_ORBITALS: &str = "orbitals";

const PREFIX_BASIS: &str = "basis";
const PREFIX_SITE: &str = "site";

const TOP_LEVEL: [&str; 6] = [
    GROUP_META,
    GROUP_UNITS,
    GROUP_HASHES,
    GROUP_GEOMETRY,
    GROUP_RADIAL,
    GROUP_INITIAL,
];

const POTENTIAL_COMPONENTS: [&str; 4] = ["v0", "bx", "by", "bz"];
const DENSITY_COMPONENTS: [&str; 4] = ["n", "mx", "my", "mz"];

/// One hashed producer source file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpexSnapshotHashV1 {
    pub role: String,
    pub name: String,
    pub sha256: String,
}

/// SPEX scalar LO kind. SPEX does not own rLO/HDLO tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpexScalarLoKind {
    Lo,
}

/// One SPEX-owned scalar LO: $(l,E)$ and optional principal $n$ when `pbas>0`.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexScalarLoV1 {
    pub kind: SpexScalarLoKind,
    pub l: u32,
    pub energy: f64,
    pub n: Option<u32>,
}

/// Scalar LO table for one `/radial_basis/basis_XXX` record.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexScalarLoTableV1 {
    pub site_id: String,
    pub spin: RadialBasisSpinV2,
    pub orbitals: Vec<SpexScalarLoV1>,
}

/// Caller-owned signed-$\kappa$ / rLO / HDLO recipe. Never stored in SPEX HDF.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpexMaterialChannelKind {
    Lo,
    Rlo,
    Hdlo,
}

/// One libmuffintin-owned channel used to materialize Snapshot V2.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexMaterialChannelV1 {
    pub site_id: String,
    pub n: u32,
    pub l: u32,
    pub kappa: i32,
    pub kind: SpexMaterialChannelKind,
    pub derivative_order: u32,
    pub energy: f64,
}

/// Explicit material basis recipe. `recipe_sha256` is caller-recorded 64 hex.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexMaterialBasisRecipeV1 {
    pub producer: String,
    pub recipe_sha256: String,
    pub channels: Vec<SpexMaterialChannelV1>,
}

/// SPEX-owned frozen fields. Not a completed material Snapshot V2.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexFrozenFieldsV1 {
    pub snapshot: SnapshotV2,
    pub source_revision: String,
    pub source_kind: String,
    pub plane_wave_cutoff: f64,
    pub coefficient_cutoff: f64,
    pub spin_layout: String,
    pub interstitial_phase: String,
    pub hashes: Vec<SpexSnapshotHashV1>,
    pub scalar_los: Vec<SpexScalarLoTableV1>,
}

/// Snapshot V2 after SPEX fields plus a hashed recipe pass compatibility.
#[derive(Clone, Debug, PartialEq)]
pub struct SpexMaterializedSnapshotV1 {
    pub snapshot: SnapshotV2,
    pub spex_hashes: Vec<SpexSnapshotHashV1>,
    pub recipe_sha256: String,
}

/// Read SPEX-owned frozen fields. Does not materialize Snapshot V2 as a
/// signed-$\kappa$ material result.
pub fn read_spex_snapshot_hdf(path: &Path) -> Result<SpexFrozenFieldsV1, IoError> {
    let file = File::open(path)?;
    require_exact_members(&file, &TOP_LEVEL)?;
    let schema_name = read_spex_str_attr(&file, ATTR_SCHEMA_NAME)?;
    if schema_name != SPEX_SNAPSHOT_HDF_SCHEMA_NAME {
        return Err(IoError::InvalidFormat {
            expected: SPEX_SNAPSHOT_HDF_SCHEMA_NAME,
            found: schema_name,
        });
    }
    let version = read_spex_i64_attr(&file, ATTR_SCHEMA_VERSION, "/@schema_version")?;
    if version != SPEX_SNAPSHOT_HDF_SCHEMA_VERSION {
        return Err(IoError::UnsupportedVersion {
            format: SPEX_SNAPSHOT_HDF_SCHEMA_NAME,
            supported: u32::try_from(SPEX_SNAPSHOT_HDF_SCHEMA_VERSION).unwrap_or(1),
            found: u32::try_from(version).unwrap_or(u32::MAX),
        });
    }
    read_units(&file.group(GROUP_UNITS)?)?;
    let (meta, source_revision, source_kind) = read_meta(&file.group(GROUP_META)?)?;
    let hashes = read_hashes(&file.group(GROUP_HASHES)?)?;
    let geometry = read_geometry(&file.group(GROUP_GEOMETRY)?)?;
    let (radial_basis, orbitals) = read_radial_basis(&file.group(GROUP_RADIAL)?, &geometry)?;
    let geometry = GeometryV2 {
        lattice: geometry.lattice,
        sites: geometry.sites,
        radial_basis,
    };
    let (initial, plane_wave_cutoff, coefficient_cutoff, spin_layout, interstitial_phase) =
        read_initial(
            &file.group(GROUP_INITIAL)?,
            &geometry,
            meta.potential_convention.angular_basis,
        )?;
    let snapshot = SnapshotV2 {
        format: SNAPSHOT_FORMAT.to_owned(),
        version: SNAPSHOT_VERSION_V2,
        meta,
        geometry,
        initial,
    };
    snapshot.validate()?;
    Ok(SpexFrozenFieldsV1 {
        snapshot,
        source_revision,
        source_kind,
        plane_wave_cutoff,
        coefficient_cutoff,
        spin_layout,
        interstitial_phase,
        hashes,
        scalar_los: orbitals,
    })
}

/// Write SPEX-owned frozen fields. Does not write signed $\kappa$.
pub fn write_spex_snapshot_hdf(path: &Path, file: &SpexFrozenFieldsV1) -> Result<(), IoError> {
    file.snapshot.validate()?;
    if file.source_kind != SPEX_SNAPSHOT_HDF_SOURCE_KIND {
        return Err(ValidationError::InvalidValue {
            path: "/meta/@source_kind".to_owned(),
            expected: SPEX_SNAPSHOT_HDF_SOURCE_KIND.to_owned(),
            actual: file.source_kind.clone(),
        }
        .into());
    }
    positive("/initial/@plane_wave_cutoff", file.plane_wave_cutoff)?;
    positive("/initial/@coefficient_cutoff", file.coefficient_cutoff)?;
    let hdf = File::create(path)?;
    write_spex_str_attr(&hdf, ATTR_SCHEMA_NAME, SPEX_SNAPSHOT_HDF_SCHEMA_NAME)?;
    write_i64_attr(&hdf, ATTR_SCHEMA_VERSION, SPEX_SNAPSHOT_HDF_SCHEMA_VERSION)?;
    write_units(&hdf.create_group(GROUP_UNITS)?)?;
    write_meta(
        &hdf.create_group(GROUP_META)?,
        &file.snapshot.meta,
        &file.source_revision,
        &file.source_kind,
    )?;
    write_hashes(&hdf.create_group(GROUP_HASHES)?, &file.hashes)?;
    write_geometry(&hdf.create_group(GROUP_GEOMETRY)?, &file.snapshot.geometry)?;
    write_radial_basis(
        &hdf.create_group(GROUP_RADIAL)?,
        &file.snapshot.geometry,
        &file.scalar_los,
    )?;
    write_initial(&hdf.create_group(GROUP_INITIAL)?, file)
}

fn read_units(group: &Group) -> Result<(), IoError> {
    require_unit(group, "length", "Bohr")?;
    require_unit(group, "inverse_length", "Bohr^-1")?;
    require_unit(group, "energy", "Hartree")?;
    require_unit(group, "density", "Bohr^-3")?;
    require_unit(group, "k_q", "fractional_reciprocal")?;
    require_unit(group, "g", "integer_reciprocal_lattice")?;
    Ok(())
}

fn write_units(group: &Group) -> Result<(), IoError> {
    write_spex_str_attr(group, "length", "Bohr")?;
    write_spex_str_attr(group, "inverse_length", "Bohr^-1")?;
    write_spex_str_attr(group, "energy", "Hartree")?;
    write_spex_str_attr(group, "density", "Bohr^-3")?;
    write_spex_str_attr(group, "k_q", "fractional_reciprocal")?;
    write_spex_str_attr(group, "g", "integer_reciprocal_lattice")
}

fn require_unit(group: &Group, name: &str, expected: &str) -> Result<(), IoError> {
    let actual = read_spex_str_attr(group, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("/units/@{name}"),
            expected: expected.to_owned(),
            actual,
        }
        .into())
    }
}

fn read_meta(group: &Group) -> Result<(MetaV1, String, String), IoError> {
    let title = require_nonempty_attr(group, "title", "/meta/@title")?;
    let producer = require_nonempty_attr(group, "producer", "/meta/@producer")?;
    let producer_version =
        require_nonempty_attr(group, "producer_version", "/meta/@producer_version")?;
    let energy_zero = require_nonempty_attr(group, "energy_zero", "/meta/@energy_zero")?;
    let source_revision =
        require_nonempty_attr(group, "source_revision", "/meta/@source_revision")?;
    let source_kind = require_nonempty_attr(group, "source_kind", "/meta/@source_kind")?;
    if source_kind != SPEX_SNAPSHOT_HDF_SOURCE_KIND {
        return Err(ValidationError::InvalidValue {
            path: "/meta/@source_kind".to_owned(),
            expected: SPEX_SNAPSHOT_HDF_SOURCE_KIND.to_owned(),
            actual: source_kind,
        }
        .into());
    }
    require_token(
        group,
        "angular_basis",
        "/meta/@angular_basis",
        "complex-condon-shortley",
    )?;
    require_token(
        group,
        "radial_quantity",
        "/meta/@radial_quantity",
        "potential",
    )?;
    require_token(
        group,
        "spherical_channel",
        "/meta/@spherical_channel",
        "physical-value",
    )?;
    require_token(
        group,
        "external_basis_required",
        "/meta/@external_basis_required",
        "true",
    )?;
    let members = group.member_names()?;
    let has_keys = members.iter().any(|name| name == "annotation_keys");
    let has_values = members.iter().any(|name| name == "annotation_values");
    let annotations = match (has_keys, has_values) {
        (false, false) => BTreeMap::new(),
        (true, true) => {
            let keys = read_str_vec(group, "annotation_keys", "hash")?;
            let values = read_str_vec(group, "annotation_values", "hash")?;
            require_len("/meta/annotation_values", keys.len(), values.len())?;
            let mut annotations = BTreeMap::new();
            for (key, value) in keys.into_iter().zip(values) {
                nonempty("/meta/annotation_keys[]", &key)?;
                if annotations.insert(key.clone(), value).is_some() {
                    return Err(ValidationError::Duplicate {
                        path: "/meta/annotation_keys".to_owned(),
                        key,
                    }
                    .into());
                }
            }
            annotations
        }
        _ => {
            return Err(ValidationError::InvalidValue {
                path: "/meta".to_owned(),
                expected: "annotation_keys and annotation_values together or absent".to_owned(),
                actual: members.join(","),
            }
            .into());
        }
    };
    Ok((
        MetaV1 {
            title,
            producer,
            producer_version: Some(producer_version),
            energy_zero,
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasisV1::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConventionV1::PhysicalValue,
            },
            annotations,
        },
        source_revision,
        source_kind,
    ))
}

fn write_meta(
    group: &Group,
    meta: &MetaV1,
    source_revision: &str,
    source_kind: &str,
) -> Result<(), IoError> {
    let producer_version =
        meta.producer_version
            .as_deref()
            .ok_or_else(|| ValidationError::Missing {
                path: "/meta/@producer_version".to_owned(),
                key: "producer_version".to_owned(),
            })?;
    write_spex_str_attr(group, "title", &meta.title)?;
    write_spex_str_attr(group, "producer", &meta.producer)?;
    write_spex_str_attr(group, "producer_version", producer_version)?;
    write_spex_str_attr(group, "energy_zero", &meta.energy_zero)?;
    write_spex_str_attr(group, "source_revision", source_revision)?;
    write_spex_str_attr(group, "source_kind", source_kind)?;
    write_spex_str_attr(group, "angular_basis", "complex-condon-shortley")?;
    write_spex_str_attr(group, "radial_quantity", "potential")?;
    write_spex_str_attr(group, "spherical_channel", "physical-value")?;
    write_spex_str_attr(group, "external_basis_required", "true")?;
    if !meta.annotations.is_empty() {
        let keys = meta.annotations.keys().cloned().collect::<Vec<_>>();
        let values = keys
            .iter()
            .map(|key| meta.annotations[key].clone())
            .collect::<Vec<_>>();
        write_str_vec(group, "annotation_keys", &keys, "hash")?;
        write_str_vec(group, "annotation_values", &values, "hash")?;
    }
    Ok(())
}

fn read_hashes(group: &Group) -> Result<Vec<SpexSnapshotHashV1>, IoError> {
    let roles = read_str_vec(group, "roles", "hash")?;
    let names = read_str_vec(group, "names", "hash")?;
    let sha256 = read_str_vec(group, "sha256", "hash")?;
    if roles.is_empty() {
        return Err(ValidationError::Empty {
            path: "/hashes/roles".to_owned(),
        }
        .into());
    }
    require_len("/hashes/names", roles.len(), names.len())?;
    require_len("/hashes/sha256", roles.len(), sha256.len())?;
    let mut unique = BTreeSet::new();
    let mut hashes = Vec::with_capacity(roles.len());
    for ((role, name), digest) in roles.into_iter().zip(names).zip(sha256) {
        nonempty("/hashes/roles[]", &role)?;
        nonempty("/hashes/names[]", &name)?;
        let digest = parse_sha256_field("/hashes/sha256[]", &digest)?;
        if !unique.insert(role.clone()) {
            return Err(ValidationError::Duplicate {
                path: "/hashes/roles".to_owned(),
                key: role,
            }
            .into());
        }
        hashes.push(SpexSnapshotHashV1 {
            role,
            name,
            sha256: digest,
        });
    }
    Ok(hashes)
}

fn write_hashes(group: &Group, hashes: &[SpexSnapshotHashV1]) -> Result<(), IoError> {
    if hashes.is_empty() {
        return Err(ValidationError::Empty {
            path: "/hashes".to_owned(),
        }
        .into());
    }
    let roles = hashes
        .iter()
        .map(|hash| hash.role.clone())
        .collect::<Vec<_>>();
    let names = hashes
        .iter()
        .map(|hash| hash.name.clone())
        .collect::<Vec<_>>();
    let sha256 = hashes
        .iter()
        .map(|hash| hash.sha256.clone())
        .collect::<Vec<_>>();
    write_str_vec(group, "roles", &roles, "hash")?;
    write_str_vec(group, "names", &names, "hash")?;
    write_str_vec(group, "sha256", &sha256, "hash")
}

struct GeometryScratch {
    lattice: crate::snapshot::LatticeV1,
    sites: Vec<SiteV2>,
}

fn read_geometry(group: &Group) -> Result<GeometryScratch, IoError> {
    let lattice_flat =
        read_spex_f64_dataset(group, "lattice_vectors", &[3, 3], &["vector", "xyz"])?;
    let mut vectors = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            vectors[row][column] = lattice_flat[row * 3 + column];
        }
    }
    let lattice = crate::snapshot::LatticeV1 {
        unit: LengthUnitV1::Bohr,
        vectors,
    };
    lattice.validate()?;
    let site_ids = read_str_vec(group, "site_ids", "site")?;
    if site_ids.is_empty() {
        return Err(ValidationError::Empty {
            path: "/geometry/site_ids".to_owned(),
        }
        .into());
    }
    let n_site = site_ids.len();
    let atomic_numbers = read_spex_i32_dataset(group, "atomic_numbers", &[n_site], &["site"])?;
    let positions = read_spex_f64_dataset(
        group,
        "fractional_positions",
        &[n_site, 3],
        &["site", "xyz"],
    )?;
    let radii = read_spex_f64_dataset(group, "muffin_tin_radii", &[n_site], &["site"])?;
    require_finite_f64s("/geometry/fractional_positions", &positions)?;
    let mut unique = BTreeSet::new();
    let mut sites = Vec::with_capacity(n_site);
    for index in 0..n_site {
        nonempty(format!("/geometry/site_ids[{index}]"), &site_ids[index])?;
        if !unique.insert(site_ids[index].clone()) {
            return Err(ValidationError::Duplicate {
                path: "/geometry/site_ids".to_owned(),
                key: site_ids[index].clone(),
            }
            .into());
        }
        let atomic_number = atomic_numbers[index];
        if !(1..=103).contains(&atomic_number) {
            return Err(ValidationError::InvalidValue {
                path: format!("/geometry/atomic_numbers[{index}]"),
                expected: "1..=103".to_owned(),
                actual: atomic_number.to_string(),
            }
            .into());
        }
        positive(format!("/geometry/muffin_tin_radii[{index}]"), radii[index])?;
        sites.push(SiteV2 {
            id: site_ids[index].clone(),
            atomic_number: u16::try_from(atomic_number).unwrap_or(0),
            fractional_position: [
                positions[index * 3],
                positions[index * 3 + 1],
                positions[index * 3 + 2],
            ],
            muffin_tin_radius_unit: LengthUnitV1::Bohr,
            muffin_tin_radius: radii[index],
        });
    }
    Ok(GeometryScratch { lattice, sites })
}

fn write_geometry(group: &Group, geometry: &GeometryV2) -> Result<(), IoError> {
    let mut lattice = Vec::with_capacity(9);
    for row in geometry.lattice.vectors {
        lattice.extend(row);
    }
    write_spex_f64_dataset(
        group,
        "lattice_vectors",
        &[3, 3],
        &lattice,
        &["vector", "xyz"],
    )?;
    let ids = geometry
        .sites
        .iter()
        .map(|site| site.id.clone())
        .collect::<Vec<_>>();
    let numbers = geometry
        .sites
        .iter()
        .map(|site| i32::from(site.atomic_number))
        .collect::<Vec<_>>();
    let mut positions = Vec::with_capacity(geometry.sites.len() * 3);
    let mut radii = Vec::with_capacity(geometry.sites.len());
    for site in &geometry.sites {
        positions.extend(site.fractional_position);
        radii.push(site.muffin_tin_radius);
    }
    write_str_vec(group, "site_ids", &ids, "site")?;
    write_spex_i32_dataset(group, "atomic_numbers", &[ids.len()], &numbers, &["site"])?;
    write_spex_f64_dataset(
        group,
        "fractional_positions",
        &[ids.len(), 3],
        &positions,
        &["site", "xyz"],
    )?;
    write_spex_f64_dataset(group, "muffin_tin_radii", &[ids.len()], &radii, &["site"])
}

fn read_radial_basis(
    group: &Group,
    geometry: &GeometryScratch,
) -> Result<(Vec<SiteRadialBasisV2>, Vec<SpexScalarLoTableV1>), IoError> {
    let records = collect_index_groups(group, PREFIX_BASIS)?;
    if records.is_empty() {
        return Err(ValidationError::Empty {
            path: "/radial_basis".to_owned(),
        }
        .into());
    }
    let mut bases = Vec::with_capacity(records.len());
    let mut orbitals = Vec::with_capacity(records.len());
    for record in records {
        let site_id =
            require_nonempty_attr(&record, "site_id", &format!("{}/@site_id", record.name()))?;
        if !geometry.sites.iter().any(|site| site.id == site_id) {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/@site_id", record.name()),
                expected: "a geometry site id".to_owned(),
                actual: site_id,
            }
            .into());
        }
        let spin = parse_spin(&read_spex_str_attr(&record, "spin")?)?;
        let radial_equation =
            parse_radial_equation(&read_spex_str_attr(&record, "radial_equation")?)?;
        let first = read_spex_f64_attr(
            &record,
            "mesh_first",
            &format!("{}/@mesh_first", record.name()),
        )?;
        let log_increment = read_spex_f64_attr(
            &record,
            "mesh_log_increment",
            &format!("{}/@mesh_log_increment", record.name()),
        )?;
        let point_count = usize::try_from(read_spex_i64_attr(
            &record,
            "mesh_point_count",
            &format!("{}/@mesh_point_count", record.name()),
        )?)
        .map_err(|_| ValidationError::InvalidValue {
            path: format!("{}/@mesh_point_count", record.name()),
            expected: "nonnegative usize".to_owned(),
            actual: "negative".to_owned(),
        })?;
        let last = read_spex_f64_attr(
            &record,
            "mesh_last",
            &format!("{}/@mesh_last", record.name()),
        )?;
        let consistency_tolerance = read_spex_f64_attr(
            &record,
            "mesh_consistency_tolerance",
            &format!("{}/@mesh_consistency_tolerance", record.name()),
        )?;
        let mesh = ExponentialMeshSpecV1 {
            radius_unit: LengthUnitV1::Bohr,
            first,
            log_increment,
            point_count,
            last,
            consistency_tolerance,
        };
        mesh.validate(&format!("{}/mesh", record.name()))?;
        let lin_l = read_i32_open(&record, "linearization_l")?;
        let n_lin = lin_l.len();
        if n_lin == 0 {
            return Err(ValidationError::Empty {
                path: format!("{}/linearization_l", record.name()),
            }
            .into());
        }
        let lin_dataset = record.dataset("linearization_l")?;
        require_shape(&lin_dataset, &[n_lin])?;
        require_spex_axes(&lin_dataset, &["l"])?;
        let lin_e = read_spex_f64_dataset(&record, "linearization_energy", &[n_lin], &["l"])?;
        require_finite_f64s(&format!("{}/linearization_energy", record.name()), &lin_e)?;
        let mut unique_l = BTreeSet::new();
        let mut linearization_energies = Vec::with_capacity(n_lin);
        for (l, energy) in lin_l.into_iter().zip(lin_e) {
            if l < 0 {
                return Err(ValidationError::InvalidValue {
                    path: format!("{}/linearization_l", record.name()),
                    expected: "l>=0".to_owned(),
                    actual: l.to_string(),
                }
                .into());
            }
            if !unique_l.insert(l) {
                return Err(ValidationError::Duplicate {
                    path: format!("{}/linearization_l", record.name()),
                    key: l.to_string(),
                }
                .into());
            }
            linearization_energies.push(EnergyParameterV1 {
                l: u32::try_from(l).unwrap_or(0),
                energy,
            });
        }
        let orbital_group = require_named_group(&record, GROUP_ORBITALS)?;
        let basis_orbitals = read_orbitals(&orbital_group)?;
        let local_orbital_energies = basis_orbitals
            .iter()
            .map(|orbital| EnergyParameterV1 {
                l: orbital.l,
                energy: orbital.energy,
            })
            .collect();
        bases.push(SiteRadialBasisV2 {
            site_id: site_id.clone(),
            spin,
            mesh,
            radial_equation,
            linearization: LinearizationV1 {
                energy_unit: EnergyUnitV1::Hartree,
                linearization_energies,
                local_orbital_energies,
            },
        });
        orbitals.push(SpexScalarLoTableV1 {
            site_id,
            spin,
            orbitals: basis_orbitals,
        });
    }
    Ok((bases, orbitals))
}

fn write_radial_basis(
    group: &Group,
    geometry: &GeometryV2,
    orbitals: &[SpexScalarLoTableV1],
) -> Result<(), IoError> {
    for (index, basis) in geometry.radial_basis.iter().enumerate() {
        let record = group.create_group(&padded(PREFIX_BASIS, index))?;
        write_spex_str_attr(&record, "site_id", &basis.site_id)?;
        write_spex_str_attr(&record, "spin", spin_token(basis.spin))?;
        write_spex_str_attr(
            &record,
            "radial_equation",
            radial_token(basis.radial_equation),
        )?;
        write_f64_attr(&record, "mesh_first", basis.mesh.first)?;
        write_f64_attr(&record, "mesh_log_increment", basis.mesh.log_increment)?;
        write_i64_attr(
            &record,
            "mesh_point_count",
            usize_as_i64("/radial_basis/@mesh_point_count", basis.mesh.point_count)?,
        )?;
        write_f64_attr(&record, "mesh_last", basis.mesh.last)?;
        write_f64_attr(
            &record,
            "mesh_consistency_tolerance",
            basis.mesh.consistency_tolerance,
        )?;
        let lin_l = basis
            .linearization
            .linearization_energies
            .iter()
            .map(|parameter| i32::try_from(parameter.l).unwrap_or(-1))
            .collect::<Vec<_>>();
        let lin_e = basis
            .linearization
            .linearization_energies
            .iter()
            .map(|parameter| parameter.energy)
            .collect::<Vec<_>>();
        write_spex_i32_dataset(&record, "linearization_l", &[lin_l.len()], &lin_l, &["l"])?;
        write_spex_f64_dataset(
            &record,
            "linearization_energy",
            &[lin_e.len()],
            &lin_e,
            &["l"],
        )?;
        let matching = orbitals
            .iter()
            .find(|row| row.site_id == basis.site_id && row.spin == basis.spin)
            .ok_or_else(|| ValidationError::Missing {
                path: "/radial_basis/orbitals".to_owned(),
                key: format!("{}:{:?}", basis.site_id, basis.spin),
            })?;
        write_orbitals(&record.create_group(GROUP_ORBITALS)?, &matching.orbitals)?;
    }
    Ok(())
}

fn read_orbitals(group: &Group) -> Result<Vec<SpexScalarLoV1>, IoError> {
    let members = group.member_names()?;
    for forbidden in ["kappa", "derivative_order"] {
        if members.iter().any(|name| name == forbidden) {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/{forbidden}", group.name()),
                expected: "absent (SPEX does not own signed kappa / HDLO)".to_owned(),
                actual: "present".to_owned(),
            }
            .into());
        }
    }
    let kinds = read_str_vec(group, "kind", "orbital")?;
    let n_orb = kinds.len();
    let l = read_i64_dataset_len(group, "l", n_orb, "orbital")?;
    let energy = read_spex_f64_dataset(group, "energy", &[n_orb], &["orbital"])?;
    require_finite_f64s(&format!("{}/energy", group.name()), &energy)?;
    let principals = if members.iter().any(|name| name == "n") {
        Some(read_i64_dataset_len(group, "n", n_orb, "orbital")?)
    } else {
        None
    };
    let mut orbitals = Vec::with_capacity(n_orb);
    for index in 0..n_orb {
        if kinds[index] != "lo" {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/kind", group.name()),
                expected: "lo".to_owned(),
                actual: kinds[index].clone(),
            }
            .into());
        }
        if l[index] < 0 {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/l", group.name()),
                expected: "l>=0".to_owned(),
                actual: l[index].to_string(),
            }
            .into());
        }
        let n = match &principals {
            None => None,
            Some(values) if values[index] < 1 => {
                return Err(ValidationError::InvalidValue {
                    path: format!("{}/n", group.name()),
                    expected: "n>=1 when pbas owns n".to_owned(),
                    actual: values[index].to_string(),
                }
                .into());
            }
            Some(values) => Some(u32::try_from(values[index]).unwrap_or(0)),
        };
        orbitals.push(SpexScalarLoV1 {
            kind: SpexScalarLoKind::Lo,
            l: u32::try_from(l[index]).unwrap_or(0),
            energy: energy[index],
            n,
        });
    }
    Ok(orbitals)
}

fn write_orbitals(group: &Group, orbitals: &[SpexScalarLoV1]) -> Result<(), IoError> {
    let kinds = vec!["lo".to_owned(); orbitals.len()];
    let l = orbitals
        .iter()
        .map(|orbital| i64::from(orbital.l))
        .collect::<Vec<_>>();
    let energy = orbitals
        .iter()
        .map(|orbital| orbital.energy)
        .collect::<Vec<_>>();
    write_str_vec(group, "kind", &kinds, "orbital")?;
    write_spex_i64_dataset(group, "l", &[l.len()], &l, &["orbital"])?;
    write_spex_f64_dataset(group, "energy", &[energy.len()], &energy, &["orbital"])?;
    if orbitals.iter().all(|orbital| orbital.n.is_some()) && !orbitals.is_empty() {
        let n = orbitals
            .iter()
            .map(|orbital| i64::from(orbital.n.expect("checked")))
            .collect::<Vec<_>>();
        write_spex_i64_dataset(group, "n", &[n.len()], &n, &["orbital"])?;
    } else if orbitals.iter().any(|orbital| orbital.n.is_some()) {
        return Err(ValidationError::InvalidValue {
            path: format!("{}/n", group.name()),
            expected: "n dataset for every LO or omitted".to_owned(),
            actual: "partial n".to_owned(),
        }
        .into());
    }
    Ok(())
}

fn read_initial(
    group: &Group,
    geometry: &GeometryV2,
    angular_basis: AngularBasisV1,
) -> Result<(InitialV2, f64, f64, String, String), IoError> {
    let kind = require_nonempty_attr(group, "kind", "/initial/@kind")?;
    require_token(
        group,
        "potential_unit",
        "/initial/@potential_unit",
        "hartree",
    )?;
    require_token(
        group,
        "potential_representation",
        "/initial/@potential_representation",
        "masked-operator",
    )?;
    require_token(
        group,
        "angular_basis",
        "/initial/@angular_basis",
        "complex-condon-shortley",
    )?;
    require_token(
        group,
        "reciprocal_length_unit",
        "/initial/@reciprocal_length_unit",
        "bohr^-1",
    )?;
    require_token(
        group,
        "fourier_normalization",
        "/initial/@fourier_normalization",
        "cell-normalized",
    )?;
    require_token(
        group,
        "fourier_phase",
        "/initial/@fourier_phase",
        "negative-exponent",
    )?;
    let plane_wave_cutoff =
        read_spex_f64_attr(group, "plane_wave_cutoff", "/initial/@plane_wave_cutoff")?;
    let coefficient_cutoff =
        read_spex_f64_attr(group, "coefficient_cutoff", "/initial/@coefficient_cutoff")?;
    positive("/initial/@plane_wave_cutoff", plane_wave_cutoff)?;
    positive("/initial/@coefficient_cutoff", coefficient_cutoff)?;
    let spin_layout = require_nonempty_attr(group, "spin_layout", "/initial/@spin_layout")?;
    match spin_layout.as_str() {
        "collinear-up-down" | "spin-unpolarized" | "noncollinear-spin-density-matrix" => {}
        other => {
            return Err(ValidationError::InvalidValue {
                path: "/initial/@spin_layout".to_owned(),
                expected: "collinear-up-down|spin-unpolarized|noncollinear-spin-density-matrix"
                    .to_owned(),
                actual: other.to_owned(),
            }
            .into());
        }
    }
    let interstitial_phase =
        require_nonempty_attr(group, "interstitial_phase", "/initial/@interstitial_phase")?;
    match interstitial_phase.as_str() {
        "positive-exponent" | "negative-exponent" => {}
        other => {
            return Err(ValidationError::InvalidValue {
                path: "/initial/@interstitial_phase".to_owned(),
                expected: "positive-exponent|negative-exponent".to_owned(),
                actual: other.to_owned(),
            }
            .into());
        }
    }
    let potential_group = require_named_group(group, GROUP_POTENTIAL)?;
    let mut potential = read_potential(&potential_group, geometry, angular_basis, &spin_layout)?;
    potential.basis_hints.plane_wave_cutoff = Some(plane_wave_cutoff);
    potential.basis_hints.coefficient_cutoff = Some(coefficient_cutoff);
    let members = group.member_names()?;
    let has_density = members.iter().any(|name| name == GROUP_DENSITY);
    let initial = match kind.as_str() {
        "frozen-potential" => {
            if has_density {
                return Err(ValidationError::InvalidValue {
                    path: "/initial/density".to_owned(),
                    expected: "absent for frozen-potential".to_owned(),
                    actual: "present".to_owned(),
                }
                .into());
            }
            InitialV2::FrozenPotential { potential }
        }
        "restart" => {
            if !has_density {
                return Err(ValidationError::Missing {
                    path: "/initial".to_owned(),
                    key: GROUP_DENSITY.to_owned(),
                }
                .into());
            }
            let mut density = read_density(&group.group(GROUP_DENSITY)?, geometry, angular_basis)?;
            density.basis_hints.plane_wave_cutoff = Some(plane_wave_cutoff);
            density.basis_hints.coefficient_cutoff = Some(coefficient_cutoff);
            InitialV2::Restart { density, potential }
        }
        other => {
            return Err(ValidationError::InvalidValue {
                path: "/initial/@kind".to_owned(),
                expected: "frozen-potential or restart".to_owned(),
                actual: other.to_owned(),
            }
            .into());
        }
    };
    Ok((
        initial,
        plane_wave_cutoff,
        coefficient_cutoff,
        spin_layout,
        interstitial_phase,
    ))
}

fn write_initial(group: &Group, file: &SpexFrozenFieldsV1) -> Result<(), IoError> {
    let (kind, potential, density) = match &file.snapshot.initial {
        InitialV2::FrozenPotential { potential } => ("frozen-potential", potential, None),
        InitialV2::Restart { density, potential } => ("restart", potential, Some(density)),
    };
    write_spex_str_attr(group, "kind", kind)?;
    write_spex_str_attr(group, "potential_unit", "hartree")?;
    write_spex_str_attr(group, "potential_representation", "masked-operator")?;
    write_spex_str_attr(group, "angular_basis", "complex-condon-shortley")?;
    write_spex_str_attr(group, "reciprocal_length_unit", "bohr^-1")?;
    write_spex_str_attr(group, "fourier_normalization", "cell-normalized")?;
    write_spex_str_attr(group, "fourier_phase", "negative-exponent")?;
    write_spex_str_attr(group, "spin_layout", &file.spin_layout)?;
    write_spex_str_attr(group, "interstitial_phase", &file.interstitial_phase)?;
    write_f64_attr(group, "plane_wave_cutoff", file.plane_wave_cutoff)?;
    write_f64_attr(group, "coefficient_cutoff", file.coefficient_cutoff)?;
    write_potential(
        &group.create_group(GROUP_POTENTIAL)?,
        potential,
        &file.snapshot.geometry,
        &file.spin_layout,
    )?;
    if let Some(density) = density {
        write_density(
            &group.create_group(GROUP_DENSITY)?,
            density,
            &file.snapshot.geometry,
        )?;
    }
    Ok(())
}

fn read_potential(
    group: &Group,
    geometry: &GeometryV2,
    angular_basis: AngularBasisV1,
    spin_layout: &str,
) -> Result<PotentialV2, IoError> {
    let mut components = BTreeMap::new();
    for name in POTENTIAL_COMPONENTS {
        let child = require_named_group(group, name)?;
        let field = read_regional(&child, geometry)?;
        if name != "v0" {
            require_zero_source(&child, name, &field, spin_layout)?;
        }
        components.insert(name, field);
    }
    Ok(PotentialV2 {
        unit: FieldUnitV2::Hartree,
        representation: FieldRepresentationV2::MaskedOperator,
        angular_basis,
        basis_hints: hints_placeholder(),
        v0: components.remove("v0").expect("v0"),
        bx: components.remove("bx").expect("bx"),
        by: components.remove("by").expect("by"),
        bz: components.remove("bz").expect("bz"),
    })
}

fn write_potential(
    group: &Group,
    potential: &PotentialV2,
    geometry: &GeometryV2,
    spin_layout: &str,
) -> Result<(), IoError> {
    write_regional(&group.create_group("v0")?, &potential.v0, geometry)?;
    write_b_component(
        &group.create_group("bx")?,
        &potential.bx,
        geometry,
        spin_layout,
    )?;
    write_b_component(
        &group.create_group("by")?,
        &potential.by,
        geometry,
        spin_layout,
    )?;
    write_b_component(
        &group.create_group("bz")?,
        &potential.bz,
        geometry,
        spin_layout,
    )
}

fn write_b_component(
    group: &Group,
    field: &RegionalFieldV2,
    geometry: &GeometryV2,
    spin_layout: &str,
) -> Result<(), IoError> {
    if regional_all_zero(field) {
        let tag = match spin_layout {
            "collinear-up-down" => "collinear-spin-density-matrix",
            "spin-unpolarized" => "spin-unpolarized",
            other => {
                return Err(ValidationError::InvalidValue {
                    path: format!("{}/@zero_source", group.name()),
                    expected: "nonzero B or a collinear/unpolarized zero tag".to_owned(),
                    actual: other.to_owned(),
                }
                .into());
            }
        };
        write_spex_str_attr(group, "zero_source", tag)?;
    }
    write_regional(group, field, geometry)
}

fn require_zero_source(
    group: &Group,
    name: &str,
    field: &RegionalFieldV2,
    spin_layout: &str,
) -> Result<(), IoError> {
    let names = group.attr_names()?;
    let has_tag = names.iter().any(|attr| attr == "zero_source");
    let zero = regional_all_zero(field);
    match (zero, has_tag) {
        (true, false) => Err(ValidationError::Missing {
            path: format!("{name}/@zero_source"),
            key: "zero_source".to_owned(),
        }
        .into()),
        (false, true) => Err(ValidationError::InvalidValue {
            path: format!("{name}/@zero_source"),
            expected: "absent when B is nonzero".to_owned(),
            actual: read_spex_str_attr(group, "zero_source")?,
        }
        .into()),
        (true, true) => {
            let tag = read_spex_str_attr(group, "zero_source")?;
            let expected = match spin_layout {
                "collinear-up-down" => "collinear-spin-density-matrix",
                "spin-unpolarized" => "spin-unpolarized",
                _ => {
                    return Err(ValidationError::InvalidValue {
                        path: format!("{name}/@zero_source"),
                        expected: "nonzero B for noncollinear layout".to_owned(),
                        actual: tag,
                    }
                    .into());
                }
            };
            if tag == expected {
                Ok(())
            } else {
                Err(ValidationError::InvalidValue {
                    path: format!("{name}/@zero_source"),
                    expected: expected.to_owned(),
                    actual: tag,
                }
                .into())
            }
        }
        (false, false) => Ok(()),
    }
}

fn regional_all_zero(field: &RegionalFieldV2) -> bool {
    let mt_zero = field.muffin_tins.iter().all(|site| {
        site.channels.iter().all(|channel| {
            channel.real.iter().all(|value| *value == 0.0)
                && channel.imaginary.iter().all(|value| *value == 0.0)
        })
    });
    let pw_zero = field
        .interstitial
        .coefficients
        .iter()
        .all(|coefficient| coefficient.value.real == 0.0 && coefficient.value.imaginary == 0.0);
    mt_zero && pw_zero
}

fn read_density(
    group: &Group,
    geometry: &GeometryV2,
    angular_basis: AngularBasisV1,
) -> Result<DensityV2, IoError> {
    require_token(
        group,
        "density_unit",
        "/initial/density/@density_unit",
        "bohr^-3",
    )?;
    require_token(
        group,
        "density_representation",
        "/initial/density/@density_representation",
        "periodic-extension",
    )?;
    let mut components = BTreeMap::new();
    for name in DENSITY_COMPONENTS {
        let child = require_named_group(group, name)?;
        components.insert(name, read_regional(&child, geometry)?);
    }
    Ok(DensityV2 {
        unit: FieldUnitV2::BohrMinus3,
        representation: FieldRepresentationV2::PeriodicExtension,
        angular_basis,
        basis_hints: hints_placeholder(),
        n: components.remove("n").expect("n"),
        mx: components.remove("mx").expect("mx"),
        my: components.remove("my").expect("my"),
        mz: components.remove("mz").expect("mz"),
    })
}

fn write_density(group: &Group, density: &DensityV2, geometry: &GeometryV2) -> Result<(), IoError> {
    write_spex_str_attr(group, "density_unit", "bohr^-3")?;
    write_spex_str_attr(group, "density_representation", "periodic-extension")?;
    write_regional(&group.create_group("n")?, &density.n, geometry)?;
    write_regional(&group.create_group("mx")?, &density.mx, geometry)?;
    write_regional(&group.create_group("my")?, &density.my, geometry)?;
    write_regional(&group.create_group("mz")?, &density.mz, geometry)
}

fn hints_placeholder() -> BasisHintsV1 {
    BasisHintsV1 {
        reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
        plane_wave_cutoff: None,
        coefficient_cutoff: None,
        normalization: FourierNormalizationV1::CellNormalized,
        phase: FourierPhaseV1::NegativeExponent,
    }
}

fn read_regional(group: &Group, geometry: &GeometryV2) -> Result<RegionalFieldV2, IoError> {
    let mt_parent = require_named_group(group, GROUP_MT)?;
    let sites = collect_index_groups(&mt_parent, PREFIX_SITE)?;
    if sites.len() != geometry.sites.len() {
        return Err(ValidationError::LengthMismatch {
            path: format!("{}/muffin_tins", group.name()),
            expected: geometry.sites.len(),
            actual: sites.len(),
        }
        .into());
    }
    let mut muffin_tins = Vec::with_capacity(sites.len());
    for (index, site_group) in sites.into_iter().enumerate() {
        let site_id = require_nonempty_attr(
            &site_group,
            "site_id",
            &format!("{}/@site_id", site_group.name()),
        )?;
        if site_id != geometry.sites[index].id {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/@site_id", site_group.name()),
                expected: geometry.sites[index].id.clone(),
                actual: site_id,
            }
            .into());
        }
        let point_count = geometry
            .radial_basis
            .iter()
            .find(|basis| basis.site_id == site_id)
            .map(|basis| basis.mesh.point_count)
            .ok_or_else(|| ValidationError::Missing {
                path: "/radial_basis".to_owned(),
                key: site_id.clone(),
            })?;
        let l = read_i32_open(&site_group, "l")?;
        let n_lm = l.len();
        if n_lm == 0 {
            return Err(ValidationError::Empty {
                path: format!("{}/l", site_group.name()),
            }
            .into());
        }
        let m = read_spex_i32_dataset(&site_group, "m", &[n_lm], &["lm"])?;
        let samples = read_spex_f64_dataset(
            &site_group,
            "samples",
            &[n_lm, point_count, 2],
            &["lm", "radial", "re_im"],
        )?;
        require_finite_f64s(&format!("{}/samples", site_group.name()), &samples)?;
        let mut channels = Vec::with_capacity(n_lm);
        for lm in 0..n_lm {
            if l[lm] < 0 {
                return Err(ValidationError::InvalidLm {
                    path: format!("{}/l", site_group.name()),
                    l: 0,
                    m: m[lm],
                }
                .into());
            }
            let mut real = Vec::with_capacity(point_count);
            let mut imaginary = Vec::with_capacity(point_count);
            for radial in 0..point_count {
                let base = (lm * point_count + radial) * 2;
                real.push(samples[base]);
                imaginary.push(samples[base + 1]);
            }
            channels.push(SphericalChannelV2 {
                l: u32::try_from(l[lm]).unwrap_or(0),
                m: m[lm],
                real,
                imaginary,
            });
        }
        muffin_tins.push(MuffinTinFieldV2 { site_id, channels });
    }
    let interstitial = require_named_group(group, GROUP_INTERSTITIAL)?;
    let g = read_i32_open(&interstitial, "g")?;
    if g.len() % 3 != 0 {
        return Err(ValidationError::LengthMismatch {
            path: format!("{}/g", interstitial.name()),
            expected: g.len() / 3 * 3,
            actual: g.len(),
        }
        .into());
    }
    let n_g = g.len() / 3;
    if n_g == 0 {
        return Err(ValidationError::Empty {
            path: format!("{}/g", interstitial.name()),
        }
        .into());
    }
    require_shape_open(&interstitial, "g", &[n_g, 3])?;
    let coefficients =
        read_spex_f64_dataset(&interstitial, "coefficients", &[n_g, 2], &["g", "re_im"])?;
    require_finite_f64s(
        &format!("{}/coefficients", interstitial.name()),
        &coefficients,
    )?;
    let mut unique = BTreeSet::new();
    let mut fourier = Vec::with_capacity(n_g);
    for index in 0..n_g {
        let label = [g[index * 3], g[index * 3 + 1], g[index * 3 + 2]];
        if !unique.insert(label) {
            return Err(ValidationError::Duplicate {
                path: format!("{}/g", interstitial.name()),
                key: format!("{label:?}"),
            }
            .into());
        }
        fourier.push(FourierCoefficientV2 {
            g: label,
            value: Complex64V2 {
                real: coefficients[index * 2],
                imaginary: coefficients[index * 2 + 1],
            },
        });
    }
    Ok(RegionalFieldV2 {
        muffin_tins,
        interstitial: InterstitialFieldV2 {
            coefficients: fourier,
        },
    })
}

fn write_regional(
    group: &Group,
    field: &RegionalFieldV2,
    geometry: &GeometryV2,
) -> Result<(), IoError> {
    let mt = group.create_group(GROUP_MT)?;
    for (index, site) in field.muffin_tins.iter().enumerate() {
        let point_count = geometry
            .radial_basis
            .iter()
            .find(|basis| basis.site_id == site.site_id)
            .map(|basis| basis.mesh.point_count)
            .ok_or_else(|| ValidationError::Missing {
                path: "/radial_basis".to_owned(),
                key: site.site_id.clone(),
            })?;
        let child = mt.create_group(&padded(PREFIX_SITE, index))?;
        write_spex_str_attr(&child, "site_id", &site.site_id)?;
        let l = site
            .channels
            .iter()
            .map(|channel| i32::try_from(channel.l).unwrap_or(-1))
            .collect::<Vec<_>>();
        let m = site
            .channels
            .iter()
            .map(|channel| channel.m)
            .collect::<Vec<_>>();
        let mut samples = Vec::with_capacity(site.channels.len() * point_count * 2);
        for channel in &site.channels {
            for radial in 0..point_count {
                samples.push(channel.real[radial]);
                samples.push(channel.imaginary.get(radial).copied().unwrap_or(0.0));
            }
        }
        write_spex_i32_dataset(&child, "l", &[l.len()], &l, &["lm"])?;
        write_spex_i32_dataset(&child, "m", &[m.len()], &m, &["lm"])?;
        write_spex_f64_dataset(
            &child,
            "samples",
            &[site.channels.len(), point_count, 2],
            &samples,
            &["lm", "radial", "re_im"],
        )?;
    }
    let interstitial = group.create_group(GROUP_INTERSTITIAL)?;
    let mut g = Vec::with_capacity(field.interstitial.coefficients.len() * 3);
    let mut coefficients = Vec::with_capacity(field.interstitial.coefficients.len() * 2);
    for coefficient in &field.interstitial.coefficients {
        g.extend(coefficient.g);
        coefficients.push(coefficient.value.real);
        coefficients.push(coefficient.value.imaginary);
    }
    let n_g = field.interstitial.coefficients.len();
    write_spex_i32_dataset(&interstitial, "g", &[n_g, 3], &g, &["g", "xyz"])?;
    write_spex_f64_dataset(
        &interstitial,
        "coefficients",
        &[n_g, 2],
        &coefficients,
        &["g", "re_im"],
    )
}

fn read_i64_dataset_len(
    group: &Group,
    name: &str,
    expected: usize,
    axis: &str,
) -> Result<Vec<i64>, IoError> {
    read_spex_i64_dataset(group, name, &[expected], &[axis])
}

fn read_i32_open(group: &Group, name: &str) -> Result<Vec<i32>, IoError> {
    let dataset = group.dataset(name)?;
    crate::mldump::require_numeric_dtype::<i32>(&dataset, &format!("{}/dtype", dataset.name()))?;
    Ok(dataset.read_raw()?)
}

fn require_shape_open(group: &Group, name: &str, expected: &[usize]) -> Result<(), IoError> {
    require_shape(&group.dataset(name)?, expected)
}

fn read_str_vec(group: &Group, name: &str, axis: &str) -> Result<Vec<String>, IoError> {
    let dataset = group.dataset(name)?;
    let len = dataset.shape().first().copied().unwrap_or(0);
    require_shape(&dataset, &[len])?;
    require_spex_axes(&dataset, &[axis])?;
    let descriptor = dataset.dtype()?.to_descriptor()?;
    let values = match descriptor {
        TypeDescriptor::VarLenUnicode => dataset
            .read_raw::<VarLenUnicode>()?
            .iter()
            .map(|value| trim_hwrapper_str(value.as_str()))
            .collect(),
        TypeDescriptor::VarLenAscii => dataset
            .read_raw::<VarLenAscii>()?
            .iter()
            .map(|value| trim_hwrapper_str(value.as_str()))
            .collect(),
        TypeDescriptor::FixedAscii(elem) | TypeDescriptor::FixedUnicode(elem) => {
            read_fixed_ascii_dataset(&dataset, name, len, elem)?
        }
        other => {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/dtype", dataset.name()),
                expected: "1-d H5T_NATIVE_CHARACTER (VL fallback only)".to_owned(),
                actual: other.to_string(),
            }
            .into());
        }
    };
    if !values.is_empty() {
        for (index, value) in values.iter().enumerate() {
            nonempty(format!("{name}[{index}]"), value)?;
        }
    }
    Ok(values)
}

fn read_fixed_ascii_dataset(
    dataset: &Dataset,
    name: &str,
    len: usize,
    elem: usize,
) -> Result<Vec<String>, IoError> {
    macro_rules! arm {
        ($n:literal) => {
            if elem == $n {
                let values: Vec<hdf5_metno::types::FixedAscii<$n>> = dataset.read_raw()?;
                require_len(name, len, values.len())?;
                return Ok(values
                    .iter()
                    .map(|value| trim_hwrapper_str(value.as_str()))
                    .collect());
            }
        };
    }
    arm!(1);
    arm!(2);
    arm!(3);
    arm!(4);
    arm!(5);
    arm!(6);
    arm!(7);
    arm!(8);
    arm!(9);
    arm!(10);
    arm!(11);
    arm!(12);
    arm!(13);
    arm!(14);
    arm!(15);
    arm!(16);
    arm!(17);
    arm!(18);
    arm!(19);
    arm!(20);
    arm!(21);
    arm!(22);
    arm!(23);
    arm!(24);
    arm!(25);
    arm!(26);
    arm!(27);
    arm!(28);
    arm!(29);
    arm!(30);
    arm!(31);
    arm!(32);
    arm!(33);
    arm!(34);
    arm!(35);
    arm!(36);
    arm!(37);
    arm!(38);
    arm!(39);
    arm!(40);
    arm!(41);
    arm!(42);
    arm!(43);
    arm!(44);
    arm!(45);
    arm!(46);
    arm!(47);
    arm!(48);
    arm!(49);
    arm!(50);
    arm!(51);
    arm!(52);
    arm!(53);
    arm!(54);
    arm!(55);
    arm!(56);
    arm!(57);
    arm!(58);
    arm!(59);
    arm!(60);
    arm!(61);
    arm!(62);
    arm!(63);
    arm!(64);
    arm!(65);
    arm!(66);
    arm!(67);
    arm!(68);
    arm!(69);
    arm!(70);
    arm!(71);
    arm!(72);
    arm!(73);
    arm!(74);
    arm!(75);
    arm!(76);
    arm!(77);
    arm!(78);
    arm!(79);
    arm!(80);
    arm!(81);
    arm!(82);
    arm!(83);
    arm!(84);
    arm!(85);
    arm!(86);
    arm!(87);
    arm!(88);
    arm!(89);
    arm!(90);
    arm!(91);
    arm!(92);
    arm!(93);
    arm!(94);
    arm!(95);
    arm!(96);
    arm!(97);
    arm!(98);
    arm!(99);
    arm!(100);
    arm!(101);
    arm!(102);
    arm!(103);
    arm!(104);
    arm!(105);
    arm!(106);
    arm!(107);
    arm!(108);
    arm!(109);
    arm!(110);
    arm!(111);
    arm!(112);
    arm!(113);
    arm!(114);
    arm!(115);
    arm!(116);
    arm!(117);
    arm!(118);
    arm!(119);
    arm!(120);
    arm!(121);
    arm!(122);
    arm!(123);
    arm!(124);
    arm!(125);
    arm!(126);
    arm!(127);
    arm!(128);
    arm!(129);
    arm!(130);
    arm!(131);
    arm!(132);
    arm!(133);
    arm!(134);
    arm!(135);
    arm!(136);
    arm!(137);
    arm!(138);
    arm!(139);
    arm!(140);
    arm!(141);
    arm!(142);
    arm!(143);
    arm!(144);
    arm!(145);
    arm!(146);
    arm!(147);
    arm!(148);
    arm!(149);
    arm!(150);
    arm!(151);
    arm!(152);
    arm!(153);
    arm!(154);
    arm!(155);
    arm!(156);
    arm!(157);
    arm!(158);
    arm!(159);
    arm!(160);
    arm!(161);
    arm!(162);
    arm!(163);
    arm!(164);
    arm!(165);
    arm!(166);
    arm!(167);
    arm!(168);
    arm!(169);
    arm!(170);
    arm!(171);
    arm!(172);
    arm!(173);
    arm!(174);
    arm!(175);
    arm!(176);
    arm!(177);
    arm!(178);
    arm!(179);
    arm!(180);
    arm!(181);
    arm!(182);
    arm!(183);
    arm!(184);
    arm!(185);
    arm!(186);
    arm!(187);
    arm!(188);
    arm!(189);
    arm!(190);
    arm!(191);
    arm!(192);
    arm!(193);
    arm!(194);
    arm!(195);
    arm!(196);
    arm!(197);
    arm!(198);
    arm!(199);
    arm!(200);
    arm!(201);
    arm!(202);
    arm!(203);
    arm!(204);
    arm!(205);
    arm!(206);
    arm!(207);
    arm!(208);
    arm!(209);
    arm!(210);
    arm!(211);
    arm!(212);
    arm!(213);
    arm!(214);
    arm!(215);
    arm!(216);
    arm!(217);
    arm!(218);
    arm!(219);
    arm!(220);
    arm!(221);
    arm!(222);
    arm!(223);
    arm!(224);
    arm!(225);
    arm!(226);
    arm!(227);
    arm!(228);
    arm!(229);
    arm!(230);
    arm!(231);
    arm!(232);
    arm!(233);
    arm!(234);
    arm!(235);
    arm!(236);
    arm!(237);
    arm!(238);
    arm!(239);
    arm!(240);
    arm!(241);
    arm!(242);
    arm!(243);
    arm!(244);
    arm!(245);
    arm!(246);
    arm!(247);
    arm!(248);
    arm!(249);
    arm!(250);
    arm!(251);
    arm!(252);
    arm!(253);
    arm!(254);
    arm!(255);
    arm!(256);
    Err(ValidationError::InvalidValue {
        path: format!("{name}/dtype"),
        expected: "H5T_NATIVE_CHARACTER element length 1..=256".to_owned(),
        actual: elem.to_string(),
    }
    .into())
}

fn write_str_vec(group: &Group, name: &str, values: &[String], axis: &str) -> Result<(), IoError> {
    let encoded = values
        .iter()
        .map(|value| pad_fixed_ascii256(name, value))
        .collect::<Result<Vec<_>, _>>()?;
    let dataset =
        create_dataset::<hdf5_metno::types::FixedAscii<256>>(group, name, &[values.len()])?;
    dataset.write_raw(encoded.as_slice())?;
    write_spex_axes(&dataset, &[axis])
}

fn pad_fixed_ascii256(
    path: &str,
    value: &str,
) -> Result<hdf5_metno::types::FixedAscii<256>, IoError> {
    if value.len() > 256 {
        return Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "string length <= 256".to_owned(),
            actual: value.len().to_string(),
        }
        .into());
    }
    let mut buf = [b' '; 256];
    buf[..value.len()].copy_from_slice(value.as_bytes());
    hdf5_metno::types::FixedAscii::<256>::from_ascii(&buf).map_err(|err| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "ascii".to_owned(),
            actual: err.to_string(),
        }
        .into()
    })
}

fn require_named_group(parent: &Group, name: &str) -> Result<Group, IoError> {
    if !parent.member_names()?.iter().any(|member| member == name) {
        return Err(ValidationError::Missing {
            path: parent.name(),
            key: name.to_owned(),
        }
        .into());
    }
    Ok(parent.group(name)?)
}

fn require_exact_members(file: &File, expected: &[&str]) -> Result<(), IoError> {
    let observed = file.member_names()?.into_iter().collect::<BTreeSet<_>>();
    let expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    if observed == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: "/members".to_owned(),
            expected: expected.into_iter().collect::<Vec<_>>().join(","),
            actual: observed.into_iter().collect::<Vec<_>>().join(","),
        }
        .into())
    }
}

fn require_nonempty_attr(object: &Location, name: &str, path: &str) -> Result<String, IoError> {
    let value = read_spex_str_attr(object, name)?;
    nonempty(path, &value)?;
    Ok(value)
}

fn require_token(group: &Group, name: &str, path: &str, expected: &str) -> Result<(), IoError> {
    let actual = read_spex_str_attr(group, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: expected.to_owned(),
            actual,
        }
        .into())
    }
}

fn trim_hwrapper_str(value: &str) -> String {
    let before_nul = value.split('\0').next().unwrap_or(value);
    before_nul.trim_end_matches(['\0', ' ']).to_owned()
}

fn one_string(name: &str, values: &[VarLenUnicode]) -> Result<String, IoError> {
    if values.len() != 1 {
        return Err(ValidationError::LengthMismatch {
            path: format!("@{name}"),
            expected: 1,
            actual: values.len(),
        }
        .into());
    }
    Ok(trim_hwrapper_str(values[0].as_str()))
}

fn one_ascii(name: &str, values: &[VarLenAscii]) -> Result<String, IoError> {
    if values.len() != 1 {
        return Err(ValidationError::LengthMismatch {
            path: format!("@{name}"),
            expected: 1,
            actual: values.len(),
        }
        .into());
    }
    Ok(trim_hwrapper_str(values[0].as_str()))
}

fn require_unit_attr_shape(attr: &Attribute, name: &str) -> Result<(), IoError> {
    let shape = attr.shape();
    if shape.is_empty() || shape == [1] {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("@{name}/shape"),
            expected: "scalar or length-1".to_owned(),
            actual: format!("{shape:?}"),
        }
        .into())
    }
}

fn read_spex_string_attribute(attr: &Attribute, name: &str) -> Result<String, IoError> {
    require_unit_attr_shape(attr, name)?;
    let descriptor = attr.dtype()?.to_descriptor()?;
    match descriptor {
        TypeDescriptor::VarLenUnicode => one_string(name, &attr.read_raw::<VarLenUnicode>()?),
        TypeDescriptor::VarLenAscii => one_ascii(name, &attr.read_raw::<VarLenAscii>()?),
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            read_fixed_native_attr(attr, name, len)
        }
        other => Err(ValidationError::InvalidValue {
            path: format!("@{name}/dtype"),
            expected: "H5T_NATIVE_CHARACTER or variable-length string".to_owned(),
            actual: other.to_string(),
        }
        .into()),
    }
}

fn read_fixed_native_attr(attr: &Attribute, name: &str, len: usize) -> Result<String, IoError> {
    macro_rules! arm {
        ($n:literal) => {
            if len == $n {
                let values: Vec<hdf5_metno::types::FixedAscii<$n>> = attr.read_raw()?;
                if values.len() != 1 {
                    return Err(ValidationError::LengthMismatch {
                        path: format!("@{name}"),
                        expected: 1,
                        actual: values.len(),
                    }
                    .into());
                }
                return Ok(trim_hwrapper_str(values[0].as_str()));
            }
        };
    }
    arm!(1);
    arm!(2);
    arm!(3);
    arm!(4);
    arm!(5);
    arm!(6);
    arm!(7);
    arm!(8);
    arm!(9);
    arm!(10);
    arm!(11);
    arm!(12);
    arm!(13);
    arm!(14);
    arm!(15);
    arm!(16);
    arm!(17);
    arm!(18);
    arm!(19);
    arm!(20);
    arm!(21);
    arm!(22);
    arm!(23);
    arm!(24);
    arm!(25);
    arm!(26);
    arm!(27);
    arm!(28);
    arm!(29);
    arm!(30);
    arm!(31);
    arm!(32);
    arm!(33);
    arm!(34);
    arm!(35);
    arm!(36);
    arm!(37);
    arm!(38);
    arm!(39);
    arm!(40);
    arm!(41);
    arm!(42);
    arm!(43);
    arm!(44);
    arm!(45);
    arm!(46);
    arm!(47);
    arm!(48);
    arm!(49);
    arm!(50);
    arm!(51);
    arm!(52);
    arm!(53);
    arm!(54);
    arm!(55);
    arm!(56);
    arm!(57);
    arm!(58);
    arm!(59);
    arm!(60);
    arm!(61);
    arm!(62);
    arm!(63);
    arm!(64);
    arm!(65);
    arm!(66);
    arm!(67);
    arm!(68);
    arm!(69);
    arm!(70);
    arm!(71);
    arm!(72);
    arm!(73);
    arm!(74);
    arm!(75);
    arm!(76);
    arm!(77);
    arm!(78);
    arm!(79);
    arm!(80);
    arm!(81);
    arm!(82);
    arm!(83);
    arm!(84);
    arm!(85);
    arm!(86);
    arm!(87);
    arm!(88);
    arm!(89);
    arm!(90);
    arm!(91);
    arm!(92);
    arm!(93);
    arm!(94);
    arm!(95);
    arm!(96);
    arm!(97);
    arm!(98);
    arm!(99);
    arm!(100);
    arm!(101);
    arm!(102);
    arm!(103);
    arm!(104);
    arm!(105);
    arm!(106);
    arm!(107);
    arm!(108);
    arm!(109);
    arm!(110);
    arm!(111);
    arm!(112);
    arm!(113);
    arm!(114);
    arm!(115);
    arm!(116);
    arm!(117);
    arm!(118);
    arm!(119);
    arm!(120);
    arm!(121);
    arm!(122);
    arm!(123);
    arm!(124);
    arm!(125);
    arm!(126);
    arm!(127);
    arm!(128);
    arm!(129);
    arm!(130);
    arm!(131);
    arm!(132);
    arm!(133);
    arm!(134);
    arm!(135);
    arm!(136);
    arm!(137);
    arm!(138);
    arm!(139);
    arm!(140);
    arm!(141);
    arm!(142);
    arm!(143);
    arm!(144);
    arm!(145);
    arm!(146);
    arm!(147);
    arm!(148);
    arm!(149);
    arm!(150);
    arm!(151);
    arm!(152);
    arm!(153);
    arm!(154);
    arm!(155);
    arm!(156);
    arm!(157);
    arm!(158);
    arm!(159);
    arm!(160);
    arm!(161);
    arm!(162);
    arm!(163);
    arm!(164);
    arm!(165);
    arm!(166);
    arm!(167);
    arm!(168);
    arm!(169);
    arm!(170);
    arm!(171);
    arm!(172);
    arm!(173);
    arm!(174);
    arm!(175);
    arm!(176);
    arm!(177);
    arm!(178);
    arm!(179);
    arm!(180);
    arm!(181);
    arm!(182);
    arm!(183);
    arm!(184);
    arm!(185);
    arm!(186);
    arm!(187);
    arm!(188);
    arm!(189);
    arm!(190);
    arm!(191);
    arm!(192);
    arm!(193);
    arm!(194);
    arm!(195);
    arm!(196);
    arm!(197);
    arm!(198);
    arm!(199);
    arm!(200);
    arm!(201);
    arm!(202);
    arm!(203);
    arm!(204);
    arm!(205);
    arm!(206);
    arm!(207);
    arm!(208);
    arm!(209);
    arm!(210);
    arm!(211);
    arm!(212);
    arm!(213);
    arm!(214);
    arm!(215);
    arm!(216);
    arm!(217);
    arm!(218);
    arm!(219);
    arm!(220);
    arm!(221);
    arm!(222);
    arm!(223);
    arm!(224);
    arm!(225);
    arm!(226);
    arm!(227);
    arm!(228);
    arm!(229);
    arm!(230);
    arm!(231);
    arm!(232);
    arm!(233);
    arm!(234);
    arm!(235);
    arm!(236);
    arm!(237);
    arm!(238);
    arm!(239);
    arm!(240);
    arm!(241);
    arm!(242);
    arm!(243);
    arm!(244);
    arm!(245);
    arm!(246);
    arm!(247);
    arm!(248);
    arm!(249);
    arm!(250);
    arm!(251);
    arm!(252);
    arm!(253);
    arm!(254);
    arm!(255);
    arm!(256);
    Err(ValidationError::InvalidValue {
        path: format!("@{name}/dtype"),
        expected: "H5T_NATIVE_CHARACTER length 1..=256".to_owned(),
        actual: len.to_string(),
    }
    .into())
}

fn read_spex_str_attr(object: &Location, name: &str) -> Result<String, IoError> {
    read_spex_string_attribute(&object.attr(name)?, name)
}

fn read_spex_f64_attr(object: &Location, name: &str, path: &str) -> Result<f64, IoError> {
    let attr = object.attr(name)?;
    crate::mldump::require_numeric_dtype::<f64>(&attr, path)?;
    let values: Vec<f64> = attr.read_raw()?;
    if values.len() != 1 {
        return Err(ValidationError::LengthMismatch {
            path: path.to_owned(),
            expected: 1,
            actual: values.len(),
        }
        .into());
    }
    Ok(values[0])
}

fn read_spex_i64_attr(object: &Location, name: &str, path: &str) -> Result<i64, IoError> {
    let attr = object.attr(name)?;
    crate::mldump::require_numeric_dtype::<i64>(&attr, path)?;
    let values: Vec<i64> = attr.read_raw()?;
    if values.len() != 1 {
        return Err(ValidationError::LengthMismatch {
            path: path.to_owned(),
            expected: 1,
            actual: values.len(),
        }
        .into());
    }
    Ok(values[0])
}

fn write_spex_str_attr(object: &Location, name: &str, value: &str) -> Result<(), IoError> {
    nonempty(format!("@{name}"), value)?;
    write_hwrapper_fixed_attr(object, name, value)
}

macro_rules! write_hwrapper_len {
    ($object:expr, $name:expr, $value:expr; $($n:literal),+) => {
        match $value.len() {
            $(
                $n => {
                    let ascii = hdf5_metno::types::FixedAscii::<$n>::from_ascii($value.as_bytes())
                        .map_err(|err| ValidationError::InvalidValue {
                            path: format!("@{}", $name),
                            expected: "ascii".to_owned(),
                            actual: err.to_string(),
                        })?;
                    $object
                        .new_attr::<hdf5_metno::types::FixedAscii<$n>>()
                        .shape([1usize])
                        .create($name)?
                        .write_raw(&[ascii])?;
                    Ok(())
                }
            )+
            n => Err(ValidationError::InvalidValue {
                path: format!("@{}", $name),
                expected: "Hwrapper native-character length 1..=64".to_owned(),
                actual: n.to_string(),
            }
            .into()),
        }
    };
}

fn write_hwrapper_fixed_attr(object: &Location, name: &str, value: &str) -> Result<(), IoError> {
    write_hwrapper_len!(
        object, name, value;
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
        61, 62, 63, 64
    )
}

fn write_spex_axes(dataset: &Dataset, axes: &[&str]) -> Result<(), IoError> {
    write_spex_str_attr(dataset, "axes", &axes.join(" "))
}

fn require_spex_axes(dataset: &Dataset, expected: &[&str]) -> Result<(), IoError> {
    let attr = dataset.attr("axes")?;
    let descriptor = attr.dtype()?.to_descriptor()?;
    let observed = match descriptor {
        TypeDescriptor::FixedAscii(_) | TypeDescriptor::FixedUnicode(_) => {
            read_spex_string_attribute(&attr, "axes")?
                .split_whitespace()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        }
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
            let shape = attr.shape();
            if shape.is_empty() || shape == [1] {
                read_spex_string_attribute(&attr, "axes")?
                    .split_whitespace()
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            } else {
                let values: Vec<VarLenUnicode> = attr.read_raw()?;
                values
                    .iter()
                    .map(|value| trim_hwrapper_str(value.as_str()))
                    .collect()
            }
        }
        other => {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/@axes/dtype", dataset.name()),
                expected: "H5T_NATIVE_CHARACTER scalar or VL string array".to_owned(),
                actual: other.to_string(),
            }
            .into());
        }
    };
    if observed.iter().map(String::as_str).collect::<Vec<_>>() == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/@axes", dataset.name()),
            expected: expected.join(" "),
            actual: observed.join(" "),
        }
        .into())
    }
}

fn write_spex_f64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[f64],
    axes: &[&str],
) -> Result<(), IoError> {
    require_flat_len(name, shape, data.len())?;
    let dataset = create_dataset::<f64>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_spex_axes(&dataset, axes)
}

fn write_spex_i32_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[i32],
    axes: &[&str],
) -> Result<(), IoError> {
    require_flat_len(name, shape, data.len())?;
    let dataset = create_dataset::<i32>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_spex_axes(&dataset, axes)
}

fn write_spex_i64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[i64],
    axes: &[&str],
) -> Result<(), IoError> {
    require_flat_len(name, shape, data.len())?;
    let dataset = create_dataset::<i64>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_spex_axes(&dataset, axes)
}

fn read_spex_f64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    axes: &[&str],
) -> Result<Vec<f64>, IoError> {
    let dataset = require_numeric_dataset::<f64>(group, name, shape, None)?;
    require_spex_axes(&dataset, axes)?;
    Ok(dataset.read_raw()?)
}

fn read_spex_i32_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    axes: &[&str],
) -> Result<Vec<i32>, IoError> {
    let dataset = require_numeric_dataset::<i32>(group, name, shape, None)?;
    require_spex_axes(&dataset, axes)?;
    Ok(dataset.read_raw()?)
}

fn read_spex_i64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    axes: &[&str],
) -> Result<Vec<i64>, IoError> {
    let dataset = require_numeric_dataset::<i64>(group, name, shape, None)?;
    require_spex_axes(&dataset, axes)?;
    Ok(dataset.read_raw()?)
}

fn require_sha256(path: &str, value: &str) -> Result<(), IoError> {
    parse_sha256_field(path, value).map(|_| ())
}

/// Take a leading 64-hex digest. Trailing Fortran SPACEPAD/NUL/control
/// padding is not part of the digest and is not invented.
fn parse_sha256_field(path: &str, value: &str) -> Result<String, IoError> {
    let cleaned = trim_hwrapper_str(value);
    let bytes = cleaned.as_bytes();
    if bytes.len() >= 64
        && bytes[..64]
            .iter()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Ok(std::str::from_utf8(&bytes[..64])
            .expect("hex prefix is ascii")
            .to_owned());
    }
    Err(ValidationError::InvalidValue {
        path: path.to_owned(),
        expected: "64 lowercase hex chars (optional SPACEPAD/NUL tail)".to_owned(),
        actual: value.to_owned(),
    }
    .into())
}

fn padded(prefix: &str, index: usize) -> String {
    format!("{prefix}_{index:03}")
}

fn parse_index(name: &str, prefix: &str) -> Option<usize> {
    let rest = name.strip_prefix(prefix)?.strip_prefix('_')?;
    if rest.len() != 3 || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

fn collect_index_groups(parent: &Group, prefix: &str) -> Result<Vec<Group>, IoError> {
    let mut indexed = parent
        .groups()?
        .into_iter()
        .filter_map(|group| {
            parse_index(child_basename(&group.name()), prefix).map(|index| (index, group))
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(index, _)| *index);
    for (position, (index, _)) in indexed.iter().enumerate() {
        if *index != position {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/{prefix}_*", parent.name()),
                expected: padded(prefix, position),
                actual: padded(prefix, *index),
            }
            .into());
        }
    }
    Ok(indexed.into_iter().map(|(_, group)| group).collect())
}

fn parse_spin(token: &str) -> Result<RadialBasisSpinV2, IoError> {
    match token {
        "scalar" => Ok(RadialBasisSpinV2::Scalar),
        "up" => Ok(RadialBasisSpinV2::Up),
        "down" => Ok(RadialBasisSpinV2::Down),
        other => Err(ValidationError::InvalidValue {
            path: "spin".to_owned(),
            expected: "scalar|up|down".to_owned(),
            actual: other.to_owned(),
        }
        .into()),
    }
}

fn spin_token(spin: RadialBasisSpinV2) -> &'static str {
    match spin {
        RadialBasisSpinV2::Scalar => "scalar",
        RadialBasisSpinV2::Up => "up",
        RadialBasisSpinV2::Down => "down",
    }
}

fn parse_radial_equation(token: &str) -> Result<RadialEquationTagV1, IoError> {
    match token {
        "schroedinger" => Ok(RadialEquationTagV1::Schroedinger),
        "scalar-koelling-harmon" => Ok(RadialEquationTagV1::ScalarKoellingHarmon),
        "fully-relativistic-dirac" => Ok(RadialEquationTagV1::FullyRelativisticDirac),
        other => Err(ValidationError::InvalidValue {
            path: "radial_equation".to_owned(),
            expected: "schroedinger|scalar-koelling-harmon|fully-relativistic-dirac".to_owned(),
            actual: other.to_owned(),
        }
        .into()),
    }
}

fn radial_token(tag: RadialEquationTagV1) -> &'static str {
    match tag {
        RadialEquationTagV1::Schroedinger => "schroedinger",
        RadialEquationTagV1::ScalarKoellingHarmon => "scalar-koelling-harmon",
        RadialEquationTagV1::FullyRelativisticDirac => "fully-relativistic-dirac",
    }
}

fn hermitian_ingest_scale(left: Complex64V2, right: Complex64V2) -> f64 {
    left.real
        .abs()
        .max(left.imaginary.abs())
        .max(right.real.abs())
        .max(right.imaginary.abs())
        .max(1.0)
}

fn hermitian_discrepancy(left: Complex64V2, right: Complex64V2) -> f64 {
    let delta_re = left.real - right.real;
    let delta_im = left.imaginary + right.imaginary;
    delta_re.hypot(delta_im)
}

fn symmetrize_pair(left: Complex64V2, right: Complex64V2) -> (Complex64V2, Complex64V2) {
    let real = 0.5 * (left.real + right.real);
    let imag = 0.5 * (left.imaginary - right.imaginary);
    (
        Complex64V2 {
            real,
            imaginary: imag,
        },
        Complex64V2 {
            real,
            imaginary: -imag,
        },
    )
}

fn symmetrize_interstitial_fourier(
    field: &mut InterstitialFieldV2,
    path: &str,
) -> Result<(), IoError> {
    let mut by_g = BTreeMap::new();
    for (index, coefficient) in field.coefficients.iter().enumerate() {
        if by_g.insert(coefficient.g, index).is_some() {
            return Err(ValidationError::Duplicate {
                path: path.to_owned(),
                key: format!("{:?}", coefficient.g),
            }
            .into());
        }
    }
    let mut done = BTreeSet::new();
    let n = field.coefficients.len();
    for index in 0..n {
        let g = field.coefficients[index].g;
        if !done.insert(g) {
            continue;
        }
        if g == [0, 0, 0] {
            let value = field.coefficients[index].value;
            let zero = Complex64V2 {
                real: value.real,
                imaginary: 0.0,
            };
            let discrepancy = value.imaginary.abs();
            let tolerance = SPEX_FOURIER_HERMITIAN_TOLERANCE * hermitian_ingest_scale(value, zero);
            if discrepancy > tolerance {
                return Err(ValidationError::SpexFourierHermitian {
                    path: path.to_owned(),
                    g,
                    discrepancy,
                    tolerance,
                }
                .into());
            }
            field.coefficients[index].value = zero;
            continue;
        }
        let opposite = [-g[0], -g[1], -g[2]];
        let Some(&other) = by_g.get(&opposite) else {
            return Err(ValidationError::Missing {
                path: path.to_owned(),
                key: format!("conjugate partner of {g:?}"),
            }
            .into());
        };
        done.insert(opposite);
        let left = field.coefficients[index].value;
        let right = field.coefficients[other].value;
        let discrepancy = hermitian_discrepancy(left, right);
        let tolerance = SPEX_FOURIER_HERMITIAN_TOLERANCE * hermitian_ingest_scale(left, right);
        if discrepancy > tolerance {
            return Err(ValidationError::SpexFourierHermitian {
                path: path.to_owned(),
                g,
                discrepancy,
                tolerance,
            }
            .into());
        }
        let (sym_left, sym_right) = symmetrize_pair(left, right);
        field.coefficients[index].value = sym_left;
        field.coefficients[other].value = sym_right;
    }
    Ok(())
}

fn symmetrize_regional_interstitial(
    field: &mut RegionalFieldV2,
    path: &str,
) -> Result<(), IoError> {
    symmetrize_interstitial_fourier(&mut field.interstitial, path)
}

fn symmetrize_snapshot_interstitial_fourier(snapshot: &mut SnapshotV2) -> Result<(), IoError> {
    match &mut snapshot.initial {
        InitialV2::FrozenPotential { potential } => {
            symmetrize_regional_interstitial(&mut potential.v0, "/initial/potential/v0")?;
            symmetrize_regional_interstitial(&mut potential.bx, "/initial/potential/bx")?;
            symmetrize_regional_interstitial(&mut potential.by, "/initial/potential/by")?;
            symmetrize_regional_interstitial(&mut potential.bz, "/initial/potential/bz")?;
        }
        InitialV2::Restart { density, potential } => {
            symmetrize_regional_interstitial(&mut potential.v0, "/initial/potential/v0")?;
            symmetrize_regional_interstitial(&mut potential.bx, "/initial/potential/bx")?;
            symmetrize_regional_interstitial(&mut potential.by, "/initial/potential/by")?;
            symmetrize_regional_interstitial(&mut potential.bz, "/initial/potential/bz")?;
            symmetrize_regional_interstitial(&mut density.n, "/initial/density/n")?;
            symmetrize_regional_interstitial(&mut density.mx, "/initial/density/mx")?;
            symmetrize_regional_interstitial(&mut density.my, "/initial/density/my")?;
            symmetrize_regional_interstitial(&mut density.mz, "/initial/density/mz")?;
        }
    }
    Ok(())
}

/// Combine SPEX frozen fields with a caller-owned signed-$\kappa$ recipe.
pub fn materialize_snapshot_v2(
    fields: &SpexFrozenFieldsV1,
    recipe: &SpexMaterialBasisRecipeV1,
) -> Result<SpexMaterializedSnapshotV1, IoError> {
    require_sha256("recipe_sha256", &recipe.recipe_sha256)?;
    nonempty("recipe.producer", &recipe.producer)?;
    if recipe.channels.is_empty() {
        return Err(ValidationError::Empty {
            path: "material_basis_recipe.channels".to_owned(),
        }
        .into());
    }
    for channel in &recipe.channels {
        nonempty("material_basis_recipe.site_id", &channel.site_id)?;
        let site_ok = fields
            .snapshot
            .geometry
            .sites
            .iter()
            .any(|site| site.id == channel.site_id);
        if !site_ok {
            return Err(ValidationError::InvalidValue {
                path: "material_basis_recipe.site_id".to_owned(),
                expected: "a SPEX geometry site id".to_owned(),
                actual: channel.site_id.clone(),
            }
            .into());
        }
        match channel.kind {
            SpexMaterialChannelKind::Lo | SpexMaterialChannelKind::Rlo => {
                if channel.derivative_order != 0 {
                    return Err(ValidationError::InvalidValue {
                        path: "material_basis_recipe.derivative_order".to_owned(),
                        expected: "0 for lo/rlo".to_owned(),
                        actual: channel.derivative_order.to_string(),
                    }
                    .into());
                }
                let matched = fields.scalar_los.iter().any(|table| {
                    table.site_id == channel.site_id
                        && table.orbitals.iter().any(|lo| {
                            lo.l == channel.l && crate::mldump::approx_eq(lo.energy, channel.energy)
                        })
                });
                if !matched {
                    return Err(ValidationError::InvalidValue {
                        path: "material_basis_recipe.lo".to_owned(),
                        expected: "SPEX scalar LO with same site, l, energy".to_owned(),
                        actual: format!("{} l={} E={}", channel.site_id, channel.l, channel.energy),
                    }
                    .into());
                }
            }
            SpexMaterialChannelKind::Hdlo => {
                if channel.derivative_order != 2 {
                    return Err(ValidationError::InvalidValue {
                        path: "material_basis_recipe.derivative_order".to_owned(),
                        expected: "2 for hdlo".to_owned(),
                        actual: channel.derivative_order.to_string(),
                    }
                    .into());
                }
                let matched = fields.snapshot.geometry.radial_basis.iter().any(|basis| {
                    basis.site_id == channel.site_id
                        && basis
                            .linearization
                            .linearization_energies
                            .iter()
                            .any(|parameter| parameter.l == channel.l)
                });
                if !matched {
                    return Err(ValidationError::InvalidValue {
                        path: "material_basis_recipe.hdlo".to_owned(),
                        expected: "SPEX APW linearization l for HDLO".to_owned(),
                        actual: format!("{} l={}", channel.site_id, channel.l),
                    }
                    .into());
                }
            }
        }
    }
    let mut snapshot = fields.snapshot.clone();
    symmetrize_snapshot_interstitial_fourier(&mut snapshot)?;
    snapshot.meta.annotations.insert(
        "spex.interstitial_phase".to_owned(),
        fields.interstitial_phase.clone(),
    );
    snapshot
        .meta
        .annotations
        .insert("spex.spin_layout".to_owned(), fields.spin_layout.clone());
    snapshot.meta.annotations.insert(
        "material_basis.recipe_sha256".to_owned(),
        recipe.recipe_sha256.clone(),
    );
    snapshot.meta.annotations.insert(
        "material_basis.producer".to_owned(),
        recipe.producer.clone(),
    );
    snapshot.validate()?;
    Ok(SpexMaterializedSnapshotV1 {
        snapshot,
        spex_hashes: fields.hashes.clone(),
        recipe_sha256: recipe.recipe_sha256.clone(),
    })
}
