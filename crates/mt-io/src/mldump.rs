//! Versioned MLDUMP v1 HDF5 interchange.
//!
//! This is a libmuffintin-owned, inspectable HDF5 schema. It is not
//! CoQui-native or SPEX-native. Runtime, mixed-product, THC, and Coulomb
//! types stay out of this crate; later stages materialize those objects
//! into this stable boundary.

use std::collections::BTreeSet;
use std::path::Path;
use std::str::FromStr;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Container, Dataset, File, Group, H5Type, Location};

use crate::error::{IoError, ValidationError, finite, nonempty, positive};

/// Stable schema name written on every MLDUMP file.
pub const MLDUMP_SCHEMA_NAME: &str = "libmuffintin.mldump";
/// MLDUMP schema version implemented by this crate.
pub const MLDUMP_SCHEMA_VERSION: u32 = 1;

/// HDF5 `status` value for a group that carries its v1 payload.
pub const MLDUMP_STATUS_PRESENT: &str = "present";
/// HDF5 `status` value for a reserved group with no child payload.
pub const MLDUMP_STATUS_ABSENT_NOT_COMPUTED: &str = "absent_not_computed";

/// Length unit recorded on `/units`.
pub const MLDUMP_UNIT_LENGTH: &str = "Bohr";
/// Reciprocal-length unit recorded on `/units`.
pub const MLDUMP_UNIT_INVERSE_LENGTH: &str = "Bohr^-1";
/// Volume unit recorded on `/units`.
pub const MLDUMP_UNIT_VOLUME: &str = "Bohr^3";
/// Energy unit recorded on `/units`.
pub const MLDUMP_UNIT_ENERGY: &str = "Hartree";
/// $k$/$q$ coordinate convention recorded on `/units`.
pub const MLDUMP_UNIT_K_Q: &str = "fractional_reciprocal";
/// $G$/Umklapp convention recorded on `/units`.
pub const MLDUMP_UNIT_G_UMKLAPP: &str = "integer_reciprocal_lattice";

/// Scale-aware fractional-coordinate tolerance for $q_{\mathrm{in}}$ and $k-q$ identities.
///
/// Two values compare equal when their absolute difference is at most this
/// constant times $\max(|a|,|b|,1)$. The $10^{-12}$ floor matches the M-L1/M-L5b
/// mesh-coordinate gate; the scale factor is the same form used by parent-grid
/// radial matching.
const FRACTIONAL_EQ_TOLERANCE: f64 = 1.0e-12;

const ATTR_SCHEMA_NAME: &str = "schema_name";
const ATTR_SCHEMA_VERSION: &str = "schema_version";
const ATTR_STATUS: &str = "status";
const ATTR_AXES: &str = "axes";

const GROUP_META: &str = "meta";
const GROUP_UNITS: &str = "units";
const GROUP_GEOMETRY: &str = "geometry";
const GROUP_MESH: &str = "mesh";
const GROUP_ORBITALS: &str = "orbitals";
const GROUP_PRODUCTS: &str = "products";
const GROUP_MPB: &str = "mpb";
const GROUP_THC: &str = "thc";
const GROUP_COULOMB: &str = "coulomb";
const GROUP_EXCHANGE: &str = "exchange";
const GROUP_EXCHANGE_VALENCE: &str = "valence";
const GROUP_EXCHANGE_CORE: &str = "core";
const GROUP_EXCHANGE_TOTAL: &str = "total";

const TOP_LEVEL_GROUPS: [&str; 10] = [
    GROUP_META,
    GROUP_UNITS,
    GROUP_GEOMETRY,
    GROUP_MESH,
    GROUP_ORBITALS,
    GROUP_PRODUCTS,
    GROUP_MPB,
    GROUP_THC,
    GROUP_COULOMB,
    GROUP_EXCHANGE,
];

const ABSENT_PAYLOAD_GROUPS: [&str; 5] = [
    GROUP_ORBITALS,
    GROUP_PRODUCTS,
    GROUP_MPB,
    GROUP_THC,
    GROUP_COULOMB,
];

/// Payload/group status stored as an HDF5 group attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MldumpStatus {
    /// Reserved group with no child payload.
    AbsentNotComputed,
    /// Group carries its documented v1 payload.
    Present,
}

impl MldumpStatus {
    /// Stable HDF5 attribute value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Present => MLDUMP_STATUS_PRESENT,
            Self::AbsentNotComputed => MLDUMP_STATUS_ABSENT_NOT_COMPUTED,
        }
    }

    fn parse(path: &str, value: &str) -> Result<Self, ValidationError> {
        match value {
            MLDUMP_STATUS_PRESENT => Ok(Self::Present),
            MLDUMP_STATUS_ABSENT_NOT_COMPUTED => Ok(Self::AbsentNotComputed),
            other => Err(ValidationError::InvalidValue {
                path: path.to_owned(),
                expected: format!("{MLDUMP_STATUS_PRESENT} or {MLDUMP_STATUS_ABSENT_NOT_COMPUTED}"),
                actual: other.to_owned(),
            }),
        }
    }
}

/// Producer identity and declared numeric/complex/index conventions.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpMetaV1 {
    /// Producer program name.
    pub producer_name: String,
    /// Producer program version.
    pub producer_version: String,
    /// Source revision recorded by the producer (for example a git SHA).
    pub source_revision: String,
    /// Feature/representation tag. This is not a runtime enum.
    pub feature_representation: String,
}

/// One muffin-tin site and the radial mesh later stages need to bind.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpSiteV1 {
    /// Species token if the producer supplied one.
    pub species: Option<String>,
    /// Site label if the producer supplied one.
    pub label: Option<String>,
    /// Cartesian muffin-tin centre in Bohr.
    pub position_bohr: [f64; 3],
    /// Muffin-tin radius in Bohr.
    pub radius_bohr: f64,
    /// Exponential radial mesh bound to this site.
    pub radial_mesh: MldumpRadialMeshV1,
}

/// Exponential radial mesh identity: first radius, log increment, count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpRadialMeshV1 {
    /// First positive mesh radius in Bohr.
    pub first_bohr: f64,
    /// Logarithmic increment $h$ in $r_i=r_0 e^{i h}$.
    pub log_increment: f64,
    /// Number of stored radial samples.
    pub point_count: usize,
}

/// Direct/reciprocal lattices, cell volume, and ordered sites.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpGeometryV1 {
    /// Direct primitive vectors, Bohr, row $i$ is $\mathbf a_i$.
    pub direct_basis_bohr: [[f64; 3]; 3],
    /// Reciprocal primitive vectors including $2\pi$, inverse Bohr, row $i$ is $\mathbf b_i$.
    pub reciprocal_basis_inv_bohr: [[f64; 3]; 3],
    /// Cell volume in Bohr cubed.
    pub cell_volume_bohr3: f64,
    /// Ordered muffin-tin sites.
    pub sites: Vec<MldumpSiteV1>,
}

/// One full-BZ $k$ point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MldumpKPointV1 {
    /// Fractional reciprocal coordinates.
    pub fractional: [f64; 3],
    /// Brillouin-zone weight supplied by the producer.
    pub weight: f64,
}

/// Per-$k$ map of $k-q$ onto the stored $k$ mesh.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MldumpKMinusQV1 {
    /// Index of the right-hand $k$ point.
    pub k_index: usize,
    /// Index of the mapped $k-q$ point on the same mesh.
    pub mapped_index: usize,
    /// Per-$k$ integer wrap $G_{\mathrm{wrap}}$.
    pub g_wrap: [i32; 3],
}

/// One transfer $q$ and its per-$k$ wraps.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpQEntryV1 {
    /// Requested input $q$ in fractional reciprocal coordinates.
    pub input_fractional: [f64; 3],
    /// Canonical $q$ in $[0,1)^3$ fractional reciprocal coordinates.
    pub canonical_fractional: [f64; 3],
    /// Global transfer Umklapp $G_{\mathrm{transfer}}$.
    pub global_umklapp: [i32; 3],
    /// Ordered $k-q$ records, one per $k$ point.
    pub k_minus_q: Vec<MldumpKMinusQV1>,
}

/// Ordered full-BZ $k$ mesh and $q$ transfers.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpMeshV1 {
    pub k_points: Vec<MldumpKPointV1>,
    pub q_entries: Vec<MldumpQEntryV1>,
}

/// Reserved payload-group statuses. M-L6a writes every field as absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MldumpStatusesV1 {
    pub orbitals: MldumpStatus,
    pub products: MldumpStatus,
    pub mpb: MldumpStatus,
    pub thc: MldumpStatus,
    pub coulomb: MldumpStatus,
    pub exchange_valence: MldumpStatus,
    pub exchange_core: MldumpStatus,
    pub exchange_total: MldumpStatus,
}

impl MldumpStatusesV1 {
    /// Every reserved payload seam is `absent_not_computed`.
    pub const fn absent_not_computed() -> Self {
        Self {
            orbitals: MldumpStatus::AbsentNotComputed,
            products: MldumpStatus::AbsentNotComputed,
            mpb: MldumpStatus::AbsentNotComputed,
            thc: MldumpStatus::AbsentNotComputed,
            coulomb: MldumpStatus::AbsentNotComputed,
            exchange_valence: MldumpStatus::AbsentNotComputed,
            exchange_core: MldumpStatus::AbsentNotComputed,
            exchange_total: MldumpStatus::AbsentNotComputed,
        }
    }
}

/// MLDUMP v1 header, geometry, mesh, and reserved-group statuses.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpV1 {
    pub meta: MldumpMetaV1,
    pub geometry: MldumpGeometryV1,
    pub mesh: MldumpMeshV1,
    pub statuses: MldumpStatusesV1,
}

impl MldumpV1 {
    /// Construct a v1 dump whose reserved payload groups are absent.
    pub fn new(meta: MldumpMetaV1, geometry: MldumpGeometryV1, mesh: MldumpMeshV1) -> Self {
        Self {
            meta,
            geometry,
            mesh,
            statuses: MldumpStatusesV1::absent_not_computed(),
        }
    }

    /// Trust-boundary checks needed to interpret the in-memory record.
    pub fn validate(&self) -> Result<(), IoError> {
        nonempty("meta.producer_name", &self.meta.producer_name)?;
        nonempty("meta.producer_version", &self.meta.producer_version)?;
        nonempty("meta.source_revision", &self.meta.source_revision)?;
        nonempty(
            "meta.feature_representation",
            &self.meta.feature_representation,
        )?;
        require_no_nul("meta.producer_name", &self.meta.producer_name)?;
        require_no_nul("meta.producer_version", &self.meta.producer_version)?;
        require_no_nul("meta.source_revision", &self.meta.source_revision)?;
        require_no_nul(
            "meta.feature_representation",
            &self.meta.feature_representation,
        )?;

        validate_matrix3(
            "geometry.direct_basis_bohr",
            self.geometry.direct_basis_bohr,
        )?;
        validate_matrix3(
            "geometry.reciprocal_basis_inv_bohr",
            self.geometry.reciprocal_basis_inv_bohr,
        )?;
        positive(
            "geometry.cell_volume_bohr3",
            self.geometry.cell_volume_bohr3,
        )?;
        if self.geometry.sites.is_empty() {
            return Err(ValidationError::Empty {
                path: "geometry.sites".to_owned(),
            }
            .into());
        }
        for (site, record) in self.geometry.sites.iter().enumerate() {
            if let Some(species) = &record.species {
                nonempty(format!("geometry.sites[{site}].species"), species)?;
                require_no_nul(format!("geometry.sites[{site}].species"), species)?;
            }
            if let Some(label) = &record.label {
                nonempty(format!("geometry.sites[{site}].label"), label)?;
                require_no_nul(format!("geometry.sites[{site}].label"), label)?;
            }
            for (axis, value) in record.position_bohr.iter().enumerate() {
                finite(
                    format!("geometry.sites[{site}].position_bohr[{axis}]"),
                    *value,
                )?;
            }
            positive(
                format!("geometry.sites[{site}].radius_bohr"),
                record.radius_bohr,
            )?;
            positive(
                format!("geometry.sites[{site}].radial_mesh.first_bohr"),
                record.radial_mesh.first_bohr,
            )?;
            positive(
                format!("geometry.sites[{site}].radial_mesh.log_increment"),
                record.radial_mesh.log_increment,
            )?;
            if record.radial_mesh.point_count == 0 {
                return Err(ValidationError::NotPositive {
                    path: format!("geometry.sites[{site}].radial_mesh.point_count"),
                    value: 0.0,
                }
                .into());
            }
        }

        if self.mesh.k_points.is_empty() {
            return Err(ValidationError::Empty {
                path: "mesh.k_points".to_owned(),
            }
            .into());
        }
        if self.mesh.q_entries.is_empty() {
            return Err(ValidationError::Empty {
                path: "mesh.q_entries".to_owned(),
            }
            .into());
        }
        let n_k = self.mesh.k_points.len();
        for (k, point) in self.mesh.k_points.iter().enumerate() {
            for (axis, value) in point.fractional.iter().enumerate() {
                finite(format!("mesh.k_points[{k}].fractional[{axis}]"), *value)?;
            }
            finite(format!("mesh.k_points[{k}].weight"), point.weight)?;
            if point.weight < 0.0 {
                return Err(ValidationError::InvalidValue {
                    path: format!("mesh.k_points[{k}].weight"),
                    expected: "nonnegative full-BZ weight".to_owned(),
                    actual: point.weight.to_string(),
                }
                .into());
            }
        }
        for (q, entry) in self.mesh.q_entries.iter().enumerate() {
            for (axis, value) in entry.input_fractional.iter().enumerate() {
                finite(
                    format!("mesh.q_entries[{q}].input_fractional[{axis}]"),
                    *value,
                )?;
            }
            for (axis, value) in entry.canonical_fractional.iter().enumerate() {
                finite(
                    format!("mesh.q_entries[{q}].canonical_fractional[{axis}]"),
                    *value,
                )?;
                if !(0.0..1.0).contains(value) {
                    return Err(ValidationError::InvalidValue {
                        path: format!("mesh.q_entries[{q}].canonical_fractional[{axis}]"),
                        expected: "[0, 1)".to_owned(),
                        actual: value.to_string(),
                    }
                    .into());
                }
            }
            for axis in 0..3 {
                let expected =
                    entry.canonical_fractional[axis] + f64::from(entry.global_umklapp[axis]);
                if !approx_eq(entry.input_fractional[axis], expected) {
                    return Err(ValidationError::InvalidValue {
                        path: format!("mesh.q_entries[{q}].input_fractional[{axis}]"),
                        expected: format!("q_canonical + global_umklapp = {expected}"),
                        actual: entry.input_fractional[axis].to_string(),
                    }
                    .into());
                }
            }
            if entry.k_minus_q.len() != n_k {
                return Err(ValidationError::LengthMismatch {
                    path: format!("mesh.q_entries[{q}].k_minus_q"),
                    expected: n_k,
                    actual: entry.k_minus_q.len(),
                }
                .into());
            }
            for (ik, mapped) in entry.k_minus_q.iter().enumerate() {
                if mapped.k_index != ik {
                    return Err(ValidationError::InvalidValue {
                        path: format!("mesh.q_entries[{q}].k_minus_q[{ik}].k_index"),
                        expected: ik.to_string(),
                        actual: mapped.k_index.to_string(),
                    }
                    .into());
                }
                if mapped.mapped_index >= n_k {
                    return Err(ValidationError::InvalidValue {
                        path: format!("mesh.q_entries[{q}].k_minus_q[{ik}].mapped_index"),
                        expected: format!("index < {n_k}"),
                        actual: mapped.mapped_index.to_string(),
                    }
                    .into());
                }
                let k_frac = self.mesh.k_points[ik].fractional;
                let mapped_frac = self.mesh.k_points[mapped.mapped_index].fractional;
                for axis in 0..3 {
                    let left = k_frac[axis] - entry.canonical_fractional[axis];
                    let right = mapped_frac[axis] + f64::from(mapped.g_wrap[axis]);
                    if !approx_eq(left, right) {
                        return Err(ValidationError::InvalidValue {
                            path: format!("mesh.q_entries[{q}].k_minus_q[{ik}].g_wrap[{axis}]"),
                            expected: format!(
                                "k - q_canonical = mapped + G_wrap ({left} == {right})"
                            ),
                            actual: mapped.g_wrap[axis].to_string(),
                        }
                        .into());
                    }
                }
            }
        }
        require_absent_payload("statuses.orbitals", self.statuses.orbitals)?;
        require_absent_payload("statuses.products", self.statuses.products)?;
        require_absent_payload("statuses.mpb", self.statuses.mpb)?;
        require_absent_payload("statuses.thc", self.statuses.thc)?;
        require_absent_payload("statuses.coulomb", self.statuses.coulomb)?;
        require_absent_payload("statuses.exchange_valence", self.statuses.exchange_valence)?;
        require_absent_payload("statuses.exchange_core", self.statuses.exchange_core)?;
        require_absent_payload("statuses.exchange_total", self.statuses.exchange_total)?;
        Ok(())
    }
}

/// Write an inspectable MLDUMP v1 HDF5 file.
pub fn write_mldump_v1(path: impl AsRef<Path>, dump: &MldumpV1) -> Result<(), IoError> {
    dump.validate()?;
    let file = File::create(path)?;
    write_str_attr(&file, ATTR_SCHEMA_NAME, MLDUMP_SCHEMA_NAME)?;
    file.new_attr::<u32>()
        .create(ATTR_SCHEMA_VERSION)?
        .write_scalar(&MLDUMP_SCHEMA_VERSION)?;

    write_meta_group(&file, &dump.meta)?;
    write_units_group(&file)?;
    write_geometry_group(&file, &dump.geometry)?;
    write_mesh_group(&file, &dump.mesh)?;
    for name in ABSENT_PAYLOAD_GROUPS {
        write_absent_group(&file, name)?;
    }
    write_exchange_group(&file, dump.statuses)?;
    Ok(())
}

/// Read an MLDUMP v1 HDF5 file into the typed v1 record.
pub fn read_mldump_v1(path: impl AsRef<Path>) -> Result<MldumpV1, IoError> {
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
    if schema_version != MLDUMP_SCHEMA_VERSION {
        return Err(IoError::UnsupportedVersion {
            format: MLDUMP_SCHEMA_NAME,
            supported: MLDUMP_SCHEMA_VERSION,
            found: schema_version,
        });
    }
    require_top_level_groups(&file)?;

    let meta = read_meta_group(&file)?;
    read_units_group(&file)?;
    let geometry = read_geometry_group(&file)?;
    let mesh = read_mesh_group(&file)?;
    let mut statuses = MldumpStatusesV1::absent_not_computed();
    statuses.orbitals = read_absent_group(&file, GROUP_ORBITALS)?;
    statuses.products = read_absent_group(&file, GROUP_PRODUCTS)?;
    statuses.mpb = read_absent_group(&file, GROUP_MPB)?;
    statuses.thc = read_absent_group(&file, GROUP_THC)?;
    statuses.coulomb = read_absent_group(&file, GROUP_COULOMB)?;
    let (valence, core, total) = read_exchange_group(&file)?;
    statuses.exchange_valence = valence;
    statuses.exchange_core = core;
    statuses.exchange_total = total;

    let dump = MldumpV1 {
        meta,
        geometry,
        mesh,
        statuses,
    };
    dump.validate()?;
    Ok(dump)
}

fn require_no_nul(path: impl Into<String>, value: &str) -> Result<(), ValidationError> {
    if value.contains('\0') {
        Err(ValidationError::InvalidValue {
            path: path.into(),
            expected: "UTF-8 without interior NUL".to_owned(),
            actual: "contains NUL".to_owned(),
        })
    } else {
        Ok(())
    }
}

fn require_absent_payload(path: &str, status: MldumpStatus) -> Result<(), ValidationError> {
    if status == MldumpStatus::AbsentNotComputed {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: MLDUMP_STATUS_ABSENT_NOT_COMPUTED.to_owned(),
            actual: status.as_str().to_owned(),
        })
    }
}

fn approx_eq(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= FRACTIONAL_EQ_TOLERANCE * scale
}

fn validate_matrix3(path: &str, matrix: [[f64; 3]; 3]) -> Result<(), ValidationError> {
    for (row, vector) in matrix.iter().enumerate() {
        for (axis, value) in vector.iter().enumerate() {
            finite(format!("{path}[{row}][{axis}]"), *value)?;
        }
    }
    Ok(())
}

fn vlen(path: &str, value: &str) -> Result<VarLenUnicode, IoError> {
    require_no_nul(path, value)?;
    VarLenUnicode::from_str(value).map_err(|err| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "UTF-8 without interior NUL".to_owned(),
            actual: err.to_string(),
        }
        .into()
    })
}

fn write_str_attr(object: &Location, name: &str, value: &str) -> Result<(), IoError> {
    object
        .new_attr::<VarLenUnicode>()
        .create(name)?
        .write_scalar(&vlen(name, value)?)?;
    Ok(())
}

fn read_str_attr(object: &Location, name: &str) -> Result<String, IoError> {
    let value: VarLenUnicode = object.attr(name)?.read_scalar()?;
    Ok(value.as_str().to_owned())
}

fn write_status(group: &Group, status: MldumpStatus) -> Result<(), IoError> {
    write_str_attr(group, ATTR_STATUS, status.as_str())
}

fn read_status(group: &Group) -> Result<MldumpStatus, IoError> {
    let path = format!("{}/@{ATTR_STATUS}", group.name());
    Ok(MldumpStatus::parse(
        &path,
        &read_str_attr(group, ATTR_STATUS)?,
    )?)
}

fn write_axes(dataset: &Dataset, axes: &[&str]) -> Result<(), IoError> {
    let values = axes
        .iter()
        .map(|axis| vlen(ATTR_AXES, axis))
        .collect::<Result<Vec<_>, _>>()?;
    dataset
        .new_attr::<VarLenUnicode>()
        .shape([axes.len()])
        .create(ATTR_AXES)?
        .write_raw(values.as_slice())?;
    Ok(())
}

fn read_axes(dataset: &Dataset) -> Result<Vec<String>, IoError> {
    let values = dataset.attr(ATTR_AXES)?.read_raw::<VarLenUnicode>()?;
    Ok(values
        .iter()
        .map(|value| value.as_str().to_owned())
        .collect())
}

fn require_axes(dataset: &Dataset, expected: &[&str]) -> Result<(), IoError> {
    let observed = read_axes(dataset)?;
    if observed.iter().map(String::as_str).collect::<Vec<_>>() == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/@{ATTR_AXES}", dataset.name()),
            expected: expected.join(" "),
            actual: observed.join(" "),
        }
        .into())
    }
}

fn write_f64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[f64],
    axes: &[&str],
) -> Result<(), IoError> {
    let dataset = create_dataset::<f64>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_axes(&dataset, axes)
}

fn write_i32_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[i32],
    axes: &[&str],
) -> Result<(), IoError> {
    let dataset = create_dataset::<i32>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_axes(&dataset, axes)
}

fn write_i64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[i64],
    axes: &[&str],
) -> Result<(), IoError> {
    let dataset = create_dataset::<i64>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_axes(&dataset, axes)
}

fn write_str_dataset(
    group: &Group,
    name: &str,
    values: &[String],
    axes: &[&str],
) -> Result<(), IoError> {
    let encoded = values
        .iter()
        .map(|value| vlen(name, value))
        .collect::<Result<Vec<_>, _>>()?;
    let dataset = create_dataset::<VarLenUnicode>(group, name, &[values.len()])?;
    dataset.write_raw(encoded.as_slice())?;
    write_axes(&dataset, axes)
}

fn create_dataset<T: H5Type>(
    group: &Group,
    name: &str,
    shape: &[usize],
) -> Result<Dataset, IoError> {
    let builder = group.new_dataset::<T>();
    let dataset = match *shape {
        [] => builder.create(name)?,
        [n] => builder.shape([n]).create(name)?,
        [n, m] => builder.shape([n, m]).create(name)?,
        [n, m, p] => builder.shape([n, m, p]).create(name)?,
        _ => {
            return Err(ValidationError::InvalidValue {
                path: name.to_owned(),
                expected: "rank 0..=3".to_owned(),
                actual: shape.len().to_string(),
            }
            .into());
        }
    };
    Ok(dataset)
}

fn require_numeric_dtype<T: H5Type>(object: &Container, path: &str) -> Result<(), IoError> {
    let dtype = object.dtype()?;
    if dtype.is::<T>() {
        Ok(())
    } else {
        let actual = dtype
            .to_descriptor()
            .map(|descriptor| descriptor.to_string())
            .unwrap_or_else(|_| "unreadable HDF5 datatype".to_owned());
        Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: std::any::type_name::<T>().to_owned(),
            actual,
        }
        .into())
    }
}

fn read_numeric_attr<T: H5Type>(object: &Location, name: &str, path: &str) -> Result<T, IoError> {
    let attr = object.attr(name)?;
    require_numeric_dtype::<T>(&attr, path)?;
    Ok(attr.read_scalar()?)
}

fn require_numeric_dataset<T: H5Type>(
    group: &Group,
    name: &str,
    expected_shape: &[usize],
    axes: Option<&[&str]>,
) -> Result<Dataset, IoError> {
    let dataset = group.dataset(name)?;
    require_numeric_dtype::<T>(&dataset, &format!("{}/dtype", dataset.name()))?;
    require_shape(&dataset, expected_shape)?;
    if let Some(axes) = axes {
        require_axes(&dataset, axes)?;
    }
    Ok(dataset)
}

fn read_f64_dataset(
    group: &Group,
    name: &str,
    expected_shape: &[usize],
    axes: &[&str],
) -> Result<Vec<f64>, IoError> {
    Ok(require_numeric_dataset::<f64>(group, name, expected_shape, Some(axes))?.read_raw()?)
}

fn read_i32_dataset(
    group: &Group,
    name: &str,
    expected_shape: &[usize],
    axes: &[&str],
) -> Result<Vec<i32>, IoError> {
    Ok(require_numeric_dataset::<i32>(group, name, expected_shape, Some(axes))?.read_raw()?)
}

fn read_i64_dataset(
    group: &Group,
    name: &str,
    expected_shape: &[usize],
    axes: &[&str],
) -> Result<Vec<i64>, IoError> {
    Ok(require_numeric_dataset::<i64>(group, name, expected_shape, Some(axes))?.read_raw()?)
}

fn read_str_dataset(
    group: &Group,
    name: &str,
    expected_len: usize,
    axes: &[&str],
) -> Result<Vec<String>, IoError> {
    let dataset = group.dataset(name)?;
    require_shape(&dataset, &[expected_len])?;
    require_axes(&dataset, axes)?;
    let values = dataset.read_raw::<VarLenUnicode>()?;
    Ok(values
        .iter()
        .map(|value| value.as_str().to_owned())
        .collect())
}

fn require_shape(dataset: &Dataset, expected: &[usize]) -> Result<(), IoError> {
    let observed = dataset.shape();
    if observed.as_slice() == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/shape", dataset.name()),
            expected: format!("{expected:?}"),
            actual: format!("{observed:?}"),
        }
        .into())
    }
}

fn optional_text(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_owned())
    }
}

fn flatten_matrix3(matrix: [[f64; 3]; 3]) -> [f64; 9] {
    let mut flat = [0.0; 9];
    for (row, vector) in matrix.iter().enumerate() {
        flat[row * 3..row * 3 + 3].copy_from_slice(vector);
    }
    flat
}

fn unflatten_matrix3(values: &[f64]) -> [[f64; 3]; 3] {
    let mut matrix = [[0.0; 3]; 3];
    for row in 0..3 {
        matrix[row].copy_from_slice(&values[row * 3..row * 3 + 3]);
    }
    matrix
}

fn write_meta_group(file: &File, meta: &MldumpMetaV1) -> Result<(), IoError> {
    let group = file.create_group(GROUP_META)?;
    write_status(&group, MldumpStatus::Present)?;
    write_str_attr(&group, "producer_name", &meta.producer_name)?;
    write_str_attr(&group, "producer_version", &meta.producer_version)?;
    write_str_attr(&group, "source_revision", &meta.source_revision)?;
    write_str_attr(
        &group,
        "feature_representation",
        &meta.feature_representation,
    )?;
    write_str_attr(&group, "index_convention", "zero-based")?;
    write_str_attr(&group, "numeric_dtype", "ieee754_f64")?;
    write_str_attr(&group, "complex_encoding", "final_re_im_axis")?;
    group
        .new_attr::<i64>()
        .create("index_origin")?
        .write_scalar(&0_i64)?;
    let complex_axis = ["re", "im"]
        .iter()
        .map(|label| vlen("complex_axis", label))
        .collect::<Result<Vec<_>, _>>()?;
    group
        .new_attr::<VarLenUnicode>()
        .shape([2])
        .create("complex_axis")?
        .write_raw(complex_axis.as_slice())?;
    Ok(())
}

fn read_meta_group(file: &File) -> Result<MldumpMetaV1, IoError> {
    let group = file.group(GROUP_META)?;
    require_status_present(&group)?;
    require_no_datasets(&group)?;
    let index_origin: i64 =
        read_numeric_attr::<i64>(&group, "index_origin", "/meta/@index_origin/dtype")?;
    if index_origin != 0 {
        return Err(ValidationError::InvalidValue {
            path: "/meta/@index_origin".to_owned(),
            expected: "0".to_owned(),
            actual: index_origin.to_string(),
        }
        .into());
    }
    require_str_attr(&group, "index_convention", "zero-based")?;
    require_str_attr(&group, "numeric_dtype", "ieee754_f64")?;
    require_str_attr(&group, "complex_encoding", "final_re_im_axis")?;
    let complex_axis = group.attr("complex_axis")?.read_raw::<VarLenUnicode>()?;
    let labels = complex_axis
        .iter()
        .map(|value| value.as_str())
        .collect::<Vec<_>>();
    if labels != ["re", "im"] {
        return Err(ValidationError::InvalidValue {
            path: "/meta/@complex_axis".to_owned(),
            expected: "re im".to_owned(),
            actual: labels.join(" "),
        }
        .into());
    }
    Ok(MldumpMetaV1 {
        producer_name: read_str_attr(&group, "producer_name")?,
        producer_version: read_str_attr(&group, "producer_version")?,
        source_revision: read_str_attr(&group, "source_revision")?,
        feature_representation: read_str_attr(&group, "feature_representation")?,
    })
}

fn write_units_group(file: &File) -> Result<(), IoError> {
    let group = file.create_group(GROUP_UNITS)?;
    write_status(&group, MldumpStatus::Present)?;
    write_str_attr(&group, "length", MLDUMP_UNIT_LENGTH)?;
    write_str_attr(&group, "inverse_length", MLDUMP_UNIT_INVERSE_LENGTH)?;
    write_str_attr(&group, "volume", MLDUMP_UNIT_VOLUME)?;
    write_str_attr(&group, "energy", MLDUMP_UNIT_ENERGY)?;
    write_str_attr(&group, "k_q_coordinates", MLDUMP_UNIT_K_Q)?;
    write_str_attr(&group, "g_umklapp", MLDUMP_UNIT_G_UMKLAPP)?;
    Ok(())
}

fn read_units_group(file: &File) -> Result<(), IoError> {
    let group = file.group(GROUP_UNITS)?;
    require_status_present(&group)?;
    require_no_datasets(&group)?;
    require_str_attr(&group, "length", MLDUMP_UNIT_LENGTH)?;
    require_str_attr(&group, "inverse_length", MLDUMP_UNIT_INVERSE_LENGTH)?;
    require_str_attr(&group, "volume", MLDUMP_UNIT_VOLUME)?;
    require_str_attr(&group, "energy", MLDUMP_UNIT_ENERGY)?;
    require_str_attr(&group, "k_q_coordinates", MLDUMP_UNIT_K_Q)?;
    require_str_attr(&group, "g_umklapp", MLDUMP_UNIT_G_UMKLAPP)?;
    Ok(())
}

fn write_geometry_group(file: &File, geometry: &MldumpGeometryV1) -> Result<(), IoError> {
    let group = file.create_group(GROUP_GEOMETRY)?;
    write_status(&group, MldumpStatus::Present)?;
    write_f64_dataset(
        &group,
        "direct_basis",
        &[3, 3],
        &flatten_matrix3(geometry.direct_basis_bohr),
        &["primitive_vector", "cartesian"],
    )?;
    write_f64_dataset(
        &group,
        "reciprocal_basis",
        &[3, 3],
        &flatten_matrix3(geometry.reciprocal_basis_inv_bohr),
        &["primitive_vector", "cartesian"],
    )?;
    group
        .new_dataset::<f64>()
        .create("cell_volume")?
        .write_scalar(&geometry.cell_volume_bohr3)?;
    let n_sites = geometry.sites.len();
    let mut species = Vec::with_capacity(n_sites);
    let mut labels = Vec::with_capacity(n_sites);
    let mut positions = Vec::with_capacity(n_sites * 3);
    let mut radii = Vec::with_capacity(n_sites);
    let mut first = Vec::with_capacity(n_sites);
    let mut increment = Vec::with_capacity(n_sites);
    let mut point_count = Vec::with_capacity(n_sites);
    for site in &geometry.sites {
        species.push(site.species.clone().unwrap_or_default());
        labels.push(site.label.clone().unwrap_or_default());
        positions.extend_from_slice(&site.position_bohr);
        radii.push(site.radius_bohr);
        first.push(site.radial_mesh.first_bohr);
        increment.push(site.radial_mesh.log_increment);
        point_count.push(i64::try_from(site.radial_mesh.point_count).map_err(|_| {
            ValidationError::InvalidValue {
                path: "geometry.radial_mesh_point_count".to_owned(),
                expected: "i64".to_owned(),
                actual: site.radial_mesh.point_count.to_string(),
            }
        })?);
    }
    write_str_dataset(&group, "site_species", &species, &["site"])?;
    write_str_dataset(&group, "site_labels", &labels, &["site"])?;
    write_f64_dataset(
        &group,
        "site_positions",
        &[n_sites, 3],
        &positions,
        &["site", "cartesian"],
    )?;
    write_f64_dataset(&group, "site_radii", &[n_sites], &radii, &["site"])?;
    write_f64_dataset(&group, "radial_mesh_first", &[n_sites], &first, &["site"])?;
    write_f64_dataset(
        &group,
        "radial_mesh_log_increment",
        &[n_sites],
        &increment,
        &["site"],
    )?;
    write_i64_dataset(
        &group,
        "radial_mesh_point_count",
        &[n_sites],
        &point_count,
        &["site"],
    )?;
    Ok(())
}

fn read_geometry_group(file: &File) -> Result<MldumpGeometryV1, IoError> {
    let group = file.group(GROUP_GEOMETRY)?;
    require_status_present(&group)?;
    require_dataset_names(
        &group,
        &[
            "direct_basis",
            "reciprocal_basis",
            "cell_volume",
            "site_species",
            "site_labels",
            "site_positions",
            "site_radii",
            "radial_mesh_first",
            "radial_mesh_log_increment",
            "radial_mesh_point_count",
        ],
    )?;
    let direct = read_f64_dataset(
        &group,
        "direct_basis",
        &[3, 3],
        &["primitive_vector", "cartesian"],
    )?;
    let reciprocal = read_f64_dataset(
        &group,
        "reciprocal_basis",
        &[3, 3],
        &["primitive_vector", "cartesian"],
    )?;
    let volume_ds = require_numeric_dataset::<f64>(&group, "cell_volume", &[], None)?;
    let cell_volume_bohr3: f64 = volume_ds.read_scalar()?;
    let n_sites = group
        .dataset("site_radii")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: "/geometry/site_radii/shape".to_owned(),
            expected: "[n_sites]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    let species = read_str_dataset(&group, "site_species", n_sites, &["site"])?;
    let labels = read_str_dataset(&group, "site_labels", n_sites, &["site"])?;
    let positions = read_f64_dataset(
        &group,
        "site_positions",
        &[n_sites, 3],
        &["site", "cartesian"],
    )?;
    let radii = read_f64_dataset(&group, "site_radii", &[n_sites], &["site"])?;
    let first = read_f64_dataset(&group, "radial_mesh_first", &[n_sites], &["site"])?;
    let increment = read_f64_dataset(&group, "radial_mesh_log_increment", &[n_sites], &["site"])?;
    let point_count = read_i64_dataset(&group, "radial_mesh_point_count", &[n_sites], &["site"])?;
    let mut sites = Vec::with_capacity(n_sites);
    for index in 0..n_sites {
        let count = point_count[index];
        if count <= 0 {
            return Err(ValidationError::NotPositive {
                path: format!("/geometry/radial_mesh_point_count[{index}]"),
                value: count as f64,
            }
            .into());
        }
        sites.push(MldumpSiteV1 {
            species: optional_text(&species[index]),
            label: optional_text(&labels[index]),
            position_bohr: [
                positions[index * 3],
                positions[index * 3 + 1],
                positions[index * 3 + 2],
            ],
            radius_bohr: radii[index],
            radial_mesh: MldumpRadialMeshV1 {
                first_bohr: first[index],
                log_increment: increment[index],
                point_count: count as usize,
            },
        });
    }
    Ok(MldumpGeometryV1 {
        direct_basis_bohr: unflatten_matrix3(&direct),
        reciprocal_basis_inv_bohr: unflatten_matrix3(&reciprocal),
        cell_volume_bohr3,
        sites,
    })
}

fn write_mesh_group(file: &File, mesh: &MldumpMeshV1) -> Result<(), IoError> {
    let group = file.create_group(GROUP_MESH)?;
    write_status(&group, MldumpStatus::Present)?;
    let n_k = mesh.k_points.len();
    let n_q = mesh.q_entries.len();
    let mut k_fractional = Vec::with_capacity(n_k * 3);
    let mut k_weights = Vec::with_capacity(n_k);
    for point in &mesh.k_points {
        k_fractional.extend_from_slice(&point.fractional);
        k_weights.push(point.weight);
    }
    write_f64_dataset(
        &group,
        "k_fractional",
        &[n_k, 3],
        &k_fractional,
        &["k", "reciprocal_axis"],
    )?;
    write_f64_dataset(&group, "k_weights", &[n_k], &k_weights, &["k"])?;

    let mut q_input = Vec::with_capacity(n_q * 3);
    let mut q_canonical = Vec::with_capacity(n_q * 3);
    let mut q_umklapp = Vec::with_capacity(n_q * 3);
    let mut k_index = Vec::with_capacity(n_q * n_k);
    let mut mapped_index = Vec::with_capacity(n_q * n_k);
    let mut g_wrap = Vec::with_capacity(n_q * n_k * 3);
    for entry in &mesh.q_entries {
        q_input.extend_from_slice(&entry.input_fractional);
        q_canonical.extend_from_slice(&entry.canonical_fractional);
        q_umklapp.extend_from_slice(&entry.global_umklapp);
        for mapped in &entry.k_minus_q {
            k_index.push(i64::try_from(mapped.k_index).map_err(|_| {
                ValidationError::InvalidValue {
                    path: "mesh.k_minus_q_k_index".to_owned(),
                    expected: "i64".to_owned(),
                    actual: mapped.k_index.to_string(),
                }
            })?);
            mapped_index.push(i64::try_from(mapped.mapped_index).map_err(|_| {
                ValidationError::InvalidValue {
                    path: "mesh.k_minus_q_mapped_index".to_owned(),
                    expected: "i64".to_owned(),
                    actual: mapped.mapped_index.to_string(),
                }
            })?);
            g_wrap.extend_from_slice(&mapped.g_wrap);
        }
    }
    write_f64_dataset(
        &group,
        "q_input_fractional",
        &[n_q, 3],
        &q_input,
        &["q", "reciprocal_axis"],
    )?;
    write_f64_dataset(
        &group,
        "q_canonical_fractional",
        &[n_q, 3],
        &q_canonical,
        &["q", "reciprocal_axis"],
    )?;
    write_i32_dataset(
        &group,
        "q_global_umklapp",
        &[n_q, 3],
        &q_umklapp,
        &["q", "reciprocal_axis"],
    )?;
    write_i64_dataset(
        &group,
        "k_minus_q_k_index",
        &[n_q, n_k],
        &k_index,
        &["q", "k"],
    )?;
    write_i64_dataset(
        &group,
        "k_minus_q_mapped_index",
        &[n_q, n_k],
        &mapped_index,
        &["q", "k"],
    )?;
    write_i32_dataset(
        &group,
        "k_minus_q_g_wrap",
        &[n_q, n_k, 3],
        &g_wrap,
        &["q", "k", "reciprocal_axis"],
    )?;
    Ok(())
}

fn read_mesh_group(file: &File) -> Result<MldumpMeshV1, IoError> {
    let group = file.group(GROUP_MESH)?;
    require_status_present(&group)?;
    require_dataset_names(
        &group,
        &[
            "k_fractional",
            "k_weights",
            "q_input_fractional",
            "q_canonical_fractional",
            "q_global_umklapp",
            "k_minus_q_k_index",
            "k_minus_q_mapped_index",
            "k_minus_q_g_wrap",
        ],
    )?;
    let n_k = group
        .dataset("k_weights")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: "/mesh/k_weights/shape".to_owned(),
            expected: "[n_k]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    let n_q = group
        .dataset("q_global_umklapp")?
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: "/mesh/q_global_umklapp/shape".to_owned(),
            expected: "[n_q, 3]".to_owned(),
            actual: "scalar".to_owned(),
        })?;
    let k_fractional =
        read_f64_dataset(&group, "k_fractional", &[n_k, 3], &["k", "reciprocal_axis"])?;
    let k_weights = read_f64_dataset(&group, "k_weights", &[n_k], &["k"])?;
    let mut k_points = Vec::with_capacity(n_k);
    for k in 0..n_k {
        k_points.push(MldumpKPointV1 {
            fractional: [
                k_fractional[k * 3],
                k_fractional[k * 3 + 1],
                k_fractional[k * 3 + 2],
            ],
            weight: k_weights[k],
        });
    }
    let q_input = read_f64_dataset(
        &group,
        "q_input_fractional",
        &[n_q, 3],
        &["q", "reciprocal_axis"],
    )?;
    let q_canonical = read_f64_dataset(
        &group,
        "q_canonical_fractional",
        &[n_q, 3],
        &["q", "reciprocal_axis"],
    )?;
    let q_umklapp = read_i32_dataset(
        &group,
        "q_global_umklapp",
        &[n_q, 3],
        &["q", "reciprocal_axis"],
    )?;
    let k_index = read_i64_dataset(&group, "k_minus_q_k_index", &[n_q, n_k], &["q", "k"])?;
    let mapped_index =
        read_i64_dataset(&group, "k_minus_q_mapped_index", &[n_q, n_k], &["q", "k"])?;
    let g_wrap = read_i32_dataset(
        &group,
        "k_minus_q_g_wrap",
        &[n_q, n_k, 3],
        &["q", "k", "reciprocal_axis"],
    )?;
    let mut q_entries = Vec::with_capacity(n_q);
    for q in 0..n_q {
        let mut records = Vec::with_capacity(n_k);
        for k in 0..n_k {
            let flat = q * n_k + k;
            records.push(MldumpKMinusQV1 {
                k_index: require_nonnegative_index("mesh.k_minus_q_k_index", k_index[flat])?,
                mapped_index: require_nonnegative_index(
                    "mesh.k_minus_q_mapped_index",
                    mapped_index[flat],
                )?,
                g_wrap: [g_wrap[flat * 3], g_wrap[flat * 3 + 1], g_wrap[flat * 3 + 2]],
            });
        }
        q_entries.push(MldumpQEntryV1 {
            input_fractional: [q_input[q * 3], q_input[q * 3 + 1], q_input[q * 3 + 2]],
            canonical_fractional: [
                q_canonical[q * 3],
                q_canonical[q * 3 + 1],
                q_canonical[q * 3 + 2],
            ],
            global_umklapp: [q_umklapp[q * 3], q_umklapp[q * 3 + 1], q_umklapp[q * 3 + 2]],
            k_minus_q: records,
        });
    }
    Ok(MldumpMeshV1 {
        k_points,
        q_entries,
    })
}

fn require_nonnegative_index(path: &str, value: i64) -> Result<usize, IoError> {
    usize::try_from(value).map_err(|_| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "nonnegative index".to_owned(),
            actual: value.to_string(),
        }
        .into()
    })
}

fn write_absent_group(file: &File, name: &str) -> Result<(), IoError> {
    let group = file.create_group(name)?;
    write_status(&group, MldumpStatus::AbsentNotComputed)
}

fn read_absent_group(file: &File, name: &str) -> Result<MldumpStatus, IoError> {
    let group = file.group(name)?;
    let status = read_status(&group)?;
    require_absent_payload(&format!("/{name}/@{ATTR_STATUS}"), status)?;
    require_no_payload(&group)?;
    Ok(status)
}

fn write_exchange_group(file: &File, statuses: MldumpStatusesV1) -> Result<(), IoError> {
    let group = file.create_group(GROUP_EXCHANGE)?;
    write_status(&group, MldumpStatus::Present)?;
    write_status(
        &group.create_group(GROUP_EXCHANGE_VALENCE)?,
        statuses.exchange_valence,
    )?;
    write_status(
        &group.create_group(GROUP_EXCHANGE_CORE)?,
        statuses.exchange_core,
    )?;
    write_status(
        &group.create_group(GROUP_EXCHANGE_TOTAL)?,
        statuses.exchange_total,
    )?;
    Ok(())
}

fn read_exchange_group(file: &File) -> Result<(MldumpStatus, MldumpStatus, MldumpStatus), IoError> {
    let group = file.group(GROUP_EXCHANGE)?;
    require_status_present(&group)?;
    require_no_datasets(&group)?;
    if group.link_exists("total_relation")
        || group
            .attr_names()?
            .iter()
            .any(|name| name == "total_relation")
    {
        return Err(ValidationError::InvalidValue {
            path: "/exchange/total_relation".to_owned(),
            expected: "absent unless same-run valence and core are present".to_owned(),
            actual: "present".to_owned(),
        }
        .into());
    }
    require_group_names(
        &group,
        &[
            GROUP_EXCHANGE_VALENCE,
            GROUP_EXCHANGE_CORE,
            GROUP_EXCHANGE_TOTAL,
        ],
    )?;
    Ok((
        read_absent_child(&group, GROUP_EXCHANGE_VALENCE)?,
        read_absent_child(&group, GROUP_EXCHANGE_CORE)?,
        read_absent_child(&group, GROUP_EXCHANGE_TOTAL)?,
    ))
}

fn read_absent_child(parent: &Group, name: &str) -> Result<MldumpStatus, IoError> {
    let group = parent.group(name)?;
    let status = read_status(&group)?;
    require_absent_payload(&format!("{}/{name}/@{ATTR_STATUS}", parent.name()), status)?;
    require_no_payload(&group)?;
    Ok(status)
}

fn require_status_present(group: &Group) -> Result<(), IoError> {
    let status = read_status(group)?;
    if status == MldumpStatus::Present {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/@{ATTR_STATUS}", group.name()),
            expected: MLDUMP_STATUS_PRESENT.to_owned(),
            actual: status.as_str().to_owned(),
        }
        .into())
    }
}

fn require_str_attr(group: &Group, name: &str, expected: &str) -> Result<(), IoError> {
    let actual = read_str_attr(group, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/@{name}", group.name()),
            expected: expected.to_owned(),
            actual,
        }
        .into())
    }
}

fn require_no_datasets(group: &Group) -> Result<(), IoError> {
    let datasets = group.datasets()?;
    if datasets.is_empty() {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: group.name(),
            expected: "no numeric datasets".to_owned(),
            actual: format!("{} datasets", datasets.len()),
        }
        .into())
    }
}

fn require_no_payload(group: &Group) -> Result<(), IoError> {
    let members = group.member_names()?;
    if members.is_empty() {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: group.name(),
            expected: "no child members while absent_not_computed".to_owned(),
            actual: members.join(","),
        }
        .into())
    }
}

fn require_dataset_names(group: &Group, expected: &[&str]) -> Result<(), IoError> {
    let observed = group
        .datasets()?
        .into_iter()
        .map(|dataset| {
            dataset
                .name()
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    if observed == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/datasets", group.name()),
            expected: expected.into_iter().collect::<Vec<_>>().join(","),
            actual: observed.into_iter().collect::<Vec<_>>().join(","),
        }
        .into())
    }
}

fn require_group_names(group: &Group, expected: &[&str]) -> Result<(), IoError> {
    let observed = group
        .groups()?
        .into_iter()
        .map(|child| {
            child
                .name()
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let expected = expected
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    if observed == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/groups", group.name()),
            expected: expected.into_iter().collect::<Vec<_>>().join(","),
            actual: observed.into_iter().collect::<Vec<_>>().join(","),
        }
        .into())
    }
}

fn require_top_level_groups(file: &File) -> Result<(), IoError> {
    let observed = file.member_names()?.into_iter().collect::<BTreeSet<_>>();
    let expected = TOP_LEVEL_GROUPS
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    if observed == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: "/".to_owned(),
            expected: expected.into_iter().collect::<Vec<_>>().join(","),
            actual: observed.into_iter().collect::<Vec<_>>().join(","),
        }
        .into())
    }
}
