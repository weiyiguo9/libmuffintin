//! Versioned MLDUMP HDF5 interchange.
//!
//! This is a libmuffintin-owned, inspectable HDF5 schema. It is not
//! CoQui-native or SPEX-native. Runtime, mixed-product, THC, and Coulomb
//! types stay out of this crate; runtime materializes those objects through
//! [`ScalarMldumpStreamV1`] or [`SpinorMldumpStreamV1`].

use std::collections::BTreeSet;
use std::path::Path;
use std::str::FromStr;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Container, Dataset, File, Group, H5Type, Location};

use crate::error::{IoError, ValidationError, finite, nonempty, positive};

mod response;
mod scalar_orbitals;
mod scalar_products;
mod session;
mod spinor_orbitals;
mod spinor_products;
mod v2;

pub use response::{
    ComplexF64V1, MLDUMP_INTERSTITIAL_SENTINEL, MLDUMP_PARENT_REGION_INTERSTITIAL,
    MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY, MLDUMP_THC_ENGINE_QRCP,
    MLDUMP_THC_STRATEGY_ALL_QL2, MldumpCoulombBeginV1, MldumpCoulombGammaRefV1,
    MldumpCoulombGammaV1, MldumpCoulombQRecordRefV1, MldumpCoulombQRecordV1, MldumpCoulombV1,
    MldumpThcBeginV1, MldumpThcParentGridRefV1, MldumpThcParentGridV1, MldumpThcQRecordRefV1,
    MldumpThcQRecordV1, MldumpThcResidualV1, MldumpThcSelectionRefV1, MldumpThcSelectionV1,
    MldumpThcV1, MldumpThcVertexTableRefV1, MldumpThcVertexV1, ScalarMldumpV1, SpinorMldumpV1,
};
pub use scalar_orbitals::{
    MLDUMP_OCCUPATIONS_NOT_EXPORTED, MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON,
    ScalarApwSiteMatchRefV1, ScalarApwSiteMatchV1, ScalarLocalOrbitalRowV1,
    ScalarLocalOrbitalTableRefV1, ScalarOrbitalKRecordV1, ScalarOrbitalKRefV1, ScalarOrbitalSpinV1,
    ScalarOrbitalsBeginV1, ScalarOrbitalsV1,
};
pub use scalar_products::{
    MLDUMP_CORE_EMPTY_NOT_FITTED, MLDUMP_PAIR_ORDER_K_LEFT_RIGHT, MLDUMP_RADIAL_KIND_CORE,
    MLDUMP_RADIAL_KIND_VALENCE, ScalarProductQRecordRefV1, ScalarProductQRecordV1,
    ScalarProductSiteRefV1, ScalarProductSiteV1, ScalarProductsBeginV1, ScalarProductsV1,
};
pub use session::{ScalarMldumpStreamV1, SpinorMldumpStreamV1};
pub use spinor_orbitals::{
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, SpinorLocalOrbitalRowV1,
    SpinorLocalOrbitalTableRefV1, SpinorOrbitalKRecordV1, SpinorOrbitalKRefV1,
    SpinorOrbitalsBeginV1, SpinorOrbitalsV1, SpinorPauliRowMapRefV1, SpinorPauliRowMapV1,
    SpinorProjectionCoordV1, SpinorSiteMatchRefV1, SpinorSiteMatchV1,
};
pub use spinor_products::{
    SpinorProductQRecordRefV1, SpinorProductQRecordV1, SpinorProductSiteRefV1, SpinorProductSiteV1,
    SpinorProductsBeginV1, SpinorProductsV1,
};
pub use v2::{
    MLDUMP_EXCHANGE_BACKEND_V2, MLDUMP_EXCHANGE_SOURCE_FRAME_V2, MLDUMP_EXCHANGE_TOTAL_RELATION_V2,
    MldumpCoreOccupationV2, MldumpExchangeFitResidualV2, MldumpExchangeLayoutV2,
    MldumpExchangeMpbQuadraticV2, MldumpExchangeProvenanceV2, MldumpExchangeRankScalingV2,
    MldumpExchangeSectorV2, MldumpExchangeSpaceV2, MldumpExchangeV2, MldumpFileV2,
    MldumpGammaPolicyV2, MldumpRequestedRankV2, MldumpSelectorEngineV2, MldumpSelectorStrategyV2,
    read_mldump_v2, upgrade_mldump_v1_with_exchange_v2,
};

/// Stable schema name written on every MLDUMP file.
pub const MLDUMP_SCHEMA_NAME: &str = "libmuffintin.mldump";
/// MLDUMP schema version implemented by this crate.
pub const MLDUMP_SCHEMA_VERSION: u32 = 1;
/// Explicit MLDUMP v1 schema version.
pub const MLDUMP_SCHEMA_VERSION_V1: u32 = 1;
/// Explicit MLDUMP v2 schema version.
pub const MLDUMP_SCHEMA_VERSION_V2: u32 = 2;

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
/// constant times $\max(|a|,|b|,1)$. The $10^{-12}$ floor matches the product-input
/// mesh-coordinate gate; the scale factor is the same form used by parent-grid
/// radial matching.
pub(crate) const FRACTIONAL_EQ_TOLERANCE: f64 = 1.0e-12;

const ATTR_SCHEMA_NAME: &str = "schema_name";
const ATTR_SCHEMA_VERSION: &str = "schema_version";
pub(crate) const ATTR_STATUS: &str = "status";
pub(crate) const ATTR_AXES: &str = "axes";

const GROUP_META: &str = "meta";
const GROUP_UNITS: &str = "units";
const GROUP_GEOMETRY: &str = "geometry";
const GROUP_MESH: &str = "mesh";
pub(crate) const GROUP_ORBITALS: &str = "orbitals";
pub(crate) const GROUP_PRODUCTS: &str = "products";
pub(crate) const GROUP_MPB: &str = "mpb";
pub(crate) const GROUP_THC: &str = "thc";
pub(crate) const GROUP_COULOMB: &str = "coulomb";
const GROUP_EXCHANGE: &str = "exchange";
const GROUP_EXCHANGE_VALENCE: &str = "valence";
const GROUP_EXCHANGE_CORE: &str = "core";
const GROUP_EXCHANGE_TOTAL: &str = "total";
pub(crate) const PREFIX_SPIN: &str = "spin";
pub(crate) const PREFIX_K: &str = "k";
pub(crate) const PREFIX_SITE: &str = "site";
pub(crate) const PREFIX_Q: &str = "q";
pub(crate) const PREFIX_BASIS: &str = "basis";
pub(crate) const PREFIX_GAMMA: &str = "gamma";
pub(crate) const PREFIX_PARENT_GRID: &str = "parent_grid";

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

/// File-level exchange valence/core/total statuses. Scalar MLDUMP writes all three absent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MldumpExchangeStatusesV1 {
    pub valence: MldumpStatus,
    pub core: MldumpStatus,
    pub total: MldumpStatus,
}

impl MldumpExchangeStatusesV1 {
    /// Every exchange seam is `absent_not_computed`.
    pub const fn absent_not_computed() -> Self {
        Self {
            valence: MldumpStatus::AbsentNotComputed,
            core: MldumpStatus::AbsentNotComputed,
            total: MldumpStatus::AbsentNotComputed,
        }
    }
}

/// Accepted MLDUMP v1 header: producer metadata, geometry, and $k$/$q$ mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpHeaderV1 {
    pub meta: MldumpMetaV1,
    pub geometry: MldumpGeometryV1,
    pub mesh: MldumpMeshV1,
}

/// Owned MLDUMP v1 file: header, representation payload, and exchange statuses.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpFileV1 {
    pub header: MldumpHeaderV1,
    pub payload: MldumpPayloadV1,
    pub exchange: MldumpExchangeStatusesV1,
}

/// Representation-discriminated MLDUMP v1 payload.
///
/// `/orbitals/@representation` is the authoritative branch tag. Header-only
/// files have all four payload groups absent. Scalar files have all four
/// groups present: companion `/products`,`/thc`,`/coulomb` representation
/// attrs are either all absent (earlier published scalar files) or all
/// `scalar_koelling_harmon`. Spinor files require those three attrs present
/// and equal to `spinor_full_first_variation`. The writer still emits all
/// four tags.
#[derive(Clone, Debug, PartialEq)]
pub enum MldumpPayloadV1 {
    /// All of `/orbitals`, `/products`, `/thc`, and `/coulomb` are absent.
    HeaderOnly,
    /// Scalar Koelling–Harmon payload.
    Scalar(ScalarMldumpV1),
    /// Full first-variation spinor payload.
    Spinor(SpinorMldumpV1),
}

/// Stateful writer for a v1 file. Header-only files call [`Self::finish`].
/// Populated scalar files continue through [`Self::begin_scalar`].
#[derive(Debug)]
pub struct MldumpWriterV1 {
    file: File,
    header: MldumpHeaderV1,
}

impl MldumpHeaderV1 {
    /// Construct a v1 header. Reserved payload groups are created absent by [`MldumpWriterV1::create`].
    pub fn new(meta: MldumpMetaV1, geometry: MldumpGeometryV1, mesh: MldumpMeshV1) -> Self {
        Self {
            meta,
            geometry,
            mesh,
        }
    }

    /// Trust-boundary checks needed to interpret the in-memory header.
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
        Ok(())
    }
}

impl MldumpWriterV1 {
    /// Create a v1 file, write the accepted header, and leave reserved groups absent.
    ///
    /// `/exchange` is present as the three-child status table; each child is
    /// `absent_not_computed`. A failed or interrupted write may leave an incomplete
    /// file; this API does not publish atomically.
    pub fn create(path: impl AsRef<Path>, header: &MldumpHeaderV1) -> Result<Self, IoError> {
        header.validate()?;
        let file = File::create(path)?;
        write_str_attr(&file, ATTR_SCHEMA_NAME, MLDUMP_SCHEMA_NAME)?;
        file.new_attr::<u32>()
            .create(ATTR_SCHEMA_VERSION)?
            .write_scalar(&MLDUMP_SCHEMA_VERSION)?;
        write_meta_group(&file, &header.meta)?;
        write_units_group(&file)?;
        write_geometry_group(&file, &header.geometry)?;
        write_mesh_group(&file, &header.mesh)?;
        for name in ABSENT_PAYLOAD_GROUPS {
            write_absent_group(&file, name)?;
        }
        write_exchange_group(&file, MldumpExchangeStatusesV1::absent_not_computed())?;
        Ok(Self {
            file,
            header: header.clone(),
        })
    }

    /// Start a streaming scalar session. All four sections must then be written
    /// record-wise before [`ScalarMldumpStreamV1::finish`].
    pub fn begin_scalar(self) -> Result<ScalarMldumpStreamV1, IoError> {
        Ok(ScalarMldumpStreamV1::new(self.file, self.header))
    }

    /// Close a header-only writer. Reserved payload groups stay
    /// `absent_not_computed`.
    pub fn finish(self) -> Result<(), IoError> {
        Ok(())
    }

    /// Start a streaming spinor session. All four sections must then be written
    /// record-wise before [`SpinorMldumpStreamV1::finish`].
    pub fn begin_spinor(self) -> Result<SpinorMldumpStreamV1, IoError> {
        Ok(SpinorMldumpStreamV1::new(self.file, self.header))
    }
}

pub(crate) fn stream_state_error(path: &str, method: &str, expected: &str) -> IoError {
    ValidationError::InvalidValue {
        path: path.to_owned(),
        expected: expected.to_owned(),
        actual: format!("{method} in unexpected session state"),
    }
    .into()
}

pub(crate) fn require_record_capacity(
    path: &str,
    next: usize,
    expected: usize,
) -> Result<(), IoError> {
    if next < expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: format!("{expected} records"),
            actual: format!("record {next}"),
        }
        .into())
    }
}

pub(crate) fn section_already_written(path: &str) -> IoError {
    ValidationError::InvalidValue {
        path: path.to_owned(),
        expected: "section written at most once".to_owned(),
        actual: "already written".to_owned(),
    }
    .into()
}

/// Read an MLDUMP v1 HDF5 file into the typed v1 record.
pub fn read_mldump_v1(path: impl AsRef<Path>) -> Result<MldumpFileV1, IoError> {
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
    let header = MldumpHeaderV1 {
        meta,
        geometry,
        mesh,
    };
    header.validate()?;

    let orbitals_status = read_payload_status(&file, GROUP_ORBITALS)?;
    let products_status = read_payload_status(&file, GROUP_PRODUCTS)?;
    let thc_status = read_payload_status(&file, GROUP_THC)?;
    let coulomb_status = read_payload_status(&file, GROUP_COULOMB)?;
    read_absent_group(&file, GROUP_MPB)?;
    let exchange = read_exchange_group(&file)?;
    require_absent_payload("/exchange/valence/@status", exchange.valence)?;
    require_absent_payload("/exchange/core/@status", exchange.core)?;
    require_absent_payload("/exchange/total/@status", exchange.total)?;

    let n_present = [orbitals_status, products_status, thc_status, coulomb_status]
        .into_iter()
        .filter(|status| *status == MldumpStatus::Present)
        .count();
    let payload = match n_present {
        0 => {
            require_absent_status_and_payload(&file, GROUP_ORBITALS, orbitals_status)?;
            require_absent_status_and_payload(&file, GROUP_PRODUCTS, products_status)?;
            require_absent_status_and_payload(&file, GROUP_THC, thc_status)?;
            require_absent_status_and_payload(&file, GROUP_COULOMB, coulomb_status)?;
            MldumpPayloadV1::HeaderOnly
        }
        4 => read_present_payload(&file, &header)?,
        _ => {
            return Err(ValidationError::InvalidValue {
                path: "payload".to_owned(),
                expected: "all four of /orbitals,/products,/thc,/coulomb present or all absent_not_computed"
                    .to_owned(),
                actual: scalar_section_names(
                    orbitals_status == MldumpStatus::Present,
                    products_status == MldumpStatus::Present,
                    thc_status == MldumpStatus::Present,
                    coulomb_status == MldumpStatus::Present,
                ),
            }
            .into());
        }
    };

    Ok(MldumpFileV1 {
        header,
        payload,
        exchange,
    })
}

fn read_present_payload(file: &File, header: &MldumpHeaderV1) -> Result<MldumpPayloadV1, IoError> {
    let orbitals_group = file.group(GROUP_ORBITALS)?;
    let products_group = file.group(GROUP_PRODUCTS)?;
    let thc_group = file.group(GROUP_THC)?;
    let coulomb_group = file.group(GROUP_COULOMB)?;
    let representation = read_str_attr(&orbitals_group, "representation")?;
    match representation.as_str() {
        MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON => {
            require_scalar_companion_representation(
                &products_group,
                &thc_group,
                &coulomb_group,
            )?;
            let scalar = ScalarMldumpV1 {
                orbitals: scalar_orbitals::read_scalar_orbitals(file, header)?,
                products: scalar_products::read_scalar_products(file, header)?,
                thc: response::read_mldump_thc(file, header, &representation)?,
                coulomb: response::read_mldump_coulomb(file, header, &representation)?,
            };
            response::validate_owned_thc_vertex_identity(
                scalar.products.n_k,
                scalar.products.n_orb,
                &scalar.thc,
            )?;
            response::validate_scalar_alignment(
                header,
                &response::OrbitalAlignmentSummary::from_owned(&scalar.orbitals),
                &response::ProductAlignmentSummary::from_owned(&scalar.products),
                &response::ThcAlignmentSummary::from_owned(&scalar.thc),
                &response::CoulombAlignmentSummary::from_owned(&scalar.coulomb),
            )?;
            Ok(MldumpPayloadV1::Scalar(scalar))
        }
        MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION => {
            require_spinor_companion_representation(
                &products_group,
                &thc_group,
                &coulomb_group,
            )?;
            let spinor = SpinorMldumpV1 {
                orbitals: spinor_orbitals::read_spinor_orbitals(file, header)?,
                products: spinor_products::read_spinor_products(file, header)?,
                thc: response::read_mldump_thc(file, header, &representation)?,
                coulomb: response::read_mldump_coulomb(file, header, &representation)?,
            };
            response::validate_owned_thc_vertex_identity(
                spinor.products.n_k,
                spinor.products.n_orb,
                &spinor.thc,
            )?;
            response::validate_payload_alignment(
                header,
                spinor.orbitals.band_window_count,
                &response::ProductAlignmentSummary::from_q_bindings(
                    spinor.products.n_k,
                    spinor.products.n_orb,
                    spinor
                        .products
                        .q_records
                        .iter()
                        .map(|record| response::ProductQAlignment {
                            q_index: record.q_index,
                            transfer_cartesian: record.transfer_cartesian,
                            global_transfer: record.global_transfer,
                        })
                        .collect(),
                ),
                &response::ThcAlignmentSummary::from_owned(&spinor.thc),
                &response::CoulombAlignmentSummary::from_owned(&spinor.coulomb),
            )?;
            Ok(MldumpPayloadV1::Spinor(spinor))
        }
        other => Err(ValidationError::InvalidValue {
            path: "/orbitals/@representation".to_owned(),
            expected: format!(
                "{MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON} or {MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION}"
            ),
            actual: other.to_owned(),
        }
        .into()),
    }
}

fn require_scalar_companion_representation(
    products: &Group,
    thc: &Group,
    coulomb: &Group,
) -> Result<(), IoError> {
    let products_tag = read_optional_str_attr(products, "representation")?;
    let thc_tag = read_optional_str_attr(thc, "representation")?;
    let coulomb_tag = read_optional_str_attr(coulomb, "representation")?;
    let companions = [
        ("/products/@representation", products_tag.as_deref()),
        ("/thc/@representation", thc_tag.as_deref()),
        ("/coulomb/@representation", coulomb_tag.as_deref()),
    ];
    let present = companions.iter().filter(|(_, tag)| tag.is_some()).count();
    if present == 0 {
        return Ok(());
    }
    if present == 3 {
        for (path, tag) in companions {
            match tag {
                Some(MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON) => {}
                Some(actual) => {
                    return Err(ValidationError::InvalidValue {
                        path: path.to_owned(),
                        expected: MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON.to_owned(),
                        actual: actual.to_owned(),
                    }
                    .into());
                }
                None => {
                    return Err(scalar_companion_mixture_error(&companions));
                }
            }
        }
        return Ok(());
    }
    Err(scalar_companion_mixture_error(&companions))
}

fn scalar_companion_mixture_error(companions: &[(&str, Option<&str>); 3]) -> IoError {
    let (path, _) = companions
        .iter()
        .find(|(_, tag)| tag.is_none())
        .copied()
        .unwrap_or(companions[0]);
    ValidationError::InvalidValue {
        path: path.to_owned(),
        expected: format!(
            "all three companion representation attrs absent (earlier published scalar files) or all present as {MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON}"
        ),
        actual: format!(
            "products={} thc={} coulomb={}",
            companions[0].1.unwrap_or("absent"),
            companions[1].1.unwrap_or("absent"),
            companions[2].1.unwrap_or("absent"),
        ),
    }
    .into()
}

fn require_spinor_companion_representation(
    products: &Group,
    thc: &Group,
    coulomb: &Group,
) -> Result<(), IoError> {
    for group in [products, thc, coulomb] {
        match read_optional_str_attr(group, "representation")? {
            Some(actual) if actual == MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION => {}
            Some(actual) => {
                return Err(ValidationError::InvalidValue {
                    path: format!("{}/@representation", group.name()),
                    expected: MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION.to_owned(),
                    actual,
                }
                .into());
            }
            None => {
                return Err(ValidationError::InvalidValue {
                    path: format!("{}/@representation", group.name()),
                    expected: MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION.to_owned(),
                    actual: "absent".to_owned(),
                }
                .into());
            }
        }
    }
    Ok(())
}

fn scalar_section_names(orbitals: bool, products: bool, thc: bool, coulomb: bool) -> String {
    let mut names = Vec::new();
    if orbitals {
        names.push("orbitals");
    }
    if products {
        names.push("products");
    }
    if thc {
        names.push("thc");
    }
    if coulomb {
        names.push("coulomb");
    }
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(",")
    }
}

fn read_payload_status(file: &File, name: &str) -> Result<MldumpStatus, IoError> {
    read_status(&file.group(name)?)
}

fn require_absent_status_and_payload(
    file: &File,
    name: &str,
    status: MldumpStatus,
) -> Result<(), IoError> {
    require_absent_payload(&format!("/{name}/@{ATTR_STATUS}"), status)?;
    require_no_payload(&file.group(name)?)?;
    Ok(())
}

pub(crate) fn require_no_nul(path: impl Into<String>, value: &str) -> Result<(), ValidationError> {
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

pub(crate) fn require_absent_payload(
    path: &str,
    status: MldumpStatus,
) -> Result<(), ValidationError> {
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

pub(crate) fn approx_eq(left: f64, right: f64) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= FRACTIONAL_EQ_TOLERANCE * scale
}

/// Shared $k+\mathbf G\cdot\mathbf b=q$ identity for scalar and spinor plane-wave tables.
pub(crate) fn validate_plane_wave_identity(
    reciprocal: &[[f64; 3]; 3],
    n_plane_waves: usize,
    plane_wave_g: &[i32],
    plane_wave_k_cartesian: &[f64],
    plane_wave_q_cartesian: &[f64],
    path: &str,
) -> Result<(), IoError> {
    for pw in 0..n_plane_waves {
        let g = [
            plane_wave_g[pw * 3],
            plane_wave_g[pw * 3 + 1],
            plane_wave_g[pw * 3 + 2],
        ];
        let reconstructed: [f64; 3] = std::array::from_fn(|axis| {
            plane_wave_k_cartesian[pw * 3 + axis]
                + f64::from(g[0]) * reciprocal[0][axis]
                + f64::from(g[1]) * reciprocal[1][axis]
                + f64::from(g[2]) * reciprocal[2][axis]
        });
        for (axis, (stored, expected)) in plane_wave_q_cartesian[pw * 3..pw * 3 + 3]
            .iter()
            .zip(reconstructed)
            .enumerate()
        {
            if !approx_eq(*stored, expected) {
                return Err(ValidationError::InvalidValue {
                    path: format!("{path}.plane_wave_q_cartesian[{pw}][{axis}]"),
                    expected: format!("k + G·b = {expected}"),
                    actual: stored.to_string(),
                }
                .into());
            }
        }
    }
    Ok(())
}

pub(crate) fn dataset_leading_len(
    dataset: &Dataset,
    path: impl Into<String>,
    expected_shape: &str,
) -> Result<usize, IoError> {
    dataset
        .shape()
        .first()
        .copied()
        .ok_or_else(|| ValidationError::InvalidValue {
            path: path.into(),
            expected: expected_shape.to_owned(),
            actual: "scalar".to_owned(),
        })
        .map_err(IoError::from)
}

/// Convert a fractional reciprocal coordinate with the stored reciprocal basis.
///
/// The geometry reciprocal rows already include $2\pi$; this does not add a
/// second factor and does not insert a global Umklapp label.
pub(crate) fn fractional_to_cartesian(reciprocal: [[f64; 3]; 3], fractional: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| {
        fractional[0] * reciprocal[0][axis]
            + fractional[1] * reciprocal[1][axis]
            + fractional[2] * reciprocal[2][axis]
    })
}

fn validate_matrix3(path: &str, matrix: [[f64; 3]; 3]) -> Result<(), ValidationError> {
    for (row, vector) in matrix.iter().enumerate() {
        for (axis, value) in vector.iter().enumerate() {
            finite(format!("{path}[{row}][{axis}]"), *value)?;
        }
    }
    Ok(())
}

pub(crate) fn vlen(path: &str, value: &str) -> Result<VarLenUnicode, IoError> {
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

pub(crate) fn write_str_attr(object: &Location, name: &str, value: &str) -> Result<(), IoError> {
    object
        .new_attr::<VarLenUnicode>()
        .create(name)?
        .write_scalar(&vlen(name, value)?)?;
    Ok(())
}

pub(crate) fn read_str_attr(object: &Location, name: &str) -> Result<String, IoError> {
    let value: VarLenUnicode = object.attr(name)?.read_scalar()?;
    Ok(value.as_str().to_owned())
}

pub(crate) fn read_optional_str_attr(
    object: &Location,
    name: &str,
) -> Result<Option<String>, IoError> {
    if !object.attr_names()?.iter().any(|attr| attr == name) {
        return Ok(None);
    }
    Ok(Some(read_str_attr(object, name)?))
}

pub(crate) fn write_status(group: &Group, status: MldumpStatus) -> Result<(), IoError> {
    write_str_attr(group, ATTR_STATUS, status.as_str())
}

pub(crate) fn read_status(group: &Group) -> Result<MldumpStatus, IoError> {
    let path = format!("{}/@{ATTR_STATUS}", group.name());
    Ok(MldumpStatus::parse(
        &path,
        &read_str_attr(group, ATTR_STATUS)?,
    )?)
}

pub(crate) fn write_axes(dataset: &Dataset, axes: &[&str]) -> Result<(), IoError> {
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

pub(crate) fn require_axes(dataset: &Dataset, expected: &[&str]) -> Result<(), IoError> {
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

pub(crate) fn write_f64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[f64],
    axes: &[&str],
) -> Result<(), IoError> {
    require_flat_len(name, shape, data.len())?;
    let dataset = create_dataset::<f64>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_axes(&dataset, axes)
}

pub(crate) fn write_i32_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[i32],
    axes: &[&str],
) -> Result<(), IoError> {
    require_flat_len(name, shape, data.len())?;
    let dataset = create_dataset::<i32>(group, name, shape)?;
    dataset.write_raw(data)?;
    write_axes(&dataset, axes)
}

pub(crate) fn write_i64_dataset(
    group: &Group,
    name: &str,
    shape: &[usize],
    data: &[i64],
    axes: &[&str],
) -> Result<(), IoError> {
    require_flat_len(name, shape, data.len())?;
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

pub(crate) fn create_dataset<T: H5Type>(
    group: &Group,
    name: &str,
    shape: &[usize],
) -> Result<Dataset, IoError> {
    let builder = group.new_dataset::<T>();
    let dataset = if shape.is_empty() {
        builder.create(name)?
    } else if shape.len() <= 4 {
        builder.shape(shape).create(name)?
    } else {
        return Err(ValidationError::InvalidValue {
            path: name.to_owned(),
            expected: "rank 0..=4".to_owned(),
            actual: shape.len().to_string(),
        }
        .into());
    };
    Ok(dataset)
}

pub(crate) fn require_numeric_dtype<T: H5Type>(
    object: &Container,
    path: &str,
) -> Result<(), IoError> {
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

pub(crate) fn read_numeric_attr<T: H5Type>(
    object: &Location,
    name: &str,
    path: &str,
) -> Result<T, IoError> {
    let attr = object.attr(name)?;
    require_numeric_dtype::<T>(&attr, path)?;
    Ok(attr.read_scalar()?)
}

pub(crate) fn require_numeric_dataset<T: H5Type>(
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

pub(crate) fn read_f64_dataset(
    group: &Group,
    name: &str,
    expected_shape: &[usize],
    axes: &[&str],
) -> Result<Vec<f64>, IoError> {
    Ok(require_numeric_dataset::<f64>(group, name, expected_shape, Some(axes))?.read_raw()?)
}

pub(crate) fn read_i32_dataset(
    group: &Group,
    name: &str,
    expected_shape: &[usize],
    axes: &[&str],
) -> Result<Vec<i32>, IoError> {
    Ok(require_numeric_dataset::<i32>(group, name, expected_shape, Some(axes))?.read_raw()?)
}

pub(crate) fn read_i64_dataset(
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

pub(crate) fn require_shape(dataset: &Dataset, expected: &[usize]) -> Result<(), IoError> {
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

pub(crate) fn require_nonnegative_index(path: &str, value: i64) -> Result<usize, IoError> {
    usize::try_from(value).map_err(|_| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "nonnegative index".to_owned(),
            actual: value.to_string(),
        }
        .into()
    })
}

pub(crate) fn write_absent_group(parent: &Group, name: &str) -> Result<Group, IoError> {
    let group = parent.create_group(name)?;
    write_status(&group, MldumpStatus::AbsentNotComputed)?;
    Ok(group)
}

fn read_absent_group(file: &File, name: &str) -> Result<MldumpStatus, IoError> {
    let group = file.group(name)?;
    let status = read_status(&group)?;
    require_absent_payload(&format!("/{name}/@{ATTR_STATUS}"), status)?;
    require_no_payload(&group)?;
    Ok(status)
}

fn write_exchange_group(file: &File, statuses: MldumpExchangeStatusesV1) -> Result<(), IoError> {
    let group = file.create_group(GROUP_EXCHANGE)?;
    write_status(&group, MldumpStatus::Present)?;
    write_status(
        &group.create_group(GROUP_EXCHANGE_VALENCE)?,
        statuses.valence,
    )?;
    write_status(&group.create_group(GROUP_EXCHANGE_CORE)?, statuses.core)?;
    write_status(&group.create_group(GROUP_EXCHANGE_TOTAL)?, statuses.total)?;
    Ok(())
}

fn read_exchange_group(file: &File) -> Result<MldumpExchangeStatusesV1, IoError> {
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
    Ok(MldumpExchangeStatusesV1 {
        valence: read_absent_child(&group, GROUP_EXCHANGE_VALENCE)?,
        core: read_absent_child(&group, GROUP_EXCHANGE_CORE)?,
        total: read_absent_child(&group, GROUP_EXCHANGE_TOTAL)?,
    })
}

fn read_absent_child(parent: &Group, name: &str) -> Result<MldumpStatus, IoError> {
    let group = parent.group(name)?;
    let status = read_status(&group)?;
    require_absent_payload(&format!("{}/{name}/@{ATTR_STATUS}", parent.name()), status)?;
    require_no_payload(&group)?;
    Ok(status)
}

pub(crate) fn require_status_present(group: &Group) -> Result<(), IoError> {
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

pub(crate) fn require_str_attr(group: &Group, name: &str, expected: &str) -> Result<(), IoError> {
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

pub(crate) fn require_str_attr_if_present(
    group: &Group,
    name: &str,
    expected: &str,
) -> Result<(), IoError> {
    match read_optional_str_attr(group, name)? {
        None => Ok(()),
        Some(actual) if actual == expected => Ok(()),
        Some(actual) => Err(ValidationError::InvalidValue {
            path: format!("{}/@{name}", group.name()),
            expected: expected.to_owned(),
            actual,
        }
        .into()),
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

pub(crate) fn require_no_payload(group: &Group) -> Result<(), IoError> {
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

pub(crate) fn require_dataset_names(group: &Group, expected: &[&str]) -> Result<(), IoError> {
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

pub(crate) fn require_group_names(group: &Group, expected: &[&str]) -> Result<(), IoError> {
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

pub(crate) fn child_basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

pub(crate) fn padded_child(prefix: &str, index: usize) -> String {
    format!("{prefix}_{index:06}")
}

pub(crate) fn parse_padded(name: &str, prefix: &str) -> Option<usize> {
    let rest = name.strip_prefix(prefix)?.strip_prefix('_')?;
    if rest.len() != 6 || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    rest.parse().ok()
}

pub(crate) fn usize_as_i64(path: &str, value: usize) -> Result<i64, IoError> {
    i64::try_from(value).map_err(|_| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "i64".to_owned(),
            actual: value.to_string(),
        }
        .into()
    })
}

pub(crate) fn require_len(path: &str, expected: usize, actual: usize) -> Result<(), IoError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ValidationError::LengthMismatch {
            path: path.to_owned(),
            expected,
            actual,
        }
        .into())
    }
}

pub(crate) fn require_flat_len(path: &str, shape: &[usize], actual: usize) -> Result<(), IoError> {
    let expected = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim));
    match expected {
        Some(expected) => require_len(path, expected, actual),
        None => Err(ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "shape product fitting usize".to_owned(),
            actual: format!("{shape:?}"),
        }
        .into()),
    }
}

pub(crate) fn require_finite_f64s(path: &str, values: &[f64]) -> Result<(), IoError> {
    for (index, value) in values.iter().enumerate() {
        finite(format!("{path}[{index}]"), *value)?;
    }
    Ok(())
}

pub(crate) fn write_i64_attr(object: &Location, name: &str, value: i64) -> Result<(), IoError> {
    object
        .new_attr::<i64>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

pub(crate) fn write_f64_attr(object: &Location, name: &str, value: f64) -> Result<(), IoError> {
    finite(format!("@{name}"), value)?;
    object
        .new_attr::<f64>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

pub(crate) fn read_i64_attr(object: &Location, name: &str, path: &str) -> Result<i64, IoError> {
    read_numeric_attr::<i64>(object, name, &format!("{path}/dtype"))
}

pub(crate) fn read_f64_attr(object: &Location, name: &str, path: &str) -> Result<f64, IoError> {
    let value = read_numeric_attr::<f64>(object, name, &format!("{path}/dtype"))?;
    finite(path, value)?;
    Ok(value)
}

pub(crate) fn read_usize_attr(object: &Location, name: &str, path: &str) -> Result<usize, IoError> {
    require_nonnegative_index(path, read_i64_attr(object, name, path)?)
}

pub(crate) fn write_str_array_attr(
    object: &Location,
    name: &str,
    values: &[&str],
) -> Result<(), IoError> {
    let encoded = values
        .iter()
        .map(|value| vlen(name, value))
        .collect::<Result<Vec<_>, _>>()?;
    object
        .new_attr::<VarLenUnicode>()
        .shape([values.len()])
        .create(name)?
        .write_raw(encoded.as_slice())?;
    Ok(())
}

pub(crate) fn require_str_array_attr(
    object: &Location,
    name: &str,
    expected: &[&str],
) -> Result<(), IoError> {
    let values = object.attr(name)?.read_raw::<VarLenUnicode>()?;
    let actual = values
        .iter()
        .map(|value| value.as_str())
        .collect::<Vec<_>>();
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("@{name}"),
            expected: expected.join(" "),
            actual: actual.join(" "),
        }
        .into())
    }
}

pub(crate) fn require_exact_members(
    group: &Group,
    expected: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<(), IoError> {
    let observed = group.member_names()?.into_iter().collect::<BTreeSet<_>>();
    let expected = expected
        .into_iter()
        .map(|name| name.as_ref().to_owned())
        .collect::<BTreeSet<_>>();
    if observed == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{}/members", group.name()),
            expected: expected.into_iter().collect::<Vec<_>>().join(","),
            actual: observed.into_iter().collect::<Vec<_>>().join(","),
        }
        .into())
    }
}

pub(crate) fn reopen_present_group(parent: &Group, name: &str) -> Result<Group, IoError> {
    {
        let existing = parent.group(name)?;
        let status = read_status(&existing)?;
        require_absent_payload(&format!("{}/{name}/@{ATTR_STATUS}", parent.name()), status)?;
        require_no_payload(&existing)?;
    }
    parent.unlink(name)?;
    let group = parent.create_group(name)?;
    write_status(&group, MldumpStatus::Present)?;
    Ok(group)
}

pub(crate) fn create_padded_group(
    parent: &Group,
    prefix: &str,
    index: usize,
) -> Result<Group, IoError> {
    Ok(parent.create_group(&padded_child(prefix, index))?)
}

pub(crate) fn collect_padded_groups(parent: &Group, prefix: &str) -> Result<Vec<Group>, IoError> {
    let mut indexed = parent
        .groups()?
        .into_iter()
        .filter_map(|group| {
            parse_padded(child_basename(&group.name()), prefix).map(|index| (index, group))
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(index, _)| *index);
    for (position, (index, _)) in indexed.iter().enumerate() {
        if *index != position {
            return Err(ValidationError::InvalidValue {
                path: format!("{}/{prefix}_*", parent.name()),
                expected: padded_child(prefix, position),
                actual: padded_child(prefix, *index),
            }
            .into());
        }
    }
    Ok(indexed.into_iter().map(|(_, group)| group).collect())
}

pub(crate) fn complex_len(logical: usize) -> Result<usize, IoError> {
    logical.checked_mul(2).ok_or_else(|| {
        ValidationError::InvalidValue {
            path: "complex_len".to_owned(),
            expected: "logical * 2 fitting usize".to_owned(),
            actual: logical.to_string(),
        }
        .into()
    })
}

pub(crate) fn triples_to_owned(
    values: &[f64],
    n: usize,
    path: &str,
) -> Result<Vec<[f64; 3]>, IoError> {
    require_len(path, n * 3, values.len())?;
    Ok((0..n)
        .map(|index| {
            [
                values[index * 3],
                values[index * 3 + 1],
                values[index * 3 + 2],
            ]
        })
        .collect())
}

pub(crate) fn i32_triples_to_owned(
    values: &[i32],
    n: usize,
    path: &str,
) -> Result<Vec<[i32; 3]>, IoError> {
    require_len(path, n * 3, values.len())?;
    Ok((0..n)
        .map(|index| {
            [
                values[index * 3],
                values[index * 3 + 1],
                values[index * 3 + 2],
            ]
        })
        .collect())
}
