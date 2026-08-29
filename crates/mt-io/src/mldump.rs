//! Versioned MLDUMP v1 HDF5 interchange.
//!
//! This is a libmuffintin-owned, inspectable HDF5 schema. It is not
//! CoQui-native or SPEX-native. Runtime, mixed-product, THC, and Coulomb
//! types stay out of this crate; runtime materializes those objects through
//! [`ScalarMldumpStreamV1`].

use std::collections::BTreeSet;
use std::path::Path;
use std::str::FromStr;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Container, Dataset, File, Group, H5Type, Location};

use crate::error::{IoError, ValidationError, finite, nonempty, positive};

mod scalar_orbitals;
mod scalar_products;
mod scalar_response;

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
pub use scalar_response::{
    ComplexF64V1, MLDUMP_INTERSTITIAL_SENTINEL, MLDUMP_PARENT_REGION_INTERSTITIAL,
    MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY, MLDUMP_THC_ENGINE_QRCP,
    MLDUMP_THC_STRATEGY_ALL_QL2, ScalarCoulombBeginV1, ScalarCoulombGammaRefV1,
    ScalarCoulombGammaV1, ScalarCoulombQRecordRefV1, ScalarCoulombQRecordV1, ScalarCoulombV1,
    ScalarMldumpV1, ScalarThcBeginV1, ScalarThcParentGridRefV1, ScalarThcParentGridV1,
    ScalarThcQRecordRefV1, ScalarThcQRecordV1, ScalarThcResidualV1, ScalarThcSelectionRefV1,
    ScalarThcSelectionV1, ScalarThcV1, ScalarThcVertexTableRefV1, ScalarThcVertexV1,
};

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

/// File-level exchange valence/core/total statuses. M-L6b1 writes all three absent.
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

/// Owned MLDUMP v1 file: header, optional scalar payload, and exchange statuses.
#[derive(Clone, Debug, PartialEq)]
pub struct MldumpFileV1 {
    pub header: MldumpHeaderV1,
    pub scalar: Option<ScalarMldumpV1>,
    pub exchange: MldumpExchangeStatusesV1,
}

/// Stateful writer for a v1 file. Header-only files call [`Self::finish`].
/// Populated scalar files continue through [`Self::begin_scalar`].
#[derive(Debug)]
pub struct MldumpWriterV1 {
    file: File,
    header: MldumpHeaderV1,
}

/// Streaming scalar payload session. Large records are written immediately;
/// only small counters, $q$ bindings, pair-layout counts, and provenance
/// strings are retained. Vertex tables are not kept after each $q$ write.
#[derive(Debug)]
pub struct ScalarMldumpStreamV1 {
    file: File,
    header: MldumpHeaderV1,
    phase: ScalarStreamPhase,
    orbital_summary: Option<scalar_response::OrbitalAlignmentSummary>,
    product_summary: Option<scalar_response::ProductAlignmentSummary>,
    thc_summary: Option<scalar_response::ThcAlignmentSummary>,
    coulomb_summary: Option<scalar_response::CoulombAlignmentSummary>,
    orbitals_band_window: usize,
    orbitals_spin_count: usize,
    products_n_site: usize,
    thc_n_parent: usize,
    thc_effective_rank: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarStreamPhase {
    Start,
    Orbitals { next_spin: usize, next_k: usize },
    Products { next_site: usize, next_q: usize },
    Thc { next_q: usize },
    Coulomb { next_q: usize },
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
        Ok(ScalarMldumpStreamV1 {
            file: self.file,
            header: self.header,
            phase: ScalarStreamPhase::Start,
            orbital_summary: None,
            product_summary: None,
            thc_summary: None,
            coulomb_summary: None,
            orbitals_band_window: 0,
            orbitals_spin_count: 0,
            products_n_site: 0,
            thc_n_parent: 0,
            thc_effective_rank: 0,
        })
    }

    /// Close a header-only writer. Reserved scalar groups stay
    /// `absent_not_computed`.
    pub fn finish(self) -> Result<(), IoError> {
        Ok(())
    }
}

impl ScalarMldumpStreamV1 {
    /// Open `/orbitals` and write shared attributes. Spin/$k$ records follow.
    pub fn begin_orbitals(&mut self, begin: &ScalarOrbitalsBeginV1) -> Result<(), IoError> {
        self.require_idle("begin_orbitals")?;
        if self.orbital_summary.is_some() {
            return Err(section_already_written("/orbitals"));
        }
        scalar_orbitals::begin_scalar_orbitals(&self.file, begin)?;
        self.orbitals_band_window = begin.band_window_count;
        self.orbitals_spin_count = begin.spin_count;
        self.orbital_summary = Some(scalar_response::OrbitalAlignmentSummary::new(
            begin.spin_count,
            self.header.mesh.k_points.len(),
            begin.band_window_count,
        ));
        self.phase = ScalarStreamPhase::Orbitals {
            next_spin: 0,
            next_k: 0,
        };
        Ok(())
    }

    /// Write one spin/$k$ orbital record immediately.
    pub fn write_orbital_k(
        &mut self,
        spin: usize,
        record: &ScalarOrbitalKRefV1<'_>,
    ) -> Result<(), IoError> {
        let (next_spin, next_k) = match self.phase {
            ScalarStreamPhase::Orbitals { next_spin, next_k } => (next_spin, next_k),
            _ => {
                return Err(stream_phase_error(
                    "write_orbital_k",
                    "orbitals section in progress",
                ));
            }
        };
        require_record_capacity("orbitals.record", next_spin, self.orbitals_spin_count)?;
        require_record_capacity("orbitals.record", next_k, self.header.mesh.k_points.len())?;
        if spin != next_spin || record.k_index != next_k {
            return Err(ValidationError::InvalidValue {
                path: "orbitals.record".to_owned(),
                expected: format!("spin={next_spin} k={next_k}"),
                actual: format!("spin={spin} k={}", record.k_index),
            }
            .into());
        }
        scalar_orbitals::write_scalar_orbital_k(
            &self.file,
            &self.header,
            spin,
            self.orbitals_band_window,
            record,
        )?;
        let n_k = self.header.mesh.k_points.len();
        let (next_spin, next_k) = if next_k + 1 == n_k {
            (next_spin + 1, 0)
        } else {
            (next_spin, next_k + 1)
        };
        self.phase = ScalarStreamPhase::Orbitals { next_spin, next_k };
        Ok(())
    }

    /// Close `/orbitals` after every spin/$k$ record has been written.
    pub fn finish_orbitals(&mut self) -> Result<(), IoError> {
        let ScalarStreamPhase::Orbitals { next_spin, next_k } = self.phase else {
            return Err(stream_phase_error(
                "finish_orbitals",
                "orbitals section in progress",
            ));
        };
        if next_spin != self.orbitals_spin_count || next_k != 0 {
            return Err(ValidationError::InvalidValue {
                path: "orbitals".to_owned(),
                expected: format!(
                    "{} spins × {} k records",
                    self.orbitals_spin_count,
                    self.header.mesh.k_points.len()
                ),
                actual: format!("next spin={next_spin} k={next_k}"),
            }
            .into());
        }
        self.phase = ScalarStreamPhase::Start;
        Ok(())
    }

    /// Open `/products` and write shared partition binding.
    pub fn begin_products(&mut self, begin: &ScalarProductsBeginV1<'_>) -> Result<(), IoError> {
        self.require_idle("begin_products")?;
        if self.product_summary.is_some() {
            return Err(section_already_written("/products"));
        }
        scalar_products::begin_scalar_products(&self.file, &self.header, begin)?;
        self.products_n_site = self.header.geometry.sites.len();
        self.product_summary = Some(scalar_response::ProductAlignmentSummary::new(
            begin.n_k,
            begin.n_orb,
        ));
        self.phase = ScalarStreamPhase::Products {
            next_site: 0,
            next_q: 0,
        };
        Ok(())
    }

    /// Write one site radial record immediately.
    pub fn write_product_site(
        &mut self,
        record: &ScalarProductSiteRefV1<'_>,
    ) -> Result<(), IoError> {
        let next_site = match self.phase {
            ScalarStreamPhase::Products {
                next_site,
                next_q: 0,
            } => next_site,
            _ => {
                return Err(stream_phase_error(
                    "write_product_site",
                    "product sites before q records",
                ));
            }
        };
        require_record_capacity("products.sites", next_site, self.products_n_site)?;
        if record.site_index != next_site {
            return Err(ValidationError::InvalidValue {
                path: "products.sites".to_owned(),
                expected: next_site.to_string(),
                actual: record.site_index.to_string(),
            }
            .into());
        }
        scalar_products::write_scalar_product_site(&self.file, &self.header, next_site, record)?;
        self.phase = ScalarStreamPhase::Products {
            next_site: next_site + 1,
            next_q: 0,
        };
        Ok(())
    }

    /// Write one positional product $q$ record immediately.
    pub fn write_product_q(
        &mut self,
        record: &ScalarProductQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        let (next_site, next_q) = match self.phase {
            ScalarStreamPhase::Products { next_site, next_q } => (next_site, next_q),
            _ => {
                return Err(stream_phase_error(
                    "write_product_q",
                    "products section in progress",
                ));
            }
        };
        if next_site != self.products_n_site {
            return Err(ValidationError::InvalidValue {
                path: "products.q_records".to_owned(),
                expected: format!("{} site records first", self.products_n_site),
                actual: format!("{next_site} sites written"),
            }
            .into());
        }
        require_record_capacity(
            "products.q_records",
            next_q,
            self.header.mesh.q_entries.len(),
        )?;
        if record.q_index != next_q {
            return Err(ValidationError::InvalidValue {
                path: "products.q_records".to_owned(),
                expected: next_q.to_string(),
                actual: record.q_index.to_string(),
            }
            .into());
        }
        scalar_products::write_scalar_product_q(&self.file, next_q, record)?;
        if let Some(summary) = self.product_summary.as_mut() {
            summary.push_q(record);
        }
        self.phase = ScalarStreamPhase::Products {
            next_site,
            next_q: next_q + 1,
        };
        Ok(())
    }

    /// Close `/products` after every site and $q$ record has been written.
    pub fn finish_products(&mut self) -> Result<(), IoError> {
        let ScalarStreamPhase::Products { next_site, next_q } = self.phase else {
            return Err(stream_phase_error(
                "finish_products",
                "products section in progress",
            ));
        };
        let n_q = self.header.mesh.q_entries.len();
        if next_site != self.products_n_site || next_q != n_q {
            return Err(ValidationError::InvalidValue {
                path: "products".to_owned(),
                expected: format!("{} sites and {n_q} q records", self.products_n_site),
                actual: format!("sites={next_site} q={next_q}"),
            }
            .into());
        }
        self.phase = ScalarStreamPhase::Start;
        Ok(())
    }

    /// Open `/thc` and write the shared parent grid and selection.
    pub fn begin_thc(&mut self, begin: &ScalarThcBeginV1<'_>) -> Result<(), IoError> {
        self.require_idle("begin_thc")?;
        if self.thc_summary.is_some() {
            return Err(section_already_written("/thc"));
        }
        if self.product_summary.is_none() {
            return Err(ValidationError::InvalidValue {
                path: "/thc".to_owned(),
                expected: "products section written before thc".to_owned(),
                actual: "products summary missing".to_owned(),
            }
            .into());
        }
        scalar_response::begin_scalar_thc(&self.file, &self.header, begin)?;
        self.thc_n_parent = begin.parent_grid.n_points;
        self.thc_effective_rank = begin.effective_rank;
        self.thc_summary = Some(scalar_response::ThcAlignmentSummary::new());
        self.phase = ScalarStreamPhase::Thc { next_q: 0 };
        Ok(())
    }

    /// Write one THC $q$ record immediately.
    pub fn write_thc_q(&mut self, record: &ScalarThcQRecordRefV1<'_>) -> Result<(), IoError> {
        let next_q = match self.phase {
            ScalarStreamPhase::Thc { next_q } => next_q,
            _ => return Err(stream_phase_error("write_thc_q", "thc section in progress")),
        };
        require_record_capacity("thc.q_records", next_q, self.header.mesh.q_entries.len())?;
        if record.q_index != next_q {
            return Err(ValidationError::InvalidValue {
                path: "thc.q_records".to_owned(),
                expected: next_q.to_string(),
                actual: record.q_index.to_string(),
            }
            .into());
        }
        let (n_k, n_orb) = match self.product_summary.as_ref() {
            Some(products) => (products.n_k, products.n_orb),
            None => {
                return Err(ValidationError::InvalidValue {
                    path: "thc.q_records".to_owned(),
                    expected: "products section written before thc".to_owned(),
                    actual: "products summary missing".to_owned(),
                }
                .into());
            }
        };
        scalar_response::write_scalar_thc_q(
            &self.file,
            next_q,
            self.thc_n_parent,
            self.thc_effective_rank,
            n_k,
            n_orb,
            record,
        )?;
        if let Some(summary) = self.thc_summary.as_mut() {
            summary.push_q(record);
        }
        self.phase = ScalarStreamPhase::Thc { next_q: next_q + 1 };
        Ok(())
    }

    /// Close `/thc` after every mesh $q$ record has been written.
    pub fn finish_thc(&mut self) -> Result<(), IoError> {
        let ScalarStreamPhase::Thc { next_q } = self.phase else {
            return Err(stream_phase_error("finish_thc", "thc section in progress"));
        };
        let n_q = self.header.mesh.q_entries.len();
        if next_q != n_q {
            return Err(ValidationError::InvalidValue {
                path: "thc".to_owned(),
                expected: format!("{n_q} q records"),
                actual: next_q.to_string(),
            }
            .into());
        }
        self.phase = ScalarStreamPhase::Start;
        Ok(())
    }

    /// Open `/coulomb` and write request/projection attributes.
    pub fn begin_coulomb(&mut self, begin: &ScalarCoulombBeginV1) -> Result<(), IoError> {
        self.require_idle("begin_coulomb")?;
        if self.coulomb_summary.is_some() {
            return Err(section_already_written("/coulomb"));
        }
        scalar_response::begin_scalar_coulomb(&self.file, begin)?;
        self.coulomb_summary = Some(scalar_response::CoulombAlignmentSummary::new());
        self.phase = ScalarStreamPhase::Coulomb { next_q: 0 };
        Ok(())
    }

    /// Write one Coulomb $q$ record immediately.
    pub fn write_coulomb_q(
        &mut self,
        record: &ScalarCoulombQRecordRefV1<'_>,
    ) -> Result<(), IoError> {
        let next_q = match self.phase {
            ScalarStreamPhase::Coulomb { next_q } => next_q,
            _ => {
                return Err(stream_phase_error(
                    "write_coulomb_q",
                    "coulomb section in progress",
                ));
            }
        };
        require_record_capacity(
            "coulomb.q_records",
            next_q,
            self.header.mesh.q_entries.len(),
        )?;
        if record.q_index != next_q {
            return Err(ValidationError::InvalidValue {
                path: "coulomb.q_records".to_owned(),
                expected: next_q.to_string(),
                actual: record.q_index.to_string(),
            }
            .into());
        }
        scalar_response::write_scalar_coulomb_q(&self.file, next_q, record)?;
        if let Some(summary) = self.coulomb_summary.as_mut() {
            summary.push_q(record);
        }
        self.phase = ScalarStreamPhase::Coulomb { next_q: next_q + 1 };
        Ok(())
    }

    /// Close `/coulomb` after every mesh $q$ record has been written.
    pub fn finish_coulomb(&mut self) -> Result<(), IoError> {
        let ScalarStreamPhase::Coulomb { next_q } = self.phase else {
            return Err(stream_phase_error(
                "finish_coulomb",
                "coulomb section in progress",
            ));
        };
        let n_q = self.header.mesh.q_entries.len();
        if next_q != n_q {
            return Err(ValidationError::InvalidValue {
                path: "coulomb".to_owned(),
                expected: format!("{n_q} q records"),
                actual: next_q.to_string(),
            }
            .into());
        }
        self.phase = ScalarStreamPhase::Start;
        Ok(())
    }

    /// Finish the populated scalar file after all four sections.
    pub fn finish(self) -> Result<(), IoError> {
        if self.phase != ScalarStreamPhase::Start {
            return Err(stream_phase_error("finish", "no section left open"));
        }
        match (
            self.orbital_summary.as_ref(),
            self.product_summary.as_ref(),
            self.thc_summary.as_ref(),
            self.coulomb_summary.as_ref(),
        ) {
            (Some(orbitals), Some(products), Some(thc), Some(coulomb)) => {
                scalar_response::validate_scalar_alignment(
                    &self.header,
                    orbitals,
                    products,
                    thc,
                    coulomb,
                )
            }
            _ => Err(ValidationError::InvalidValue {
                path: "scalar".to_owned(),
                expected: "alignment summaries for all four written sections".to_owned(),
                actual: "missing retained summary".to_owned(),
            }
            .into()),
        }
    }

    fn require_idle(&self, method: &str) -> Result<(), IoError> {
        if self.phase == ScalarStreamPhase::Start {
            Ok(())
        } else {
            Err(stream_phase_error(method, "no section currently open"))
        }
    }
}

fn stream_phase_error(method: &str, expected: &str) -> IoError {
    ValidationError::InvalidValue {
        path: "scalar".to_owned(),
        expected: expected.to_owned(),
        actual: format!("{method} in unexpected session state"),
    }
    .into()
}

fn require_record_capacity(path: &str, next: usize, expected: usize) -> Result<(), IoError> {
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

fn section_already_written(path: &str) -> IoError {
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
    let scalar = match n_present {
        0 => {
            require_absent_status_and_payload(&file, GROUP_ORBITALS, orbitals_status)?;
            require_absent_status_and_payload(&file, GROUP_PRODUCTS, products_status)?;
            require_absent_status_and_payload(&file, GROUP_THC, thc_status)?;
            require_absent_status_and_payload(&file, GROUP_COULOMB, coulomb_status)?;
            None
        }
        4 => {
            let scalar = ScalarMldumpV1 {
                orbitals: scalar_orbitals::read_scalar_orbitals(&file, &header)?,
                products: scalar_products::read_scalar_products(&file, &header)?,
                thc: scalar_response::read_scalar_thc(&file, &header)?,
                coulomb: scalar_response::read_scalar_coulomb(&file, &header)?,
            };
            scalar_response::validate_owned_thc_vertex_identity(
                scalar.products.n_k,
                scalar.products.n_orb,
                &scalar.thc,
            )?;
            scalar_response::validate_scalar_alignment(
                &header,
                &scalar_response::OrbitalAlignmentSummary::from_owned(&scalar.orbitals),
                &scalar_response::ProductAlignmentSummary::from_owned(&scalar.products),
                &scalar_response::ThcAlignmentSummary::from_owned(&scalar.thc),
                &scalar_response::CoulombAlignmentSummary::from_owned(&scalar.coulomb),
            )?;
            Some(scalar)
        }
        _ => {
            return Err(ValidationError::InvalidValue {
                path: "scalar".to_owned(),
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
        scalar,
        exchange,
    })
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
