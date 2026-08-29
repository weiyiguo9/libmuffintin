//! Explicit, versioned, human-diffable interchange formats.
//!
//! [`SnapshotFile`] dispatches legacy scalar/collinear [`SnapshotV1`] and
//! noncollinear Pauli-field [`SnapshotV2`] files. [`GridArtifactV1`] is
//! deliberately a separate format for materialized integration grids and is
//! never embedded in a snapshot. [`MldumpFileV1`] is the libmuffintin-owned
//! MLDUMP v1 HDF5 schema; it is not CoQui-native or SPEX-native.

mod error;
mod grid;
mod mldump;
mod snapshot;
mod snapshot_v2;
mod units;

pub use error::{IoError, ValidationError};
pub use grid::{
    GRID_ARTIFACT_FORMAT, GRID_ARTIFACT_VERSION, GridArtifactV1, grid_artifact_from_toml,
    grid_artifact_to_toml,
};
pub use mldump::{
    ComplexF64V1, MLDUMP_CORE_EMPTY_NOT_FITTED, MLDUMP_INTERSTITIAL_SENTINEL,
    MLDUMP_OCCUPATIONS_NOT_EXPORTED, MLDUMP_PAIR_ORDER_K_LEFT_RIGHT,
    MLDUMP_PARENT_REGION_INTERSTITIAL, MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_RADIAL_KIND_CORE,
    MLDUMP_RADIAL_KIND_VALENCE, MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON, MLDUMP_SCHEMA_NAME,
    MLDUMP_SCHEMA_VERSION, MLDUMP_STATUS_ABSENT_NOT_COMPUTED, MLDUMP_STATUS_PRESENT,
    MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY, MLDUMP_THC_ENGINE_QRCP, MLDUMP_THC_STRATEGY_ALL_QL2,
    MLDUMP_UNIT_ENERGY, MLDUMP_UNIT_G_UMKLAPP, MLDUMP_UNIT_INVERSE_LENGTH, MLDUMP_UNIT_K_Q,
    MLDUMP_UNIT_LENGTH, MLDUMP_UNIT_VOLUME, MldumpExchangeStatusesV1, MldumpFileV1,
    MldumpGeometryV1, MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1, MldumpMeshV1, MldumpMetaV1,
    MldumpQEntryV1, MldumpRadialMeshV1, MldumpSiteV1, MldumpStatus, MldumpWriterV1,
    ScalarApwSiteMatchRefV1, ScalarApwSiteMatchV1, ScalarCoulombGammaRefV1, ScalarCoulombGammaV1,
    ScalarCoulombQRecordRefV1, ScalarCoulombQRecordV1, ScalarCoulombRefV1, ScalarCoulombV1,
    ScalarLocalOrbitalRowV1, ScalarLocalOrbitalTableRefV1, ScalarMldumpV1, ScalarOrbitalKRecordV1,
    ScalarOrbitalKRefV1, ScalarOrbitalSpinRefV1, ScalarOrbitalSpinV1, ScalarOrbitalsRefV1,
    ScalarOrbitalsV1, ScalarProductQRecordRefV1, ScalarProductQRecordV1, ScalarProductSiteRefV1,
    ScalarProductSiteV1, ScalarProductsRefV1, ScalarProductsV1, ScalarThcParentGridRefV1,
    ScalarThcParentGridV1, ScalarThcQRecordRefV1, ScalarThcQRecordV1, ScalarThcRefV1,
    ScalarThcResidualV1, ScalarThcSelectionRefV1, ScalarThcSelectionV1, ScalarThcV1,
    ScalarThcVertexTableRefV1, ScalarThcVertexV1, read_mldump_v1,
};
pub use snapshot::{
    AngularBasisV1, BasisHintsV1, Complex64V1, EnergyParameterV1, ExponentialMeshSpecV1,
    FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1, GeometryV1, InterstitialV1,
    LatticeV1, LinearizationV1, MetaV1, PotentialChannelV1, PotentialConventionV1,
    PotentialRadialQuantityV1, RadialEquationTagV1, SNAPSHOT_FORMAT, SNAPSHOT_VERSION, SiteSpinV1,
    SiteV1, SnapshotV1, SphericalChannelConventionV1, SpinTagV1, snapshot_from_toml,
    snapshot_to_toml,
};
pub use snapshot_v2::{
    Complex64V2, DensityV2, FieldRepresentationV2, FieldUnitV2, FourierCoefficientV2, GeometryV2,
    InitialV2, InterstitialFieldV2, MuffinTinFieldV2, PotentialV2, RadialBasisSpinV2,
    RegionalFieldV2, SNAPSHOT_VERSION_V2, SiteRadialBasisV2, SiteV2, SnapshotFile, SnapshotV2,
    SphericalChannelV2, snapshot_file_from_toml, snapshot_file_to_toml,
};
pub use units::{EnergyUnitV1, InverseLengthUnitV1, LengthUnitV1, VolumeUnitV1};
