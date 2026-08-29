//! Explicit, versioned, human-diffable interchange formats.
//!
//! [`SnapshotFile`] dispatches legacy scalar/collinear [`SnapshotV1`] and
//! noncollinear Pauli-field [`SnapshotV2`] files. [`GridArtifactV1`] is
//! deliberately a separate format for materialized integration grids and is
//! never embedded in a snapshot. [`MldumpFileV1`] is the libmuffintin-owned
//! MLDUMP v1 HDF5 schema; it is not CoQui-native or SPEX-native. Populated
//! files are written through [`ScalarMldumpStreamV1`] or
//! [`SpinorMldumpStreamV1`].

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
    MLDUMP_RADIAL_KIND_VALENCE, MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON,
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, MLDUMP_SCHEMA_NAME, MLDUMP_SCHEMA_VERSION,
    MLDUMP_STATUS_ABSENT_NOT_COMPUTED, MLDUMP_STATUS_PRESENT, MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY,
    MLDUMP_THC_ENGINE_QRCP, MLDUMP_THC_STRATEGY_ALL_QL2, MLDUMP_UNIT_ENERGY, MLDUMP_UNIT_G_UMKLAPP,
    MLDUMP_UNIT_INVERSE_LENGTH, MLDUMP_UNIT_K_Q, MLDUMP_UNIT_LENGTH, MLDUMP_UNIT_VOLUME,
    MldumpCoulombBeginV1, MldumpCoulombGammaRefV1, MldumpCoulombGammaV1, MldumpCoulombQRecordRefV1,
    MldumpCoulombQRecordV1, MldumpCoulombV1, MldumpExchangeStatusesV1, MldumpFileV1,
    MldumpGeometryV1, MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1, MldumpMeshV1, MldumpMetaV1,
    MldumpPayloadV1, MldumpQEntryV1, MldumpRadialMeshV1, MldumpSiteV1, MldumpStatus,
    MldumpThcBeginV1, MldumpThcParentGridRefV1, MldumpThcParentGridV1, MldumpThcQRecordRefV1,
    MldumpThcQRecordV1, MldumpThcResidualV1, MldumpThcSelectionRefV1, MldumpThcSelectionV1,
    MldumpThcV1, MldumpThcVertexTableRefV1, MldumpThcVertexV1, MldumpWriterV1,
    ScalarApwSiteMatchRefV1, ScalarApwSiteMatchV1, ScalarLocalOrbitalRowV1,
    ScalarLocalOrbitalTableRefV1, ScalarMldumpStreamV1, ScalarMldumpV1, ScalarOrbitalKRecordV1,
    ScalarOrbitalKRefV1, ScalarOrbitalSpinV1, ScalarOrbitalsBeginV1, ScalarOrbitalsV1,
    ScalarProductQRecordRefV1, ScalarProductQRecordV1, ScalarProductSiteRefV1, ScalarProductSiteV1,
    ScalarProductsBeginV1, ScalarProductsV1, SpinorLocalOrbitalRowV1, SpinorLocalOrbitalTableRefV1,
    SpinorMldumpStreamV1, SpinorMldumpV1, SpinorOrbitalKRecordV1, SpinorOrbitalKRefV1,
    SpinorOrbitalsBeginV1, SpinorOrbitalsV1, SpinorPauliRowMapRefV1, SpinorPauliRowMapV1,
    SpinorProductQRecordRefV1, SpinorProductQRecordV1, SpinorProductSiteRefV1, SpinorProductSiteV1,
    SpinorProductsBeginV1, SpinorProductsV1, SpinorProjectionCoordV1, SpinorSiteMatchRefV1,
    SpinorSiteMatchV1, read_mldump_v1,
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
