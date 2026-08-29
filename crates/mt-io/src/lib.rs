//! Explicit, versioned, human-diffable interchange formats.
//!
//! [`SnapshotFile`] dispatches legacy scalar/collinear [`SnapshotV1`] and
//! noncollinear Pauli-field [`SnapshotV2`] files. [`GridArtifactV1`] is
//! deliberately a separate format for materialized integration grids and is
//! never embedded in a snapshot. [`MldumpV1`] is the libmuffintin-owned
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
    MLDUMP_SCHEMA_NAME, MLDUMP_SCHEMA_VERSION, MLDUMP_STATUS_ABSENT_NOT_COMPUTED,
    MLDUMP_STATUS_PRESENT, MLDUMP_UNIT_ENERGY, MLDUMP_UNIT_G_UMKLAPP, MLDUMP_UNIT_INVERSE_LENGTH,
    MLDUMP_UNIT_K_Q, MLDUMP_UNIT_LENGTH, MLDUMP_UNIT_VOLUME, MldumpGeometryV1, MldumpKMinusQV1,
    MldumpKPointV1, MldumpMeshV1, MldumpMetaV1, MldumpQEntryV1, MldumpRadialMeshV1, MldumpSiteV1,
    MldumpStatus, MldumpStatusesV1, MldumpV1, read_mldump_v1, write_mldump_v1,
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
