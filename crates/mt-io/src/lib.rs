//! Explicit, versioned, human-diffable interchange formats.
//!
//! [`SnapshotV1`] contains the physical input needed to reconstruct a
//! muffin-tin calculation. [`GridArtifactV1`] is deliberately a separate
//! format for materialized integration grids and is never embedded in a
//! snapshot.

mod error;
mod grid;
mod snapshot;
mod units;

pub use error::{IoError, ValidationError};
pub use grid::{
    GRID_ARTIFACT_FORMAT, GRID_ARTIFACT_VERSION, GridArtifactV1, grid_artifact_from_toml,
    grid_artifact_to_toml,
};
pub use snapshot::{
    AngularBasisV1, BasisHintsV1, Complex64V1, EnergyParameterV1, ExponentialMeshSpecV1,
    FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1, GeometryV1, InterstitialV1,
    LatticeV1, LinearizationV1, MetaV1, PotentialChannelV1, PotentialConventionV1,
    PotentialRadialQuantityV1, RadialEquationTagV1, SNAPSHOT_FORMAT, SNAPSHOT_VERSION, SiteSpinV1,
    SiteV1, SnapshotV1, SphericalChannelConventionV1, SpinTagV1, snapshot_from_toml,
    snapshot_to_toml,
};
pub use units::{EnergyUnitV1, InverseLengthUnitV1, LengthUnitV1, VolumeUnitV1};
