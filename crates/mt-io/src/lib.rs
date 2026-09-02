//! Explicit, versioned, human-diffable interchange formats.
//!
//! [`CheckpointFile`] dispatches legacy scalar/collinear [`CheckpointV1`] and
//! noncollinear Pauli-field [`CheckpointV2`] files. [`GridArtifactV1`] is
//! deliberately a separate format for materialized integration grids and is
//! never embedded in a checkpoint. [`MldumpFileV1`] is the libmuffintin-owned
//! MLDUMP v1 HDF5 schema; [`MldumpFileV2`] keeps the complete v1 spinor common
//! payload and adds the core-aware exchange summary. Neither is CoQui-native
//! or SPEX-native.
//! [`read_spex_snapshot_hdf`] reads SPEX-owned `spex.snapshot_hdf` v1
//! frozen fields; [`materialize_checkpoint_v2`] builds [`CheckpointV2`] only
//! with an explicit signed-$\kappa$ recipe and a tight Hermitian ingest of
//! interstitial Fourier pairs. Populated
//! files are written through [`ScalarMldumpStreamV1`] or
//! [`SpinorMldumpStreamV1`]. [`CoquiCholeskyFile`] is a separate CoQui-native
//! single-file Cholesky ERI tree and is not MLDUMP.

mod checkpoint;
mod checkpoint_v2;
mod coqui_cholesky;
mod error;
mod grid;
mod mldump;
mod spex_snapshot;
mod spex_symmetry;
mod units;

pub use checkpoint::{
    AngularBasis, BasisHints, CHECKPOINT_FORMAT, CHECKPOINT_VERSION, CheckpointMeta, CheckpointV1,
    Complex64V1, EnergyParameterV1, ExponentialMeshSpec, FourierCoefficientV1,
    FourierNormalization, FourierPhase, GeometryV1, InterstitialV1, LatticeV1, LinearizationV1,
    PotentialChannelV1, PotentialConventionV1, PotentialRadialQuantityV1, RadialEquationTag,
    SiteSpinV1, SiteV1, SphericalChannelConvention, SpinTag, checkpoint_from_toml,
    checkpoint_to_toml,
};
pub use checkpoint_v2::{
    CHECKPOINT_VERSION_V2, CheckpointFile, CheckpointV2, Complex64V2, DensityV2,
    FieldRepresentationV2, FieldUnitV2, FourierCoefficientV2, GeometryV2, InitialV2,
    InterstitialFieldV2, MuffinTinFieldV2, PotentialV2, RadialBasisSpinV2, RegionalFieldV2,
    SiteRadialBasisV2, SiteV2, SphericalChannelV2, checkpoint_file_from_toml,
    checkpoint_file_to_toml,
};
pub use coqui_cholesky::{
    COQUI_CHOLESKY_COMPLEX_ATTR, COQUI_CHOLESKY_COMPLEX_VALUE, COQUI_CHOLESKY_GROUP,
    CoquiCholeskyFile, CoquiCholeskyHeader, CoquiCholeskyVq, CoquiCholeskyVqRef,
    CoquiCholeskyWriter, read_coqui_cholesky,
};
pub use error::{IoError, ValidationError};
pub use grid::{
    GRID_ARTIFACT_FORMAT, GRID_ARTIFACT_VERSION, GridArtifactV1, grid_artifact_from_toml,
    grid_artifact_to_toml,
};
pub use mldump::{
    ComplexF64V1, MLDUMP_CORE_EMPTY_NOT_FITTED, MLDUMP_EXCHANGE_BACKEND_V2,
    MLDUMP_EXCHANGE_SOURCE_FRAME_V2, MLDUMP_EXCHANGE_TOTAL_RELATION_V2,
    MLDUMP_INTERSTITIAL_SENTINEL, MLDUMP_OCCUPATIONS_NOT_EXPORTED, MLDUMP_PAIR_ORDER_K_LEFT_RIGHT,
    MLDUMP_PARENT_REGION_INTERSTITIAL, MLDUMP_PARENT_REGION_MUFFIN_TIN, MLDUMP_RADIAL_KIND_CORE,
    MLDUMP_RADIAL_KIND_VALENCE, MLDUMP_REPRESENTATION_SCALAR_KOELLING_HARMON,
    MLDUMP_REPRESENTATION_SPINOR_FULL_FIRST_VARIATION, MLDUMP_SCHEMA_NAME, MLDUMP_SCHEMA_VERSION,
    MLDUMP_SCHEMA_VERSION_V1, MLDUMP_SCHEMA_VERSION_V2, MLDUMP_STATUS_ABSENT_NOT_COMPUTED,
    MLDUMP_STATUS_PRESENT, MLDUMP_THC_ENGINE_PIVOTED_CHOLESKY, MLDUMP_THC_ENGINE_QRCP,
    MLDUMP_THC_STRATEGY_ALL_QL2, MLDUMP_UNIT_ENERGY, MLDUMP_UNIT_G_UMKLAPP,
    MLDUMP_UNIT_INVERSE_LENGTH, MLDUMP_UNIT_K_Q, MLDUMP_UNIT_LENGTH, MLDUMP_UNIT_VOLUME,
    MldumpCoreOccupationV2, MldumpCoulombBeginV1, MldumpCoulombGammaRefV1, MldumpCoulombGammaV1,
    MldumpCoulombQRecordRefV1, MldumpCoulombQRecordV1, MldumpCoulombV1,
    MldumpExchangeFitResidualV2, MldumpExchangeLayoutV2, MldumpExchangeMpbQuadraticV2,
    MldumpExchangeProvenanceV2, MldumpExchangeRankScalingV2, MldumpExchangeSectorV2,
    MldumpExchangeSpaceV2, MldumpExchangeStatusesV1, MldumpExchangeV2, MldumpFileV1, MldumpFileV2,
    MldumpGammaPolicyV2, MldumpGeometryV1, MldumpHeaderV1, MldumpKMinusQV1, MldumpKPointV1,
    MldumpMeshV1, MldumpMetaV1, MldumpPayloadV1, MldumpQEntryV1, MldumpRadialMeshV1,
    MldumpRequestedRankV2, MldumpSelectorEngineV2, MldumpSelectorStrategyV2, MldumpSiteV1,
    MldumpStatus, MldumpThcBeginV1, MldumpThcParentGridRefV1, MldumpThcParentGridV1,
    MldumpThcQRecordRefV1, MldumpThcQRecordV1, MldumpThcResidualV1, MldumpThcSelectionRefV1,
    MldumpThcSelectionV1, MldumpThcV1, MldumpThcVertexTableRefV1, MldumpThcVertexV1,
    MldumpWriterV1, ScalarApwSiteMatchRefV1, ScalarApwSiteMatchV1, ScalarLocalOrbitalRowV1,
    ScalarLocalOrbitalTableRefV1, ScalarMldumpStreamV1, ScalarMldumpV1, ScalarOrbitalKRecordV1,
    ScalarOrbitalKRefV1, ScalarOrbitalSpinV1, ScalarOrbitalsBeginV1, ScalarOrbitalsV1,
    ScalarProductQRecordRefV1, ScalarProductQRecordV1, ScalarProductSiteRefV1, ScalarProductSiteV1,
    ScalarProductsBeginV1, ScalarProductsV1, SpinorLocalOrbitalRowV1, SpinorLocalOrbitalTableRefV1,
    SpinorMldumpStreamV1, SpinorMldumpV1, SpinorOrbitalKRecordV1, SpinorOrbitalKRefV1,
    SpinorOrbitalsBeginV1, SpinorOrbitalsV1, SpinorPauliRowMapRefV1, SpinorPauliRowMapV1,
    SpinorProductQRecordRefV1, SpinorProductQRecordV1, SpinorProductSiteRefV1, SpinorProductSiteV1,
    SpinorProductsBeginV1, SpinorProductsV1, SpinorProjectionCoordV1, SpinorSiteMatchRefV1,
    SpinorSiteMatchV1, read_mldump_v1, read_mldump_v2, upgrade_mldump_v1_with_exchange_v2,
};
pub use spex_symmetry::{
    SPEX_SYMMETRY_SCHEMA_NAME, SPEX_SYMMETRY_SCHEMA_VERSION, SpexSymmetryFileV1,
    read_spex_symmetry_v1, write_spex_symmetry_v1,
};

pub use spex_snapshot::{
    SPEX_FOURIER_HERMITIAN_TOLERANCE, SPEX_SNAPSHOT_HDF_SCHEMA_NAME,
    SPEX_SNAPSHOT_HDF_SCHEMA_VERSION, SPEX_SNAPSHOT_HDF_SOURCE_KIND, SpexFrozenFieldsV1,
    SpexMaterialBasisRecipeV1, SpexMaterialChannelKind, SpexMaterialChannelV1,
    SpexMaterializedSnapshotV1, SpexScalarLoKind, SpexScalarLoTableV1, SpexScalarLoV1,
    SpexSnapshotHashV1, materialize_checkpoint_v2, read_spex_snapshot_hdf, write_spex_snapshot_hdf,
};
pub use units::{EnergyUnit, InverseLengthUnit, LengthUnit, VolumeUnit};
