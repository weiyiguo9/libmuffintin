//! Versioned workflow input, preparation, and the unified libmuffintin runtime boundary.

#![forbid(unsafe_code)]

#[cfg(test)]
extern crate self as muffintin;

mod channel_recipe;
mod channel_token;
mod error;
mod input;
mod mldump_header;
mod q_mesh;
mod runner;
mod scalar_coulomb;
mod scalar_mldump;
mod scalar_mpb;
mod scalar_product;
mod scalar_thc;
mod site_coords;
mod snapshot_dft;
mod spinor_coulomb;
mod spinor_mldump;
mod spinor_mpb;
mod spinor_product;
mod spinor_thc;
mod thc_grid;

pub use channel_recipe::{
    ChannelRecipeArtifact, ChannelRecipeError, CompiledChannelRecipe, CompiledSiteRecipe,
    ExternalChannelRecipe, RecipeSite, channel_recipe_to_toml, compile_channel_recipe,
    parse_channel_recipe_toml,
};
pub use channel_token::{
    ChannelEnergyGenerator, ChannelIdentity, ChannelProvenance, ChannelRecipeRecord, ChannelScope,
    ChannelTokenContext, ChannelTokenError, ChannelTreatment, ParsedChannelToken,
    parse_channel_token,
};
pub use error::{InputError, InputValidationError};
pub use input::{
    BandPathPoint, Basis, BasisEnvelope, BasisEnvelopeKind, Convergence, EnergyWindow,
    ExchangeCorrelation, INPUT_FORMAT, INPUT_VERSION, Input, KMesh, Mixing, NoncollinearXcRoute,
    Occupations, Relativity, Task, TaskKind, Workflow, input_to_toml, parse_input_toml,
};
pub use muffintin_thc::RankPolicy;
pub use runner::{
    PreparedSource, PreparedTask, PreparedWorkflow, TaskResult, WorkflowResult,
    execute_prepared_with, load_input_path, prepare_input, prepare_input_with_recipes,
};
pub use scalar_coulomb::{
    SCALAR_COULOMB_EXACTNESS_FLOOR, ScalarCoulombDiscrepancy, ScalarCoulombError,
    ScalarCoulombPairDiagnostic, ScalarCoulombPairMatch, ScalarCoulombQRecord, ScalarCoulombResult,
    ScalarCoulombSpec, build_scalar_coulomb,
};
pub use scalar_mldump::{ScalarMldumpError, write_scalar_mldump};
pub use scalar_mpb::{
    SCALAR_MPB_NSPIN, ScalarMpbError, ScalarMpbPairVertex, ScalarMpbResult, ScalarMpbSelection,
    ScalarMpbSpec, build_scalar_mpb,
};
pub use scalar_product::{
    SCALAR_RADIAL_LO0, SCALAR_RADIAL_U, SCALAR_RADIAL_UDOT, ScalarBandWindow, ScalarFrozenOrbitals,
    ScalarKMinusQ, ScalarProductInput, ScalarSpinChannel,
};
pub use scalar_thc::{ScalarThcError, ScalarThcResult, ScalarThcSpec, build_scalar_thc};
pub use snapshot_dft::{SnapshotDftError, SnapshotDftPhysics, snapshot_v2_from_state};
pub use spinor_coulomb::{
    SPINOR_COULOMB_EXACTNESS_FLOOR, SpinorCoulombDiscrepancy, SpinorCoulombError,
    SpinorCoulombPairDiagnostic, SpinorCoulombPairMatch, SpinorCoulombQRecord, SpinorCoulombResult,
    SpinorCoulombSpec, build_spinor_coulomb,
};
pub use spinor_mldump::{SpinorMldumpError, write_spinor_mldump};
pub use spinor_mpb::{
    SPINOR_MPB_NSPIN, SpinorMpbError, SpinorMpbPairVertex, SpinorMpbResult, SpinorMpbSelection,
    SpinorMpbSpec, build_spinor_mpb,
};
pub use spinor_product::{
    SPINOR_RADIAL_LO0, SPINOR_RADIAL_P, SPINOR_RADIAL_PDOT, SpinorBandWindow, SpinorFrozenOrbitals,
    SpinorKMinusQ, SpinorProductInput,
};
pub use spinor_thc::{SpinorThcError, SpinorThcResult, SpinorThcSpec, build_spinor_thc};
pub use thc_grid::{
    ThcCandidates, ThcEngine, ThcGridError, ThcParentGrid, ThcPoint, ThcQRecord, ThcRegion,
};
