//! Versioned workflow input, preparation, and the unified libmuffintin runtime boundary.

#![forbid(unsafe_code)]

mod channel_recipe;
mod channel_token;
mod error;
mod input;
mod runner;
mod scalar_mpb;
mod scalar_product;
mod scalar_thc;
mod site_coords;
mod snapshot_dft;

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
pub use scalar_mpb::{
    SCALAR_MPB_NSPIN, ScalarMpbError, ScalarMpbPairVertex, ScalarMpbResult, ScalarMpbSelection,
    ScalarMpbSpec, build_scalar_mpb,
};
pub use scalar_product::{
    SCALAR_RADIAL_LO0, SCALAR_RADIAL_U, SCALAR_RADIAL_UDOT, ScalarBandWindow, ScalarFrozenOrbitals,
    ScalarKMinusQ, ScalarProductInput, ScalarSpinChannel,
};
pub use scalar_thc::{
    ScalarThcCandidates, ScalarThcEngine, ScalarThcError, ScalarThcGrid, ScalarThcPoint,
    ScalarThcQRecord, ScalarThcRegion, ScalarThcResult, ScalarThcSpec, build_scalar_thc,
};
pub use snapshot_dft::{SnapshotDftError, SnapshotDftPhysics, snapshot_v2_from_state};
