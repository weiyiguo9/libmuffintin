//! Versioned workflow input, preparation, and the unified libmuffintin runtime boundary.

#![forbid(unsafe_code)]

mod channel_token;
mod error;
mod input;
mod runner;
mod snapshot_dft;

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
pub use runner::{
    PreparedSource, PreparedTask, PreparedWorkflow, TaskResult, WorkflowResult,
    execute_prepared_with, load_input_path, prepare_input,
};
pub use snapshot_dft::{SnapshotDftError, SnapshotDftPhysics, snapshot_v2_from_state};
