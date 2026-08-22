//! Versioned workflow input, preparation, and the unified libmuffintin runtime boundary.

#![forbid(unsafe_code)]

mod error;
mod input;
mod runner;
mod snapshot_dft;

pub use error::{InputError, InputValidationError};
pub use input::{
    BandPathPointV1, BasisV1, ConvergenceV1, CoreStateV1, EnergyWindowV1, ExchangeCorrelationV1,
    INPUT_FORMAT, INPUT_VERSION, InputV1, KMeshV1, LocalOrbitalKindV1, LocalOrbitalV1, MixingV1,
    OccupationsV1, RelativityV1, TaskKindV1, TaskV1, WorkflowV1, input_to_toml, parse_input_toml,
};
pub use runner::{
    PreparedSource, PreparedTask, PreparedWorkflow, TaskResult, WorkflowResult,
    execute_prepared_with, load_input_path, prepare_input,
};
pub use snapshot_dft::{SnapshotDftError, SnapshotDftPhysics};
