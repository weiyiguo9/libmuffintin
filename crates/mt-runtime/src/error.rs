use std::path::PathBuf;

use muffintin_core::Hartree;
use muffintin_io::IoError;
use thiserror::Error;

use crate::{ChannelEnergyGenerator, ChannelIdentity, ChannelRecipeError, TaskKind};

/// A syntactically valid input whose values violate the workflow contract.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum InputValidationError {
    #[error("checkpoint path must not be empty")]
    EmptyCheckpointPath,
    #[error("checkpoint path must be relative to the input file, got {path:?}")]
    AbsoluteCheckpointPath { path: PathBuf },
    #[error("basis recipe path must not be empty")]
    EmptyRecipePath,
    #[error("basis recipe path must be relative to the input file, got {path:?}")]
    AbsoluteRecipePath { path: PathBuf },
    #[error("workflow.tasks must not be empty")]
    EmptyWorkflow,
    #[error("invalid task id {id:?}; expected [A-Za-z][A-Za-z0-9_-]*")]
    InvalidTaskId { id: String },
    #[error("workflow.tasks contains duplicate task id {id:?}")]
    DuplicateTaskId { id: String },
    #[error("workflow task {id:?} has no matching [task.{id}] block")]
    MissingTaskBlock { id: String },
    #[error("[task.{id}] is not listed in workflow.tasks")]
    OrphanTaskBlock { id: String },
    #[error("task {task_id:?} has invalid source {source_ref:?}; expected <task-id>.<output>")]
    InvalidSource { task_id: String, source_ref: String },
    #[error("task {task_id:?} references unknown source task {source_task:?}")]
    MissingSourceTask {
        task_id: String,
        source_task: String,
    },
    #[error("task {task_id:?} source {source_task:?} must appear earlier in workflow.tasks")]
    ForwardSource {
        task_id: String,
        source_task: String,
    },
    #[error(
        "task {task_id:?} ({task_kind}) cannot consume source {source_ref:?}; expected an earlier dft-scf state output"
    )]
    IncompatibleSource {
        task_id: String,
        task_kind: TaskKind,
        source_ref: String,
    },
    #[error("{path} must not be empty")]
    Empty { path: String },
    #[error("{path} must be finite, got {value}")]
    NonFinite { path: String, value: f64 },
    #[error("{path} must be positive, got {value}")]
    NotPositive { path: String, value: f64 },
    #[error("{path} requires exactly one of g-cutoff or energy-cutoff")]
    MissingPlaneWaveCutoff { path: String },
    #[error("{path} accepts only one of g-cutoff or energy-cutoff")]
    ConflictingPlaneWaveCutoffs { path: String },
    #[error("{path} must be in the interval (0, 1], got {value}")]
    InvalidFraction { path: String, value: f64 },
    #[error("{path} must be nonzero")]
    Zero { path: String },
    #[error("{path} must contain at least {minimum} entries, got {actual}")]
    TooShort {
        path: String,
        minimum: usize,
        actual: usize,
    },
    #[error("{path}.minimum must be less than {path}.maximum, got {minimum} >= {maximum}")]
    InvalidRange {
        path: String,
        minimum: f64,
        maximum: f64,
    },
}

/// Input decoding, preparation, loading, or execution failure.
#[derive(Debug, Error)]
pub enum InputError {
    #[error("could not decode input TOML: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("could not encode input TOML: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("expected input format {expected:?}, found {found:?}")]
    InvalidFormat {
        expected: &'static str,
        found: String,
    },
    #[error("unsupported {format} version {found}; supported version is {supported}")]
    UnsupportedVersion {
        format: &'static str,
        supported: u32,
        found: u32,
    },
    #[error(
        "input version 1 requires migration to version = 3: replace plane-wave-cutoff with [task.<id>.basis.envelope] kind plus g-cutoff or energy-cutoff, and replace local-orbitals/state-overrides with [task.<id>.basis.channels]"
    )]
    V1MigrationRequired,
    #[error(
        "input version 2 requires migration to version = 3: replace [task.<id>.basis.envelope].cutoff with exactly one of g-cutoff or energy-cutoff"
    )]
    V2MigrationRequired,
    #[error(transparent)]
    Validation(#[from] InputValidationError),
    #[error("could not read input file {path:?}: {source}")]
    ReadInput {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not read checkpoint file {path:?}: {source}")]
    ReadCheckpoint {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid checkpoint file {path:?}: {source}")]
    Checkpoint {
        path: PathBuf,
        #[source]
        source: IoError,
    },
    #[error("invalid in-memory checkpoint: {0}")]
    InvalidCheckpoint(#[source] IoError),
    #[error("task {task_id:?} requires preloaded channel recipe artifact {path:?}")]
    MissingRecipeArtifact { task_id: String, path: PathBuf },
    #[error("task {task_id:?} could not read channel recipe file {path:?}: {source}")]
    ReadRecipe {
        task_id: String,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("task {task_id:?} has an invalid channel recipe at {path:?}: {source}")]
    ChannelRecipe {
        task_id: String,
        path: Option<PathBuf>,
        #[source]
        source: Box<ChannelRecipeError>,
    },
    #[error("task {task_id:?} has no FLEUR atomic default for Z={atomic_number} on site {site:?}")]
    UnsupportedAtomicNumber {
        task_id: String,
        site: String,
        atomic_number: u16,
    },
    #[error(
        "task {task_id:?} site {site:?} core channel {identity:?} has no matching occupation in the FLEUR neutral-atom default for Z={atomic_number}"
    )]
    MissingCoreOccupation {
        task_id: String,
        site: String,
        atomic_number: u8,
        identity: ChannelIdentity,
    },
    #[error(
        "task {task_id:?} cannot inject built-in base valence channel {identity:?} on site {site:?} under task-level explicit generation because the channel has no explicit Hartree seed; add explicit valence coverage for this angular channel"
    )]
    MissingExplicitBaseValenceSeed {
        task_id: String,
        site: String,
        identity: ChannelIdentity,
    },
    #[error(
        "task {task_id:?} site {site:?} has inconsistent built-in valence partners for n={n}, l={l}: first generator/seed are {first_generator:?}/{first_seed:?}, conflicting generator/seed are {conflicting_generator:?}/{conflicting_seed:?}"
    )]
    InconsistentBuiltInValencePartners {
        task_id: String,
        site: String,
        n: u32,
        l: u32,
        first_generator: ChannelEnergyGenerator,
        first_seed: Option<Hartree>,
        conflicting_generator: ChannelEnergyGenerator,
        conflicting_seed: Option<Hartree>,
    },
    #[error(
        "task {task_id:?} site {site:?} channel {identity:?} requests derivative order {derivative_order}, which is not implemented"
    )]
    DerivativeOrderNotImplemented {
        task_id: String,
        site: String,
        identity: ChannelIdentity,
        derivative_order: u32,
    },
    #[error("task {task_id:?} could not consume its prepared SCF state source")]
    UnavailableScfSource { task_id: String },
    #[error("task {task_id:?} ({kind}) failed: {source}")]
    TaskExecution {
        task_id: String,
        kind: TaskKind,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

pub(crate) fn finite(path: impl Into<String>, value: f64) -> Result<(), InputValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(InputValidationError::NonFinite {
            path: path.into(),
            value,
        })
    }
}

pub(crate) fn positive(path: impl Into<String>, value: f64) -> Result<(), InputValidationError> {
    let path = path.into();
    finite(path.clone(), value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(InputValidationError::NotPositive { path, value })
    }
}

pub(crate) fn fraction(path: impl Into<String>, value: f64) -> Result<(), InputValidationError> {
    let path = path.into();
    finite(path.clone(), value)?;
    if value > 0.0 && value <= 1.0 {
        Ok(())
    } else {
        Err(InputValidationError::InvalidFraction { path, value })
    }
}

pub(crate) fn nonempty(path: impl Into<String>, value: &str) -> Result<(), InputValidationError> {
    if value.trim().is_empty() {
        Err(InputValidationError::Empty { path: path.into() })
    } else {
        Ok(())
    }
}
