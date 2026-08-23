use std::path::PathBuf;

use muffintin_io::IoError;
use thiserror::Error;

use crate::TaskKindV1;

/// A syntactically valid input whose values violate the V1 workflow contract.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum InputValidationError {
    #[error("snapshot path must not be empty")]
    EmptySnapshotPath,
    #[error("snapshot path must be relative to the input file, got {path:?}")]
    AbsoluteSnapshotPath { path: PathBuf },
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
        task_kind: TaskKindV1,
        source_ref: String,
    },
    #[error("{path} must not be empty")]
    Empty { path: String },
    #[error("{path} must be finite, got {value}")]
    NonFinite { path: String, value: f64 },
    #[error("{path} must be positive, got {value}")]
    NotPositive { path: String, value: f64 },
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
    #[error(
        "{path}.kappa must be +1, +2, or +3 for relativistic-local-orbital treatment, got {kappa}"
    )]
    InvalidRelativisticLocalOrbitalKappa { path: String, kappa: i32 },
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
    #[error(transparent)]
    Validation(#[from] InputValidationError),
    #[error("could not read input file {path:?}: {source}")]
    ReadInput {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not read snapshot file {path:?}: {source}")]
    ReadSnapshot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid snapshot file {path:?}: {source}")]
    Snapshot {
        path: PathBuf,
        #[source]
        source: IoError,
    },
    #[error("invalid in-memory snapshot: {0}")]
    InvalidSnapshot(#[source] IoError),
    #[error("task {task_id:?} names unknown electronic-state override site {site:?}")]
    UnknownElectronicStateSite { task_id: String, site: String },
    #[error("task {task_id:?} has no FLEUR atomic default for Z={atomic_number} on site {site:?}")]
    UnsupportedAtomicNumber {
        task_id: String,
        site: String,
        atomic_number: u16,
    },
    #[error(
        "task {task_id:?} override names an unoccupied or invalid state n={principal_quantum_number}, kappa={kappa} on site {site:?}"
    )]
    UnknownElectronicState {
        task_id: String,
        site: String,
        principal_quantum_number: u32,
        kappa: i32,
    },
    #[error(
        "task {task_id:?} contains duplicate overrides for n={principal_quantum_number}, kappa={kappa} on site {site:?}"
    )]
    DuplicateElectronicStateOverride {
        task_id: String,
        site: String,
        principal_quantum_number: u32,
        kappa: i32,
    },
    #[error("task {task_id:?} names unknown local-orbital site {site:?}")]
    UnknownLocalOrbitalSite { task_id: String, site: String },
    #[error("task {task_id:?} could not consume its prepared SCF state source")]
    UnavailableScfSource { task_id: String },
    #[error("task {task_id:?} ({kind}) failed: {source}")]
    TaskExecution {
        task_id: String,
        kind: TaskKindV1,
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
