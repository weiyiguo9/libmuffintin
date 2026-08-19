use thiserror::Error;

/// Syntax, version, or semantic failure while reading or writing an artifact.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("could not decode TOML: {0}")]
    Decode(#[from] toml::de::Error),
    #[error("could not encode TOML: {0}")]
    Encode(#[from] toml::ser::Error),
    #[error("expected format {expected:?}, found {found:?}")]
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
    Validation(#[from] ValidationError),
}

/// A structurally valid TOML document that violates the format contract.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ValidationError {
    #[error("{path} must not be empty")]
    Empty { path: String },
    #[error("{path} must be finite, got {value}")]
    NonFinite { path: String, value: f64 },
    #[error("{path} must be positive, got {value}")]
    NotPositive { path: String, value: f64 },
    #[error("{path} must be nonzero, got {value}")]
    Zero { path: String, value: f64 },
    #[error("{path} has length {actual}, expected {expected}")]
    LengthMismatch {
        path: String,
        expected: usize,
        actual: usize,
    },
    #[error("{path} contains duplicate key {key:?}")]
    Duplicate { path: String, key: String },
    #[error("invalid LM channel at {path}: l={l}, m={m}")]
    InvalidLm { path: String, l: u32, m: i32 },
    #[error("{path} has {points} points; exponential meshes require at least 7")]
    MeshTooShort { path: String, points: usize },
    #[error(
        "{path}.last={actual} is inconsistent with first*exp((N-1)*h)={expected} at tolerance {tolerance}"
    )]
    MeshEndpoint {
        path: String,
        expected: f64,
        actual: f64,
        tolerance: f64,
    },
    #[error("lattice determinant must be finite and positive, got {determinant}")]
    InvalidLattice { determinant: f64 },
}

pub(crate) fn finite(path: impl Into<String>, value: f64) -> Result<(), ValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ValidationError::NonFinite {
            path: path.into(),
            value,
        })
    }
}

pub(crate) fn positive(path: impl Into<String>, value: f64) -> Result<(), ValidationError> {
    let path = path.into();
    finite(path.clone(), value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ValidationError::NotPositive { path, value })
    }
}

pub(crate) fn nonempty(path: impl Into<String>, value: &str) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        Err(ValidationError::Empty { path: path.into() })
    } else {
        Ok(())
    }
}
