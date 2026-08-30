use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use muffintin_io::{CheckpointV2, checkpoint_file_from_toml};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

#[pyclass(name = "Checkpoint", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct Checkpoint {
    pub(crate) inner: Arc<CheckpointV2>,
}

#[pyclass(name = "CheckpointPhysics", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct CheckpointPhysics {
    pub(crate) checkpoint: Arc<CheckpointV2>,
    physics: Arc<muffintin::CheckpointPhysics>,
}

#[pyclass(name = "ScalarProductInput", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct ScalarProductInput {
    pub(crate) checkpoint: Arc<CheckpointV2>,
    pub(crate) inner: Arc<muffintin::ScalarProductInput>,
}

#[pyfunction]
pub(crate) fn load_checkpoint(path: PathBuf) -> PyResult<Checkpoint> {
    let text = fs::read_to_string(&path).map_err(|error| {
        PyOSError::new_err(format!(
            "could not read checkpoint {}: {error}",
            path.display()
        ))
    })?;
    let checkpoint = checkpoint_file_from_toml(&text)
        .and_then(|file| file.into_v2_prevalidated())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(Checkpoint {
        inner: Arc::new(checkpoint),
    })
}

#[pymethods]
impl CheckpointPhysics {
    #[new]
    fn new(checkpoint: PyRef<'_, Checkpoint>) -> PyResult<Self> {
        let physics = muffintin::CheckpointPhysics::new(checkpoint.inner.as_ref())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            checkpoint: Arc::clone(&checkpoint.inner),
            physics: Arc::new(physics),
        })
    }

    fn scalar_product_input(
        &self,
        input_path: PathBuf,
        q: [f64; 3],
    ) -> PyResult<ScalarProductInput> {
        let workflow = muffintin::load_input_path(input_path)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let config = muffintin::single_dft_scf_config(&workflow)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let input = self
            .physics
            .scalar_product_input(&config, q)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(ScalarProductInput {
            checkpoint: Arc::clone(&self.checkpoint),
            inner: Arc::new(input),
        })
    }
}
