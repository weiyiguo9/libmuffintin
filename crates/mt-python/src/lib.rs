//! Thin Python data-export ABI over frozen libmuffintin runtime objects.

mod checkpoint;
mod export;

use checkpoint::{Checkpoint, CheckpointPhysics, ScalarProductInput, load_checkpoint};
use pyo3::prelude::*;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Checkpoint>()?;
    module.add_class::<CheckpointPhysics>()?;
    module.add_class::<ScalarProductInput>()?;
    module.add_function(wrap_pyfunction!(load_checkpoint, module)?)?;
    Ok(())
}
