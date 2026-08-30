//! Thin Python data-export ABI over frozen libmuffintin runtime objects.

mod checkpoint;
mod core;
mod coulomb;
mod energy;
mod export;
mod mixing;
mod products;
mod regional;
mod scf;
mod spinor;
mod thc;
mod writers;

use checkpoint::{
    AtomicStart, Checkpoint, CheckpointPhysics, FreeAtomControls, RegionalFieldLayout,
    ScalarProductInput, ScalarProductSlice, Structure, load_checkpoint, materialize_atomic_start,
};
use coulomb::{
    ScalarCoulombResult, ScalarMpbCoulombResult, build_scalar_coulomb, build_scalar_mpb_coulomb,
};
use products::{ScalarMpbResult, build_scalar_mpb};
use pyo3::prelude::*;
use thc::{ScalarThcResult, build_scalar_thc, sample_scalar_orbitals};

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Checkpoint>()?;
    module.add_class::<CheckpointPhysics>()?;
    module.add_class::<Structure>()?;
    module.add_class::<RegionalFieldLayout>()?;
    module.add_class::<FreeAtomControls>()?;
    module.add_class::<AtomicStart>()?;
    module.add_class::<ScalarProductInput>()?;
    module.add_class::<ScalarProductSlice>()?;
    module.add_class::<ScalarMpbResult>()?;
    module.add_class::<ScalarThcResult>()?;
    module.add_class::<ScalarCoulombResult>()?;
    module.add_class::<ScalarMpbCoulombResult>()?;
    module.add_function(wrap_pyfunction!(load_checkpoint, module)?)?;
    module.add_function(wrap_pyfunction!(materialize_atomic_start, module)?)?;
    module.add_function(wrap_pyfunction!(build_scalar_mpb, module)?)?;
    module.add_function(wrap_pyfunction!(build_scalar_thc, module)?)?;
    module.add_function(wrap_pyfunction!(build_scalar_coulomb, module)?)?;
    module.add_function(wrap_pyfunction!(build_scalar_mpb_coulomb, module)?)?;
    module.add_function(wrap_pyfunction!(sample_scalar_orbitals, module)?)?;
    core::register(module)?;
    energy::register(module)?;
    mixing::register(module)?;
    spinor::register(module)?;
    writers::register(module)?;
    regional::register(module)?;
    scf::register(module)?;
    Ok(())
}
