use std::sync::Arc;

use muffintin_core::{Bohr, Cell, InverseBohr};
use muffintin_coulomb::{CoulombOperator, CoulombRequest, InterpolationProjection};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::checkpoint::ScalarProductSlice;
use crate::products::ScalarMpbResult;
use crate::thc::ScalarThcResult;

#[pyclass(name = "ScalarCoulombResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct ScalarCoulombResult {
    pub(crate) _checkpoint: Arc<muffintin_io::CheckpointV2>,
    pub(crate) _slice: ScalarProductSlice,
    pub(crate) _thc: ScalarThcResult,
    pub(crate) inner: Arc<muffintin::ScalarCoulombResult>,
    pub(crate) spec: muffintin::ScalarCoulombSpec,
}

#[pyclass(
    name = "ScalarMpbCoulombResult",
    module = "libmuffintin._native",
    frozen
)]
#[derive(Clone, Debug)]
pub(crate) struct ScalarMpbCoulombResult {
    pub(crate) _checkpoint: Arc<muffintin_io::CheckpointV2>,
    pub(crate) _mpb: ScalarMpbResult,
    pub(crate) inner: Arc<CoulombOperator>,
}

fn checkpoint_cell(checkpoint: &muffintin_io::CheckpointV2) -> PyResult<Cell> {
    Cell::new(checkpoint.geometry.lattice.vectors.map(|row| row.map(Bohr)))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
pub(crate) fn build_scalar_mpb_coulomb(
    mpb: PyRef<'_, ScalarMpbResult>,
    lexp: u32,
) -> PyResult<ScalarMpbCoulombResult> {
    let request = CoulombRequest::new(checkpoint_cell(mpb.checkpoint.as_ref())?, lexp)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let operator = muffintin_coulomb::assemble_coulomb(&mpb.inner.auxiliary, &request)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ScalarMpbCoulombResult {
        _checkpoint: Arc::clone(&mpb.checkpoint),
        _mpb: ScalarMpbResult::clone(&*mpb),
        inner: Arc::new(operator),
    })
}

#[pyfunction]
#[pyo3(signature = (slice, thc, lexp, interpolation_pw_cutoff, interpolation_l_max, comparisons=None))]
pub(crate) fn build_scalar_coulomb(
    py: Python<'_>,
    slice: PyRef<'_, ScalarProductSlice>,
    thc: PyRef<'_, ScalarThcResult>,
    lexp: u32,
    interpolation_pw_cutoff: f64,
    interpolation_l_max: u32,
    comparisons: Option<Vec<(usize, Py<ScalarMpbResult>, usize)>>,
) -> PyResult<ScalarCoulombResult> {
    let cell = checkpoint_cell(slice.checkpoint.as_ref())?;
    let spec = muffintin::ScalarCoulombSpec {
        request: CoulombRequest::new(cell, lexp)
            .map_err(|error| PyValueError::new_err(error.to_string()))?,
        projection: InterpolationProjection::new(
            InverseBohr(interpolation_pw_cutoff),
            interpolation_l_max,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?,
    };
    let owned = comparisons
        .unwrap_or_default()
        .into_iter()
        .map(|(q_index, handle, vertex)| {
            let handle = handle.borrow(py);
            (q_index, Arc::clone(&handle.inner), vertex)
        })
        .collect::<Vec<_>>();
    let comparisons = owned
        .iter()
        .map(|(q_index, mpb, vertex)| muffintin::ScalarCoulombPairMatch {
            q_index: *q_index,
            mpb: mpb.as_ref(),
            mpb_vertex: *vertex,
        })
        .collect::<Vec<_>>();
    let result = muffintin::build_scalar_coulomb(
        slice.inner.as_slice(),
        thc.inner.as_ref(),
        &spec,
        &comparisons,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ScalarCoulombResult {
        _checkpoint: Arc::clone(&slice.checkpoint),
        _slice: ScalarProductSlice::clone(&*slice),
        _thc: ScalarThcResult::clone(&*thc),
        inner: Arc::new(result),
        spec,
    })
}
