use std::sync::Arc;

use muffintin_core::InverseBohr;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::checkpoint::ScalarProductInput;

#[pyclass(name = "ScalarMpbResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct ScalarMpbResult {
    pub(crate) checkpoint: Arc<muffintin_io::CheckpointV2>,
    pub(crate) _input: Arc<muffintin::ScalarProductInput>,
    pub(crate) inner: Arc<muffintin::ScalarMpbResult>,
    pub(crate) _spec: muffintin::ScalarMpbSpec,
}

#[pyfunction]
pub(crate) fn build_scalar_mpb(
    input: PyRef<'_, ScalarProductInput>,
    selections: PyReadonlyArray2<'_, i64>,
    product_l_max: u32,
    product_g_max: f64,
    overlap_tolerance: f64,
) -> PyResult<ScalarMpbResult> {
    let selections = selections.as_array();
    if selections.shape()[1] != 4 || selections.shape()[0] == 0 {
        return Err(PyValueError::new_err(
            "selections must have shape (S, 4) with S > 0",
        ));
    }
    let mut parsed = Vec::with_capacity(selections.shape()[0]);
    for row in selections.rows() {
        parsed.push(muffintin::ScalarMpbSelection {
            spin: u8::try_from(row[0])
                .map_err(|_| PyValueError::new_err("selection spin is outside uint8"))?,
            k: usize::try_from(row[1])
                .map_err(|_| PyValueError::new_err("selection k must be nonnegative"))?,
            left_band: usize::try_from(row[2])
                .map_err(|_| PyValueError::new_err("selection left band must be nonnegative"))?,
            right_band: usize::try_from(row[3])
                .map_err(|_| PyValueError::new_err("selection right band must be nonnegative"))?,
        });
    }
    let spec = muffintin::ScalarMpbSpec {
        lattice: input.inner.reciprocal,
        product_l_max,
        product_g_max: InverseBohr(product_g_max),
        overlap_tolerance,
        selections: parsed,
    };
    let result = muffintin::build_scalar_mpb(input.inner.as_ref(), &spec)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(ScalarMpbResult {
        checkpoint: Arc::clone(&input.checkpoint),
        _input: Arc::clone(&input.inner),
        inner: Arc::new(result),
        _spec: spec,
    })
}
