use std::sync::{Arc, Mutex};

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::regional::RegionalDensity;

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "DensityMixer", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct DensityMixer {
    inner: Mutex<muffintin_dft::DensityMixer>,
}

#[pyclass(name = "DensityMixStep", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct DensityMixStep {
    inner: muffintin_dft::MixStep,
    structure: Arc<muffintin::Structure>,
}

#[pymethods]
impl DensityMixer {
    #[classmethod]
    fn linear(_class: &Bound<'_, PyType>, alpha: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(muffintin_dft::DensityMixer::linear(alpha).map_err(py_error)?),
        })
    }

    #[classmethod]
    fn broyden2(_class: &Bound<'_, PyType>, alpha: f64, history: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(
                muffintin_dft::DensityMixer::broyden2(alpha, history).map_err(py_error)?,
            ),
        })
    }

    #[classmethod]
    fn pulay_anderson(_class: &Bound<'_, PyType>, alpha: f64, history: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Mutex::new(
                muffintin_dft::DensityMixer::pulay_anderson(alpha, history).map_err(py_error)?,
            ),
        })
    }

    fn step(
        &self,
        input: PyRef<'_, RegionalDensity>,
        output: PyRef<'_, RegionalDensity>,
    ) -> PyResult<DensityMixStep> {
        if input.structure.geometry() != output.structure.geometry() {
            return Err(PyValueError::new_err(
                "input and output densities belong to different structures",
            ));
        }
        let inner = self
            .inner
            .lock()
            .expect("density mixer mutex is not poisoned")
            .mix(input.inner.as_ref(), output.inner.as_ref())
            .map_err(py_error)?;
        Ok(DensityMixStep {
            inner,
            structure: Arc::clone(&input.structure),
        })
    }

    #[getter]
    fn history_length(&self) -> usize {
        self.inner
            .lock()
            .expect("density mixer mutex is not poisoned")
            .history()
            .len()
    }

    fn last_pulay_coefficients<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(
            py,
            self.inner
                .lock()
                .expect("density mixer mutex is not poisoned")
                .last_pulay_coefficients()
                .to_vec(),
        )
    }
}

#[pymethods]
impl DensityMixStep {
    fn density(&self) -> RegionalDensity {
        RegionalDensity::from_runtime(self.inner.density.clone(), Arc::clone(&self.structure))
    }

    #[getter]
    fn status(&self) -> &'static str {
        match self.inner.status {
            muffintin_dft::MixStatus::Linear => "linear",
            muffintin_dft::MixStatus::NonlinearWarmup => "nonlinear-warmup",
            muffintin_dft::MixStatus::Nonlinear => "nonlinear",
            muffintin_dft::MixStatus::RankDeficientLinearFallback => {
                "rank-deficient-linear-fallback"
            }
            muffintin_dft::MixStatus::NotMixed => "not-mixed",
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DensityMixer>()?;
    module.add_class::<DensityMixStep>()?;
    Ok(())
}
