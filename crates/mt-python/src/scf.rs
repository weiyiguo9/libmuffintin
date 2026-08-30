use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::checkpoint::{Checkpoint, CheckpointPhysics};
use crate::export::export_dict;

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn take<T>(slot: &Mutex<Option<T>>, name: &str) -> PyResult<T> {
    slot.lock()
        .expect("SCF handle mutex is not poisoned")
        .take()
        .ok_or_else(|| PyValueError::new_err(format!("{name} has already been consumed")))
}

fn with_session<T>(
    slot: &Mutex<Option<muffintin::DftScfSession>>,
    operation: impl FnOnce(&mut muffintin::DftScfSession) -> Result<T, muffintin::DftScfError>,
) -> PyResult<T> {
    let mut guard = slot.lock().expect("SCF session mutex is not poisoned");
    let session = guard
        .as_mut()
        .ok_or_else(|| PyValueError::new_err("SCF session has already been consumed"))?;
    operation(session).map_err(py_error)
}

fn export_regional<'py>(
    py: Python<'py>,
    fields: muffintin::DftRegionalFourier,
) -> PyResult<Py<PyDict>> {
    let g = fields
        .g_vectors
        .iter()
        .flat_map(|vector| *vector)
        .collect::<Vec<_>>();
    let values = fields.components.into_iter().flatten().collect::<Vec<_>>();
    let dict = export_dict(py)?;
    dict.set_item(
        "g_vectors",
        PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((fields.g_vectors.len(), 3), g)
                .expect("each reciprocal vector has three coordinates"),
        ),
    )?;
    dict.set_item(
        "components",
        PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((4, fields.g_vectors.len()), values)
                .expect("four regional components share one Fourier layout"),
        ),
    )?;
    Ok(dict.unbind())
}

#[pyclass(name = "DftScfPlan", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct DftScfPlan {
    inner: Arc<muffintin::DftScfPlan>,
}

#[pyclass(name = "ScfSession", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct ScfSession {
    inner: Mutex<Option<muffintin::DftScfSession>>,
}

#[pyclass(name = "RegionalDensity", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct RegionalDensity {
    inner: Mutex<Option<muffintin::DftRegionalDensity>>,
}

#[pyclass(
    name = "RegionalPotentialStep",
    module = "libmuffintin._native",
    frozen
)]
#[derive(Debug)]
pub(crate) struct RegionalPotentialStep {
    inner: Mutex<Option<muffintin::DftRegionalPotentialStep>>,
}

#[pyclass(name = "CoreStep", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct CoreStep {
    inner: Mutex<Option<muffintin::DftCoreStep>>,
}

#[pyclass(name = "LapwSolution", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct LapwSolution {
    inner: Mutex<Option<muffintin::DftLapwSolution>>,
}

#[pyclass(name = "Occupations", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct Occupations {
    inner: Mutex<Option<muffintin::DftOccupations>>,
}

#[pyclass(name = "LapwDensityAssembly", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct LapwDensityAssembly {
    inner: Mutex<Option<muffintin::DftLapwDensityAssembly>>,
}

#[pyclass(name = "EnergyRecord", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct EnergyRecord {
    inner: Mutex<Option<muffintin::DftEnergyRecord>>,
}

#[pyclass(name = "ConvergenceDecision", module = "libmuffintin._native", frozen)]
#[derive(Debug)]
pub(crate) struct ConvergenceDecision {
    inner: Mutex<Option<muffintin::DftConvergenceDecision>>,
}

#[pyclass(name = "ScfResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct ScfResult {
    inner: Arc<muffintin::DftScfResult>,
}

#[pyfunction]
pub(crate) fn prepare_dft_scf(path: PathBuf) -> PyResult<DftScfPlan> {
    Ok(DftScfPlan {
        inner: Arc::new(muffintin::prepare_dft_scf(path).map_err(py_error)?),
    })
}

#[pyfunction]
pub(crate) fn run_dft_scf(path: PathBuf) -> PyResult<ScfResult> {
    Ok(ScfResult {
        inner: Arc::new(muffintin::run_dft_scf(path).map_err(py_error)?),
    })
}

#[pymethods]
impl DftScfPlan {
    fn session(&self) -> PyResult<ScfSession> {
        Ok(ScfSession {
            inner: Mutex::new(Some(self.inner.session().map_err(py_error)?)),
        })
    }
}

#[pymethods]
impl CheckpointPhysics {
    /// Create the staged SCF session selected by Input V2 for this checkpoint context.
    fn scf_session(&self, input_path: PathBuf) -> PyResult<ScfSession> {
        let plan = muffintin::prepare_dft_scf(input_path).map_err(py_error)?;
        let session = plan
            .session_for_checkpoint(self.checkpoint.as_ref())
            .map_err(py_error)?;
        Ok(ScfSession {
            inner: Mutex::new(Some(session)),
        })
    }
}

#[pymethods]
impl ScfSession {
    fn initial_density(&self) -> PyResult<RegionalDensity> {
        Ok(RegionalDensity {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.initial_density()
            })?)),
        })
    }

    fn potential(&self, density: PyRef<'_, RegionalDensity>) -> PyResult<RegionalPotentialStep> {
        let density = take(&density.inner, "RegionalDensity")?;
        Ok(RegionalPotentialStep {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.potential(density)
            })?)),
        })
    }

    fn core(&self, potential: PyRef<'_, RegionalPotentialStep>) -> PyResult<CoreStep> {
        let potential = take(&potential.inner, "RegionalPotentialStep")?;
        Ok(CoreStep {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.core(potential)
            })?)),
        })
    }

    fn lapw(&self, core: PyRef<'_, CoreStep>) -> PyResult<LapwSolution> {
        let core = take(&core.inner, "CoreStep")?;
        Ok(LapwSolution {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.lapw(core)
            })?)),
        })
    }

    fn occupations(&self, solution: PyRef<'_, LapwSolution>) -> PyResult<Occupations> {
        let solution = take(&solution.inner, "LapwSolution")?;
        Ok(Occupations {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.occupations(solution)
            })?)),
        })
    }

    fn density(&self, occupations: PyRef<'_, Occupations>) -> PyResult<LapwDensityAssembly> {
        let occupations = take(&occupations.inner, "Occupations")?;
        Ok(LapwDensityAssembly {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.density(occupations)
            })?)),
        })
    }

    fn energy(&self, density: PyRef<'_, LapwDensityAssembly>) -> PyResult<EnergyRecord> {
        let density = take(&density.inner, "LapwDensityAssembly")?;
        Ok(EnergyRecord {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.energy(density)
            })?)),
        })
    }

    fn convergence(&self, energy: PyRef<'_, EnergyRecord>) -> PyResult<ConvergenceDecision> {
        let energy = take(&energy.inner, "EnergyRecord")?;
        Ok(ConvergenceDecision {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.convergence(energy)
            })?)),
        })
    }

    fn mix(&self, decision: PyRef<'_, ConvergenceDecision>) -> PyResult<RegionalDensity> {
        let decision = take(&decision.inner, "ConvergenceDecision")?;
        Ok(RegionalDensity {
            inner: Mutex::new(Some(with_session(&self.inner, |session| {
                session.mix(decision)
            })?)),
        })
    }

    fn run(&self) -> PyResult<ScfResult> {
        let session = take(&self.inner, "ScfSession")?;
        Ok(ScfResult {
            inner: Arc::new(session.run().map_err(py_error)?),
        })
    }
}

#[pymethods]
impl RegionalDensity {
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("density mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RegionalDensity has already been consumed"))?
            .iteration())
    }

    fn export_interstitial(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let guard = self.inner.lock().expect("density mutex is not poisoned");
        let density = guard
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RegionalDensity has already been consumed"))?;
        export_regional(py, density.export_interstitial())
    }
}

#[pymethods]
impl RegionalPotentialStep {
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("potential mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| {
                PyValueError::new_err("RegionalPotentialStep has already been consumed")
            })?
            .iteration())
    }

    fn export_interstitial(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let guard = self.inner.lock().expect("potential mutex is not poisoned");
        let potential = guard.as_ref().ok_or_else(|| {
            PyValueError::new_err("RegionalPotentialStep has already been consumed")
        })?;
        export_regional(py, potential.export_interstitial())
    }
}

#[pymethods]
impl CoreStep {
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("core mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("CoreStep has already been consumed"))?
            .iteration())
    }
    #[getter]
    fn core_eigenvalue_sum(&self) -> PyResult<f64> {
        Ok(self
            .inner
            .lock()
            .expect("core mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("CoreStep has already been consumed"))?
            .core_eigenvalue_sum()
            .get())
    }
}

#[pymethods]
impl LapwSolution {
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("LAPW mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("LapwSolution has already been consumed"))?
            .iteration())
    }
}

#[pymethods]
impl Occupations {
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("occupations mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Occupations has already been consumed"))?
            .iteration())
    }
    #[getter]
    fn chemical_potential(&self) -> PyResult<f64> {
        Ok(self
            .inner
            .lock()
            .expect("occupations mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Occupations has already been consumed"))?
            .chemical_potential()
            .get())
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self
            .inner
            .lock()
            .expect("occupations mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Occupations has already been consumed"))?
            .values()
            .to_vec();
        Ok(PyArray1::from_vec(py, values))
    }
}

#[pymethods]
impl LapwDensityAssembly {
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("density assembly mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("LapwDensityAssembly has already been consumed"))?
            .iteration())
    }
    fn export_interstitial(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let guard = self
            .inner
            .lock()
            .expect("density assembly mutex is not poisoned");
        let density = guard.as_ref().ok_or_else(|| {
            PyValueError::new_err("LapwDensityAssembly has already been consumed")
        })?;
        export_regional(py, density.export_interstitial())
    }
}

#[pymethods]
impl EnergyRecord {
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("energy mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("EnergyRecord has already been consumed"))?
            .iteration())
    }
    #[getter]
    fn total_energy(&self) -> PyResult<f64> {
        Ok(self
            .inner
            .lock()
            .expect("energy mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("EnergyRecord has already been consumed"))?
            .total_energy()
            .get())
    }
    #[getter]
    fn density_rms(&self) -> PyResult<f64> {
        Ok(self
            .inner
            .lock()
            .expect("energy mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("EnergyRecord has already been consumed"))?
            .density_rms())
    }
    #[getter]
    fn energy_change(&self) -> PyResult<Option<f64>> {
        Ok(self
            .inner
            .lock()
            .expect("energy mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("EnergyRecord has already been consumed"))?
            .energy_change()
            .map(|value| value.get()))
    }
}

#[pymethods]
impl ConvergenceDecision {
    #[getter]
    fn converged(&self) -> PyResult<bool> {
        Ok(self
            .inner
            .lock()
            .expect("decision mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ConvergenceDecision has already been consumed"))?
            .is_converged())
    }
    #[getter]
    fn iteration(&self) -> PyResult<usize> {
        Ok(self
            .inner
            .lock()
            .expect("decision mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ConvergenceDecision has already been consumed"))?
            .iteration())
    }
    fn result(&self) -> PyResult<Option<ScfResult>> {
        Ok(self
            .inner
            .lock()
            .expect("decision mutex is not poisoned")
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ConvergenceDecision has already been consumed"))?
            .result()
            .cloned()
            .map(|inner| ScfResult {
                inner: Arc::new(inner),
            }))
    }
}

#[pymethods]
impl ScfResult {
    #[getter]
    fn converged(&self) -> bool {
        true
    }
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.state.iterations()
    }
    #[getter]
    fn total_energy(&self) -> f64 {
        self.inner.state.energy.total.get()
    }
    fn energy_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(
            py,
            self.inner
                .diagnostics()
                .iter()
                .map(|item| item.energy.total.get())
                .collect(),
        )
    }
    fn convergence_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let values = self
            .inner
            .diagnostics()
            .iter()
            .flat_map(|item| {
                [
                    item.density_rms,
                    item.energy_change.map_or(f64::NAN, |value| value.get()),
                ]
            })
            .collect();
        PyArray2::from_owned_array(
            py,
            Array2::from_shape_vec((self.inner.diagnostics().len(), 2), values)
                .expect("two convergence values per iteration"),
        )
    }
    fn restart_checkpoint(&self) -> Checkpoint {
        Checkpoint {
            inner: Arc::new(self.inner.checkpoint.clone()),
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DftScfPlan>()?;
    module.add_class::<ScfSession>()?;
    module.add_class::<RegionalDensity>()?;
    module.add_class::<RegionalPotentialStep>()?;
    module.add_class::<CoreStep>()?;
    module.add_class::<LapwSolution>()?;
    module.add_class::<Occupations>()?;
    module.add_class::<LapwDensityAssembly>()?;
    module.add_class::<EnergyRecord>()?;
    module.add_class::<ConvergenceDecision>()?;
    module.add_class::<ScfResult>()?;
    module.add_function(wrap_pyfunction!(prepare_dft_scf, module)?)?;
    module.add_function(wrap_pyfunction!(run_dft_scf, module)?)?;
    Ok(())
}
