use muffintin_core::Hartree;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::regional::{RegionalDensity, RegionalPotential};

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(
    name = "TotalEnergyEvaluation",
    module = "libmuffintin._native",
    frozen
)]
#[derive(Clone, Debug)]
pub(crate) struct TotalEnergyEvaluation {
    inner: muffintin_dft::TotalEnergyEvaluation,
}

#[pymethods]
impl TotalEnergyEvaluation {
    #[getter]
    fn band(&self) -> f64 {
        self.inner.energy.band.get()
    }

    #[getter]
    fn core_eigenvalues(&self) -> f64 {
        self.inner.energy.core_eigenvalues.get()
    }

    #[getter]
    fn madelung(&self) -> f64 {
        self.inner.energy.madelung.get()
    }

    #[getter]
    fn coulomb(&self) -> f64 {
        self.inner.energy.coulomb.get()
    }

    #[getter]
    fn exchange_correlation(&self) -> f64 {
        self.inner.energy.exchange_correlation.get()
    }

    #[getter]
    fn exchange_correlation_potential(&self) -> f64 {
        self.inner.energy.exchange_correlation_potential.get()
    }

    #[getter]
    fn occupation_correction(&self) -> f64 {
        self.inner.energy.occupation.correction().get()
    }

    #[getter]
    fn total(&self) -> f64 {
        self.inner.energy.total.get()
    }

    #[getter]
    fn density_rms(&self) -> f64 {
        self.inner.density_rms
    }

    #[getter]
    fn energy_change(&self) -> Option<f64> {
        self.inner.energy_change.map(Hartree::get)
    }
}

#[pyfunction]
#[pyo3(signature = (
    potential,
    output_density,
    band_energy,
    core_eigenvalue_sum,
    occupation_correction,
    previous_total=None
))]
pub(crate) fn evaluate_total_energy(
    potential: PyRef<'_, RegionalPotential>,
    output_density: PyRef<'_, RegionalDensity>,
    band_energy: f64,
    core_eigenvalue_sum: f64,
    occupation_correction: f64,
    previous_total: Option<f64>,
) -> PyResult<TotalEnergyEvaluation> {
    if potential.structure.geometry() != output_density.structure.geometry() {
        return Err(PyValueError::new_err(
            "regional potential and output density belong to different structures",
        ));
    }
    let inner = muffintin_dft::evaluate_total_energy(
        potential.inner.as_ref(),
        output_density.inner.as_ref(),
        muffintin_dft::TotalEnergyInput {
            band_energy: Hartree(band_energy),
            core_eigenvalue_sum: Hartree(core_eigenvalue_sum),
            occupation_correction: Hartree(occupation_correction),
        },
        previous_total.map(Hartree),
    )
    .map_err(py_error)?;
    Ok(TotalEnergyEvaluation { inner })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TotalEnergyEvaluation>()?;
    module.add_function(wrap_pyfunction!(evaluate_total_energy, module)?)?;
    Ok(())
}
