use std::sync::Arc;

use muffintin_core::Kappa;
use muffintin_dft::{CoreSiteRequest, CoreSpinPartition, CoreStateRequest, RegionalCoreResult};
use muffintin_sphere::CoreState as SphereCoreState;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::regional::{RegionalDensity, RegionalPotential};

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyclass(name = "CoreState", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct CoreState {
    request: CoreStateRequest,
}

#[pyclass(name = "CoreSite", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct CoreSite {
    request: CoreSiteRequest,
}

#[pyclass(name = "CoreStation", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct CoreStation {
    sites: Vec<CoreSiteRequest>,
}

#[pyclass(name = "CoreResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct CoreResult {
    inner: Arc<RegionalCoreResult>,
    structure: Arc<muffintin::Structure>,
}

#[pymethods]
impl CoreState {
    #[new]
    #[pyo3(signature = (n, kappa, occupation, spin_up=None, spin_down=None))]
    fn new(
        n: u32,
        kappa: i32,
        occupation: f64,
        spin_up: Option<f64>,
        spin_down: Option<f64>,
    ) -> PyResult<Self> {
        let spin = match (spin_up, spin_down) {
            (None, None) => CoreSpinPartition::ClosedShellAverage,
            (Some(up), Some(down)) => CoreSpinPartition::ExplicitCollinear { up, down },
            _ => {
                return Err(PyValueError::new_err(
                    "spin_up and spin_down must either both be absent or both be present",
                ));
            }
        };
        let kappa = Kappa::new(kappa).map_err(py_error)?;
        let state = SphereCoreState::new(n, kappa).map_err(py_error)?;
        Ok(Self {
            request: CoreStateRequest {
                state,
                occupation,
                spin,
            },
        })
    }
}

#[pymethods]
impl CoreSite {
    #[new]
    fn new(py: Python<'_>, site_index: usize, site_id: String, states: Vec<Py<CoreState>>) -> Self {
        Self {
            request: CoreSiteRequest {
                site_index,
                site_id,
                states: states
                    .iter()
                    .map(|state| state.borrow(py).request.clone())
                    .collect(),
            },
        }
    }
}

#[pymethods]
impl CoreStation {
    #[new]
    fn new(py: Python<'_>, sites: Vec<Py<CoreSite>>) -> Self {
        Self {
            sites: sites
                .iter()
                .map(|site| site.borrow(py).request.clone())
                .collect(),
        }
    }

    fn solve(&self, potential: PyRef<'_, RegionalPotential>) -> PyResult<CoreResult> {
        for site in &self.sites {
            let Some(expected) = potential.structure.geometry().sites.get(site.site_index) else {
                continue;
            };
            if expected.id != site.site_id {
                return Err(PyValueError::new_err(format!(
                    "core site index {} is {:?}, not {:?}",
                    site.site_index, expected.id, site.site_id
                )));
            }
        }
        let inner = muffintin_dft::solve_regional_core(potential.inner.as_ref(), &self.sites)
            .map_err(py_error)?;
        Ok(CoreResult {
            inner: Arc::new(inner),
            structure: Arc::clone(&potential.structure),
        })
    }
}

#[pymethods]
impl CoreResult {
    fn density(&self) -> RegionalDensity {
        RegionalDensity::from_runtime(self.inner.density.clone(), Arc::clone(&self.structure))
    }

    #[getter]
    fn core_eigenvalue_sum(&self) -> f64 {
        self.inner.eigenvalue_sum.get()
    }

    fn site_ids(&self) -> Vec<String> {
        self.inner
            .sites
            .iter()
            .map(|site| site.contribution.site_id.clone())
            .collect()
    }

    fn requested_charges(&self) -> Vec<f64> {
        self.inner
            .sites
            .iter()
            .map(|site| site.diagnostics.requested_charge)
            .collect()
    }

    fn represented_charges(&self) -> Vec<f64> {
        self.inner
            .sites
            .iter()
            .map(|site| site.diagnostics.represented_charge)
            .collect()
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CoreState>()?;
    module.add_class::<CoreSite>()?;
    module.add_class::<CoreStation>()?;
    module.add_class::<CoreResult>()?;
    Ok(())
}
