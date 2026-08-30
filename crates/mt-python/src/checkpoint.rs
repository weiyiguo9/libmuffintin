use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use muffintin_io::{CheckpointV2, checkpoint_file_from_toml};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::export::export_dict;
use crate::scf::export_regional;

#[pyclass(name = "Checkpoint", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct Checkpoint {
    pub(crate) inner: Arc<CheckpointV2>,
}

#[pyclass(name = "CheckpointPhysics", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct CheckpointPhysics {
    pub(crate) checkpoint: Arc<CheckpointV2>,
    pub(crate) physics: Arc<muffintin::CheckpointPhysics>,
}

#[pyclass(name = "ScalarProductInput", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct ScalarProductInput {
    pub(crate) checkpoint: Arc<CheckpointV2>,
    pub(crate) inner: Arc<muffintin::ScalarProductInput>,
}

#[pyclass(name = "ScalarProductSlice", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct ScalarProductSlice {
    pub(crate) checkpoint: Arc<CheckpointV2>,
    pub(crate) inner: Arc<Vec<muffintin::ScalarProductInput>>,
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

    fn export_frozen_potential(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        export_regional(py, self.physics.export_frozen_potential())
    }

    fn export_restart_density(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.physics
            .export_restart_density()
            .map(|density| export_regional(py, density))
            .transpose()
    }

    #[pyo3(signature = (site_id, l, energies, hard_radius=None))]
    fn sample_frozen_scalar_radials(
        &self,
        py: Python<'_>,
        site_id: String,
        l: u32,
        energies: Vec<f64>,
        hard_radius: Option<f64>,
    ) -> PyResult<Py<PyDict>> {
        let energies = energies
            .into_iter()
            .map(muffintin_core::Hartree)
            .collect::<Vec<_>>();
        let samples = self
            .physics
            .sample_frozen_scalar_radials(
                &site_id,
                l,
                &energies,
                hard_radius.map(muffintin_core::Bohr),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let boundary_radial = samples
            .boundary_radial
            .iter()
            .flat_map(|boundary| *boundary)
            .collect::<Vec<_>>();
        let energy_derivative_boundary_radial = samples
            .energy_derivative_boundary_radial
            .iter()
            .flat_map(|boundary| *boundary)
            .collect::<Vec<_>>();
        let dict = export_dict(py)?;
        dict.set_item("site_index", samples.site_index)?;
        dict.set_item("site_id", samples.site_id)?;
        dict.set_item("l", samples.angular_momentum)?;
        dict.set_item(
            "energies",
            PyArray1::from_vec(
                py,
                samples.energies.iter().map(|value| value.get()).collect(),
            ),
        )?;
        dict.set_item("mesh_first", samples.mesh_first.get())?;
        dict.set_item("mesh_increment", samples.mesh_increment)?;
        dict.set_item("mesh_count", samples.mesh_count)?;
        dict.set_item(
            "mesh_radii",
            PyArray1::from_vec(
                py,
                samples
                    .mesh_radii
                    .iter()
                    .map(|radius| radius.get())
                    .collect(),
            ),
        )?;
        dict.set_item(
            "radial_samples",
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec(
                    (samples.energies.len(), samples.mesh_count),
                    samples.radial_samples,
                )
                .expect("one radial row is exported for every requested energy"),
            ),
        )?;
        dict.set_item("boundary_radius", samples.boundary_radius.get())?;
        dict.set_item(
            "boundary_radial",
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((samples.energies.len(), 2), boundary_radial)
                    .expect("each radial solution has a value and derivative"),
            ),
        )?;
        dict.set_item(
            "log_derivative",
            samples
                .log_derivative
                .iter()
                .map(|value| value.map(|value| value.get()))
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "energy_derivative_boundary_radial",
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec(
                    (samples.energies.len(), 2),
                    energy_derivative_boundary_radial,
                )
                .expect("each energy derivative has a boundary value and derivative"),
            ),
        )?;
        Ok(dict.unbind())
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

    fn scalar_q_slice(&self, input_path: PathBuf) -> PyResult<ScalarProductSlice> {
        let workflow = muffintin::load_input_path(input_path)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let config = muffintin::single_dft_scf_config(&workflow)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let gamma = self
            .physics
            .scalar_product_input(&config, [0.0, 0.0, 0.0])
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let q_points = gamma.orbitals.k_fractional.clone();
        let mut inputs = Vec::with_capacity(q_points.len());
        for q in q_points {
            if q.iter().all(|component| component.abs() <= 1.0e-12) {
                inputs.push(gamma.clone());
            } else {
                inputs.push(
                    self.physics
                        .scalar_product_input(&config, q)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                );
            }
        }
        Ok(ScalarProductSlice {
            checkpoint: Arc::clone(&self.checkpoint),
            inner: Arc::new(inputs),
        })
    }
}
