use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use muffintin_core::{AngularGrid, Bohr, ExponentialMesh, InverseBohr};
use muffintin_dft::{FreeAtomScfSpec, NoncollinearXcRoute, ScfExchangeCorrelation, XcFunctional};
use muffintin_io::{
    AngularBasis, CheckpointFile, CheckpointMeta, CheckpointV2, EnergyParameterV1, EnergyUnit,
    ExponentialMeshSpec, GeometryV2, LatticeV1, LengthUnit, LinearizationV1, PotentialConventionV1,
    PotentialRadialQuantityV1, RadialBasisSpinV2, RadialEquationTag, SiteRadialBasisV2, SiteV2,
    SphericalChannelConvention, checkpoint_file_from_toml, checkpoint_file_to_toml,
};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};

use crate::export::export_dict;
use crate::regional::RegionalDensity;
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

#[pyclass(name = "Structure", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct Structure {
    pub(crate) inner: Arc<muffintin::Structure>,
}

#[pyclass(name = "RegionalFieldLayout", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct RegionalFieldLayout {
    pub(crate) inner: Arc<muffintin::RegionalFieldLayout>,
}

#[pyclass(name = "FreeAtomControls", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct FreeAtomControls {
    free_atom_scf: FreeAtomScfSpec,
    angular_points: usize,
}

#[pyclass(name = "AtomicStart", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct AtomicStart {
    checkpoint: Arc<CheckpointV2>,
    charge_closure: muffintin_dft::AtomicSuperpositionChargeClosure,
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

#[pymethods]
impl Structure {
    #[new]
    #[pyo3(signature = (
        lattice,
        site_ids,
        atomic_numbers,
        fractional_positions,
        radial_meshes,
        radial_equations,
        linearization_energies=None
    ))]
    fn new(
        lattice: [[f64; 3]; 3],
        site_ids: Vec<String>,
        atomic_numbers: Vec<u16>,
        fractional_positions: Vec<[f64; 3]>,
        radial_meshes: Vec<(f64, f64, usize)>,
        radial_equations: Vec<String>,
        linearization_energies: Option<Vec<Vec<(u32, f64)>>>,
    ) -> PyResult<Self> {
        let site_count = site_ids.len();
        require_site_count("atomic_numbers", atomic_numbers.len(), site_count)?;
        require_site_count(
            "fractional_positions",
            fractional_positions.len(),
            site_count,
        )?;
        require_site_count("radial_meshes", radial_meshes.len(), site_count)?;
        require_site_count("radial_equations", radial_equations.len(), site_count)?;
        let linearization_energies =
            linearization_energies.unwrap_or_else(|| vec![Vec::new(); site_count]);
        require_site_count(
            "linearization_energies",
            linearization_energies.len(),
            site_count,
        )?;

        let mut sites = Vec::with_capacity(site_count);
        let mut radial_basis = Vec::with_capacity(site_count);
        for (index, site_id) in site_ids.iter().enumerate() {
            let (first, log_increment, point_count) = radial_meshes[index];
            let mesh = ExponentialMesh::new(Bohr(first), log_increment, point_count)
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            let radius = mesh.last().get();
            sites.push(SiteV2 {
                id: site_id.clone(),
                atomic_number: atomic_numbers[index],
                fractional_position: fractional_positions[index],
                muffin_tin_radius_unit: LengthUnit::Bohr,
                muffin_tin_radius: radius,
            });
            radial_basis.push(SiteRadialBasisV2 {
                site_id: site_id.clone(),
                spin: RadialBasisSpinV2::Scalar,
                mesh: ExponentialMeshSpec {
                    radius_unit: LengthUnit::Bohr,
                    first,
                    log_increment,
                    point_count,
                    last: radius,
                    consistency_tolerance: 1.0e-12,
                },
                radial_equation: parse_radial_equation(&radial_equations[index])?,
                linearization: LinearizationV1 {
                    energy_unit: EnergyUnit::Hartree,
                    linearization_energies: linearization_energies[index]
                        .iter()
                        .map(|&(l, energy)| EnergyParameterV1 { l, energy })
                        .collect(),
                    local_orbital_energies: Vec::new(),
                },
            });
        }
        let inner = muffintin::Structure::new(GeometryV2 {
            lattice: LatticeV1 {
                unit: LengthUnit::Bohr,
                vectors: lattice,
            },
            sites,
            radial_basis,
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

#[pymethods]
impl RegionalFieldLayout {
    #[new]
    #[pyo3(signature = (structure, g_vectors, muffin_tin_l_max))]
    fn new(
        structure: PyRef<'_, Structure>,
        g_vectors: Vec<[i32; 3]>,
        muffin_tin_l_max: u32,
    ) -> PyResult<Self> {
        let inner = muffintin::RegionalFieldLayout::new(
            structure.inner.as_ref(),
            g_vectors,
            muffin_tin_l_max,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    #[classmethod]
    #[pyo3(signature = (structure, g_cutoff, muffin_tin_l_max))]
    fn from_g_cutoff(
        _class: &Bound<'_, PyType>,
        structure: PyRef<'_, Structure>,
        g_cutoff: f64,
        muffin_tin_l_max: u32,
    ) -> PyResult<Self> {
        let inner = muffintin::RegionalFieldLayout::from_g_cutoff(
            structure.inner.as_ref(),
            InverseBohr(g_cutoff),
            muffin_tin_l_max,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

#[pymethods]
impl FreeAtomControls {
    #[new]
    #[pyo3(signature = (
        mesh_first,
        mesh_log_increment,
        mesh_point_count,
        mixing,
        potential_tolerance,
        tail_tolerance,
        max_iterations,
        angular_points
    ))]
    fn new(
        mesh_first: f64,
        mesh_log_increment: f64,
        mesh_point_count: usize,
        mixing: f64,
        potential_tolerance: f64,
        tail_tolerance: f64,
        max_iterations: usize,
        angular_points: usize,
    ) -> PyResult<Self> {
        let mesh = ExponentialMesh::new(Bohr(mesh_first), mesh_log_increment, mesh_point_count)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        AngularGrid::fibonacci(angular_points)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            free_atom_scf: FreeAtomScfSpec {
                mesh,
                mixing,
                potential_tolerance,
                tail_tolerance,
                max_iterations,
            },
            angular_points,
        })
    }
}

#[pymethods]
impl AtomicStart {
    #[getter]
    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            inner: Arc::clone(&self.checkpoint),
        }
    }

    #[getter]
    fn charge_closure(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let closure = self.charge_closure;
        let dict = PyDict::new(py);
        dict.set_item("interstitial_fraction", closure.interstitial_fraction)?;
        dict.set_item("response_volume", closure.response_volume)?;
        dict.set_item("target_electron_count", closure.target_electron_count)?;
        dict.set_item(
            "uncorrected_electron_count",
            closure.uncorrected_electron_count,
        )?;
        dict.set_item(
            "zero_mode_coefficient_correction",
            closure.zero_mode_coefficient_correction,
        )?;
        dict.set_item(
            "represented_electron_count",
            closure.represented_electron_count,
        )?;
        Ok(dict.unbind())
    }
}

#[pyfunction]
#[pyo3(signature = (structure, field_layout, xc, free_atom_controls))]
pub(crate) fn materialize_atomic_start(
    structure: PyRef<'_, Structure>,
    field_layout: PyRef<'_, RegionalFieldLayout>,
    xc: String,
    free_atom_controls: PyRef<'_, FreeAtomControls>,
) -> PyResult<AtomicStart> {
    let functional = match xc.as_str() {
        "lda-pw92" => XcFunctional::LdaPw92,
        "pbe" => XcFunctional::Pbe,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown exchange-correlation functional {xc:?}; expected 'lda-pw92' or 'pbe'"
            )));
        }
    };
    let angular_grid = AngularGrid::fibonacci(free_atom_controls.angular_points)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let start = muffintin::materialize_atomic_start(muffintin::AtomicStartRequest {
        meta: CheckpointMeta {
            title: "neutral atomic-superposition start".to_owned(),
            producer: "libmuffintin-python".to_owned(),
            producer_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
            energy_zero: "periodic crystal electrostatic reference".to_owned(),
            potential_convention: PotentialConventionV1 {
                angular_basis: AngularBasis::ComplexCondonShortley,
                radial_quantity: PotentialRadialQuantityV1::Potential,
                spherical_channel: SphericalChannelConvention::PhysicalValue,
            },
            annotations: BTreeMap::new(),
        },
        structure: structure.inner.as_ref().clone(),
        field_layout: field_layout.inner.as_ref().clone(),
        exchange_correlation: ScfExchangeCorrelation {
            functional,
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        },
        free_atom_scf: free_atom_controls.free_atom_scf.clone(),
        angular_grid,
    })
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(AtomicStart {
        checkpoint: Arc::new(start.checkpoint),
        charge_closure: start.charge_closure,
    })
}

fn require_site_count(field: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{field} has {actual} rows, expected {expected} sites"
        )))
    }
}

fn parse_radial_equation(value: &str) -> PyResult<RadialEquationTag> {
    match value {
        "schroedinger" => Ok(RadialEquationTag::Schroedinger),
        "scalar-koelling-harmon" => Ok(RadialEquationTag::ScalarKoellingHarmon),
        "fully-relativistic-dirac" => Ok(RadialEquationTag::FullyRelativisticDirac),
        _ => Err(PyValueError::new_err(format!(
            "unknown radial equation {value:?}; expected 'schroedinger', 'scalar-koelling-harmon', or 'fully-relativistic-dirac'"
        ))),
    }
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
impl Checkpoint {
    /// Serialize this validated V2 checkpoint as canonical TOML.
    fn write(&self, path: PathBuf) -> PyResult<()> {
        let text = checkpoint_file_to_toml(&CheckpointFile::V2(self.inner.as_ref().clone()))
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        fs::write(&path, text).map_err(|error| {
            PyOSError::new_err(format!(
                "could not write checkpoint {}: {error}",
                path.display()
            ))
        })
    }
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

    fn restart_density(&self) -> PyResult<Option<RegionalDensity>> {
        let Some(density) = self.physics.restart_density() else {
            return Ok(None);
        };
        let structure = muffintin::Structure::new(self.checkpoint.geometry.clone())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Some(RegionalDensity::from_runtime(
            density,
            Arc::new(structure),
        )))
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
