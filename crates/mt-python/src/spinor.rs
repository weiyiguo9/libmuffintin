use std::path::PathBuf;
use std::sync::Arc;

use muffintin_core::{Bohr, Cell, InverseBohr};
use muffintin_coulomb::{CoulombRequest, InterpolationProjection};
use muffintin_io::CheckpointV2;
use muffintin_operators::lapw::Provenance;
use muffintin_prodbasis::{AuxiliaryRegion, InterpolationRegion, OrbitalPair};
use num_complex::Complex64;
use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::checkpoint::CheckpointPhysics;

const SCHEMA: &str = "libmuffintin.pyexport";
const VERSION: i64 = 1;

#[pyclass(name = "SpinorProductInput", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct SpinorProductInput {
    pub(crate) checkpoint: Arc<CheckpointV2>,
    pub(crate) inner: Arc<muffintin::SpinorProductInput>,
}

#[pyclass(name = "SpinorProductSlice", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct SpinorProductSlice {
    pub(crate) checkpoint: Arc<CheckpointV2>,
    pub(crate) inner: Arc<Vec<muffintin::SpinorProductInput>>,
}

#[pyclass(name = "SpinorMpbResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct SpinorMpbResult {
    pub(crate) _checkpoint: Arc<CheckpointV2>,
    pub(crate) _input: Arc<muffintin::SpinorProductInput>,
    pub(crate) inner: Arc<muffintin::SpinorMpbResult>,
    pub(crate) _spec: muffintin::SpinorMpbSpec,
}

#[pyclass(name = "SpinorThcResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct SpinorThcResult {
    pub(crate) _checkpoint: Arc<CheckpointV2>,
    pub(crate) _slice: SpinorProductSlice,
    pub(crate) inner: Arc<muffintin::SpinorThcResult>,
    pub(crate) grid: muffintin::ThcParentGrid,
    pub(crate) _spec: muffintin::SpinorThcSpec,
    pub(crate) candidates: Vec<usize>,
}

#[pyclass(name = "SpinorCoulombResult", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct SpinorCoulombResult {
    pub(crate) _checkpoint: Arc<CheckpointV2>,
    pub(crate) _slice: SpinorProductSlice,
    pub(crate) _thc: SpinorThcResult,
    pub(crate) inner: Arc<muffintin::SpinorCoulombResult>,
    pub(crate) spec: muffintin::SpinorCoulombSpec,
}

fn export_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("schema", SCHEMA)?;
    dict.set_item("version", VERSION)?;
    Ok(dict)
}

fn array2<'py, T: Element>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    values: Vec<T>,
) -> Bound<'py, PyArray2<T>> {
    let array = Array2::from_shape_vec((rows, columns), values)
        .expect("export row count and flattened data length agree");
    PyArray2::from_owned_array(py, array)
}

fn fortran_array2<'py>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    values: Vec<Complex64>,
) -> Bound<'py, PyArray2<Complex64>> {
    let array = Array2::from_shape_vec((rows, columns).f(), values)
        .expect("eigenvector shape matches column-major storage");
    PyArray2::from_owned_array(py, array)
}

#[pymethods]
impl CheckpointPhysics {
    fn spinor_product_input(
        &self,
        input_path: PathBuf,
        q: [f64; 3],
    ) -> PyResult<SpinorProductInput> {
        let workflow = muffintin::load_input_path(input_path)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let config = muffintin::single_dft_scf_config(&workflow)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let input = self
            .physics
            .spinor_product_input(&config, q)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(SpinorProductInput {
            checkpoint: Arc::clone(&self.checkpoint),
            inner: Arc::new(input),
        })
    }

    fn spinor_q_slice(&self, input_path: PathBuf) -> PyResult<SpinorProductSlice> {
        let workflow = muffintin::load_input_path(input_path)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let config = muffintin::single_dft_scf_config(&workflow)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let seed = self
            .physics
            .spinor_product_input(&config, [0.0; 3])
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let q_points = seed.orbitals.k_fractional.clone();
        let mut inputs = Vec::with_capacity(q_points.len());
        for q in q_points {
            if q.iter().all(|component| component.abs() <= 1.0e-12) {
                inputs.push(seed.clone());
            } else {
                inputs.push(
                    self.physics
                        .spinor_product_input(&config, q)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?,
                );
            }
        }
        Ok(SpinorProductSlice {
            checkpoint: Arc::clone(&self.checkpoint),
            inner: Arc::new(inputs),
        })
    }
}

#[pymethods]
impl SpinorProductInput {
    fn export_orbitals(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let orbitals = &self.inner.orbitals;
        let dict = export_dict(py)?;
        dict.set_item(
            "k_fractional",
            array2(
                py,
                orbitals.k_fractional.len(),
                3,
                orbitals
                    .k_fractional
                    .iter()
                    .flat_map(|point| point.iter().copied())
                    .collect(),
            ),
        )?;
        dict.set_item("band_window_start", orbitals.band_window.start)?;
        dict.set_item("band_window_count", orbitals.band_window.count)?;
        dict.set_item(
            "energies",
            array2(
                py,
                orbitals.energies.len(),
                orbitals.band_window.count,
                orbitals
                    .energies
                    .iter()
                    .flat_map(|row| row.iter().map(|energy| energy.get()))
                    .collect(),
            ),
        )?;
        let eigenvectors = PyList::empty(py);
        for matrix in &orbitals.eigenvectors {
            eigenvectors.append(fortran_array2(
                py,
                matrix.rows(),
                matrix.columns(),
                matrix.to_host_column_major(),
            ))?;
        }
        dict.set_item("eigenvectors", eigenvectors)?;
        dict.set_item(
            "available_bands",
            PyArray1::from_vec(
                py,
                orbitals
                    .available_bands
                    .iter()
                    .map(|&count| count as i64)
                    .collect(),
            ),
        )?;
        Ok(dict.unbind())
    }

    fn export_basis(&self, py: Python<'_>, k: isize) -> PyResult<Py<PyDict>> {
        if k < 0 || k as usize >= self.inner.orbitals.bases.len() {
            return Err(PyIndexError::new_err(format!(
                "k index {k} is out of range"
            )));
        }
        let k_index = k as usize;
        let basis = &self.inner.orbitals.bases[k_index];
        let dict = export_dict(py)?;
        let n_pw = basis.plane_waves.len();
        dict.set_item("k_index", k_index)?;
        dict.set_item("basis_dimension", basis.layout.dimension())?;
        dict.set_item("spatial_plane_wave_count", n_pw)?;
        dict.set_item(
            "plane_wave_g",
            array2(
                py,
                n_pw,
                3,
                basis
                    .plane_waves
                    .iter()
                    .flat_map(|wave| wave.g.index)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "plane_wave_k_cartesian",
            array2(
                py,
                n_pw,
                3,
                basis
                    .plane_waves
                    .iter()
                    .flat_map(|wave| wave.k.map(|component| component.get()))
                    .collect(),
            ),
        )?;
        dict.set_item(
            "plane_wave_k_plus_g",
            array2(
                py,
                n_pw,
                3,
                basis
                    .plane_waves
                    .iter()
                    .flat_map(|wave| wave.q.map(|component| component.get()))
                    .collect(),
            ),
        )?;

        let mut pauli_rows = Vec::with_capacity(2 * n_pw * 3);
        for pauli in 0..2 {
            for g in 0..n_pw {
                pauli_rows.extend_from_slice(&[(pauli * n_pw + g) as i64, pauli as i64, g as i64]);
            }
        }
        dict.set_item("pauli_rows", array2(py, 2 * n_pw, 3, pauli_rows))?;

        let mut local_rows = Vec::new();
        let mut n_local = 0;
        for site in 0..basis.layout.site_count() {
            let layout = basis
                .layout
                .site_layout(site)
                .expect("site index comes from the compiled layout");
            for &(kappa, count) in layout.counts_by_kappa() {
                for twice_mu in kappa.twice_mu_values() {
                    for ordinal in 0..count {
                        let row = basis
                            .layout
                            .site_spinor_index(site, kappa, twice_mu, ordinal)
                            .expect("enumerated local orbital has a global row");
                        local_rows.extend_from_slice(&[
                            row as i64,
                            site as i64,
                            i64::from(kappa.get()),
                            twice_mu.get(),
                            ordinal as i64,
                            (muffintin::SPINOR_RADIAL_LO0 + ordinal) as i64,
                        ]);
                        n_local += 1;
                    }
                }
            }
        }
        dict.set_item("local_orbital_rows", array2(py, n_local, 6, local_rows))?;

        let mut projection_rows = Vec::new();
        let mut matching_labels = Vec::new();
        let mut matching_coefficients = Vec::new();
        for (site, waves) in basis.site_augmentations.iter().enumerate() {
            let channels = waves
                .first()
                .map(|wave| wave.channels.as_slice())
                .unwrap_or(&[]);
            let mut coordinate = 0_i64;
            for channel in channels {
                for radial_n in 0..2_i64 {
                    projection_rows.extend_from_slice(&[
                        site as i64,
                        coordinate,
                        i64::from(channel.kappa().get()),
                        channel.twice_mu().get(),
                        radial_n,
                    ]);
                    coordinate += 1;
                }
            }
            if let Some(layout) = basis.layout.site_layout(site) {
                for &(kappa, count) in layout.counts_by_kappa() {
                    for twice_mu in kappa.twice_mu_values() {
                        for ordinal in 0..count {
                            projection_rows.extend_from_slice(&[
                                site as i64,
                                coordinate,
                                i64::from(kappa.get()),
                                twice_mu.get(),
                                (muffintin::SPINOR_RADIAL_LO0 + ordinal) as i64,
                            ]);
                            coordinate += 1;
                        }
                    }
                }
            }
            for (g, wave) in waves.iter().enumerate() {
                for pauli in 0..2 {
                    for (channel_index, channel) in wave.channels.iter().enumerate() {
                        matching_labels.extend_from_slice(&[
                            site as i64,
                            g as i64,
                            pauli as i64,
                            i64::from(channel.kappa().get()),
                            channel.twice_mu().get(),
                        ]);
                        matching_coefficients
                            .extend_from_slice(wave.coefficient(pauli, channel_index));
                    }
                }
            }
        }
        dict.set_item(
            "projection_rows",
            array2(py, projection_rows.len() / 5, 5, projection_rows),
        )?;
        dict.set_item(
            "matching_labels",
            array2(py, matching_labels.len() / 5, 5, matching_labels),
        )?;
        dict.set_item(
            "matching_coefficients",
            array2(
                py,
                matching_coefficients.len() / 2,
                2,
                matching_coefficients,
            ),
        )?;
        Ok(dict.unbind())
    }

    fn export_radials(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let radials = &self.inner.source.radials;
        let dict = export_dict(py)?;
        dict.set_item(
            "mesh_site",
            PyArray1::from_vec(py, (0..radials.len()).map(|site| site as i64).collect()),
        )?;
        dict.set_item(
            "mesh_first",
            PyArray1::from_vec(
                py,
                radials.iter().map(|site| site.mesh.first().get()).collect(),
            ),
        )?;
        dict.set_item(
            "mesh_increment",
            PyArray1::from_vec(
                py,
                radials.iter().map(|site| site.mesh.increment()).collect(),
            ),
        )?;
        dict.set_item(
            "mesh_count",
            PyArray1::from_vec(
                py,
                radials.iter().map(|site| site.mesh.len() as i64).collect(),
            ),
        )?;
        let mut mesh_offsets = vec![0_i64];
        let mut mesh_radii = Vec::new();
        let mut mesh_weights = Vec::new();
        let mut radial_labels = Vec::new();
        let mut sample_offsets = vec![0_i64];
        let mut p = Vec::new();
        let mut q = Vec::new();
        let mut n_functions = 0;
        for (site_index, site) in radials.iter().enumerate() {
            mesh_radii.extend(site.mesh.radii().iter().map(|radius| radius.get()));
            mesh_weights.extend_from_slice(site.mesh.weights());
            mesh_offsets.push(mesh_radii.len() as i64);
            for (kind, functions) in [(0_i64, &site.valence), (1_i64, &site.cores)] {
                for function in functions {
                    radial_labels.extend_from_slice(&[
                        site_index as i64,
                        kind,
                        i64::from(function.kappa.get()),
                        function.n as i64,
                    ]);
                    p.extend_from_slice(&function.samples.large);
                    q.extend_from_slice(&function.samples.small);
                    sample_offsets.push(p.len() as i64);
                    n_functions += 1;
                }
            }
        }
        dict.set_item("mesh_offsets", PyArray1::from_vec(py, mesh_offsets))?;
        dict.set_item("mesh_radii", PyArray1::from_vec(py, mesh_radii))?;
        dict.set_item("mesh_weights", PyArray1::from_vec(py, mesh_weights))?;
        dict.set_item("radial_labels", array2(py, n_functions, 4, radial_labels))?;
        dict.set_item("sample_offsets", PyArray1::from_vec(py, sample_offsets))?;
        dict.set_item("p", PyArray1::from_vec(py, p))?;
        dict.set_item("q", PyArray1::from_vec(py, q))?;
        Ok(dict.unbind())
    }

    fn export_geometry(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let checkpoint = self.checkpoint.as_ref();
        let partition = &self.inner.source.partition;
        let dict = export_dict(py)?;
        dict.set_item(
            "site_id",
            checkpoint
                .geometry
                .sites
                .iter()
                .map(|site| site.id.as_str())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "atomic_number",
            PyArray1::from_vec(
                py,
                checkpoint
                    .geometry
                    .sites
                    .iter()
                    .map(|site| i64::from(site.atomic_number))
                    .collect(),
            ),
        )?;
        dict.set_item(
            "site_fractional",
            array2(
                py,
                checkpoint.geometry.sites.len(),
                3,
                checkpoint
                    .geometry
                    .sites
                    .iter()
                    .flat_map(|site| site.fractional_position)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "site_cartesian",
            array2(
                py,
                partition.site_count(),
                3,
                partition
                    .sites()
                    .iter()
                    .flat_map(|site| site.position.map(|coordinate| coordinate.get()))
                    .collect(),
            ),
        )?;
        dict.set_item(
            "muffin_tin_radius",
            PyArray1::from_vec(
                py,
                partition
                    .sites()
                    .iter()
                    .map(|site| site.radius.get())
                    .collect(),
            ),
        )?;
        dict.set_item(
            "direct_lattice",
            array2(
                py,
                3,
                3,
                checkpoint
                    .geometry
                    .lattice
                    .vectors
                    .iter()
                    .flat_map(|row| row.iter().copied())
                    .collect(),
            ),
        )?;
        dict.set_item(
            "reciprocal_lattice",
            array2(
                py,
                3,
                3,
                self.inner
                    .reciprocal
                    .basis()
                    .iter()
                    .flat_map(|row| row.iter().map(|component| component.get()))
                    .collect(),
            ),
        )?;
        dict.set_item("cell_volume", partition.interstitial().cell_volume().get())?;
        Ok(dict.unbind())
    }

    fn export_kq_map(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_dict(py)?;
        let map = &self.inner.k_minus_q;
        dict.set_item(
            "k_index",
            PyArray1::from_vec(py, map.iter().map(|entry| entry.k_index as i64).collect()),
        )?;
        dict.set_item(
            "kq_index",
            PyArray1::from_vec(py, map.iter().map(|entry| entry.kq_index as i64).collect()),
        )?;
        dict.set_item(
            "g_wrap_index",
            array2(
                py,
                map.len(),
                3,
                map.iter().flat_map(|entry| entry.umklapp.index).collect(),
            ),
        )?;
        dict.set_item(
            "g_wrap_cartesian",
            array2(
                py,
                map.len(),
                3,
                map.iter()
                    .flat_map(|entry| entry.umklapp.cartesian.map(|value| value.get()))
                    .collect(),
            ),
        )?;
        dict.set_item(
            "transfer_cartesian",
            PyArray1::from_vec(
                py,
                self.inner
                    .source
                    .q
                    .cartesian
                    .map(|component| component.get())
                    .to_vec(),
            ),
        )?;
        dict.set_item(
            "global_transfer_index",
            PyArray1::from_vec(py, self.inner.source.q.umklapp.index.to_vec()),
        )?;
        Ok(dict.unbind())
    }

    fn export_pair_support(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_dict(py)?;
        let components = &self.inner.source.interstitial_pair_support.components;
        dict.set_item(
            "g_relative_index",
            array2(
                py,
                components.len(),
                3,
                components
                    .iter()
                    .flat_map(|component| component.g_relative.index)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "g_relative_cartesian",
            array2(
                py,
                components.len(),
                3,
                components
                    .iter()
                    .flat_map(|component| {
                        component
                            .g_relative
                            .cartesian
                            .map(|coordinate| coordinate.get())
                    })
                    .collect(),
            ),
        )?;
        dict.set_item(
            "g_relative_norm",
            PyArray1::from_vec(
                py,
                components
                    .iter()
                    .map(|component| component.g_relative.norm.get())
                    .collect(),
            ),
        )?;
        Ok(dict.unbind())
    }

    fn export_pair_layout(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_dict(py)?;
        let layout = self.inner.pair_columns;
        dict.set_item("n_k", layout.n_k)?;
        dict.set_item("n_orb", layout.n_orb)?;
        dict.set_item(
            "n_columns",
            layout
                .n_columns()
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )?;
        dict.set_item("core_orbital", layout.core_orbital)?;
        dict.set_item("pair_order", "k*n_orb^2 + i*n_orb + j")?;
        Ok(dict.unbind())
    }
}

fn set_q(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    q: muffintin_prodbasis::TransferQ,
) -> PyResult<()> {
    dict.set_item(
        "q_cartesian",
        PyArray1::from_vec(py, q.cartesian.map(|value| value.get()).to_vec()),
    )?;
    dict.set_item(
        "q_umklapp_index",
        PyArray1::from_vec(py, q.umklapp.index.to_vec()),
    )?;
    dict.set_item(
        "q_umklapp_cartesian",
        PyArray1::from_vec(py, q.umklapp.cartesian.map(|value| value.get()).to_vec()),
    )?;
    Ok(())
}

fn region_rows(regions: impl IntoIterator<Item = AuxiliaryRegion>) -> Vec<i64> {
    let mut rows = Vec::new();
    for region in regions {
        let row = match region {
            AuxiliaryRegion::MuffinTin { site, l, m, n } => {
                [0, site as i64, i64::from(l), i64::from(m), n as i64]
            }
            AuxiliaryRegion::Interstitial { g } => [
                1,
                i64::from(g.index[0]),
                i64::from(g.index[1]),
                i64::from(g.index[2]),
                -1,
            ],
            AuxiliaryRegion::InterpolationPoint { id, region } => match region {
                InterpolationRegion::MuffinTin { site } => [2, id as i64, 0, site as i64, -1],
                InterpolationRegion::Interstitial => [2, id as i64, 1, -1, -1],
                InterpolationRegion::Uniform => [2, id as i64, 2, -1, -1],
            },
        };
        rows.extend_from_slice(&row);
    }
    rows
}

fn vertex_coefficients<'py, 'a>(
    py: Python<'py>,
    vertices: impl IntoIterator<Item = &'a muffintin_prodbasis::PairVertex>,
    dimension: usize,
) -> Bound<'py, PyArray2<Complex64>> {
    let values = vertices
        .into_iter()
        .flat_map(|vertex| vertex.coefficients().iter().copied())
        .collect::<Vec<_>>();
    let rows = values.len().checked_div(dimension).unwrap_or(0);
    array2(py, rows, dimension, values)
}

fn grid_region_rows(
    grid: &muffintin::ThcParentGrid,
    ids: impl IntoIterator<Item = usize>,
) -> Vec<i64> {
    let mut rows = Vec::new();
    for id in ids {
        match grid.points()[id].region {
            muffintin::ThcRegion::MuffinTin { site, radial_index } => {
                rows.extend_from_slice(&[0, site as i64, radial_index as i64]);
            }
            muffintin::ThcRegion::Interstitial => rows.extend_from_slice(&[1, -1, -1]),
        }
    }
    rows
}

fn set_optional_residual(
    dict: &Bound<'_, PyDict>,
    py: Python<'_>,
    key: &str,
    residual: Option<muffintin_prodbasis::thc::WeightedResidual>,
) -> PyResult<()> {
    match residual {
        Some(residual) => dict.set_item(
            key,
            PyArray1::from_vec(py, vec![residual.frobenius, residual.column_max]),
        ),
        None => dict.set_item(key, py.None()),
    }
}

#[pymethods]
impl SpinorMpbResult {
    fn export_auxiliary(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let auxiliary = &self.inner.auxiliary;
        let payload = auxiliary
            .require_mixed_product()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let dict = export_dict(py)?;
        dict.set_item("dimension", auxiliary.dimension())?;
        dict.set_item("mt_dimension", auxiliary.mt_dimension())?;
        dict.set_item("interstitial_dimension", auxiliary.interstitial_dimension())?;
        let regions = auxiliary.regions();
        set_q(&dict, py, auxiliary.q)?;
        dict.set_item(
            "regions",
            array2(py, regions.len(), 5, region_rows(regions)),
        )?;

        let mut mesh_site = Vec::new();
        let mut mesh_first = Vec::new();
        let mut mesh_increment = Vec::new();
        let mut mesh_count = Vec::new();
        let mut mesh_offsets = vec![0_i64];
        let mut mesh_radii = Vec::new();
        let mut mesh_weights = Vec::new();
        let mut mode_labels = Vec::new();
        let mut mode_offsets = vec![0_i64];
        let mut mode_radial = Vec::new();
        for site in &payload.sites {
            mesh_site.push(site.site as i64);
            mesh_first.push(site.mesh.first().get());
            mesh_increment.push(site.mesh.increment());
            mesh_count.push(site.mesh.len() as i64);
            mesh_radii.extend(site.mesh.radii().iter().map(|radius| radius.get()));
            mesh_weights.extend_from_slice(site.mesh.weights());
            mesh_offsets.push(mesh_radii.len() as i64);
            for mode in &site.modes {
                mode_labels.extend_from_slice(&[
                    site.site as i64,
                    i64::from(mode.l),
                    mode.n as i64,
                ]);
                mode_radial.extend_from_slice(&mode.radial);
                mode_offsets.push(mode_radial.len() as i64);
            }
        }
        dict.set_item("mt_mesh_site", PyArray1::from_vec(py, mesh_site))?;
        dict.set_item("mt_mesh_first", PyArray1::from_vec(py, mesh_first))?;
        dict.set_item("mt_mesh_increment", PyArray1::from_vec(py, mesh_increment))?;
        dict.set_item("mt_mesh_count", PyArray1::from_vec(py, mesh_count))?;
        dict.set_item("mt_mesh_offsets", PyArray1::from_vec(py, mesh_offsets))?;
        dict.set_item("mt_mesh_radii", PyArray1::from_vec(py, mesh_radii))?;
        dict.set_item("mt_mesh_weights", PyArray1::from_vec(py, mesh_weights))?;
        dict.set_item(
            "mt_mode_labels",
            array2(py, mode_labels.len() / 3, 3, mode_labels),
        )?;
        dict.set_item("mt_mode_offsets", PyArray1::from_vec(py, mode_offsets))?;
        dict.set_item("mt_mode_radial", PyArray1::from_vec(py, mode_radial))?;
        let waves = &payload.interstitial.waves;
        dict.set_item(
            "interstitial_g_index",
            array2(
                py,
                waves.len(),
                3,
                waves.iter().flat_map(|wave| wave.g.index).collect(),
            ),
        )?;
        dict.set_item(
            "interstitial_g_cartesian",
            array2(
                py,
                waves.len(),
                3,
                waves
                    .iter()
                    .flat_map(|wave| wave.g.cartesian.map(|value| value.get()))
                    .collect(),
            ),
        )?;
        dict.set_item(
            "interstitial_q_plus_g",
            array2(
                py,
                waves.len(),
                3,
                waves
                    .iter()
                    .flat_map(|wave| wave.q_plus_g.map(|value| value.get()))
                    .collect(),
            ),
        )?;
        dict.set_item(
            "interstitial_q_plus_g_norm",
            PyArray1::from_vec(
                py,
                waves.iter().map(|wave| wave.q_plus_g_norm.get()).collect(),
            ),
        )?;
        match payload.cutoff {
            Some(cutoff) => {
                dict.set_item("cutoff_kind", "spectral-overlap")?;
                dict.set_item("cutoff_value", cutoff.value)?;
                dict.set_item("cutoff_nspin_factor", cutoff.nspin_factor)?;
            }
            None => {
                dict.set_item("cutoff_kind", py.None())?;
                dict.set_item("cutoff_value", py.None())?;
                dict.set_item("cutoff_nspin_factor", py.None())?;
            }
        }
        Ok(dict.unbind())
    }

    fn export_vertices(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_dict(py)?;
        let mut labels = Vec::with_capacity(self.inner.vertices.len() * 4);
        for selected in &self.inner.vertices {
            labels.extend_from_slice(&[
                selected.k as i64,
                selected.left_band as i64,
                selected.right_band as i64,
                selected.column as i64,
            ]);
        }
        dict.set_item("labels", array2(py, self.inner.vertices.len(), 4, labels))?;
        dict.set_item(
            "coefficients",
            vertex_coefficients(
                py,
                self.inner.vertices.iter().map(|selected| &selected.vertex),
                self.inner.auxiliary.dimension(),
            ),
        )?;
        let regions = self.inner.auxiliary.regions();
        dict.set_item(
            "regions",
            array2(py, regions.len(), 5, region_rows(regions)),
        )?;
        Ok(dict.unbind())
    }
}

#[pymethods]
impl SpinorThcResult {
    fn export_selection(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_dict(py)?;
        let selection = &self.inner.selection;
        dict.set_item(
            "requested_rank",
            match self.inner.requested_rank {
                muffintin::RankPolicy::Exact { n_mu } => n_mu,
                muffintin::RankPolicy::Threshold { n_max, .. } => n_max,
            },
        )?;
        dict.set_item("effective_rank", self.inner.effective_rank)?;
        dict.set_item(
            "point_ids",
            PyArray1::from_vec(
                py,
                selection
                    .points
                    .iter()
                    .map(|point| point.id as i64)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "point_regions",
            array2(
                py,
                selection.points.len(),
                3,
                grid_region_rows(&self.grid, selection.points.iter().map(|point| point.id)),
            ),
        )?;
        dict.set_item(
            "pivots",
            PyArray1::from_vec(py, selection.pivots.iter().map(|&id| id as i64).collect()),
        )?;
        dict.set_item(
            "candidates",
            PyArray1::from_vec(py, self.candidates.iter().map(|&id| id as i64).collect()),
        )?;
        dict.set_item("strategy", selection.provenance.strategy.as_str())?;
        dict.set_item(
            "engine",
            match selection.provenance.engine {
                muffintin_prodbasis::thc::L2Engine::FullColumnPivotedQr => "qrcp",
                muffintin_prodbasis::thc::L2Engine::FullPivotedCholesky => "pivoted-cholesky",
                muffintin_prodbasis::thc::L2Engine::StructuredSketch { .. } => "structured-sketch",
            },
        )?;
        dict.set_item("q_set", selection.provenance.q_set)?;
        dict.set_item("weights", selection.provenance.weights)?;
        dict.set_item("seed", selection.provenance.seed)?;
        dict.set_item("n_points", self.grid.points().len())?;
        dict.set_item("n_candidates", self.candidates.len())?;
        Ok(dict.unbind())
    }

    fn export_records(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_dict(py)?;
        let n_points = self.grid.points().len();
        dict.set_item(
            "coordinates",
            array2(
                py,
                n_points,
                3,
                self.grid
                    .points()
                    .iter()
                    .flat_map(|point| point.coordinate.map(|value| value.get()))
                    .collect(),
            ),
        )?;
        dict.set_item(
            "weights",
            PyArray1::from_vec(
                py,
                self.grid
                    .points()
                    .iter()
                    .map(|point| point.weight)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "regions",
            array2(py, n_points, 3, grid_region_rows(&self.grid, 0..n_points)),
        )?;
        let records = PyList::empty(py);
        for record in &self.inner.records {
            let item = export_dict(py)?;
            set_q(&item, py, record.q)?;
            item.set_item("q_index", record.q_index)?;
            item.set_item("rank", record.fit.rank)?;
            item.set_item("n_points", record.fit.n_points)?;
            item.set_item("n_mu", record.fit.n_mu)?;
            item.set_item(
                "zeta",
                array2(
                    py,
                    record.fit.n_points,
                    record.fit.n_mu,
                    record.fit.zeta.clone(),
                ),
            )?;
            item.set_item(
                "l2_all",
                PyArray1::from_vec(
                    py,
                    vec![record.fit.l2_all.frobenius, record.fit.l2_all.column_max],
                ),
            )?;
            set_optional_residual(&item, py, "l2_core", record.fit.l2_core)?;
            set_optional_residual(&item, py, "l2_valence", record.fit.l2_valence)?;
            set_optional_residual(&item, py, "coulomb_residual", record.fit.coulomb)?;
            let n_columns = record
                .layout
                .n_columns()
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            item.set_item(
                "vertices",
                vertex_coefficients(py, record.vertices.iter(), record.fit.n_mu),
            )?;
            item.set_item(
                "vertex_labels",
                array2(
                    py,
                    n_columns,
                    4,
                    (0..n_columns)
                        .flat_map(|column| {
                            let (k, left, right) = record.layout.decode(column);
                            [k as i64, left as i64, right as i64, column as i64]
                        })
                        .collect(),
                ),
            )?;
            let ids = record
                .auxiliary
                .require_interpolation_points()
                .map_err(|error| PyValueError::new_err(error.to_string()))?
                .iter()
                .map(|point| point.id)
                .collect::<Vec<_>>();
            item.set_item(
                "point_ids",
                PyArray1::from_vec(py, ids.iter().map(|&id| id as i64).collect()),
            )?;
            item.set_item(
                "point_regions",
                array2(py, ids.len(), 3, grid_region_rows(&self.grid, ids)),
            )?;
            records.append(item)?;
        }
        dict.set_item("records", records)?;
        Ok(dict.unbind())
    }
}

#[pymethods]
impl SpinorCoulombResult {
    fn export_matrix(&self, py: Python<'_>, q_index: isize) -> PyResult<Py<PyDict>> {
        if q_index < 0 || q_index as usize >= self.inner.records().len() {
            return Err(PyIndexError::new_err(format!(
                "q index {q_index} is out of range"
            )));
        }
        let record = &self.inner.records()[q_index as usize];
        let dict = crate::export::export_coulomb_operator(py, &record.operator)?;
        dict.set_item("q_index", record.q_index)?;
        dict.set_item("spin", py.None())?;
        Ok(dict.unbind())
    }

    fn export_diagnostics(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let diagnostics = &self.inner.diagnostics;
        let dict = export_dict(py)?;
        let mut pairs = Vec::with_capacity(diagnostics.len() * 3);
        let mut columns = Vec::with_capacity(diagnostics.len());
        let mut mpb_quadratic = Vec::with_capacity(diagnostics.len());
        let mut thc_quadratic = Vec::with_capacity(diagnostics.len());
        let mut mpb_action_norm = Vec::with_capacity(diagnostics.len());
        let mut thc_action_norm = Vec::with_capacity(diagnostics.len());
        let mut absolute = Vec::with_capacity(diagnostics.len());
        let mut relative = Vec::with_capacity(diagnostics.len());
        for diagnostic in diagnostics {
            let OrbitalPair::Bloch {
                k_index,
                left,
                right,
            } = diagnostic.pair
            else {
                return Err(PyValueError::new_err("spinor diagnostic pair is not Bloch"));
            };
            pairs.extend_from_slice(&[k_index as i64, left as i64, right as i64]);
            columns.push(diagnostic.column as i64);
            mpb_quadratic.push(diagnostic.mpb_quadratic);
            thc_quadratic.push(diagnostic.thc_quadratic);
            mpb_action_norm.push(diagnostic.mpb_action_norm);
            thc_action_norm.push(diagnostic.thc_action_norm);
            absolute.push(diagnostic.quadratic_discrepancy.absolute);
            relative.push(diagnostic.quadratic_discrepancy.relative);
        }
        dict.set_item("pairs", array2(py, diagnostics.len(), 3, pairs))?;
        dict.set_item("columns", PyArray1::from_vec(py, columns))?;
        dict.set_item("mpb_quadratic", PyArray1::from_vec(py, mpb_quadratic))?;
        dict.set_item("thc_quadratic", PyArray1::from_vec(py, thc_quadratic))?;
        dict.set_item("mpb_action_norm", PyArray1::from_vec(py, mpb_action_norm))?;
        dict.set_item("thc_action_norm", PyArray1::from_vec(py, thc_action_norm))?;
        dict.set_item("absolute", PyArray1::from_vec(py, absolute))?;
        dict.set_item("relative", PyArray1::from_vec(py, relative))?;
        Ok(dict.unbind())
    }
}

#[pyfunction]
pub(crate) fn build_spinor_mpb(
    input: PyRef<'_, SpinorProductInput>,
    selections: PyReadonlyArray2<'_, i64>,
    product_l_max: u32,
    product_g_max: f64,
    overlap_tolerance: f64,
) -> PyResult<SpinorMpbResult> {
    let selections = selections.as_array();
    if selections.shape()[1] != 3 || selections.shape()[0] == 0 {
        return Err(PyValueError::new_err(
            "selections must have shape (S, 3) with S > 0",
        ));
    }
    let mut parsed = Vec::with_capacity(selections.shape()[0]);
    for row in selections.rows() {
        parsed.push(muffintin::SpinorMpbSelection {
            k: usize::try_from(row[0])
                .map_err(|_| PyValueError::new_err("selection k must be nonnegative"))?,
            left_band: usize::try_from(row[1])
                .map_err(|_| PyValueError::new_err("selection left band must be nonnegative"))?,
            right_band: usize::try_from(row[2])
                .map_err(|_| PyValueError::new_err("selection right band must be nonnegative"))?,
        });
    }
    let spec = muffintin::SpinorMpbSpec {
        product_l_max,
        product_g_max: InverseBohr(product_g_max),
        overlap_tolerance,
        selections: parsed,
    };
    let result = muffintin::build_spinor_mpb(input.inner.as_ref(), &spec)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(SpinorMpbResult {
        _checkpoint: Arc::clone(&input.checkpoint),
        _input: Arc::clone(&input.inner),
        inner: Arc::new(result),
        _spec: spec,
    })
}

fn parse_parent_grid(
    input: &muffintin::SpinorProductInput,
    coordinates: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    regions: PyReadonlyArray2<'_, i64>,
) -> PyResult<muffintin::ThcParentGrid> {
    let coordinates = coordinates.as_array();
    let weights = weights.as_array();
    let regions = regions.as_array();
    let n_points = coordinates.shape()[0];
    if coordinates.shape()[1] != 3 || regions.shape() != [n_points, 3] || weights.len() != n_points
    {
        return Err(PyValueError::new_err(
            "coordinates, weights, and regions must have shapes (P,3), (P,), and (P,3)",
        ));
    }
    let mut points = Vec::with_capacity(n_points);
    for point in 0..n_points {
        let region = match (
            regions[[point, 0]],
            regions[[point, 1]],
            regions[[point, 2]],
        ) {
            (0, site, radial) => muffintin::ThcRegion::MuffinTin {
                site: usize::try_from(site)
                    .map_err(|_| PyValueError::new_err("muffin-tin site must be nonnegative"))?,
                radial_index: usize::try_from(radial).map_err(|_| {
                    PyValueError::new_err("muffin-tin radial index must be nonnegative")
                })?,
            },
            (1, -1, -1) => muffintin::ThcRegion::Interstitial,
            _ => {
                return Err(PyValueError::new_err(
                    "regions rows must be (0, site, radial) or (1, -1, -1)",
                ));
            }
        };
        points.push(muffintin::ThcPoint {
            coordinate: [
                Bohr(coordinates[[point, 0]]),
                Bohr(coordinates[[point, 1]]),
                Bohr(coordinates[[point, 2]]),
            ],
            weight: weights[point],
            region,
        });
    }
    muffintin::ThcParentGrid::new(
        input.source.partition.clone(),
        Provenance::default(),
        points,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (slice, coordinates, weights, regions, rank, engine, candidates=None))]
pub(crate) fn build_spinor_thc(
    slice: PyRef<'_, SpinorProductSlice>,
    coordinates: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    regions: PyReadonlyArray2<'_, i64>,
    rank: usize,
    engine: &str,
    candidates: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<SpinorThcResult> {
    let first = slice
        .inner
        .first()
        .ok_or_else(|| PyValueError::new_err("spinor product slice is empty"))?;
    let grid = parse_parent_grid(first, coordinates, weights, regions)?;
    let candidates = match candidates {
        Some(values) => values
            .as_array()
            .iter()
            .map(|&value| {
                usize::try_from(value)
                    .map_err(|_| PyValueError::new_err("candidate indices must be nonnegative"))
            })
            .collect::<PyResult<Vec<_>>>()?,
        None => grid
            .points()
            .iter()
            .enumerate()
            .filter_map(|(index, point)| (point.weight > 0.0).then_some(index))
            .collect(),
    };
    let engine = match engine {
        "qrcp" => muffintin::ThcEngine::FullColumnPivotedQr,
        "pivoted-cholesky" => muffintin::ThcEngine::FullPivotedCholesky,
        _ => {
            return Err(PyValueError::new_err(
                "engine must be 'qrcp' or 'pivoted-cholesky'",
            ));
        }
    };
    let spec = muffintin::SpinorThcSpec {
        rank: muffintin::RankPolicy::Exact { n_mu: rank },
        candidates: muffintin::ThcCandidates::Indices(candidates.clone()),
        engine,
    };
    let result = muffintin::build_spinor_thc(slice.inner.as_slice(), &grid, &spec)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(SpinorThcResult {
        _checkpoint: Arc::clone(&slice.checkpoint),
        _slice: SpinorProductSlice::clone(&*slice),
        inner: Arc::new(result),
        grid,
        _spec: spec,
        candidates,
    })
}

#[pyfunction]
#[pyo3(signature = (slice, thc, lexp, interpolation_pw_cutoff, interpolation_l_max, comparisons=None))]
pub(crate) fn build_spinor_coulomb(
    py: Python<'_>,
    slice: PyRef<'_, SpinorProductSlice>,
    thc: PyRef<'_, SpinorThcResult>,
    lexp: u32,
    interpolation_pw_cutoff: f64,
    interpolation_l_max: u32,
    comparisons: Option<Vec<(usize, Py<SpinorMpbResult>, usize)>>,
) -> PyResult<SpinorCoulombResult> {
    let cell = Cell::new(
        slice
            .checkpoint
            .geometry
            .lattice
            .vectors
            .map(|row| row.map(Bohr)),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let spec = muffintin::SpinorCoulombSpec {
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
        .map(|(q_index, mpb, vertex)| muffintin::SpinorCoulombPairMatch {
            q_index: *q_index,
            mpb: mpb.as_ref(),
            mpb_vertex: *vertex,
        })
        .collect::<Vec<_>>();
    let result = muffintin::build_spinor_coulomb(
        slice.inner.as_slice(),
        thc.inner.as_ref(),
        &spec,
        &comparisons,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(SpinorCoulombResult {
        _checkpoint: Arc::clone(&slice.checkpoint),
        _slice: SpinorProductSlice::clone(&*slice),
        _thc: SpinorThcResult::clone(&*thc),
        inner: Arc::new(result),
        spec,
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SpinorProductInput>()?;
    module.add_class::<SpinorProductSlice>()?;
    module.add_class::<SpinorMpbResult>()?;
    module.add_class::<SpinorThcResult>()?;
    module.add_class::<SpinorCoulombResult>()?;
    module.add_function(wrap_pyfunction!(build_spinor_mpb, module)?)?;
    module.add_function(wrap_pyfunction!(build_spinor_thc, module)?)?;
    module.add_function(wrap_pyfunction!(build_spinor_coulomb, module)?)?;
    Ok(())
}
