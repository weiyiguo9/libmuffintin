use muffintin_core::lm_from_index;
use num_complex::Complex64;
use numpy::ndarray::{Array2, ShapeBuilder};
use numpy::{Element, PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::checkpoint::ScalarProductInput;

const SCHEMA: &str = "libmuffintin.pyexport";
const VERSION: i64 = 1;

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
impl ScalarProductInput {
    fn export_orbitals(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let input = self.inner.as_ref();
        let dict = export_dict(py)?;
        let k = input
            .orbitals
            .k_fractional
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        dict.set_item(
            "k_fractional",
            array2(py, input.orbitals.k_fractional.len(), 3, k),
        )?;
        dict.set_item("band_window_start", input.orbitals.band_window.start)?;
        dict.set_item("band_window_count", input.orbitals.band_window.count)?;

        let channels = PyList::empty(py);
        for source in &input.orbitals.channels {
            let channel = PyDict::new(py);
            channel.set_item("spin", source.spin)?;
            let energies = source
                .energies
                .iter()
                .flat_map(|row| row.iter().map(|energy| energy.get()))
                .collect();
            channel.set_item(
                "energies",
                array2(
                    py,
                    source.energies.len(),
                    input.orbitals.band_window.count,
                    energies,
                ),
            )?;
            let eigenvectors = PyList::empty(py);
            for matrix in &source.eigenvectors {
                eigenvectors.append(fortran_array2(
                    py,
                    matrix.rows(),
                    matrix.columns(),
                    matrix.to_host_column_major(),
                ))?;
            }
            channel.set_item("eigenvectors", eigenvectors)?;
            channel.set_item(
                "available_bands",
                PyArray1::from_vec(
                    py,
                    source
                        .available_bands
                        .iter()
                        .map(|&count| count as i64)
                        .collect(),
                ),
            )?;
            channels.append(channel)?;
        }
        dict.set_item("channels", channels)?;
        Ok(dict.unbind())
    }

    fn export_basis(&self, py: Python<'_>, k: isize, spin: i64) -> PyResult<Py<PyDict>> {
        let channel = self
            .inner
            .orbitals
            .channels
            .iter()
            .find(|channel| i64::from(channel.spin) == spin)
            .ok_or_else(|| PyValueError::new_err(format!("invalid scalar spin {spin}")))?;
        if k < 0 || k as usize >= channel.bases.len() {
            return Err(PyIndexError::new_err(format!(
                "k index {k} is out of range"
            )));
        }
        let k_index = k as usize;
        let basis = &channel.bases[k_index];
        let dict = export_dict(py)?;
        dict.set_item("k_index", k_index)?;
        dict.set_item("spin", spin)?;
        dict.set_item("basis_dimension", basis.layout.dimension())?;
        dict.set_item("plane_wave_count", basis.plane_waves.len())?;

        let plane_wave_g = basis
            .plane_waves
            .iter()
            .flat_map(|wave| wave.g.index)
            .collect();
        let plane_wave_k = basis
            .plane_waves
            .iter()
            .flat_map(|wave| wave.k.map(|component| component.get()))
            .collect();
        let plane_wave_k_plus_g = basis
            .plane_waves
            .iter()
            .flat_map(|wave| wave.q.map(|component| component.get()))
            .collect();
        dict.set_item(
            "plane_wave_g",
            array2(py, basis.plane_waves.len(), 3, plane_wave_g),
        )?;
        dict.set_item(
            "plane_wave_k_cartesian",
            array2(py, basis.plane_waves.len(), 3, plane_wave_k),
        )?;
        dict.set_item(
            "plane_wave_k_plus_g",
            array2(py, basis.plane_waves.len(), 3, plane_wave_k_plus_g),
        )?;

        let mut apw_labels = Vec::new();
        let mut apw_coefficients = Vec::new();
        for (site, waves) in basis.site_augmentations.iter().enumerate() {
            for (g, wave) in waves.iter().enumerate() {
                for (index, coefficients) in wave.coefficients.iter().enumerate() {
                    let lm = lm_from_index(index);
                    apw_labels.extend_from_slice(&[
                        site as i64,
                        g as i64,
                        i64::from(lm.l),
                        i64::from(lm.m),
                    ]);
                    apw_coefficients.extend_from_slice(coefficients);
                }
            }
        }
        let n_apw = apw_coefficients.len() / 2;
        dict.set_item("apw_labels", array2(py, n_apw, 4, apw_labels))?;
        dict.set_item("apw_coefficients", array2(py, n_apw, 2, apw_coefficients))?;

        let mut local_rows = Vec::new();
        let mut n_local = 0;
        for site in 0..basis.layout.site_count() {
            let layout = basis
                .layout
                .site_layout(site)
                .expect("site index comes from the compiled layout");
            for (l, &count) in layout.counts_by_l().iter().enumerate() {
                for m in -(l as i32)..=l as i32 {
                    for ordinal in 0..count {
                        let global_row = basis
                            .layout
                            .local_orbital_index(site, l as u32, m, ordinal)
                            .expect("enumerated local orbital has a global row");
                        local_rows.extend_from_slice(&[
                            global_row as i64,
                            site as i64,
                            l as i64,
                            i64::from(m),
                            ordinal as i64,
                            (muffintin::SCALAR_RADIAL_LO0 + ordinal) as i64,
                        ]);
                        n_local += 1;
                    }
                }
            }
        }
        dict.set_item("local_orbital_rows", array2(py, n_local, 6, local_rows))?;
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
        let mut large = Vec::new();
        let mut small_present = Vec::new();
        let mut small_offsets = vec![0_i64];
        let mut small = Vec::new();
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
                        i64::from(function.l),
                        function.n as i64,
                        i64::from(function.spin),
                    ]);
                    large.extend_from_slice(&function.samples.large);
                    sample_offsets.push(large.len() as i64);
                    if let Some(samples) = &function.samples.small {
                        small_present.push(true);
                        small.extend_from_slice(samples);
                    } else {
                        small_present.push(false);
                    }
                    small_offsets.push(small.len() as i64);
                    n_functions += 1;
                }
            }
        }
        dict.set_item("mesh_offsets", PyArray1::from_vec(py, mesh_offsets))?;
        dict.set_item("mesh_radii", PyArray1::from_vec(py, mesh_radii))?;
        dict.set_item("mesh_weights", PyArray1::from_vec(py, mesh_weights))?;
        dict.set_item("radial_labels", array2(py, n_functions, 5, radial_labels))?;
        dict.set_item("sample_offsets", PyArray1::from_vec(py, sample_offsets))?;
        dict.set_item("large", PyArray1::from_vec(py, large))?;
        dict.set_item("small_present", PyArray1::from_vec(py, small_present))?;
        dict.set_item("small_offsets", PyArray1::from_vec(py, small_offsets))?;
        dict.set_item("small", PyArray1::from_vec(py, small))?;
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
        let site_fractional = checkpoint
            .geometry
            .sites
            .iter()
            .flat_map(|site| site.fractional_position)
            .collect();
        let site_cartesian = partition
            .sites()
            .iter()
            .flat_map(|site| site.position.map(|coordinate| coordinate.get()))
            .collect();
        let muffin_tin_radius = partition
            .sites()
            .iter()
            .map(|site| site.radius.get())
            .collect();
        dict.set_item(
            "site_fractional",
            array2(py, checkpoint.geometry.sites.len(), 3, site_fractional),
        )?;
        dict.set_item(
            "site_cartesian",
            array2(py, partition.site_count(), 3, site_cartesian),
        )?;
        dict.set_item(
            "muffin_tin_radius",
            PyArray1::from_vec(py, muffin_tin_radius),
        )?;
        let direct = checkpoint
            .geometry
            .lattice
            .vectors
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let reciprocal = self
            .inner
            .reciprocal
            .basis()
            .iter()
            .flat_map(|row| row.iter().map(|component| component.get()))
            .collect();
        dict.set_item("direct_lattice", array2(py, 3, 3, direct))?;
        dict.set_item("reciprocal_lattice", array2(py, 3, 3, reciprocal))?;
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
