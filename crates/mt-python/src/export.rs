use muffintin_core::lm_from_index;
use num_complex::Complex64;
use numpy::ndarray::{Array2, Array3, ShapeBuilder};
use numpy::{Element, PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::checkpoint::ScalarProductInput;
use crate::coulomb::{ScalarCoulombResult, ScalarMpbCoulombResult};
use crate::products::ScalarMpbResult;
use crate::thc::ScalarThcResult;

pub(crate) const SCHEMA: &str = "libmuffintin.pyexport";
pub(crate) const VERSION: i64 = 1;

pub(crate) fn export_dict(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("schema", SCHEMA)?;
    dict.set_item("version", VERSION)?;
    Ok(dict)
}

pub(crate) fn array2<'py, T: Element>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    values: Vec<T>,
) -> Bound<'py, PyArray2<T>> {
    let array = Array2::from_shape_vec((rows, columns), values)
        .expect("export row count and flattened data length agree");
    PyArray2::from_owned_array(py, array)
}

fn array3<'py, T: Element>(
    py: Python<'py>,
    first: usize,
    second: usize,
    third: usize,
    values: Vec<T>,
) -> Bound<'py, PyArray3<T>> {
    let array = Array3::from_shape_vec((first, second, third), values)
        .expect("export dimensions and flattened data length agree");
    PyArray3::from_owned_array(py, array)
}

pub(crate) fn export_orbital_samples(
    py: Python<'_>,
    samples: &muffintin::ScalarOrbitalSamples,
) -> PyResult<Py<PyDict>> {
    let dict = export_dict(py)?;
    dict.set_item(
        "large",
        array3(
            py,
            samples.n_points,
            samples.n_k,
            samples.n_orb,
            samples.large.clone(),
        ),
    )?;
    dict.set_item(
        "small",
        array3(
            py,
            samples.n_points,
            samples.n_k,
            samples.n_orb,
            samples.small.clone(),
        ),
    )?;
    Ok(dict.unbind())
}

pub(crate) fn fortran_array2<'py>(
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

fn auxiliary_region_table(
    regions: impl IntoIterator<Item = muffintin_prodbasis::AuxiliaryRegion>,
) -> Vec<i64> {
    let mut table = Vec::new();
    for region in regions {
        match region {
            muffintin_prodbasis::AuxiliaryRegion::MuffinTin { site, l, m, n } => {
                table.extend_from_slice(&[0, site as i64, i64::from(l), i64::from(m), n as i64]);
            }
            muffintin_prodbasis::AuxiliaryRegion::Interstitial { g } => {
                table.extend_from_slice(&[
                    1,
                    i64::from(g.index[0]),
                    i64::from(g.index[1]),
                    i64::from(g.index[2]),
                    -1,
                ]);
            }
            muffintin_prodbasis::AuxiliaryRegion::InterpolationPoint { id, region } => {
                let (kind, site) = match region {
                    muffintin_prodbasis::InterpolationRegion::MuffinTin { site } => {
                        (0_i64, site as i64)
                    }
                    muffintin_prodbasis::InterpolationRegion::Interstitial => (1, -1),
                    muffintin_prodbasis::InterpolationRegion::Uniform => (2, -1),
                };
                table.extend_from_slice(&[2, id as i64, kind, site, -1]);
            }
        }
    }
    table
}

fn grid_region_table(
    grid: &muffintin::ThcParentGrid,
    ids: impl IntoIterator<Item = usize>,
) -> Vec<i64> {
    let mut table = Vec::new();
    for id in ids {
        match grid.points()[id].region {
            muffintin::ThcRegion::MuffinTin { site, radial_index } => {
                table.extend_from_slice(&[0, site as i64, radial_index as i64]);
            }
            muffintin::ThcRegion::Interstitial => table.extend_from_slice(&[1, -1, -1]),
        }
    }
    table
}

#[pymethods]
impl ScalarMpbResult {
    fn export_auxiliary(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let auxiliary = &self.inner.auxiliary;
        let payload = auxiliary
            .require_mixed_product()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let dict = export_dict(py)?;
        set_q(&dict, py, auxiliary.q)?;
        dict.set_item("dimension", auxiliary.dimension())?;
        dict.set_item("mt_dimension", auxiliary.mt_dimension())?;
        dict.set_item("interstitial_dimension", auxiliary.interstitial_dimension())?;
        dict.set_item(
            "regions",
            array2(
                py,
                auxiliary.dimension(),
                5,
                auxiliary_region_table(auxiliary.regions()),
            ),
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
        for block in &payload.sites {
            mesh_site.push(block.site as i64);
            mesh_first.push(block.mesh.first().get());
            mesh_increment.push(block.mesh.increment());
            mesh_count.push(block.mesh.len() as i64);
            mesh_radii.extend(block.mesh.radii().iter().map(|radius| radius.get()));
            mesh_weights.extend_from_slice(block.mesh.weights());
            mesh_offsets.push(mesh_radii.len() as i64);
            for mode in &block.modes {
                mode_labels.extend_from_slice(&[
                    block.site as i64,
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
            array2(py, mode_offsets.len() - 1, 3, mode_labels),
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
        let n_aux = self.inner.auxiliary.dimension();
        let mut labels = Vec::with_capacity(self.inner.vertices.len() * 5);
        let mut coefficients = Vec::with_capacity(self.inner.vertices.len() * n_aux);
        for vertex in &self.inner.vertices {
            labels.extend_from_slice(&[
                i64::from(vertex.spin),
                vertex.k as i64,
                vertex.left_band as i64,
                vertex.right_band as i64,
                vertex.column as i64,
            ]);
            coefficients.extend_from_slice(vertex.vertex.coefficients());
        }
        dict.set_item(
            "regions",
            array2(
                py,
                n_aux,
                5,
                auxiliary_region_table(self.inner.auxiliary.regions()),
            ),
        )?;
        dict.set_item("labels", array2(py, self.inner.vertices.len(), 5, labels))?;
        dict.set_item(
            "coefficients",
            array2(py, self.inner.vertices.len(), n_aux, coefficients),
        )?;
        Ok(dict.unbind())
    }
}

#[pymethods]
impl ScalarThcResult {
    fn export_selection(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let selection = &self.inner.selection;
        let dict = export_dict(py)?;
        dict.set_item("spin", self.inner.spin)?;
        dict.set_item(
            "requested_rank",
            match self.spec.rank {
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
                grid_region_table(&self.grid, selection.points.iter().map(|point| point.id)),
            ),
        )?;
        dict.set_item(
            "pivots",
            PyArray1::from_vec(py, selection.pivots.iter().map(|&id| id as i64).collect()),
        )?;
        dict.set_item(
            "diagonal",
            PyArray1::from_vec(py, self.selection_diagonal.clone()),
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
            array2(py, n_points, 3, grid_region_table(&self.grid, 0..n_points)),
        )?;
        let records = PyList::empty(py);
        for (record, block) in self.inner.records.iter().zip(self.pair_blocks.iter()) {
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
                array2(
                    py,
                    n_columns,
                    record.fit.n_mu,
                    record
                        .vertices
                        .iter()
                        .flat_map(|vertex| vertex.coefficients().iter().copied())
                        .collect(),
                ),
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
            item.set_item(
                "pair_samples",
                array2(
                    py,
                    block.n_points,
                    block.n_columns(),
                    block.values().to_vec(),
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
                array2(py, ids.len(), 3, grid_region_table(&self.grid, ids)),
            )?;
            records.append(item)?;
        }
        dict.set_item("records", records)?;
        Ok(dict.unbind())
    }
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
impl ScalarCoulombResult {
    fn export_matrix(&self, py: Python<'_>, q_index: isize) -> PyResult<Py<PyDict>> {
        if q_index < 0 || q_index as usize >= self.inner.records.len() {
            return Err(PyIndexError::new_err(format!(
                "q index {q_index} is out of range"
            )));
        }
        let record = &self.inner.records[q_index as usize];
        let dict = export_coulomb_operator(py, &record.operator)?;
        dict.set_item("q_index", record.q_index)?;
        dict.set_item("spin", record.spin)?;
        Ok(dict.unbind())
    }

    fn export_diagnostics(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let diagnostics = &self.inner.diagnostics;
        let dict = export_dict(py)?;
        let mut k_index = Vec::with_capacity(diagnostics.len());
        let mut left = Vec::with_capacity(diagnostics.len());
        let mut right = Vec::with_capacity(diagnostics.len());
        for diagnostic in diagnostics {
            match diagnostic.pair {
                muffintin_prodbasis::OrbitalPair::Bloch {
                    k_index: k,
                    left: i,
                    right: j,
                } => {
                    k_index.push(k as i64);
                    left.push(i as i64);
                    right.push(j as i64);
                }
                _ => return Err(PyValueError::new_err("scalar diagnostic pair is not Bloch")),
            }
        }
        dict.set_item(
            "q_index",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.q_index as i64)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "spin",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| i64::from(value.spin))
                    .collect(),
            ),
        )?;
        dict.set_item("k_index", PyArray1::from_vec(py, k_index))?;
        dict.set_item("left_band", PyArray1::from_vec(py, left))?;
        dict.set_item("right_band", PyArray1::from_vec(py, right))?;
        dict.set_item(
            "column",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.column as i64)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "mpb_quadratic",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.mpb_quadratic)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "thc_quadratic",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.thc_quadratic)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "mpb_action_norm",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.mpb_action_norm)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "thc_action_norm",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.thc_action_norm)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "quadratic_absolute",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.quadratic_discrepancy.absolute)
                    .collect(),
            ),
        )?;
        dict.set_item(
            "quadratic_relative",
            PyArray1::from_vec(
                py,
                diagnostics
                    .iter()
                    .map(|value| value.quadratic_discrepancy.relative)
                    .collect(),
            ),
        )?;
        Ok(dict.unbind())
    }
}

pub(crate) fn export_coulomb_operator<'py>(
    py: Python<'py>,
    operator: &muffintin_coulomb::CoulombOperator,
) -> PyResult<Bound<'py, PyDict>> {
    let dimension = operator.dimension();
    let dict = export_dict(py)?;
    set_q(&dict, py, operator.q())?;
    dict.set_item("dimension", dimension)?;
    dict.set_item("mt_dimension", operator.mt_dimension())?;
    dict.set_item("interstitial_dimension", operator.interstitial_dimension())?;
    dict.set_item(
        "matrix",
        array2(py, dimension, dimension, operator.matrix().to_vec()),
    )?;
    dict.set_item(
        "regions",
        array2(
            py,
            dimension,
            5,
            auxiliary_region_table(operator.regions().iter().copied()),
        ),
    )?;
    match operator.gamma() {
        Some(gamma) => {
            dict.set_item("gamma_present", true)?;
            dict.set_item(
                "gamma_spherical_average_subtracted",
                gamma.spherical_average_subtracted,
            )?;
            dict.set_item("gamma_head_prefactor", gamma.head_prefactor)?;
            dict.set_item(
                "gamma_constant_coefficients",
                PyArray1::from_vec(py, gamma.constant_coefficients.clone()),
            )?;
        }
        None => {
            dict.set_item("gamma_present", false)?;
            dict.set_item("gamma_spherical_average_subtracted", py.None())?;
            dict.set_item("gamma_head_prefactor", py.None())?;
            dict.set_item(
                "gamma_constant_coefficients",
                PyArray1::from_vec(py, Vec::<Complex64>::new()),
            )?;
        }
    }
    Ok(dict)
}

#[pymethods]
impl ScalarMpbCoulombResult {
    fn export_matrix(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_coulomb_operator(py, self.inner.as_ref())?;
        dict.set_item("q_index", py.None())?;
        dict.set_item("spin", py.None())?;
        Ok(dict.unbind())
    }
}
