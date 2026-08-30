use std::sync::Arc;

use muffintin_core::HermitianFourierField;
use muffintin_dft::{
    InterstitialField, MuffinTinField, NoncollinearXcRoute, RegionalScalarField,
    ScfExchangeCorrelation, ScfPotentialBuild, XcFunctional,
};
use muffintin_sphere::{HarmonicConvention, SphereField};
use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::checkpoint::{RegionalFieldLayout, Structure};
use crate::scf::export_regional;

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Reusable method-neutral regional Pauli density.
#[pyclass(name = "RegionalDensity", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct RegionalDensity {
    pub(crate) inner: Arc<muffintin_dft::RegionalDensity>,
    pub(crate) structure: Arc<muffintin::Structure>,
}

impl RegionalDensity {
    pub(crate) fn from_runtime(
        inner: muffintin_dft::RegionalDensity,
        structure: Arc<muffintin::Structure>,
    ) -> Self {
        Self {
            inner: Arc::new(inner),
            structure,
        }
    }
}

/// Reusable method-neutral regional Pauli potential and its energy contractions.
#[pyclass(name = "RegionalPotential", module = "libmuffintin._native", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct RegionalPotential {
    pub(crate) inner: Arc<ScfPotentialBuild>,
    pub(crate) structure: Arc<muffintin::Structure>,
}

#[pymethods]
impl RegionalDensity {
    #[new]
    #[pyo3(signature = (
        structure,
        layout,
        angular_basis,
        interstitial_components,
        mt_channel_labels,
        mt_sample_offsets,
        mt_components
    ))]
    fn new(
        structure: PyRef<'_, Structure>,
        layout: PyRef<'_, RegionalFieldLayout>,
        angular_basis: &str,
        interstitial_components: PyReadonlyArray2<'_, Complex64>,
        mt_channel_labels: PyReadonlyArray2<'_, i64>,
        mt_sample_offsets: PyReadonlyArray1<'_, i64>,
        mt_components: PyReadonlyArray2<'_, Complex64>,
    ) -> PyResult<Self> {
        if layout.inner.fourier().reciprocal() != structure.inner.reciprocal() {
            return Err(PyValueError::new_err(
                "regional-field layout belongs to a different structure lattice",
            ));
        }
        let convention = match angular_basis {
            "complex-condon-shortley" => HarmonicConvention::Complex,
            "real-tesseral-condon-shortley" => HarmonicConvention::Real,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown angular basis {angular_basis:?}; expected 'complex-condon-shortley' or 'real-tesseral-condon-shortley'"
                )));
            }
        };

        let interstitial_components = interstitial_components.as_array();
        let expected_g_count = layout.inner.fourier().len();
        if interstitial_components.shape() != [4, expected_g_count] {
            return Err(PyValueError::new_err(format!(
                "interstitial_components has shape {:?}, expected (4, {expected_g_count})",
                interstitial_components.shape()
            )));
        }

        let site_meshes = structure.inner.site_meshes().cloned().collect::<Vec<_>>();
        let mut expected_labels = Vec::new();
        let mut expected_offsets = vec![0_i64];
        for (site, mesh) in site_meshes.iter().enumerate() {
            for l in 0..=layout.inner.muffin_tin_l_max() {
                for m in -(i64::from(l))..=i64::from(l) {
                    expected_labels.push([site as i64, i64::from(l), m]);
                    expected_offsets
                        .push(expected_offsets.last().copied().unwrap() + mesh.len() as i64);
                }
            }
        }

        let mt_channel_labels = mt_channel_labels.as_array();
        if mt_channel_labels.shape() != [expected_labels.len(), 3] {
            return Err(PyValueError::new_err(format!(
                "mt_channel_labels has shape {:?}, expected ({}, 3)",
                mt_channel_labels.shape(),
                expected_labels.len()
            )));
        }
        for (row, expected) in expected_labels.iter().enumerate() {
            let actual = [
                mt_channel_labels[(row, 0)],
                mt_channel_labels[(row, 1)],
                mt_channel_labels[(row, 2)],
            ];
            if actual != *expected {
                return Err(PyValueError::new_err(format!(
                    "mt_channel_labels row {row} is {actual:?}, expected {expected:?}"
                )));
            }
        }

        let mt_sample_offsets = mt_sample_offsets.as_array();
        if mt_sample_offsets.len() != expected_offsets.len()
            || !mt_sample_offsets
                .iter()
                .copied()
                .eq(expected_offsets.iter().copied())
        {
            return Err(PyValueError::new_err(format!(
                "mt_sample_offsets must equal the exact site-mesh offsets {expected_offsets:?}"
            )));
        }
        let expected_sample_count = expected_offsets.last().copied().unwrap() as usize;
        let mt_components = mt_components.as_array();
        if mt_components.shape() != [4, expected_sample_count] {
            return Err(PyValueError::new_err(format!(
                "mt_components has shape {:?}, expected (4, {expected_sample_count})",
                mt_components.shape()
            )));
        }

        let mut components = Vec::with_capacity(4);
        for component in 0..4 {
            let interstitial = InterstitialField::from_fourier_field(
                HermitianFourierField::new(
                    layout.inner.fourier().clone(),
                    (0..expected_g_count)
                        .map(|g| interstitial_components[(component, g)])
                        .collect(),
                )
                .map_err(py_error)?,
            );
            let mut muffin_tins = Vec::with_capacity(site_meshes.len());
            for (site, mesh) in site_meshes.iter().enumerate() {
                let channels = expected_labels
                    .iter()
                    .enumerate()
                    .filter(|(_, label)| label[0] == site as i64)
                    .map(|(channel, label)| {
                        let start = expected_offsets[channel] as usize;
                        let end = expected_offsets[channel + 1] as usize;
                        (
                            (label[1] as u32, label[2] as i32),
                            (start..end)
                                .map(|sample| mt_components[(component, sample)])
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let field = SphereField::new(convention, channels).map_err(py_error)?;
                muffin_tins.push(MuffinTinField::new(mesh.clone(), field).map_err(py_error)?);
            }
            components.push(
                RegionalScalarField::new(
                    structure.inner.interstitial_geometry().clone(),
                    muffin_tins,
                    interstitial,
                )
                .map_err(py_error)?,
            );
        }
        let [charge, mx, my, mz]: [RegionalScalarField; 4] = components
            .try_into()
            .expect("exactly four regional components were constructed");
        let density =
            muffintin_dft::RegionalDensity::new(charge, [mx, my, mz]).map_err(py_error)?;
        Ok(Self::from_runtime(density, Arc::clone(&structure.inner)))
    }

    fn export_interstitial(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        export_regional(py, muffintin::density_fourier(self.inner.as_ref()))
    }

    fn physical_inner_product(&self, other: PyRef<'_, Self>) -> PyResult<f64> {
        self.require_same_structure(&other)?;
        self.inner
            .physical_inner_product(other.inner.as_ref())
            .map_err(py_error)
    }

    fn residual_rms(&self) -> PyResult<f64> {
        self.inner.residual_rms().map_err(py_error)
    }

    fn difference_rms(&self, other: PyRef<'_, Self>) -> PyResult<f64> {
        self.require_same_structure(&other)?;
        self.inner
            .difference_rms(other.inner.as_ref())
            .map_err(py_error)
    }

    fn add_scaled(&self, scale: f64, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.require_same_structure(&other)?;
        let mut result = self.inner.as_ref().clone();
        result
            .add_scaled(scale, other.inner.as_ref())
            .map_err(py_error)?;
        Ok(Self::from_runtime(result, Arc::clone(&self.structure)))
    }

    fn difference(&self, other: PyRef<'_, Self>) -> PyResult<Self> {
        self.require_same_structure(&other)?;
        let result = self
            .inner
            .difference(other.inner.as_ref())
            .map_err(py_error)?;
        Ok(Self::from_runtime(result, Arc::clone(&self.structure)))
    }
}

impl RegionalDensity {
    fn require_same_structure(&self, other: &Self) -> PyResult<()> {
        if self.structure.geometry() == other.structure.geometry() {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "regional densities belong to different structures",
            ))
        }
    }
}

#[pymethods]
impl RegionalPotential {
    fn export_interstitial(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        export_regional(py, muffintin::potential_fourier(&self.inner.potential))
    }

    #[getter]
    fn madelung(&self) -> f64 {
        self.inner.energy_terms.madelung.get()
    }

    #[getter]
    fn coulomb(&self) -> f64 {
        self.inner.energy_terms.coulomb.get()
    }

    #[getter]
    fn exchange_correlation(&self) -> f64 {
        self.inner.energy_terms.exchange_correlation.get()
    }

    #[getter]
    fn exchange_correlation_potential(&self) -> f64 {
        self.inner.energy_terms.exchange_correlation_potential.get()
    }
}

#[pyfunction]
#[pyo3(signature = (density, xc, noncollinear_route="local-spin-frame"))]
pub(crate) fn build_regional_potential(
    density: PyRef<'_, RegionalDensity>,
    xc: &str,
    noncollinear_route: &str,
) -> PyResult<RegionalPotential> {
    let functional = match xc {
        "lda-pw92" => XcFunctional::LdaPw92,
        "pbe" => XcFunctional::Pbe,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown exchange-correlation functional {xc:?}; expected 'lda-pw92' or 'pbe'"
            )));
        }
    };
    let noncollinear_route = match noncollinear_route {
        "local-spin-frame" => NoncollinearXcRoute::LocalSpinFrame,
        "magnetization-field" => NoncollinearXcRoute::MagnetizationField,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown noncollinear XC route {noncollinear_route:?}; expected 'local-spin-frame' or 'magnetization-field'"
            )));
        }
    };
    let inner = muffintin_dft::build_scf_potential(
        density.inner.as_ref(),
        density.structure.nuclear_charges(),
        ScfExchangeCorrelation {
            functional,
            noncollinear_route,
        },
    )
    .map_err(py_error)?;
    Ok(RegionalPotential {
        inner: Arc::new(inner),
        structure: Arc::clone(&density.structure),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RegionalDensity>()?;
    module.add_class::<RegionalPotential>()?;
    module.add_function(wrap_pyfunction!(build_regional_potential, module)?)?;
    Ok(())
}
