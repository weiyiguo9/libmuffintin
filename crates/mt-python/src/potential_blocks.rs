use std::sync::Arc;
use std::time::Instant;

use muffintin_coulomb::WeinertHartreeSpec;
use muffintin_dft::{
    ElectrostaticSpec, NoncollinearXcRoute, PreparedScalarRegionalXc, RegionalElectrostaticResult,
    XcFunctional, assemble_scalar_regional_xc, assemble_scf_potential_from_parts,
    evaluate_regional_electrostatics, evaluate_scalar_interstitial_xc_point_block,
    evaluate_scalar_muffin_tin_xc_shell_block, prepare_scalar_regional_xc, xc_spec_for_density,
};
use muffintin_sphere::HarmonicConvention;
use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2,
    PyReadwriteArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::export::export_dict;
use crate::regional::{RegionalDensity, RegionalPotential};

const RADIAL_BLOCK_SIZE: usize = 32;
const INTERSTITIAL_BLOCK_SIZE: usize = 256;

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn parse_functional(value: &str) -> PyResult<XcFunctional> {
    match value {
        "lda-pw92" => Ok(XcFunctional::LdaPw92),
        "pbe" => Ok(XcFunctional::Pbe),
        _ => Err(PyValueError::new_err(format!(
            "unknown exchange-correlation functional {value:?}; expected 'lda-pw92' or 'pbe'"
        ))),
    }
}

fn functional_name(functional: XcFunctional) -> &'static str {
    match functional {
        XcFunctional::LdaPw92 => "lda-pw92",
        XcFunctional::Pbe => "pbe",
    }
}

fn parse_noncollinear_route(value: &str) -> PyResult<NoncollinearXcRoute> {
    match value {
        "local-spin-frame" => Ok(NoncollinearXcRoute::LocalSpinFrame),
        "magnetization-field" => Ok(NoncollinearXcRoute::MagnetizationField),
        _ => Err(PyValueError::new_err(format!(
            "unknown noncollinear XC route {value:?}; expected 'local-spin-frame' or 'magnetization-field'"
        ))),
    }
}

fn parse_convention(value: &str) -> PyResult<HarmonicConvention> {
    match value {
        "complex-condon-shortley" => Ok(HarmonicConvention::Complex),
        "real-tesseral-condon-shortley" => Ok(HarmonicConvention::Real),
        _ => Err(PyValueError::new_err(format!(
            "unknown angular basis {value:?}; expected 'complex-condon-shortley' or 'real-tesseral-condon-shortley'"
        ))),
    }
}

fn convention_name(convention: HarmonicConvention) -> &'static str {
    match convention {
        HarmonicConvention::Complex => "complex-condon-shortley",
        HarmonicConvention::Real => "real-tesseral-condon-shortley",
    }
}

/// Root-only state retaining the exact density and electrostatic fields while
/// scalar XC point and shell blocks are evaluated by Python-owned MPI ranks.
#[pyclass(
    name = "PreparedScfPotential",
    module = "libmuffintin._native",
    unsendable
)]
pub(crate) struct PreparedScfPotential {
    source_density: Arc<muffintin_dft::RegionalDensity>,
    structure: Arc<muffintin::Structure>,
    electrostatic: Option<RegionalElectrostaticResult>,
    exchange_correlation: PreparedScalarRegionalXc,
    hartree_seconds: f64,
    xc_prepare_seconds: f64,
    xc_assemble_seconds: Option<f64>,
}

#[pymethods]
impl PreparedScfPotential {
    fn export_xc_inputs(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let prepared = &self.exchange_correlation;
        let spec = prepared.spec();
        let site_count = prepared.muffin_tin_site_count();
        let output_lm_count = (spec.output_l_max as usize + 1).pow(2);
        let mut mt_channel_labels = Vec::new();
        let mut mt_sample_offsets = vec![0_i64];
        let mut mt_site_channel_offsets = vec![0_i64];
        let mut mt_site_sample_offsets = vec![0_i64];
        let mut mt_radial_offsets = vec![0_i64];
        let mut mt_mesh_radii = Vec::new();
        let mut mt_input_l_max = Vec::with_capacity(site_count);
        let mut mt_charge_channels = Vec::new();
        let mut convention = None;

        for site in 0..site_count {
            let site_convention = prepared.muffin_tin_convention(site);
            if convention.is_some_and(|value| value != site_convention) {
                return Err(PyValueError::new_err(
                    "scalar XC block preparation requires one angular basis across sites",
                ));
            }
            convention = Some(site_convention);
            let input_l_max = prepared.muffin_tin_input_l_max(site);
            let radii = prepared.muffin_tin_radii(site);
            let channels = prepared.muffin_tin_charge_channels(site);
            mt_input_l_max.push(i64::from(input_l_max));
            mt_mesh_radii.extend_from_slice(radii);
            mt_charge_channels.extend_from_slice(channels);
            for l in 0..=input_l_max {
                for m in -(l as i32)..=l as i32 {
                    mt_channel_labels.push([site as i64, i64::from(l), i64::from(m)]);
                    mt_sample_offsets
                        .push(mt_sample_offsets.last().copied().unwrap() + radii.len() as i64);
                }
            }
            mt_site_channel_offsets.push(mt_channel_labels.len() as i64);
            mt_site_sample_offsets.push(mt_charge_channels.len() as i64);
            mt_radial_offsets.push(mt_mesh_radii.len() as i64);
        }

        let total_mt_samples = mt_charge_channels.len();
        let mut mt_components = vec![Complex64::default(); 4 * total_mt_samples];
        mt_components[..total_mt_samples].copy_from_slice(&mt_charge_channels);
        let point_count = prepared.interstitial_point_count();
        let derivative_count = prepared.interstitial_density_derivatives().len() / point_count;
        let dict = export_dict(py)?;
        dict.set_item("functional", functional_name(prepared.functional()))?;
        dict.set_item(
            "angular_basis",
            convention_name(convention.unwrap_or(HarmonicConvention::Complex)),
        )?;
        dict.set_item("angular_point_count", spec.angular_point_count)?;
        dict.set_item("output_l_max", spec.output_l_max)?;
        dict.set_item("output_lm_count", output_lm_count)?;
        dict.set_item("radial_block_size", RADIAL_BLOCK_SIZE)?;
        dict.set_item("interstitial_block_size", INTERSTITIAL_BLOCK_SIZE)?;
        dict.set_item("mt_input_l_max", PyArray1::from_vec(py, mt_input_l_max))?;
        dict.set_item(
            "mt_components",
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((4, total_mt_samples), mt_components)
                    .expect("four scalar-density components have one common sample count"),
            ),
        )?;
        dict.set_item(
            "mt_channel_labels",
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec(
                    (mt_channel_labels.len(), 3),
                    mt_channel_labels.into_iter().flatten().collect(),
                )
                .expect("every muffin-tin channel label has three entries"),
            ),
        )?;
        dict.set_item(
            "mt_sample_offsets",
            PyArray1::from_vec(py, mt_sample_offsets),
        )?;
        dict.set_item(
            "mt_site_channel_offsets",
            PyArray1::from_vec(py, mt_site_channel_offsets),
        )?;
        dict.set_item(
            "mt_site_sample_offsets",
            PyArray1::from_vec(py, mt_site_sample_offsets),
        )?;
        dict.set_item(
            "mt_radial_offsets",
            PyArray1::from_vec(py, mt_radial_offsets),
        )?;
        dict.set_item("mt_mesh_radii", PyArray1::from_vec(py, mt_mesh_radii))?;
        dict.set_item(
            "is_density_derivatives",
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec(
                    (derivative_count, point_count),
                    prepared.interstitial_density_derivatives().to_vec(),
                )
                .expect("interstitial derivatives are field-major"),
            ),
        )?;
        dict.set_item(
            "is_theta",
            PyArray1::from_vec(py, prepared.interstitial_step().to_vec()),
        )?;
        dict.set_item("is_point_weight", prepared.interstitial_point_weight())?;
        Ok(dict.unbind())
    }

    fn timings(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = export_dict(py)?;
        dict.set_item("hartree", self.hartree_seconds)?;
        dict.set_item("xc_prepare", self.xc_prepare_seconds)?;
        dict.set_item("fft_plan", self.exchange_correlation.fft_plan_seconds())?;
        dict.set_item(
            "fft_transforms",
            self.exchange_correlation.fft_transform_seconds(),
        )?;
        dict.set_item("xc_assemble", self.xc_assemble_seconds)?;
        Ok(dict.unbind())
    }
}

#[pyfunction]
#[pyo3(signature = (density, xc, noncollinear_route="local-spin-frame"))]
fn prepare_scf_potential_blocks(
    density: PyRef<'_, RegionalDensity>,
    xc: &str,
    noncollinear_route: &str,
) -> PyResult<PreparedScfPotential> {
    let functional = parse_functional(xc)?;
    let noncollinear_route = parse_noncollinear_route(noncollinear_route)?;
    let output_l_max = std::iter::once(density.inner.charge())
        .chain(density.inner.magnetization())
        .flat_map(muffintin_dft::RegionalScalarField::muffin_tins)
        .flat_map(|field| field.field().channels().map(|(channel, _)| channel.l))
        .max()
        .unwrap_or(0);
    let xc_spec = xc_spec_for_density(density.inner.as_ref(), output_l_max, noncollinear_route);

    let started = Instant::now();
    let electrostatic = evaluate_regional_electrostatics(
        density.inner.charge(),
        &ElectrostaticSpec::new(
            WeinertHartreeSpec::electronic(4).map_err(py_error)?,
            density.structure.nuclear_charges().to_vec(),
        )
        .map_err(py_error)?,
    )
    .map_err(py_error)?;
    let hartree_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let exchange_correlation =
        prepare_scalar_regional_xc(functional, density.inner.as_ref(), xc_spec)
            .map_err(py_error)?;
    let xc_prepare_seconds = started.elapsed().as_secs_f64();
    Ok(PreparedScfPotential {
        source_density: Arc::clone(&density.inner),
        structure: Arc::clone(&density.structure),
        electrostatic: Some(electrostatic),
        exchange_correlation,
        hartree_seconds,
        xc_prepare_seconds,
        xc_assemble_seconds: None,
    })
}

#[pyfunction]
#[pyo3(signature = (
    functional,
    angular_basis,
    radii,
    charge_channels,
    input_l_max,
    output_l_max,
    angular_point_count,
    radial_start,
    potential_out,
    integrands_out
))]
fn evaluate_muffin_tin_xc_block(
    functional: &str,
    angular_basis: &str,
    radii: PyReadonlyArray1<'_, f64>,
    charge_channels: PyReadonlyArray2<'_, Complex64>,
    input_l_max: u32,
    output_l_max: u32,
    angular_point_count: usize,
    radial_start: usize,
    mut potential_out: PyReadwriteArray3<'_, Complex64>,
    mut integrands_out: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let output_lm_count = (output_l_max as usize + 1).pow(2);
    let block_count = potential_out.shape()[0];
    if charge_channels.shape() != [(input_l_max as usize + 1).pow(2), radii.len()]
        || potential_out.shape() != [block_count, 4, output_lm_count]
        || integrands_out.shape() != [block_count, 2]
    {
        return Err(PyValueError::new_err(
            "invalid muffin-tin XC block array shape",
        ));
    }
    evaluate_scalar_muffin_tin_xc_shell_block(
        parse_functional(functional)?,
        parse_convention(angular_basis)?,
        radii.as_slice()?,
        charge_channels.as_slice()?,
        input_l_max,
        output_l_max,
        angular_point_count,
        radial_start,
        potential_out.as_slice_mut()?,
        integrands_out.as_slice_mut()?,
    )
    .map_err(py_error)
}

#[pyfunction]
#[pyo3(signature = (
    functional,
    density_derivatives,
    theta,
    point_weight,
    point_start,
    potential_out,
    integrands_out
))]
fn evaluate_interstitial_xc_block(
    functional: &str,
    density_derivatives: PyReadonlyArray2<'_, f64>,
    theta: PyReadonlyArray1<'_, f64>,
    point_weight: f64,
    point_start: usize,
    mut potential_out: PyReadwriteArray2<'_, f64>,
    mut integrands_out: PyReadwriteArray2<'_, f64>,
) -> PyResult<()> {
    let total_points = theta.len();
    let field_count = if parse_functional(functional)? == XcFunctional::Pbe {
        10
    } else {
        1
    };
    let block_count = potential_out.shape()[0];
    if density_derivatives.shape() != [field_count, total_points]
        || potential_out.shape() != [block_count, 4]
        || integrands_out.shape() != [block_count, 2]
    {
        return Err(PyValueError::new_err(
            "invalid interstitial XC block array shape",
        ));
    }
    evaluate_scalar_interstitial_xc_point_block(
        parse_functional(functional)?,
        density_derivatives.as_slice()?,
        total_points,
        theta.as_slice()?,
        point_weight,
        point_start,
        potential_out.as_slice_mut()?,
        integrands_out.as_slice_mut()?,
    )
    .map_err(py_error)
}

#[pyfunction]
fn assemble_scf_potential_blocks(
    mut prepared: PyRefMut<'_, PreparedScfPotential>,
    mt_potential: PyReadonlyArray3<'_, Complex64>,
    mt_integrands: PyReadonlyArray2<'_, f64>,
    is_potential: PyReadonlyArray2<'_, f64>,
    is_integrands: PyReadonlyArray2<'_, f64>,
) -> PyResult<RegionalPotential> {
    if prepared.electrostatic.is_none() {
        return Err(PyValueError::new_err(
            "prepared SCF potential has already been assembled",
        ));
    }
    let spec = prepared.exchange_correlation.spec();
    let radial_count = (0..prepared.exchange_correlation.muffin_tin_site_count())
        .map(|site| prepared.exchange_correlation.muffin_tin_radii(site).len())
        .sum::<usize>();
    let lm_count = (spec.output_l_max as usize + 1).pow(2);
    let point_count = prepared.exchange_correlation.interstitial_point_count();
    if mt_potential.shape() != [radial_count, 4, lm_count]
        || mt_integrands.shape() != [radial_count, 2]
        || is_potential.shape() != [point_count, 4]
        || is_integrands.shape() != [point_count, 2]
    {
        return Err(PyValueError::new_err("invalid assembled XC array shape"));
    }
    let functional = prepared.exchange_correlation.functional();
    let started = Instant::now();
    let exchange_correlation = assemble_scalar_regional_xc(
        &mut prepared.exchange_correlation,
        mt_potential.as_slice()?,
        mt_integrands.as_slice()?,
        is_potential.as_slice()?,
        is_integrands.as_slice()?,
    )
    .map_err(py_error)?;
    prepared.xc_assemble_seconds = Some(started.elapsed().as_secs_f64());
    let electrostatic = prepared.electrostatic.take().unwrap();
    let inner = assemble_scf_potential_from_parts(
        prepared.source_density.as_ref(),
        electrostatic,
        exchange_correlation,
        functional,
        spec,
    )
    .map_err(py_error)?;
    Ok(RegionalPotential {
        inner: Arc::new(inner),
        structure: Arc::clone(&prepared.structure),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedScfPotential>()?;
    module.add_function(wrap_pyfunction!(prepare_scf_potential_blocks, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_muffin_tin_xc_block, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_interstitial_xc_block, module)?)?;
    module.add_function(wrap_pyfunction!(assemble_scf_potential_blocks, module)?)?;
    Ok(())
}
