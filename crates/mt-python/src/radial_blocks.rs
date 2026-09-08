use std::sync::Arc;

use muffintin_core::{Bohr, ExponentialMesh, Hartree, Kappa};
use muffintin_dft::{
    PreparedRegionalCorePotentials, PreparedScalarRadialPotential, RadialEquation,
    RegionalCoreResult, SolvedCoreState, assemble_regional_core_site, core_state_search_provenance,
    prepare_regional_core_potentials, prepare_scalar_radial_potential,
    sample_scalar_radials_on_potential, solve_core_state_block,
};
use muffintin_sphere::{CoreDiracSpec, CoreState, EnergyBracket, SPEX_SPEED_OF_LIGHT};
use numpy::ndarray::Array2;
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::{CoreResult, CoreStation};
use crate::export::export_dict;
use crate::regional::RegionalPotential;

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn parse_radial_equation(name: &str) -> PyResult<RadialEquation> {
    match name {
        "schroedinger" => Ok(RadialEquation::Schroedinger),
        "scalar-koelling-harmon" => Ok(RadialEquation::ScalarKoellingHarmon),
        _ => Err(PyValueError::new_err(format!(
            "unknown radial equation {name:?}; expected 'schroedinger' or 'scalar-koelling-harmon'"
        ))),
    }
}

#[pyfunction]
pub(crate) fn prepare_scalar_radial_blocks(
    py: Python<'_>,
    potential: PyRef<'_, RegionalPotential>,
) -> PyResult<Py<PyDict>> {
    let site_count = potential.inner.potential.scalar().muffin_tins().len();
    let prepared = (0..site_count)
        .map(|site| prepare_scalar_radial_potential(&potential.inner.potential, site))
        .collect::<Result<Vec<_>, _>>()
        .map_err(py_error)?;
    export_scalar_radial_blocks(py, &prepared)
}

fn export_scalar_radial_blocks(
    py: Python<'_>,
    prepared: &[PreparedScalarRadialPotential],
) -> PyResult<Py<PyDict>> {
    let mut offsets = Vec::with_capacity(prepared.len() + 1);
    let mut mesh_first = Vec::with_capacity(prepared.len());
    let mut mesh_increment = Vec::with_capacity(prepared.len());
    let mut mesh_count = Vec::with_capacity(prepared.len());
    let mut mesh_radii = Vec::new();
    let mut potential_values = Vec::new();
    offsets.push(0_i64);
    for site in prepared {
        mesh_first.push(site.mesh.first().get());
        mesh_increment.push(site.mesh.increment());
        mesh_count.push(site.mesh.len() as i64);
        mesh_radii.extend(site.mesh.radii().iter().map(|radius| radius.get()));
        potential_values.extend_from_slice(&site.values);
        offsets.push(potential_values.len() as i64);
    }
    let dict = export_dict(py)?;
    dict.set_item("site_offsets", PyArray1::from_vec(py, offsets))?;
    dict.set_item("mesh_first", PyArray1::from_vec(py, mesh_first))?;
    dict.set_item("mesh_increment", PyArray1::from_vec(py, mesh_increment))?;
    dict.set_item("mesh_count", PyArray1::from_vec(py, mesh_count))?;
    dict.set_item("mesh_radii", PyArray1::from_vec(py, mesh_radii))?;
    dict.set_item("potential_values", PyArray1::from_vec(py, potential_values))?;
    Ok(dict.unbind())
}

#[pyfunction]
#[pyo3(signature = (
    potential_values,
    mesh_first,
    mesh_increment,
    site_index,
    radial_equation,
    l,
    energy,
    radial_samples,
    small_radial_samples,
    inverse_mass_samples,
    boundary_jets,
    speed_of_light=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_scalar_radial_block(
    potential_values: PyReadonlyArray1<'_, f64>,
    mesh_first: f64,
    mesh_increment: f64,
    site_index: usize,
    radial_equation: &str,
    l: u32,
    energy: f64,
    mut radial_samples: PyReadwriteArray1<'_, f64>,
    mut small_radial_samples: PyReadwriteArray1<'_, f64>,
    mut inverse_mass_samples: PyReadwriteArray1<'_, f64>,
    mut boundary_jets: PyReadwriteArray1<'_, f64>,
    speed_of_light: Option<f64>,
) -> PyResult<()> {
    let values = potential_values
        .as_slice()
        .map_err(|_| PyValueError::new_err("potential_values must be contiguous"))?;
    let mesh =
        ExponentialMesh::new(Bohr(mesh_first), mesh_increment, values.len()).map_err(py_error)?;
    let equation = parse_radial_equation(radial_equation)?;
    let speed_of_light = speed_of_light.unwrap_or(SPEX_SPEED_OF_LIGHT);
    let samples = sample_scalar_radials_on_potential(
        site_index,
        &mesh,
        values,
        equation,
        l,
        &[Hartree(energy)],
        speed_of_light,
    )
    .map_err(py_error)?;
    if radial_samples.len() != values.len()
        || small_radial_samples.len() != values.len()
        || inverse_mass_samples.len() != values.len()
        || boundary_jets.len() != 7
    {
        return Err(PyValueError::new_err(
            "scalar radial outputs must have shapes (nr), (nr), (nr), and (7)",
        ));
    }
    radial_samples
        .as_array_mut()
        .iter_mut()
        .zip(samples.radial_samples)
        .for_each(|(target, source)| *target = source);
    small_radial_samples
        .as_array_mut()
        .iter_mut()
        .zip(samples.small_radial_samples)
        .for_each(|(target, source)| *target = source);
    let c_inverse_squared = match equation {
        RadialEquation::Schroedinger => 0.0,
        RadialEquation::ScalarKoellingHarmon => 1.0 / (speed_of_light * speed_of_light),
    };
    inverse_mass_samples
        .as_array_mut()
        .iter_mut()
        .zip(values)
        .for_each(|(target, potential)| {
            *target = (2.0 + (energy - potential) * c_inverse_squared).recip();
        });
    let boundary = samples.boundary_radial[0];
    let energy_boundary = samples.energy_derivative_boundary_radial[0];
    let boundary_inverse_mass =
        (2.0 + (energy - values[values.len() - 1]) * c_inverse_squared).recip();
    let mut jets = boundary_jets.as_array_mut();
    jets[0] = boundary[0];
    jets[1] = boundary[1];
    jets[2] = energy_boundary[0];
    jets[3] = energy_boundary[1];
    jets[4] = boundary_inverse_mass;
    jets[5] = -c_inverse_squared * boundary_inverse_mass * boundary_inverse_mass;
    jets[6] = match equation {
        RadialEquation::Schroedinger => 0.0,
        RadialEquation::ScalarKoellingHarmon => speed_of_light.recip(),
    };
    Ok(())
}

#[pyclass(
    name = "PreparedCoreRadialBlocks",
    module = "libmuffintin._native",
    frozen
)]
#[derive(Clone, Debug)]
pub(crate) struct PreparedCoreRadialBlocks {
    potential: RegionalPotential,
    sites: Vec<muffintin_dft::CoreSiteRequest>,
    speed_of_light: f64,
    prepared: PreparedRegionalCorePotentials,
}

#[pyfunction]
pub(crate) fn prepare_core_radial_blocks(
    station: PyRef<'_, CoreStation>,
    potential: PyRef<'_, RegionalPotential>,
) -> PyResult<PreparedCoreRadialBlocks> {
    for site in &station.sites {
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
    let prepared = prepare_regional_core_potentials(potential.inner.as_ref(), &station.sites)
        .map_err(py_error)?;
    Ok(PreparedCoreRadialBlocks {
        potential: RegionalPotential {
            inner: Arc::clone(&potential.inner),
            structure: Arc::clone(&potential.structure),
        },
        sites: station.sites.clone(),
        speed_of_light: station.speed_of_light,
        prepared,
    })
}

#[pymethods]
impl PreparedCoreRadialBlocks {
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let density = self.potential.inner.source_density();
        let mut site_indices = Vec::with_capacity(self.sites.len());
        let mut site_ids = Vec::with_capacity(self.sites.len());
        let mut site_offsets = Vec::with_capacity(self.sites.len() + 1);
        let mut mesh_first = Vec::with_capacity(self.sites.len());
        let mut mesh_increment = Vec::with_capacity(self.sites.len());
        let mut mesh_count = Vec::with_capacity(self.sites.len());
        let mut nuclear_charges = Vec::with_capacity(self.sites.len());
        let mut muffin_tin_radii = Vec::with_capacity(self.sites.len());
        let mut potential_values = Vec::new();
        let task_count = self.sites.iter().map(|site| site.states.len()).sum();
        let mut tasks = Vec::with_capacity(task_count * 4);
        let mut task_radial_offsets = Vec::with_capacity(task_count + 1);
        site_offsets.push(0_i64);
        task_radial_offsets.push(0_i64);
        for (site_slot, request) in self.sites.iter().enumerate() {
            let extended = &self.prepared.sites[request.site_index].potential;
            let nuclear_charge = self.prepared.nuclear_charges[request.site_index];
            let radius = density.geometry().spheres()[request.site_index]
                .radius
                .get();
            site_indices.push(request.site_index as i64);
            site_ids.push(request.site_id.clone());
            mesh_first.push(extended.mesh.first().get());
            mesh_increment.push(extended.mesh.increment());
            mesh_count.push(extended.mesh.len() as i64);
            nuclear_charges.push(nuclear_charge);
            muffin_tin_radii.push(radius);
            potential_values.extend_from_slice(&extended.values);
            site_offsets.push(potential_values.len() as i64);
            for (state_ordinal, requested) in request.states.iter().enumerate() {
                tasks.extend_from_slice(&[
                    site_slot as i64,
                    state_ordinal as i64,
                    requested.state.n as i64,
                    requested.state.kappa.get() as i64,
                ]);
                let next =
                    task_radial_offsets.last().copied().unwrap() + extended.mesh.len() as i64;
                task_radial_offsets.push(next);
            }
        }
        let dict = export_dict(py)?;
        dict.set_item("site_indices", PyArray1::from_vec(py, site_indices))?;
        dict.set_item("site_ids", site_ids)?;
        dict.set_item("site_offsets", PyArray1::from_vec(py, site_offsets))?;
        dict.set_item("mesh_first", PyArray1::from_vec(py, mesh_first))?;
        dict.set_item("mesh_increment", PyArray1::from_vec(py, mesh_increment))?;
        dict.set_item("mesh_count", PyArray1::from_vec(py, mesh_count))?;
        dict.set_item("nuclear_charges", PyArray1::from_vec(py, nuclear_charges))?;
        dict.set_item("muffin_tin_radii", PyArray1::from_vec(py, muffin_tin_radii))?;
        dict.set_item("potential_values", PyArray1::from_vec(py, potential_values))?;
        dict.set_item(
            "tasks",
            PyArray2::from_owned_array(
                py,
                Array2::from_shape_vec((task_count, 4), tasks)
                    .expect("four integer columns are exported per core task"),
            ),
        )?;
        dict.set_item(
            "task_radial_offsets",
            PyArray1::from_vec(py, task_radial_offsets),
        )?;
        dict.set_item("speed_of_light", self.speed_of_light)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (energies, p, q, norm_total, norm_mt, spill, brackets))]
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        &self,
        energies: PyReadonlyArray1<'_, f64>,
        p: PyReadonlyArray1<'_, f64>,
        q: PyReadonlyArray1<'_, f64>,
        norm_total: PyReadonlyArray1<'_, f64>,
        norm_mt: PyReadonlyArray1<'_, f64>,
        spill: PyReadonlyArray1<'_, f64>,
        brackets: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<CoreResult> {
        let task_count = self
            .sites
            .iter()
            .map(|site| site.states.len())
            .sum::<usize>();
        let expected_radial = self
            .sites
            .iter()
            .map(|site| {
                self.prepared.sites[site.site_index].potential.mesh.len() * site.states.len()
            })
            .sum::<usize>();
        let energies = energies.as_array();
        let p = p.as_array();
        let q = q.as_array();
        let norm_total = norm_total.as_array();
        let norm_mt = norm_mt.as_array();
        let spill = spill.as_array();
        let brackets = brackets.as_array();
        if energies.len() != task_count
            || norm_total.len() != task_count
            || norm_mt.len() != task_count
            || spill.len() != task_count
            || brackets.shape() != [task_count, 2]
        {
            return Err(PyValueError::new_err(
                "core scalar outputs must contain one row per exported task",
            ));
        }
        if p.len() != expected_radial || q.len() != expected_radial {
            return Err(PyValueError::new_err(
                "core p and q outputs must match the exported task_radial_offsets",
            ));
        }

        let density = self.potential.inner.source_density();
        let mut result_density = density.zero_like();
        let mut eigenvalue_sum = Hartree(0.0);
        let mut built_sites = Vec::with_capacity(self.sites.len());
        let mut orbitals = Vec::with_capacity(self.sites.len());
        let mut task = 0;
        let mut radial = 0;
        for request in &self.sites {
            let extended = &self.prepared.sites[request.site_index].potential;
            let nuclear_charge = self.prepared.nuclear_charges[request.site_index];
            let muffin_tin_radius = density.geometry().spheres()[request.site_index].radius;
            let mut solved = Vec::with_capacity(request.states.len());
            for requested in &request.states {
                let next_radial = radial + extended.mesh.len();
                let bracket = EnergyBracket::from_values(brackets[[task, 0]], brackets[[task, 1]])
                    .map_err(py_error)?;
                solved.push(SolvedCoreState {
                    state: requested.state,
                    energy: Hartree(energies[task]),
                    p: p.iter()
                        .skip(radial)
                        .take(extended.mesh.len())
                        .copied()
                        .collect(),
                    q: q.iter()
                        .skip(radial)
                        .take(extended.mesh.len())
                        .copied()
                        .collect(),
                    norm_total: norm_total[task],
                    norm_mt: norm_mt[task],
                    spill: spill[task],
                    spec: CoreDiracSpec::new(
                        requested.state,
                        nuclear_charge,
                        bracket,
                        muffin_tin_radius,
                        self.speed_of_light,
                    ),
                    sourced_search: core_state_search_provenance(
                        &extended.values,
                        requested.state,
                        nuclear_charge,
                    )
                    .map_err(py_error)?,
                });
                radial = next_radial;
                task += 1;
            }
            let solved = assemble_regional_core_site(density, request, extended, solved)
                .map_err(py_error)?;
            result_density
                .add_scaled(1.0, &solved.contribution.contribution.density)
                .map_err(py_error)?;
            eigenvalue_sum += solved.contribution.contribution.eigenvalue_sum;
            built_sites.push(solved.contribution);
            orbitals.push(solved.orbitals);
        }
        Ok(CoreResult {
            inner: Arc::new(RegionalCoreResult {
                density: result_density,
                eigenvalue_sum,
                sites: built_sites,
                orbitals,
            }),
            structure: Arc::clone(&self.potential.structure),
        })
    }
}

#[pyfunction]
#[pyo3(signature = (
    potential_values,
    mesh_first,
    mesh_increment,
    n,
    kappa,
    nuclear_charge,
    muffin_tin_radius,
    p,
    q,
    scalars,
    speed_of_light=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_core_radial_block(
    potential_values: PyReadonlyArray1<'_, f64>,
    mesh_first: f64,
    mesh_increment: f64,
    n: u32,
    kappa: i32,
    nuclear_charge: f64,
    muffin_tin_radius: f64,
    mut p: PyReadwriteArray1<'_, f64>,
    mut q: PyReadwriteArray1<'_, f64>,
    mut scalars: PyReadwriteArray1<'_, f64>,
    speed_of_light: Option<f64>,
) -> PyResult<()> {
    let values = potential_values
        .as_slice()
        .map_err(|_| PyValueError::new_err("potential_values must be contiguous"))?;
    let mesh =
        ExponentialMesh::new(Bohr(mesh_first), mesh_increment, values.len()).map_err(py_error)?;
    let state = CoreState::new(n, Kappa::new(kappa).map_err(py_error)?).map_err(py_error)?;
    let solved = solve_core_state_block(
        &mesh,
        values,
        state,
        nuclear_charge,
        Bohr(muffin_tin_radius),
        speed_of_light.unwrap_or(SPEX_SPEED_OF_LIGHT),
    )
    .map_err(py_error)?;
    if p.len() != values.len() || q.len() != values.len() || scalars.len() != 6 {
        return Err(PyValueError::new_err(
            "core radial outputs must have shapes (nr), (nr), and (6)",
        ));
    }
    p.as_array_mut()
        .iter_mut()
        .zip(solved.p)
        .for_each(|(target, source)| *target = source);
    q.as_array_mut()
        .iter_mut()
        .zip(solved.q)
        .for_each(|(target, source)| *target = source);
    let mut output = scalars.as_array_mut();
    output[0] = solved.energy.get();
    output[1] = solved.norm_total;
    output[2] = solved.norm_mt;
    output[3] = solved.spill;
    output[4] = solved.spec.bracket.lower.get();
    output[5] = solved.spec.bracket.upper.get();
    Ok(())
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedCoreRadialBlocks>()?;
    module.add_function(wrap_pyfunction!(prepare_scalar_radial_blocks, module)?)?;
    module.add_function(wrap_pyfunction!(solve_scalar_radial_block, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_core_radial_blocks, module)?)?;
    module.add_function(wrap_pyfunction!(solve_core_radial_block, module)?)?;
    Ok(())
}
