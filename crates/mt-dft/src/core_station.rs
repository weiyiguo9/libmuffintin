//! Method-neutral four-component bound-core station.

use std::collections::BTreeSet;

use muffintin_core::{Bohr, ExponentialMesh, Hartree};
use muffintin_sphere::{
    CoreBracketSearch, CoreDiracSpec, CoreState, DiracError, EnergyBracket, ExtendedCorePotential,
    isolate_core_dirac_bracket, solve_core_dirac,
};
use thiserror::Error;

use crate::{
    BuiltRegionalCoreContribution, CoreDensityError, CorePotentialBuildError, CoreSpinPartition,
    RegionalCoreShellInput, RegionalDensity, RegionalError, ScfPotentialBuild,
    build_extended_core_potentials, build_regional_core_contribution,
};

/// One occupied spherical bound-core channel.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreStateRequest {
    pub state: CoreState,
    pub occupation: f64,
    pub spin: CoreSpinPartition,
}

/// Bound-core channels requested at one physical site.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreSiteRequest {
    pub site_index: usize,
    pub site_id: String,
    pub states: Vec<CoreStateRequest>,
}

/// Aggregate regional core density, energy sum, and site diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalCoreResult {
    pub density: RegionalDensity,
    pub eigenvalue_sum: Hartree,
    pub sites: Vec<BuiltRegionalCoreContribution>,
}

/// Solve all requested sites from one complete regional potential build.
pub fn solve_regional_core(
    potential: &ScfPotentialBuild,
    sites: &[CoreSiteRequest],
) -> Result<RegionalCoreResult, CoreStationError> {
    let density = potential.source_density();
    let site_count = density.geometry().spheres().len();
    let mut site_indices = BTreeSet::new();
    let mut site_ids = BTreeSet::new();
    for site in sites {
        if site.site_index >= site_count {
            return Err(CoreStationError::SiteIndex {
                site: site.site_index,
                site_count,
            });
        }
        if !site_indices.insert(site.site_index) {
            return Err(CoreStationError::DuplicateSiteIndex(site.site_index));
        }
        if !site_ids.insert(site.site_id.clone()) {
            return Err(CoreStationError::DuplicateSiteId(site.site_id.clone()));
        }
    }
    if sites.is_empty() {
        return Ok(RegionalCoreResult {
            density: density.zero_like(),
            eigenvalue_sum: Hartree(0.0),
            sites: Vec::new(),
        });
    }

    let nuclear_charges = potential.electrostatic.raw_nuclear.nuclear_charges();
    if nuclear_charges.len() != site_count {
        return Err(CoreStationError::NuclearSiteCount {
            expected: site_count,
            actual: nuclear_charges.len(),
        });
    }
    let mut maximum_n = vec![1_u32; site_count];
    for site in sites {
        for state in &site.states {
            maximum_n[site.site_index] = maximum_n[site.site_index].max(state.state.n);
        }
    }
    let extended_meshes = density
        .charge()
        .muffin_tins()
        .iter()
        .enumerate()
        .map(|(site, field)| {
            let orbital_scale = f64::from(maximum_n[site]).powi(2) / nuclear_charges[site].max(1.0);
            let radius = density.geometry().spheres()[site].radius;
            let outer_radius = (4.0 * radius.get()).max(40.0 * orbital_scale);
            extend_core_mesh(field.mesh(), outer_radius)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let extended = build_extended_core_potentials(
        &potential.electrostatic,
        &potential.exchange_correlation,
        density,
        &extended_meshes,
        potential.core_spec,
    )?;

    let mut result_density = density.zero_like();
    let mut eigenvalue_sum = Hartree(0.0);
    let mut built_sites = Vec::with_capacity(sites.len());
    for site in sites {
        let built = solve_regional_core_site(
            density,
            nuclear_charges,
            site,
            &extended[site.site_index].potential,
        )?;
        result_density.add_scaled(1.0, &built.contribution.density)?;
        eigenvalue_sum += built.contribution.eigenvalue_sum;
        built_sites.push(built);
    }
    Ok(RegionalCoreResult {
        density: result_density,
        eigenvalue_sum,
        sites: built_sites,
    })
}

pub(crate) fn solve_regional_core_site(
    density: &RegionalDensity,
    nuclear_charges: &[f64],
    request: &CoreSiteRequest,
    extended: &ExtendedCorePotential,
) -> Result<BuiltRegionalCoreContribution, CoreStationError> {
    let site_count = density.geometry().spheres().len();
    let sphere = density.geometry().spheres().get(request.site_index).ok_or(
        CoreStationError::SiteIndex {
            site: request.site_index,
            site_count,
        },
    )?;
    let charge =
        *nuclear_charges
            .get(request.site_index)
            .ok_or(CoreStationError::NuclearSiteCount {
                expected: site_count,
                actual: nuclear_charges.len(),
            })?;
    let muffin_tin = density
        .charge()
        .muffin_tins()
        .get(request.site_index)
        .ok_or(CoreStationError::SiteIndex {
            site: request.site_index,
            site_count: density.charge().muffin_tins().len(),
        })?;

    let solved = request
        .states
        .iter()
        .map(|requested| solve_bound_core_state(requested.state, extended, charge, sphere.radius))
        .collect::<Result<Vec<_>, _>>()?;
    let shells = solved
        .iter()
        .zip(&request.states)
        .map(|(solution, requested)| RegionalCoreShellInput {
            mesh: &extended.mesh,
            solution,
            occupation: requested.occupation,
            spin: requested.spin,
        })
        .collect::<Vec<_>>();
    Ok(build_regional_core_contribution(
        request.site_id.clone(),
        density.geometry(),
        request.site_index,
        muffin_tin.mesh(),
        &shells,
        density,
    )?)
}

fn solve_bound_core_state(
    state: CoreState,
    extended: &ExtendedCorePotential,
    nuclear_charge: f64,
    muffin_tin_radius: Bohr,
) -> Result<muffintin_sphere::CoreDiracSolution, CoreStationError> {
    let continuum = *extended
        .values
        .last()
        .expect("extended core potential follows a nonempty mesh");
    let atomic_scale = (nuclear_charge * nuclear_charge / f64::from(state.n).powi(2)).max(1.0);
    let window = EnergyBracket::from_values(
        continuum - 2.0 * nuclear_charge * nuclear_charge,
        continuum - 1.0e-8 * atomic_scale,
    )?;
    let bracket = isolate_core_dirac_bracket(
        &extended.mesh,
        &extended.values,
        CoreBracketSearch::new(state, muffin_tin_radius, window).with_intervals(512),
    )?;
    Ok(solve_core_dirac(
        &extended.mesh,
        &extended.values,
        CoreDiracSpec::new(state, bracket, muffin_tin_radius),
    )?)
}

pub(crate) fn extend_core_mesh(
    muffin_tin: &ExponentialMesh,
    target_radius: f64,
) -> Result<ExponentialMesh, CoreStationError> {
    let extra = (target_radius / muffin_tin.last().get()).ln().max(0.0) / muffin_tin.increment();
    let count = muffin_tin
        .len()
        .checked_add(extra.ceil() as usize)
        .and_then(|count| count.checked_add(1))
        .ok_or(CoreStationError::MeshCountOverflow)?;
    Ok(ExponentialMesh::new(
        muffin_tin.first(),
        muffin_tin.increment(),
        count,
    )?)
}

/// Invalid core-station request or failure in the shared physical core path.
#[derive(Debug, Error)]
pub enum CoreStationError {
    #[error("core site index {site} is outside 0..{site_count}")]
    SiteIndex { site: usize, site_count: usize },
    #[error("core site index {0} is requested more than once")]
    DuplicateSiteIndex(usize),
    #[error("core site id {0:?} is requested more than once")]
    DuplicateSiteId(String),
    #[error("core potential carries {actual} nuclear charges, expected {expected}")]
    NuclearSiteCount { expected: usize, actual: usize },
    #[error("extended core mesh point count overflows usize")]
    MeshCountOverflow,
    #[error(transparent)]
    Mesh(#[from] muffintin_core::MeshError),
    #[error(transparent)]
    Dirac(#[from] DiracError),
    #[error(transparent)]
    CorePotential(#[from] CorePotentialBuildError),
    #[error(transparent)]
    CoreDensity(#[from] CoreDensityError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
}
