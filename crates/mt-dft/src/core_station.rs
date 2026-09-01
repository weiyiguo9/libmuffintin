//! Method-neutral four-component bound-core station.

use std::collections::BTreeSet;

use muffintin_core::{Bohr, ExponentialMesh, Hartree, TwiceMu};
use muffintin_sphere::{
    CoreBracketSearch, CoreDiracSolution, CoreDiracSpec, CoreState, DiracError, EnergyBracket,
    ExtendedCorePotential, isolate_core_dirac_bracket, solve_core_dirac,
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

/// One physical four-component bound-core shell retained on its solve mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreShellOrbital {
    pub state: CoreState,
    pub energy: Hartree,
    pub p: Vec<f64>,
    pub q: Vec<f64>,
    pub norm_total: f64,
    pub norm_mt: f64,
    pub spill: f64,
    pub occupations: Vec<(TwiceMu, f64)>,
}

/// Exact potential samples and solver specifications that produced a site sidecar.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreShellOrbitalsProvenance {
    pub extended_potential: Vec<Hartree>,
    pub solve_specs: Vec<CoreDiracSpec>,
}

/// Bound-core radial sidecar for one physical site.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreShellOrbitals {
    pub site_index: usize,
    pub site_id: String,
    pub extended_mesh: ExponentialMesh,
    pub shells: Vec<CoreShellOrbital>,
    pub provenance: CoreShellOrbitalsProvenance,
}

/// Aggregate regional core density, energy sum, site diagnostics, and radial sidecars.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalCoreResult {
    pub density: RegionalDensity,
    pub eigenvalue_sum: Hartree,
    pub sites: Vec<BuiltRegionalCoreContribution>,
    pub orbitals: Vec<CoreShellOrbitals>,
}

pub(crate) struct SolvedRegionalCoreSite {
    pub contribution: BuiltRegionalCoreContribution,
    pub orbitals: CoreShellOrbitals,
}

struct SolvedBoundCoreState {
    solution: CoreDiracSolution,
    spec: CoreDiracSpec,
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
            orbitals: Vec::new(),
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
    let mut orbitals = Vec::with_capacity(sites.len());
    for site in sites {
        let solved = solve_regional_core_site(
            density,
            nuclear_charges,
            site,
            &extended[site.site_index].potential,
        )?;
        result_density.add_scaled(1.0, &solved.contribution.contribution.density)?;
        eigenvalue_sum += solved.contribution.contribution.eigenvalue_sum;
        built_sites.push(solved.contribution);
        orbitals.push(solved.orbitals);
    }
    Ok(RegionalCoreResult {
        density: result_density,
        eigenvalue_sum,
        sites: built_sites,
        orbitals,
    })
}

pub(crate) fn solve_regional_core_site(
    density: &RegionalDensity,
    nuclear_charges: &[f64],
    request: &CoreSiteRequest,
    extended: &ExtendedCorePotential,
) -> Result<SolvedRegionalCoreSite, CoreStationError> {
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

    let occupations = request
        .states
        .iter()
        .map(expand_mu_occupations)
        .collect::<Result<Vec<_>, _>>()?;

    let solved = request
        .states
        .iter()
        .map(|requested| solve_bound_core_state(requested.state, extended, charge, sphere.radius))
        .collect::<Result<Vec<_>, _>>()?;
    build_regional_core_site(
        density,
        request,
        extended,
        muffin_tin.mesh(),
        solved,
        occupations,
    )
}

fn build_regional_core_site(
    density: &RegionalDensity,
    request: &CoreSiteRequest,
    extended: &ExtendedCorePotential,
    muffin_tin_mesh: &ExponentialMesh,
    solved: Vec<SolvedBoundCoreState>,
    occupations: Vec<Vec<(TwiceMu, f64)>>,
) -> Result<SolvedRegionalCoreSite, CoreStationError> {
    let shells = solved
        .iter()
        .zip(&request.states)
        .map(|(solved, requested)| RegionalCoreShellInput {
            mesh: &extended.mesh,
            solution: &solved.solution,
            occupation: requested.occupation,
            spin: requested.spin,
        })
        .collect::<Vec<_>>();
    let contribution = build_regional_core_contribution(
        request.site_id.clone(),
        density.geometry(),
        request.site_index,
        muffin_tin_mesh,
        &shells,
        density,
    )?;
    let solve_specs = solved.iter().map(|solved| solved.spec).collect();
    let shells = solved
        .into_iter()
        .zip(occupations)
        .map(|(solved, occupations)| CoreShellOrbital {
            state: solved.solution.state,
            energy: solved.solution.energy,
            p: solved.solution.p,
            q: solved.solution.q,
            norm_total: solved.solution.norm_total,
            norm_mt: solved.solution.norm_mt,
            spill: solved.solution.spill,
            occupations,
        })
        .collect();
    Ok(SolvedRegionalCoreSite {
        contribution,
        orbitals: CoreShellOrbitals {
            site_index: request.site_index,
            site_id: request.site_id.clone(),
            extended_mesh: extended.mesh.clone(),
            shells,
            provenance: CoreShellOrbitalsProvenance {
                extended_potential: extended.values.iter().copied().map(Hartree).collect(),
                solve_specs,
            },
        },
    })
}

fn expand_mu_occupations(
    requested: &CoreStateRequest,
) -> Result<Vec<(TwiceMu, f64)>, CoreStationError> {
    match requested.spin {
        CoreSpinPartition::ClosedShellAverage => {
            let occupation = requested.occupation / f64::from(requested.state.kappa.degeneracy());
            Ok(requested
                .state
                .kappa
                .twice_mu_values()
                .map(|twice_mu| (twice_mu, occupation))
                .collect())
        }
        CoreSpinPartition::ExplicitCollinear { .. } => {
            Err(CoreStationError::AmbiguousCollinearMuOccupation {
                state: requested.state,
            })
        }
    }
}

fn solve_bound_core_state(
    state: CoreState,
    extended: &ExtendedCorePotential,
    nuclear_charge: f64,
    muffin_tin_radius: Bohr,
) -> Result<SolvedBoundCoreState, CoreStationError> {
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
    let spec = CoreDiracSpec::new(state, bracket, muffin_tin_radius);
    let solution = solve_core_dirac(&extended.mesh, &extended.values, spec)?;
    Ok(SolvedBoundCoreState { solution, spec })
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
    #[error(
        "explicit collinear occupation for core state {state:?} cannot be assigned uniquely to mu channels"
    )]
    AmbiguousCollinearMuOccupation { state: CoreState },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InterstitialField, MuffinTinField, RegionalScalarField};
    use muffintin_core::{
        FourierLayout, HermitianFourierField, InterstitialGeometry, InverseBohr, Kappa,
        ReciprocalLattice, Sphere, VolumeBohr3,
    };
    use muffintin_sphere::{HarmonicConvention, SphereField};
    use num_complex::Complex64;

    const CELL_LENGTH: f64 = 20.0;

    fn extended_mesh(first: f64, last: f64, increment: f64) -> ExponentialMesh {
        let count = ((last / first).ln() / increment).ceil() as usize + 1;
        ExponentialMesh::new(Bohr(first), increment, count).unwrap()
    }

    fn density_template(muffin_tin_mesh: &ExponentialMesh) -> RegionalDensity {
        let reciprocal = ReciprocalLattice::from_direct([
            [Bohr(CELL_LENGTH), Bohr(0.0), Bohr(0.0)],
            [Bohr(0.0), Bohr(CELL_LENGTH), Bohr(0.0)],
            [Bohr(0.0), Bohr(0.0), Bohr(CELL_LENGTH)],
        ])
        .unwrap();
        let vectors = reciprocal.enumerate(InverseBohr(0.0)).unwrap();
        let layout = FourierLayout::new(reciprocal, vectors).unwrap();
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(CELL_LENGTH.powi(3)),
            vec![Sphere {
                center: [Bohr(0.0); 3],
                radius: muffin_tin_mesh.last(),
            }],
        )
        .unwrap();
        let muffin_tin = MuffinTinField::new(
            muffin_tin_mesh.clone(),
            SphereField::new(
                HarmonicConvention::Complex,
                [(
                    (0, 0),
                    vec![Complex64::new(0.0, 0.0); muffin_tin_mesh.len()],
                )],
            )
            .unwrap(),
        )
        .unwrap();
        let interstitial = InterstitialField::from_fourier_field(
            HermitianFourierField::new(
                layout.clone(),
                vec![Complex64::new(0.0, 0.0); layout.len()],
            )
            .unwrap(),
        );
        let charge = RegionalScalarField::new(geometry, vec![muffin_tin], interstitial).unwrap();
        let zero = charge.zero_like();
        RegionalDensity::new(charge, [zero.clone(), zero.clone(), zero]).unwrap()
    }

    #[test]
    fn sidecar_retains_one_solver_output_and_preserves_density_path() {
        let mesh = extended_mesh(1.0e-7, 40.0, 0.004);
        let potential = mesh
            .radii()
            .iter()
            .map(|radius| -1.0 / radius.get())
            .collect::<Vec<_>>();
        let muffin_tin_index = mesh
            .radii()
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                (left.get() - 6.0)
                    .abs()
                    .total_cmp(&(right.get() - 6.0).abs())
            })
            .map(|(index, _)| index)
            .unwrap();
        let muffin_tin_mesh =
            ExponentialMesh::new(mesh.first(), mesh.increment(), muffin_tin_index + 1).unwrap();
        let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
        let spec = CoreDiracSpec::new(
            state,
            EnergyBracket::from_values(-0.6, -0.4).unwrap(),
            muffin_tin_mesh.last(),
        );
        let solution = solve_core_dirac(&mesh, &potential, spec).unwrap();
        let solver_oracle = solution.clone();
        let extended = ExtendedCorePotential {
            mesh: mesh.clone(),
            values: potential.clone(),
            muffin_tin_points: muffin_tin_mesh.len(),
            muffin_tin_boundary: potential[muffin_tin_index],
            periodic_boundary: potential[muffin_tin_index],
            boundary_mismatch: 0.0,
            origin_coulomb_residual: 0.0,
        };
        let request = CoreSiteRequest {
            site_index: 0,
            site_id: "H".to_owned(),
            states: vec![CoreStateRequest {
                state,
                occupation: 2.0,
                spin: CoreSpinPartition::ClosedShellAverage,
            }],
        };
        let density = density_template(&muffin_tin_mesh);
        let density_oracle = build_regional_core_contribution(
            request.site_id.clone(),
            density.geometry(),
            request.site_index,
            &muffin_tin_mesh,
            &[RegionalCoreShellInput {
                mesh: &mesh,
                solution: &solver_oracle,
                occupation: 2.0,
                spin: CoreSpinPartition::ClosedShellAverage,
            }],
            &density,
        )
        .unwrap();
        let occupations = request
            .states
            .iter()
            .map(expand_mu_occupations)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let retained = build_regional_core_site(
            &density,
            &request,
            &extended,
            &muffin_tin_mesh,
            vec![SolvedBoundCoreState { solution, spec }],
            occupations,
        )
        .unwrap();

        assert_eq!(retained.contribution, density_oracle);
        assert_eq!(
            retained.contribution.contribution.eigenvalue_sum,
            solver_oracle.energy * 2.0
        );
        assert_eq!(retained.orbitals.site_index, 0);
        assert_eq!(retained.orbitals.site_id, "H");
        assert_eq!(retained.orbitals.extended_mesh, mesh);
        assert_eq!(retained.orbitals.provenance.solve_specs, vec![spec]);
        assert_eq!(
            retained.orbitals.provenance.extended_potential,
            potential.into_iter().map(Hartree).collect::<Vec<_>>()
        );
        let shell = &retained.orbitals.shells[0];
        assert_eq!(shell.state, solver_oracle.state);
        assert_eq!(shell.energy, solver_oracle.energy);
        assert_eq!(shell.p, solver_oracle.p);
        assert_eq!(shell.q, solver_oracle.q);
        assert_eq!(shell.norm_total, solver_oracle.norm_total);
        assert_eq!(shell.norm_mt, solver_oracle.norm_mt);
        assert_eq!(shell.spill, solver_oracle.spill);
        assert_eq!(
            shell.occupations,
            state
                .kappa
                .twice_mu_values()
                .map(|twice_mu| (twice_mu, 1.0))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            shell
                .occupations
                .iter()
                .map(|(_, occupation)| occupation)
                .sum::<f64>(),
            request.states[0].occupation
        );
    }
}
