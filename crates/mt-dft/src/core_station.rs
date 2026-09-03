//! Method-neutral four-component bound-core station.

use std::collections::BTreeSet;

use muffintin_core::{Bohr, ExponentialMesh, Hartree, MeshError, TwiceMu};
use muffintin_coulomb::{
    BorrowedCoreShell, ClosedCoreOccupations, CoreCoreFockAction, CoreCoreFockError,
    CoreCoreFockShell, CoreCoreFockTrace, PreweightedSiteValenceDensity, RadialSlaterSite,
    RadialValenceCoreActions, RadialValenceCoreError, RadialValenceCoreShellAction,
    core_core_fock_actions, radial_valence_core_actions,
};
use muffintin_sphere::{
    CoreBracketSearch, CoreDiracExchangeAction, CoreDiracSolution, CoreDiracSourcedSpec,
    CoreDiracSpec, CoreState, DiracError, DiracLocalHamiltonianError, EnergyBracket,
    ExtendedCorePotential, dirac_local_hamiltonian_expectation, isolate_core_dirac_bracket,
    solve_core_dirac, solve_core_dirac_with_action,
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
    pub occupations: CoreShellOccupations,
}

/// Exact occupation representation retained with a core shell.
#[derive(Clone, Debug, PartialEq)]
pub enum CoreShellOccupations {
    /// Occupations resolved over every magnetic channel of one kappa shell.
    MuResolved(Vec<(TwiceMu, f64)>),
    /// Collinear spin totals whose assignment to individual mu channels is unspecified.
    ExplicitCollinear { up: f64, down: f64 },
}

/// Exact potential samples and solver specifications that produced a site sidecar.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreShellOrbitalsProvenance {
    pub extended_potential: Vec<Hartree>,
    /// Exact isolated homogeneous specifications that produced the sidecar shells.
    pub solve_specs: Vec<CoreDiracSpec>,
    /// Broad physical windows retained for prediction-centered sourced Fock solves.
    pub sourced_searches: Vec<CoreSourcedSearchProvenance>,
}

/// Root-search provenance for one source-driven core-shell update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreSourcedSearchProvenance {
    pub energy_window: EnergyBracket,
    pub intervals: usize,
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

/// One shell's direct occupied expectation value of the immutable local H0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreLocalOneBodyShellTrace {
    pub shell_index: usize,
    pub state: CoreState,
    pub occupation: f64,
    pub expectation: Hartree,
    pub contribution: Hartree,
}

/// Direct occupied core trace of the immutable local radial Hamiltonian.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreLocalOneBodyTrace {
    pub shells: Vec<CoreLocalOneBodyShellTrace>,
    pub total: Hartree,
}

/// Controls for the M3a/M3b core inner loop at one immutable local potential.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreFixedPotentialSpec {
    /// Linear mixing applied only between consecutive radial Fock actions.
    pub action_mixing: f64,
    pub energy_tolerance: Hartree,
    /// Full-extended-mesh norm of the phase-aligned radial difference.
    pub radial_tolerance: f64,
    /// Maximum discarded imaginary VC radial-action or trace component.
    pub vc_imaginary_tolerance: f64,
    pub max_iterations: usize,
}

/// One shell's fixed-potential convergence residuals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoreFixedPotentialShellResidual {
    pub shell_index: usize,
    pub state: CoreState,
    pub energy_change: Hartree,
    pub radial_residual: f64,
}

/// One simultaneous update of every shell at a site.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreFixedPotentialIteration {
    pub iteration: usize,
    pub shells: Vec<CoreFixedPotentialShellResidual>,
    pub maximum_energy_change: Hartree,
    pub maximum_radial_residual: f64,
    /// Fresh occupied CC trace generated by the input orbitals of this update.
    pub cc_trace: CoreCoreFockTrace,
    /// Fresh occupied VC trace generated from the fixed valence density.
    pub vc_trace: Hartree,
    pub vc_imaginary_residual: f64,
}

/// Converged M3a/M3b sidecar with fresh CC and VC traces.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreFixedPotentialResult {
    pub orbitals: CoreShellOrbitals,
    pub diagnostics: Vec<CoreFixedPotentialIteration>,
    pub final_cc_trace: CoreCoreFockTrace,
    pub final_vc_trace: Hartree,
    pub final_vc_imaginary_residual: f64,
}

/// Fixed occupied valence radial-density frame for one site's VC inner action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedSiteValenceDensity<'a> {
    pub site_index: usize,
    pub muffin_tin_mesh: &'a ExponentialMesh,
    pub valence: PreweightedSiteValenceDensity<'a>,
}

pub(crate) struct SolvedRegionalCoreSite {
    pub contribution: BuiltRegionalCoreContribution,
    pub orbitals: CoreShellOrbitals,
}

struct SolvedBoundCoreState {
    solution: CoreDiracSolution,
    spec: CoreDiracSpec,
    sourced_search: CoreSourcedSearchProvenance,
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

/// Synthesize a fresh regional contribution directly from retained core radials.
///
/// The sidecar's physical `P/Q`, norms, spill, energies, and occupations are
/// borrowed without reconstructing a [`CoreDiracSolution`].
pub fn build_regional_core_contribution_from_sidecar(
    sidecar: &CoreShellOrbitals,
    zero_like_template: &RegionalDensity,
) -> Result<BuiltRegionalCoreContribution, CoreDensityError> {
    let muffin_tin_mesh = zero_like_template
        .charge()
        .muffin_tins()
        .get(sidecar.site_index)
        .ok_or(CoreDensityError::SiteIndex {
            site: sidecar.site_index,
            site_count: zero_like_template.charge().muffin_tins().len(),
        })?
        .mesh();
    let shell_partitions = sidecar
        .shells
        .iter()
        .enumerate()
        .map(|(shell_index, shell)| sidecar_density_partition(shell_index, shell))
        .collect::<Result<Vec<_>, _>>()?;
    let shells = sidecar
        .shells
        .iter()
        .zip(&shell_partitions)
        .map(|(shell, &(occupation, spin))| RegionalCoreShellInput {
            mesh: &sidecar.extended_mesh,
            state: shell.state,
            energy: shell.energy,
            p: &shell.p,
            q: &shell.q,
            norm_mt: shell.norm_mt,
            spill: shell.spill,
            occupation,
            spin,
        })
        .collect::<Vec<_>>();
    build_regional_core_contribution(
        sidecar.site_id.clone(),
        zero_like_template.geometry(),
        sidecar.site_index,
        muffin_tin_mesh,
        &shells,
        zero_like_template,
    )
}

/// Evaluate `Tr(Dc H0)` from physical retained core radials and local potential.
/// The evaluation potential is explicit and need not be the radial-generating
/// potential recorded in the immutable sidecar provenance.
pub fn core_local_one_body_trace(
    sidecar: &CoreShellOrbitals,
    potential: &[f64],
) -> Result<CoreLocalOneBodyTrace, CoreLocalOneBodyError> {
    if potential.len() != sidecar.extended_mesh.len() {
        return Err(CoreLocalOneBodyError::PotentialLength {
            expected: sidecar.extended_mesh.len(),
            actual: potential.len(),
        });
    }
    let mut total = Hartree(0.0);
    let shells = sidecar
        .shells
        .iter()
        .enumerate()
        .map(|(shell_index, shell)| {
            let occupation = sidecar_shell_occupation(shell_index, shell)?;
            let expectation = dirac_local_hamiltonian_expectation(
                &sidecar.extended_mesh,
                potential,
                shell.state.kappa,
                &shell.p,
                &shell.q,
            )?;
            let contribution = expectation * occupation;
            total += contribution;
            Ok(CoreLocalOneBodyShellTrace {
                shell_index,
                state: shell.state,
                occupation,
                expectation,
                contribution,
            })
        })
        .collect::<Result<Vec<_>, CoreLocalOneBodyError>>()?;
    Ok(CoreLocalOneBodyTrace { shells, total })
}

fn sidecar_density_partition(
    shell_index: usize,
    shell: &CoreShellOrbital,
) -> Result<(f64, CoreSpinPartition), CoreDensityError> {
    match &shell.occupations {
        CoreShellOccupations::MuResolved(occupations) => {
            let expected = shell.state.kappa.twice_mu_values().collect::<Vec<_>>();
            if occupations.len() != expected.len()
                || expected
                    .iter()
                    .any(|mu| occupations.iter().filter(|(found, _)| found == mu).count() != 1)
            {
                return Err(CoreDensityError::SidecarMagneticChannels { shell: shell_index });
            }
            let reference = occupations[0].1;
            if !reference.is_finite() || reference < 0.0 {
                return Err(CoreDensityError::SidecarOccupation {
                    shell: shell_index,
                    occupation: reference,
                });
            }
            if occupations
                .iter()
                .any(|(_, value)| value.to_bits() != reference.to_bits())
            {
                return Err(CoreDensityError::SidecarOpenShell { shell: shell_index });
            }
            Ok((
                reference * expected.len() as f64,
                CoreSpinPartition::ClosedShellAverage,
            ))
        }
        CoreShellOccupations::ExplicitCollinear { up, down } => Ok((
            *up + *down,
            CoreSpinPartition::ExplicitCollinear {
                up: *up,
                down: *down,
            },
        )),
    }
}

fn sidecar_shell_occupation(
    shell_index: usize,
    shell: &CoreShellOrbital,
) -> Result<f64, CoreLocalOneBodyError> {
    let occupation = match &shell.occupations {
        CoreShellOccupations::MuResolved(occupations) => {
            occupations.iter().map(|(_, occupation)| occupation).sum()
        }
        CoreShellOccupations::ExplicitCollinear { up, down } => *up + *down,
    };
    let capacity = f64::from(shell.state.kappa.degeneracy());
    if !occupation.is_finite() || occupation < 0.0 || occupation > capacity {
        return Err(CoreLocalOneBodyError::Occupation {
            shell: shell_index,
            occupation,
            capacity,
        });
    }
    Ok(occupation)
}

/// Relax one site's closed/uniform-mu core shells against fresh CC and VC exchange.
///
/// The extended mesh, local potential samples, and every `CoreDiracSpec` are
/// borrowed from `initial.provenance` and remain immutable. Every iteration
/// builds the fresh CC action from one input shell set, adds the fixed
/// valence-generated VC action, and solves all shells before replacing any of
/// them. Only consecutive summed CC+VC actions are mixed.
pub fn relax_core_at_fixed_potential(
    initial: &CoreShellOrbitals,
    fixed_valence: FixedSiteValenceDensity<'_>,
    spec: CoreFixedPotentialSpec,
) -> Result<CoreFixedPotentialResult, CoreRelaxationError> {
    validate_relaxation_spec(spec)?;
    validate_core_core_sidecar(initial)?;
    if fixed_valence.site_index != initial.site_index {
        return Err(CoreRelaxationError::ValenceCoreSite {
            initial: initial.site_index,
            density: fixed_valence.site_index,
        });
    }

    let potential = initial
        .provenance
        .extended_potential
        .iter()
        .map(|value| value.get())
        .collect::<Vec<_>>();
    let mut current = initial.clone();
    let mut previous_action: Option<Vec<CoreCoreFockAction>> = None;
    let mut diagnostics = Vec::with_capacity(spec.max_iterations);

    for iteration in 1..=spec.max_iterations {
        let fresh_cc = build_sidecar_core_core_action(&current)?;
        let fresh_vc = build_sidecar_valence_core_action(&current, fixed_valence)?;
        validate_vc_imaginary_residual(iteration, fresh_vc.imaginary_residual, spec)?;
        let fresh_total = sum_core_actions(&fresh_cc.actions, &fresh_vc.shells);
        let action =
            mix_core_core_actions(previous_action.as_deref(), &fresh_total, spec.action_mixing);
        let mut next_shells = Vec::with_capacity(current.shells.len());
        let mut shell_residuals = Vec::with_capacity(current.shells.len());
        let mut maximum_energy_change = 0.0_f64;
        let mut maximum_radial_residual = 0.0_f64;

        for shell_index in 0..current.shells.len() {
            let shell = &current.shells[shell_index];
            let solve_spec = initial.provenance.solve_specs[shell_index];
            let sourced_search = initial.provenance.sourced_searches[shell_index];
            let shell_action = &action[shell_index];
            let action_expectation = initial.extended_mesh.integrate(
                &shell
                    .p
                    .iter()
                    .zip(&shell.q)
                    .zip(shell_action.p.iter().zip(&shell_action.q))
                    .map(|((p, q), (action_p, action_q))| p * action_p + q * action_q)
                    .collect::<Vec<_>>(),
            )? / shell.norm_total.sqrt();
            let sourced_spec = CoreDiracSourcedSpec::new(
                CoreDiracSpec {
                    bracket: sourced_search.energy_window,
                    ..solve_spec
                },
                Hartree(shell.energy.get() + action_expectation),
            )
            .with_search_intervals(sourced_search.intervals);
            let mut solution = solve_core_dirac_with_action(
                &initial.extended_mesh,
                &potential,
                sourced_spec,
                CoreDiracExchangeAction {
                    p: &shell_action.p,
                    q: &shell_action.q,
                },
            )?;
            let energy_change = (solution.energy.get() - shell.energy.get()).abs();
            if !energy_change.is_finite() {
                return Err(CoreRelaxationError::NonFiniteEnergyChange {
                    iteration,
                    shell: shell_index,
                    value: energy_change,
                });
            }
            let radial_residual = phase_align_and_measure(
                &initial.extended_mesh,
                &shell.p,
                &shell.q,
                &mut solution.p,
                &mut solution.q,
            )?;
            if !radial_residual.is_finite() {
                return Err(CoreRelaxationError::NonFiniteRadialResidual {
                    iteration,
                    shell: shell_index,
                    value: radial_residual,
                });
            }
            maximum_energy_change = maximum_energy_change.max(energy_change);
            maximum_radial_residual = maximum_radial_residual.max(radial_residual);
            shell_residuals.push(CoreFixedPotentialShellResidual {
                shell_index,
                state: shell.state,
                energy_change: Hartree(energy_change),
                radial_residual,
            });
            next_shells.push(CoreShellOrbital {
                state: solution.state,
                energy: solution.energy,
                p: solution.p,
                q: solution.q,
                norm_total: solution.norm_total,
                norm_mt: solution.norm_mt,
                spill: solution.spill,
                occupations: shell.occupations.clone(),
            });
        }

        diagnostics.push(CoreFixedPotentialIteration {
            iteration,
            shells: shell_residuals,
            maximum_energy_change: Hartree(maximum_energy_change),
            maximum_radial_residual,
            cc_trace: fresh_cc.trace,
            vc_trace: fresh_vc.action_trace,
            vc_imaginary_residual: fresh_vc.imaginary_residual,
        });
        current.shells = next_shells;
        previous_action = Some(action);

        if maximum_energy_change <= spec.energy_tolerance.get()
            && maximum_radial_residual <= spec.radial_tolerance
        {
            let final_cc_trace = build_sidecar_core_core_action(&current)?.trace;
            let final_vc = build_sidecar_valence_core_action(&current, fixed_valence)?;
            validate_vc_imaginary_residual(iteration, final_vc.imaginary_residual, spec)?;
            return Ok(CoreFixedPotentialResult {
                orbitals: current,
                diagnostics,
                final_cc_trace,
                final_vc_trace: final_vc.action_trace,
                final_vc_imaginary_residual: final_vc.imaginary_residual,
            });
        }
    }

    let last = diagnostics
        .last()
        .expect("positive validated core-core iteration count");
    Err(CoreRelaxationError::NotConverged {
        iterations: diagnostics.len(),
        maximum_energy_change: last.maximum_energy_change.get(),
        maximum_radial_residual: last.maximum_radial_residual,
    })
}

fn validate_relaxation_spec(spec: CoreFixedPotentialSpec) -> Result<(), CoreRelaxationError> {
    if !spec.action_mixing.is_finite() || spec.action_mixing <= 0.0 || spec.action_mixing > 1.0 {
        return Err(CoreRelaxationError::InvalidActionMixing(spec.action_mixing));
    }
    if !spec.energy_tolerance.get().is_finite() || spec.energy_tolerance.get() <= 0.0 {
        return Err(CoreRelaxationError::InvalidEnergyTolerance(
            spec.energy_tolerance.get(),
        ));
    }
    if !spec.radial_tolerance.is_finite() || spec.radial_tolerance <= 0.0 {
        return Err(CoreRelaxationError::InvalidRadialTolerance(
            spec.radial_tolerance,
        ));
    }
    if !spec.vc_imaginary_tolerance.is_finite() || spec.vc_imaginary_tolerance < 0.0 {
        return Err(CoreRelaxationError::InvalidValenceCoreImaginaryTolerance(
            spec.vc_imaginary_tolerance,
        ));
    }
    if spec.max_iterations == 0 {
        return Err(CoreRelaxationError::InvalidMaximumIterations);
    }
    Ok(())
}

fn validate_vc_imaginary_residual(
    iteration: usize,
    residual: f64,
    spec: CoreFixedPotentialSpec,
) -> Result<(), CoreRelaxationError> {
    if residual > spec.vc_imaginary_tolerance {
        return Err(CoreRelaxationError::ValenceCoreImaginaryResidual {
            iteration,
            residual,
            tolerance: spec.vc_imaginary_tolerance,
        });
    }
    Ok(())
}

fn validate_core_core_sidecar(sidecar: &CoreShellOrbitals) -> Result<(), CoreRelaxationError> {
    if sidecar.provenance.extended_potential.len() != sidecar.extended_mesh.len() {
        return Err(CoreRelaxationError::PotentialLength {
            expected: sidecar.extended_mesh.len(),
            actual: sidecar.provenance.extended_potential.len(),
        });
    }
    if sidecar.provenance.solve_specs.len() != sidecar.shells.len() {
        return Err(CoreRelaxationError::SolveSpecCount {
            expected: sidecar.shells.len(),
            actual: sidecar.provenance.solve_specs.len(),
        });
    }
    if sidecar.provenance.sourced_searches.len() != sidecar.shells.len() {
        return Err(CoreRelaxationError::SourcedSearchCount {
            expected: sidecar.shells.len(),
            actual: sidecar.provenance.sourced_searches.len(),
        });
    }
    for (shell_index, ((shell, spec), search)) in sidecar
        .shells
        .iter()
        .zip(&sidecar.provenance.solve_specs)
        .zip(&sidecar.provenance.sourced_searches)
        .enumerate()
    {
        if shell.state != spec.state {
            return Err(CoreRelaxationError::ShellState {
                shell: shell_index,
                orbital: shell.state,
                solve_spec: spec.state,
            });
        }
        if search.intervals == 0 {
            return Err(CoreRelaxationError::SourcedSearchIntervals { shell: shell_index });
        }
        uniform_mu_occupation(shell_index, shell)?;
    }
    Ok(())
}

fn build_sidecar_core_core_action(
    sidecar: &CoreShellOrbitals,
) -> Result<muffintin_coulomb::CoreCoreFockResult, CoreRelaxationError> {
    let occupations = sidecar
        .shells
        .iter()
        .enumerate()
        .map(|(shell, orbital)| uniform_mu_occupation(shell, orbital))
        .collect::<Result<Vec<_>, _>>()?;
    let inputs = sidecar
        .shells
        .iter()
        .zip(&occupations)
        .map(|(shell, &occupation_per_mu)| CoreCoreFockShell {
            kappa: shell.state.kappa,
            p: &shell.p,
            q: &shell.q,
            normalization: shell.norm_total,
            occupation_per_mu,
        })
        .collect::<Vec<_>>();
    Ok(core_core_fock_actions(&sidecar.extended_mesh, &inputs)?)
}

fn uniform_mu_occupation(
    shell_index: usize,
    shell: &CoreShellOrbital,
) -> Result<f64, CoreRelaxationError> {
    let CoreShellOccupations::MuResolved(occupations) = &shell.occupations else {
        return Err(CoreRelaxationError::ExplicitCollinear { shell: shell_index });
    };
    let expected = shell.state.kappa.twice_mu_values().collect::<Vec<_>>();
    if occupations.len() != expected.len()
        || expected
            .iter()
            .any(|mu| occupations.iter().filter(|(found, _)| found == mu).count() != 1)
    {
        return Err(CoreRelaxationError::MagneticChannels { shell: shell_index });
    }
    let reference = occupations[0].1;
    if occupations
        .iter()
        .any(|(_, value)| value.to_bits() != reference.to_bits())
    {
        return Err(CoreRelaxationError::OpenShell { shell: shell_index });
    }
    Ok(reference)
}

fn build_sidecar_valence_core_action(
    sidecar: &CoreShellOrbitals,
    fixed: FixedSiteValenceDensity<'_>,
) -> Result<RadialValenceCoreActions, CoreRelaxationError> {
    let cores = sidecar
        .shells
        .iter()
        .map(|shell| BorrowedCoreShell {
            kappa: shell.state.kappa,
            p: &shell.p,
            q: &shell.q,
            normalization: shell.norm_total,
            occupations: match &shell.occupations {
                CoreShellOccupations::MuResolved(occupations) => {
                    ClosedCoreOccupations::MuResolved(occupations)
                }
                CoreShellOccupations::ExplicitCollinear { up, down } => {
                    ClosedCoreOccupations::ExplicitCollinear {
                        up: *up,
                        down: *down,
                    }
                }
            },
        })
        .collect::<Vec<_>>();
    let site = RadialSlaterSite {
        site_index: sidecar.site_index,
        mt_mesh: fixed.muffin_tin_mesh,
        extended_mesh: &sidecar.extended_mesh,
        cores: &cores,
        valence: fixed.valence,
    };
    Ok(radial_valence_core_actions(&[site])?)
}

fn sum_core_actions(
    cc: &[CoreCoreFockAction],
    vc: &[RadialValenceCoreShellAction],
) -> Vec<CoreCoreFockAction> {
    cc.iter()
        .zip(vc)
        .map(|(cc, vc)| CoreCoreFockAction {
            p: cc.p.iter().zip(&vc.p).map(|(cc, vc)| cc + vc).collect(),
            q: cc.q.iter().zip(&vc.q).map(|(cc, vc)| cc + vc).collect(),
        })
        .collect()
}

fn mix_core_core_actions(
    previous: Option<&[CoreCoreFockAction]>,
    fresh: &[CoreCoreFockAction],
    alpha: f64,
) -> Vec<CoreCoreFockAction> {
    let Some(previous) = previous else {
        return fresh.to_vec();
    };
    previous
        .iter()
        .zip(fresh)
        .map(|(previous, fresh)| CoreCoreFockAction {
            p: previous
                .p
                .iter()
                .zip(&fresh.p)
                .map(|(old, new)| (1.0 - alpha) * old + alpha * new)
                .collect(),
            q: previous
                .q
                .iter()
                .zip(&fresh.q)
                .map(|(old, new)| (1.0 - alpha) * old + alpha * new)
                .collect(),
        })
        .collect()
}

fn phase_align_and_measure(
    mesh: &ExponentialMesh,
    old_p: &[f64],
    old_q: &[f64],
    new_p: &mut [f64],
    new_q: &mut [f64],
) -> Result<f64, CoreRelaxationError> {
    let overlap = mesh.integrate(
        &old_p
            .iter()
            .zip(old_q)
            .zip(new_p.iter().zip(new_q.iter()))
            .map(|((old_p, old_q), (new_p, new_q))| old_p * new_p + old_q * new_q)
            .collect::<Vec<_>>(),
    )?;
    if overlap < 0.0 {
        new_p.iter_mut().for_each(|value| *value = -*value);
        new_q.iter_mut().for_each(|value| *value = -*value);
    }
    let residual_squared = mesh.integrate(
        &old_p
            .iter()
            .zip(old_q)
            .zip(new_p.iter().zip(new_q.iter()))
            .map(|((old_p, old_q), (new_p, new_q))| {
                (old_p - new_p).powi(2) + (old_q - new_q).powi(2)
            })
            .collect::<Vec<_>>(),
    )?;
    Ok(residual_squared.max(0.0).sqrt())
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
        .map(retain_shell_occupations)
        .collect::<Vec<_>>();

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
    occupations: Vec<CoreShellOccupations>,
) -> Result<SolvedRegionalCoreSite, CoreStationError> {
    let shells = solved
        .iter()
        .zip(&request.states)
        .map(|(solved, requested)| RegionalCoreShellInput {
            mesh: &extended.mesh,
            state: solved.solution.state,
            energy: solved.solution.energy,
            p: &solved.solution.p,
            q: &solved.solution.q,
            norm_mt: solved.solution.norm_mt,
            spill: solved.solution.spill,
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
    let sourced_searches = solved.iter().map(|solved| solved.sourced_search).collect();
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
                sourced_searches,
            },
        },
    })
}

fn retain_shell_occupations(requested: &CoreStateRequest) -> CoreShellOccupations {
    match requested.spin {
        CoreSpinPartition::ClosedShellAverage => {
            let occupation = requested.occupation / f64::from(requested.state.kappa.degeneracy());
            CoreShellOccupations::MuResolved(
                requested
                    .state
                    .kappa
                    .twice_mu_values()
                    .map(|twice_mu| (twice_mu, occupation))
                    .collect(),
            )
        }
        CoreSpinPartition::ExplicitCollinear { up, down } => {
            CoreShellOccupations::ExplicitCollinear { up, down }
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
    let search_intervals = 512;
    let bracket = isolate_core_dirac_bracket(
        &extended.mesh,
        &extended.values,
        CoreBracketSearch::new(state, nuclear_charge, muffin_tin_radius, window)
            .with_intervals(search_intervals),
    )?;
    let spec = CoreDiracSpec::new(state, nuclear_charge, bracket, muffin_tin_radius);
    let solution = solve_core_dirac(&extended.mesh, &extended.values, spec)?;
    Ok(SolvedBoundCoreState {
        solution,
        spec,
        sourced_search: CoreSourcedSearchProvenance {
            energy_window: window,
            intervals: search_intervals,
        },
    })
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

/// Invalid sidecar input or radial failure in a direct local core trace.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CoreLocalOneBodyError {
    #[error("core sidecar potential has {actual} samples, expected {expected}")]
    PotentialLength { expected: usize, actual: usize },
    #[error("core shell {shell} occupation {occupation} is outside [0,{capacity}]")]
    Occupation {
        shell: usize,
        occupation: f64,
        capacity: f64,
    },
    #[error(transparent)]
    Radial(#[from] DiracLocalHamiltonianError),
}

/// Invalid fixed-potential core input, radial failure, or bounded nonconvergence.
#[derive(Debug, Error)]
pub enum CoreRelaxationError {
    #[error("core-core action_mixing must be finite and in (0, 1], got {0}")]
    InvalidActionMixing(f64),
    #[error("core-core energy tolerance must be finite and positive, got {0} Ha")]
    InvalidEnergyTolerance(f64),
    #[error("core-core radial tolerance must be finite and positive, got {0}")]
    InvalidRadialTolerance(f64),
    #[error("VC imaginary tolerance must be finite and nonnegative, got {0}")]
    InvalidValenceCoreImaginaryTolerance(f64),
    #[error("core-core relaxation max_iterations must be positive")]
    InvalidMaximumIterations,
    #[error("core sidecar potential has {actual} samples, expected {expected}")]
    PotentialLength { expected: usize, actual: usize },
    #[error("core sidecar has {actual} solve specs, expected {expected}")]
    SolveSpecCount { expected: usize, actual: usize },
    #[error("core sidecar has {actual} sourced searches, expected {expected}")]
    SourcedSearchCount { expected: usize, actual: usize },
    #[error("core shell {shell} sourced search interval count must be positive")]
    SourcedSearchIntervals { shell: usize },
    #[error(
        "core shell {shell} state {orbital:?} does not match provenance solve state {solve_spec:?}"
    )]
    ShellState {
        shell: usize,
        orbital: CoreState,
        solve_spec: CoreState,
    },
    #[error("core shell {shell} uses ExplicitCollinear occupations")]
    ExplicitCollinear { shell: usize },
    #[error("core shell {shell} does not contain every magnetic channel exactly once")]
    MagneticChannels { shell: usize },
    #[error("core shell {shell} is not uniform over magnetic channels")]
    OpenShell { shell: usize },
    #[error("fixed valence density site {density} does not match core sidecar site {initial}")]
    ValenceCoreSite { initial: usize, density: usize },
    #[error(transparent)]
    ValenceCore(#[from] RadialValenceCoreError),
    #[error(
        "core relaxation iteration {iteration} VC imaginary residual {residual} exceeds {tolerance}"
    )]
    ValenceCoreImaginaryResidual {
        iteration: usize,
        residual: f64,
        tolerance: f64,
    },
    #[error(
        "core-core relaxation iteration {iteration} shell {shell} has non-finite energy change {value}"
    )]
    NonFiniteEnergyChange {
        iteration: usize,
        shell: usize,
        value: f64,
    },
    #[error(
        "core-core relaxation iteration {iteration} shell {shell} has non-finite radial residual {value}"
    )]
    NonFiniteRadialResidual {
        iteration: usize,
        shell: usize,
        value: f64,
    },
    #[error(
        "core-core relaxation did not converge in {iterations} iterations: max energy change {maximum_energy_change} Ha, max full-mesh radial residual {maximum_radial_residual}"
    )]
    NotConverged {
        iterations: usize,
        maximum_energy_change: f64,
        maximum_radial_residual: f64,
    },
    #[error(transparent)]
    CoreFock(#[from] CoreCoreFockError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    Dirac(#[from] DiracError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InterstitialField, MuffinTinField, RegionalScalarField};
    use muffintin_core::{
        FourierLayout, HermitianFourierField, InterstitialGeometry, InverseBohr, Kappa,
        ReciprocalLattice, RelativisticChannel, Sphere, VolumeBohr3,
    };
    use muffintin_coulomb::BorrowedValenceRadial;
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
            1.0,
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
                state: solver_oracle.state,
                energy: solver_oracle.energy,
                p: &solver_oracle.p,
                q: &solver_oracle.q,
                norm_mt: solver_oracle.norm_mt,
                spill: solver_oracle.spill,
                occupation: 2.0,
                spin: CoreSpinPartition::ClosedShellAverage,
            }],
            &density,
        )
        .unwrap();
        let occupations = request
            .states
            .iter()
            .map(retain_shell_occupations)
            .collect::<Vec<_>>();
        let retained = build_regional_core_site(
            &density,
            &request,
            &extended,
            &muffin_tin_mesh,
            vec![SolvedBoundCoreState {
                solution,
                spec,
                sourced_search: CoreSourcedSearchProvenance {
                    energy_window: EnergyBracket::from_values(-2.0, -1.0e-8).unwrap(),
                    intervals: 512,
                },
            }],
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
            potential.iter().copied().map(Hartree).collect::<Vec<_>>()
        );
        let shell = &retained.orbitals.shells[0];
        assert_eq!(shell.state, solver_oracle.state);
        assert_eq!(shell.energy, solver_oracle.energy);
        assert_eq!(shell.p, solver_oracle.p);
        assert_eq!(shell.q, solver_oracle.q);
        assert_eq!(shell.norm_total, solver_oracle.norm_total);
        assert_eq!(shell.norm_mt, solver_oracle.norm_mt);
        assert_eq!(shell.spill, solver_oracle.spill);
        let CoreShellOccupations::MuResolved(mu_occupations) = &shell.occupations else {
            panic!("closed shell must be resolved over mu channels");
        };
        assert_eq!(
            *mu_occupations,
            state
                .kappa
                .twice_mu_values()
                .map(|twice_mu| (twice_mu, 1.0))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            mu_occupations
                .iter()
                .map(|(_, occupation)| occupation)
                .sum::<f64>(),
            request.states[0].occupation
        );
        let rebuilt =
            build_regional_core_contribution_from_sidecar(&retained.orbitals, &density).unwrap();
        assert_eq!(rebuilt, retained.contribution);
        let local_trace = core_local_one_body_trace(&retained.orbitals, &potential).unwrap();
        let shifted = potential
            .iter()
            .map(|value| value + 0.125)
            .collect::<Vec<_>>();
        let shifted_trace = core_local_one_body_trace(&retained.orbitals, &shifted).unwrap();
        assert!((shifted_trace.total.get() - local_trace.total.get() - 0.25).abs() < 1.0e-10);
        assert_eq!(local_trace.shells.len(), 1);
        assert_eq!(local_trace.shells[0].state, state);
        assert_eq!(local_trace.shells[0].occupation, 2.0);
        assert!(
            (local_trace.shells[0].expectation.get() - solver_oracle.energy.get()).abs() < 2.0e-7
        );
        assert!((local_trace.total.get() - 2.0 * solver_oracle.energy.get()).abs() < 4.0e-7);

        let explicit_spin = CoreSpinPartition::ExplicitCollinear { up: 1.5, down: 0.5 };
        let explicit_request = CoreSiteRequest {
            site_index: 0,
            site_id: "H".to_owned(),
            states: vec![CoreStateRequest {
                state,
                occupation: 2.0,
                spin: explicit_spin,
            }],
        };
        let explicit_density_oracle = build_regional_core_contribution(
            explicit_request.site_id.clone(),
            density.geometry(),
            explicit_request.site_index,
            &muffin_tin_mesh,
            &[RegionalCoreShellInput {
                mesh: &mesh,
                state: solver_oracle.state,
                energy: solver_oracle.energy,
                p: &solver_oracle.p,
                q: &solver_oracle.q,
                norm_mt: solver_oracle.norm_mt,
                spill: solver_oracle.spill,
                occupation: 2.0,
                spin: explicit_spin,
            }],
            &density,
        )
        .unwrap();
        let explicit_retained = build_regional_core_site(
            &density,
            &explicit_request,
            &extended,
            &muffin_tin_mesh,
            vec![SolvedBoundCoreState {
                solution: solver_oracle.clone(),
                spec,
                sourced_search: CoreSourcedSearchProvenance {
                    energy_window: EnergyBracket::from_values(-2.0, -1.0e-8).unwrap(),
                    intervals: 512,
                },
            }],
            explicit_request
                .states
                .iter()
                .map(retain_shell_occupations)
                .collect(),
        )
        .unwrap();

        assert_eq!(explicit_retained.contribution, explicit_density_oracle);
        assert_eq!(
            explicit_retained.orbitals.shells[0].occupations,
            CoreShellOccupations::ExplicitCollinear { up: 1.5, down: 0.5 }
        );
    }

    #[test]
    fn core_relaxation_rebuilds_cc_and_vc_actions_at_one_fixed_valence_density() {
        let mesh = extended_mesh(1.0e-6, 30.0, 0.018);
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
                (left.get() - 5.0)
                    .abs()
                    .total_cmp(&(right.get() - 5.0).abs())
            })
            .map(|(index, _)| index)
            .unwrap();
        let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
        let solve_spec = CoreDiracSpec::new(
            state,
            1.0,
            EnergyBracket::from_values(-0.6, -0.4).unwrap(),
            mesh.radii()[muffin_tin_index],
        );
        let solution = solve_core_dirac(&mesh, &potential, solve_spec).unwrap();
        let muffin_tin_mesh =
            ExponentialMesh::new(mesh.first(), mesh.increment(), muffin_tin_index + 1).unwrap();
        let valence_p = solution.p[..muffin_tin_mesh.len()].to_vec();
        let valence_q = solution.q[..muffin_tin_mesh.len()].to_vec();
        let valence_normalization = muffin_tin_mesh
            .integrate(
                &valence_p
                    .iter()
                    .zip(&valence_q)
                    .map(|(p, q)| p * p + q * q)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let valence_orbitals = [BorrowedValenceRadial {
            channel: RelativisticChannel::new(
                state.kappa,
                state.kappa.twice_mu_values().next().unwrap(),
            )
            .unwrap(),
            p: &valence_p,
            q: &valence_q,
            normalization: valence_normalization,
        }];
        let valence_matrix = [Complex64::new(0.005, 0.0)];
        let occupations = CoreShellOccupations::MuResolved(
            state.kappa.twice_mu_values().map(|mu| (mu, 0.02)).collect(),
        );
        let initial = CoreShellOrbitals {
            site_index: 3,
            site_id: "fixed-H".to_owned(),
            extended_mesh: mesh,
            shells: vec![CoreShellOrbital {
                state,
                energy: solution.energy,
                p: solution.p,
                q: solution.q,
                norm_total: solution.norm_total,
                norm_mt: solution.norm_mt,
                spill: solution.spill,
                occupations,
            }],
            provenance: CoreShellOrbitalsProvenance {
                extended_potential: potential.into_iter().map(Hartree).collect(),
                solve_specs: vec![solve_spec],
                sourced_searches: vec![CoreSourcedSearchProvenance {
                    energy_window: EnergyBracket::from_values(-2.0, -1.0e-8).unwrap(),
                    intervals: 512,
                }],
            },
        };
        let provenance = initial.provenance.clone();
        let occupations = initial.shells[0].occupations.clone();
        let initial_core = [BorrowedCoreShell {
            kappa: initial.shells[0].state.kappa,
            p: &initial.shells[0].p,
            q: &initial.shells[0].q,
            normalization: initial.shells[0].norm_total,
            occupations: match &initial.shells[0].occupations {
                CoreShellOccupations::MuResolved(occupations) => {
                    ClosedCoreOccupations::MuResolved(occupations)
                }
                CoreShellOccupations::ExplicitCollinear { .. } => unreachable!(),
            },
        }];
        let initial_vc = radial_valence_core_actions(&[RadialSlaterSite {
            site_index: initial.site_index,
            mt_mesh: &muffin_tin_mesh,
            extended_mesh: &initial.extended_mesh,
            cores: &initial_core,
            valence: PreweightedSiteValenceDensity {
                orbitals: &valence_orbitals,
                matrix: &valence_matrix,
            },
        }])
        .unwrap();

        let result = relax_core_at_fixed_potential(
            &initial,
            FixedSiteValenceDensity {
                site_index: initial.site_index,
                muffin_tin_mesh: &muffin_tin_mesh,
                valence: PreweightedSiteValenceDensity {
                    orbitals: &valence_orbitals,
                    matrix: &valence_matrix,
                },
            },
            CoreFixedPotentialSpec {
                action_mixing: 0.5,
                energy_tolerance: Hartree(1.0e-7),
                radial_tolerance: 1.0e-7,
                vc_imaginary_tolerance: 1.0e-10,
                max_iterations: 24,
            },
        )
        .unwrap();

        assert_eq!(result.orbitals.site_index, initial.site_index);
        assert_eq!(result.orbitals.site_id, initial.site_id);
        assert_eq!(result.orbitals.extended_mesh, initial.extended_mesh);
        assert_eq!(result.orbitals.provenance, provenance);
        assert_eq!(result.orbitals.shells[0].occupations, occupations);
        assert_eq!(
            result.orbitals.shells[0].p.len(),
            initial.extended_mesh.len()
        );
        assert_eq!(
            result.orbitals.shells[0].q.len(),
            initial.extended_mesh.len()
        );
        assert!(result.orbitals.shells[0].norm_total.is_finite());
        assert!(result.orbitals.shells[0].norm_mt.is_finite());
        assert!(result.orbitals.shells[0].spill.is_finite());
        let last = result.diagnostics.last().unwrap();
        assert!(last.maximum_energy_change.get() <= 1.0e-7);
        assert!(last.maximum_radial_residual <= 1.0e-7);
        assert!(last.cc_trace.total.get() < 0.0);
        assert!(last.vc_trace.get() < 0.0);
        assert!(last.vc_imaginary_residual <= 1.0e-10);
        assert!(result.final_cc_trace.total.get() < 0.0);
        assert!(result.final_vc_trace.get() < 0.0);
        assert!(result.final_vc_imaginary_residual <= 1.0e-10);
        assert_eq!(valence_matrix, [Complex64::new(0.005, 0.0)]);

        let final_core = [BorrowedCoreShell {
            kappa: result.orbitals.shells[0].state.kappa,
            p: &result.orbitals.shells[0].p,
            q: &result.orbitals.shells[0].q,
            normalization: result.orbitals.shells[0].norm_total,
            occupations: match &result.orbitals.shells[0].occupations {
                CoreShellOccupations::MuResolved(occupations) => {
                    ClosedCoreOccupations::MuResolved(occupations)
                }
                CoreShellOccupations::ExplicitCollinear { .. } => unreachable!(),
            },
        }];
        let independent_final_vc = radial_valence_core_actions(&[RadialSlaterSite {
            site_index: result.orbitals.site_index,
            mt_mesh: &muffin_tin_mesh,
            extended_mesh: &result.orbitals.extended_mesh,
            cores: &final_core,
            valence: PreweightedSiteValenceDensity {
                orbitals: &valence_orbitals,
                matrix: &valence_matrix,
            },
        }])
        .unwrap();
        assert!(
            (independent_final_vc.action_trace.get() - result.final_vc_trace.get()).abs() < 1.0e-12
        );
        let occupation_per_mu = uniform_mu_occupation(0, &result.orbitals.shells[0]).unwrap();
        let fixed_initial_source_final_bra = occupation_per_mu
            * f64::from(result.orbitals.shells[0].state.kappa.degeneracy())
            * result
                .orbitals
                .extended_mesh
                .integrate(
                    &result.orbitals.shells[0]
                        .p
                        .iter()
                        .zip(&result.orbitals.shells[0].q)
                        .zip(initial_vc.shells[0].p.iter().zip(&initial_vc.shells[0].q))
                        .map(|((p, q), (action_p, action_q))| {
                            (p * action_p + q * action_q)
                                / result.orbitals.shells[0].norm_total.sqrt()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        assert!((fixed_initial_source_final_bra - result.final_vc_trace.get()).abs() > 1.0e-12);
    }
}
