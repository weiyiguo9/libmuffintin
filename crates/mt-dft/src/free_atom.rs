//! Neutral spherical free-atom LDA/PW92 self-consistency.

use std::f64::consts::PI;

use muffintin_core::{ExponentialMesh, Hartree, Kappa, MeshError};
use muffintin_coulomb::{CoulombError, radial_primitive};
use muffintin_radial::{CoreDiracSolution, CoreState};
use thiserror::Error;

use crate::atomic_configuration::{AtomicNumber, fleur_default_atomic_configuration};
use crate::linearization::{
    AtomicEnergyRequest, LinearizationEnergyError, solve_atomic_bound_state,
};
use crate::xc::{DensityJet2, XcError, XcFunctional, evaluate_xc_point};

/// Explicit numerical controls for one neutral free-atom calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct FreeAtomScfSpec {
    pub mesh: ExponentialMesh,
    /// Linear fraction of the newly constructed effective potential.
    pub mixing: f64,
    /// Maximum absolute effective-potential residual in Hartree.
    pub potential_tolerance: f64,
    /// Dimensionless tolerance for both charge closure and the outer logarithmic-shell charge.
    pub tail_tolerance: f64,
    pub max_iterations: usize,
}

/// One occupied signed-kappa orbital of the neutral atom.
#[derive(Clone, Debug, PartialEq)]
pub struct FreeAtomOrbital {
    pub occupation: f64,
    pub solution: CoreDiracSolution,
}

/// Converged unpolarized spherical LDA/PW92 atom.
#[derive(Clone, Debug, PartialEq)]
pub struct FreeAtomState {
    pub atomic_number: AtomicNumber,
    pub mesh: ExponentialMesh,
    /// Physical electron number density in inverse cubic bohr.
    pub number_density: Vec<f64>,
    /// Total Kohn-Sham potential, including the attractive nuclear term, in Hartree.
    pub effective_potential: Vec<Hartree>,
    pub orbitals: Vec<FreeAtomOrbital>,
    pub iterations: usize,
}

/// Diagnosable free-atom input, radial-solve, and convergence failures.
#[derive(Debug, Error, PartialEq)]
pub enum FreeAtomScfError {
    #[error("free-atom potential mixing must be finite and in (0, 1], got {0}")]
    InvalidMixing(f64),
    #[error("free-atom potential tolerance must be finite and positive, got {0}")]
    InvalidPotentialTolerance(f64),
    #[error("free-atom tail tolerance must be finite and positive, got {0}")]
    InvalidTailTolerance(f64),
    #[error("free-atom SCF requires at least one iteration")]
    InvalidMaxIterations,
    #[error("free-atom bound-state solve failed at iteration {iteration} for n={}, kappa={}", state.n, state.kappa.get())]
    BoundState {
        iteration: usize,
        state: CoreState,
        #[source]
        source: LinearizationEnergyError,
    },
    #[error("free-atom exchange-correlation evaluation failed at radial index {index}")]
    ExchangeCorrelation {
        index: usize,
        #[source]
        source: XcError,
    },
    #[error(transparent)]
    Quadrature(#[from] MeshError),
    #[error(transparent)]
    Coulomb(#[from] CoulombError),
    #[error(
        "free-atom SCF did not converge after {iterations} iterations: potential residual {potential_residual} Ha, charge error {charge_error}, tail charge {tail_charge}"
    )]
    NotConverged {
        iterations: usize,
        potential_residual: f64,
        charge_error: f64,
        tail_charge: f64,
    },
}

/// Solve a neutral atom from the bare nuclear potential using spherical unpolarized LDA/PW92.
pub fn run_free_atom_lda(
    atomic_number: AtomicNumber,
    spec: &FreeAtomScfSpec,
) -> Result<FreeAtomState, FreeAtomScfError> {
    validate_spec(spec)?;

    let mesh = &spec.mesh;
    let nuclear_charge = f64::from(atomic_number.get());
    let muffin_tin_radius = mesh.radii()[mesh.len() - 2];
    let configuration = fleur_default_atomic_configuration(atomic_number);
    let mut potential = mesh
        .radii()
        .iter()
        .map(|radius| -nuclear_charge / radius.get())
        .collect::<Vec<_>>();
    let mut potential_residual = f64::INFINITY;
    let mut charge_error = f64::INFINITY;
    let mut tail_charge = f64::INFINITY;

    for iteration in 1..=spec.max_iterations {
        let mut orbitals = Vec::with_capacity(configuration.occupations().len());
        for occupation in configuration.occupations() {
            let state = CoreState::new(
                u32::from(occupation.orbital.principal_quantum_number()),
                Kappa::new(i32::from(occupation.orbital.kappa()))
                    .expect("the embedded atomic configuration has nonzero kappa"),
            )
            .expect("the embedded atomic configuration is physically admissible");
            let request = AtomicEnergyRequest::new(state, nuclear_charge, muffin_tin_radius);
            let solved = solve_atomic_bound_state(mesh, &potential, request).map_err(|source| {
                FreeAtomScfError::BoundState {
                    iteration,
                    state,
                    source,
                }
            })?;
            orbitals.push(FreeAtomOrbital {
                occupation: occupation.occupation,
                solution: solved.solution,
            });
        }

        let radial_probability = radial_probability(mesh.len(), &orbitals);
        let charge = mesh.integrate(&radial_probability)?;
        charge_error = (charge - nuclear_charge).abs();
        tail_charge = mesh.last().get() * radial_probability[mesh.len() - 1];
        let number_density = radial_probability
            .iter()
            .zip(mesh.radii())
            .map(|(&radial, radius)| radial / (4.0 * PI * radius.get().powi(2)))
            .collect::<Vec<_>>();
        let next_potential = effective_potential(
            nuclear_charge,
            mesh,
            &radial_probability,
            &number_density,
        )?;
        potential_residual = potential
            .iter()
            .zip(&next_potential)
            .map(|(&old, &new)| (new - old).abs())
            .fold(0.0, f64::max);

        if potential_residual <= spec.potential_tolerance
            && charge_error <= spec.tail_tolerance
            && tail_charge <= spec.tail_tolerance
        {
            return Ok(FreeAtomState {
                atomic_number,
                mesh: mesh.clone(),
                number_density,
                effective_potential: next_potential.into_iter().map(Hartree).collect(),
                orbitals,
                iterations: iteration,
            });
        }

        for (old, new) in potential.iter_mut().zip(next_potential) {
            *old += spec.mixing * (new - *old);
        }
    }

    Err(FreeAtomScfError::NotConverged {
        iterations: spec.max_iterations,
        potential_residual,
        charge_error,
        tail_charge,
    })
}

fn validate_spec(spec: &FreeAtomScfSpec) -> Result<(), FreeAtomScfError> {
    if !spec.mixing.is_finite() || !(0.0 < spec.mixing && spec.mixing <= 1.0) {
        return Err(FreeAtomScfError::InvalidMixing(spec.mixing));
    }
    if !spec.potential_tolerance.is_finite() || spec.potential_tolerance <= 0.0 {
        return Err(FreeAtomScfError::InvalidPotentialTolerance(
            spec.potential_tolerance,
        ));
    }
    if !spec.tail_tolerance.is_finite() || spec.tail_tolerance <= 0.0 {
        return Err(FreeAtomScfError::InvalidTailTolerance(
            spec.tail_tolerance,
        ));
    }
    if spec.max_iterations == 0 {
        return Err(FreeAtomScfError::InvalidMaxIterations);
    }
    Ok(())
}

fn radial_probability(mesh_len: usize, orbitals: &[FreeAtomOrbital]) -> Vec<f64> {
    let mut radial_probability = vec![0.0; mesh_len];
    for orbital in orbitals {
        for ((density, &large), &small) in radial_probability
            .iter_mut()
            .zip(&orbital.solution.p)
            .zip(&orbital.solution.q)
        {
            *density += orbital.occupation * (large * large + small * small);
        }
    }
    radial_probability
}

fn effective_potential(
    nuclear_charge: f64,
    mesh: &ExponentialMesh,
    radial_probability: &[f64],
    number_density: &[f64],
) -> Result<Vec<f64>, FreeAtomScfError> {
    let enclosed_charge = radial_primitive(mesh, radial_probability, false)?;
    let tail_integrand = radial_probability
        .iter()
        .zip(mesh.radii())
        .map(|(&density, radius)| density / radius.get())
        .collect::<Vec<_>>();
    let outer_potential = radial_primitive(mesh, &tail_integrand, true)?;

    mesh.radii()
        .iter()
        .zip(number_density)
        .zip(enclosed_charge.into_iter().zip(outer_potential))
        .enumerate()
        .map(|(index, ((radius, &density), (enclosed, outer)))| {
            let xc = evaluate_xc_point(
                XcFunctional::LdaPw92,
                DensityJet2 {
                    rho: [0.5 * density; 2],
                    gradient: [[0.0; 3]; 2],
                    hessian: [[0.0; 6]; 2],
                },
            )
            .map_err(|source| FreeAtomScfError::ExchangeCorrelation { index, source })?;
            Ok((enclosed - nuclear_charge) / radius.get()
                + outer
                + xc.potential[0].get())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::Bohr;

    #[test]
    fn neutral_boron_converges_with_every_occupied_signed_kappa_channel() {
        let first = 1.0e-6_f64;
        let increment = 0.004;
        let last = 30.0_f64;
        let number = ((last / first).ln() / increment).ceil() as usize + 1;
        let spec = FreeAtomScfSpec {
            mesh: ExponentialMesh::new(Bohr(first), increment, number).unwrap(),
            mixing: 0.3,
            potential_tolerance: 2.0e-5,
            tail_tolerance: 1.0e-7,
            max_iterations: 120,
        };
        let atomic_number = AtomicNumber::new(5).unwrap();
        let state = run_free_atom_lda(atomic_number, &spec).unwrap();
        let expected = fleur_default_atomic_configuration(atomic_number);

        assert!(state.iterations <= spec.max_iterations);
        assert_eq!(state.orbitals.len(), expected.occupations().len());
        for occupation in expected.occupations() {
            assert!(state.orbitals.iter().any(|orbital| {
                orbital.solution.state.n
                    == u32::from(occupation.orbital.principal_quantum_number())
                    && orbital.solution.state.kappa.get()
                        == i32::from(occupation.orbital.kappa())
                    && (orbital.occupation - occupation.occupation).abs() < 1.0e-14
            }));
        }
        assert!(state.number_density.iter().all(|value| value.is_finite()));
        assert!(
            state
                .effective_potential
                .iter()
                .all(|value| value.get().is_finite())
        );
        assert!(state.orbitals.iter().all(|orbital| {
            orbital.solution.p.iter().all(|value| value.is_finite())
                && orbital.solution.q.iter().all(|value| value.is_finite())
                && (orbital.solution.norm_total - 1.0).abs() < 1.0e-10
        }));
        let charge_samples = state
            .number_density
            .iter()
            .zip(state.mesh.radii())
            .map(|(&density, radius)| 4.0 * PI * density * radius.get().powi(2))
            .collect::<Vec<_>>();
        let charge = state.mesh.integrate(&charge_samples).unwrap();
        assert!((charge - f64::from(atomic_number.get())).abs() <= spec.tail_tolerance);
    }
}
