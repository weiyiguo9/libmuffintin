//! Direct four-index restricted Hartree-Fock for a finite AO basis.
//!
//! The spin-summed density is `P_mu,nu = 2 sum_i C_mu,i C_nu,i^*`.
//! Electron-repulsion integrals are real chemist-order `(mu nu | lambda sigma)`
//! values stored at `(((mu * n + nu) * n + lambda) * n + sigma)`. The Fock
//! matrix is `F = h + J - K / 2`, with contractions over `P_sigma,lambda`.

#![forbid(unsafe_code)]

use muffintin_core::Hartree;
use muffintin_operators::{OperatorError, solve_generalized_hermitian};
use muffintin_tensor::{Axis, DenseEigenvectors, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use thiserror::Error;

/// A closed-shell finite-basis restricted Hartree-Fock problem.
#[derive(Clone, Debug, PartialEq)]
pub struct RestrictedHfProblem {
    pub overlap: DenseHermitianMatrix,
    /// One-electron Hamiltonian in Hartree.
    pub one_electron: DenseHermitianMatrix,
    /// Real chemist-order `(mu nu | lambda sigma)` integrals in Hartree.
    pub chemist_eri: Vec<f64>,
    pub electron_count: usize,
    pub nuclear_repulsion: Hartree,
}

/// Numerical controls for restricted Hartree-Fock SCF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestrictedHfSpec {
    pub max_iterations: usize,
    pub energy_tolerance: Hartree,
    /// RMS tolerance over all AO density-matrix entries.
    pub density_tolerance: f64,
    /// Weight of the newly solved density in `(1 - alpha) P + alpha P_solved`.
    pub density_mixing: f64,
    pub overlap_threshold: f64,
}

/// One completed SCF update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestrictedHfIterationDiagnostic {
    pub iteration: usize,
    pub total_energy: Hartree,
    pub electronic_energy: Hartree,
    pub energy_change: Hartree,
    pub density_rms: f64,
}

/// Converged restricted Hartree-Fock state.
#[derive(Clone, Debug, PartialEq)]
pub struct RestrictedHfResult {
    pub total_energy: Hartree,
    pub electronic_energy: Hartree,
    pub orbital_energies: Vec<Hartree>,
    pub orbital_coefficients: DenseEigenvectors,
    /// Spin-summed AO density, including the closed-shell factor of two.
    pub density: DenseHermitianMatrix,
    pub iterations: usize,
    pub diagnostics: Vec<RestrictedHfIterationDiagnostic>,
}

/// Invalid restricted Hartree-Fock input or failed SCF solution.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum RestrictedHfError {
    #[error("the AO basis is empty")]
    EmptyBasis,
    #[error("{matrix} uses axis {actual:?}, expected GlobalBasis")]
    MatrixAxis { matrix: &'static str, actual: Axis },
    #[error("one-electron dimension {one_electron} differs from overlap dimension {overlap}")]
    MatrixDimension { one_electron: usize, overlap: usize },
    #[error("closed-shell restricted Hartree-Fock requires an even electron count, got {0}")]
    OddElectronCount(usize),
    #[error("{electrons} electrons exceed the {capacity}-electron AO basis capacity")]
    ElectronCountExceedsBasis { electrons: usize, capacity: usize },
    #[error("ERI buffer has length {actual}, expected {expected} for basis dimension {dimension}")]
    EriLength {
        dimension: usize,
        expected: usize,
        actual: usize,
    },
    #[error("ERI at flat chemist-order index {index} is not finite")]
    NonFiniteEri { index: usize },
    #[error("nuclear repulsion must be finite, got {0}")]
    NonFiniteNuclearRepulsion(f64),
    #[error("max_iterations must be positive")]
    ZeroIterations,
    #[error("energy tolerance must be finite and nonnegative, got {0}")]
    InvalidEnergyTolerance(f64),
    #[error("density tolerance must be finite and nonnegative, got {0}")]
    InvalidDensityTolerance(f64),
    #[error("density mixing must be finite and in (0, 1], got {0}")]
    InvalidDensityMixing(f64),
    #[error("overlap threshold must be finite and nonnegative, got {0}")]
    InvalidOverlapThreshold(f64),
    #[error(
        "the overlap-filtered basis retained {retained} orbitals, fewer than {occupied} occupied orbitals"
    )]
    InsufficientOrbitals { occupied: usize, retained: usize },
    #[error(
        "restricted Hartree-Fock did not converge in {iterations} iterations (energy change {energy_change} Ha, density RMS {density_rms})"
    )]
    NotConverged {
        iterations: usize,
        energy_change: f64,
        density_rms: f64,
    },
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Run a direct four-index closed-shell restricted Hartree-Fock SCF.
pub fn solve_restricted_hf(
    problem: &RestrictedHfProblem,
    spec: &RestrictedHfSpec,
) -> Result<RestrictedHfResult, RestrictedHfError> {
    validate(problem, spec)?;
    let occupied = problem.electron_count / 2;

    // Core-Hamiltonian initial guess.
    let core_solution = solve_generalized_hermitian(
        &problem.one_electron,
        &problem.overlap,
        spec.overlap_threshold,
    )?;
    ensure_occupied_capacity(occupied, core_solution.retained_dimension)?;
    let mut density = density_from_orbitals(&core_solution.eigenvectors, occupied)?;
    let mut fock = build_fock(problem, &density)?;
    let initial_electronic_energy = electronic_energy(problem, &density, &fock);
    let mut total_energy = initial_electronic_energy + problem.nuclear_repulsion;
    let mut diagnostics = Vec::with_capacity(spec.max_iterations);

    for iteration in 1..=spec.max_iterations {
        let solution =
            solve_generalized_hermitian(&fock, &problem.overlap, spec.overlap_threshold)?;
        ensure_occupied_capacity(occupied, solution.retained_dimension)?;
        let solved_density = density_from_orbitals(&solution.eigenvectors, occupied)?;
        let next_density = mix_density(&density, &solved_density, spec.density_mixing)?;
        let density_rms = density_rms_difference(&next_density, &density);
        let next_fock = build_fock(problem, &next_density)?;
        let next_electronic_energy = electronic_energy(problem, &next_density, &next_fock);
        let next_total_energy = next_electronic_energy + problem.nuclear_repulsion;
        let energy_change = (next_total_energy - total_energy).get().abs();

        diagnostics.push(RestrictedHfIterationDiagnostic {
            iteration,
            total_energy: next_total_energy,
            electronic_energy: next_electronic_energy,
            energy_change: Hartree(energy_change),
            density_rms,
        });

        if energy_change <= spec.energy_tolerance.get() && density_rms <= spec.density_tolerance {
            // Report orbitals of the Fock matrix belonging to the returned density.
            let final_solution =
                solve_generalized_hermitian(&next_fock, &problem.overlap, spec.overlap_threshold)?;
            ensure_occupied_capacity(occupied, final_solution.retained_dimension)?;
            return Ok(RestrictedHfResult {
                total_energy: next_total_energy,
                electronic_energy: next_electronic_energy,
                orbital_energies: final_solution.eigenvalues,
                orbital_coefficients: final_solution.eigenvectors,
                density: next_density,
                iterations: iteration,
                diagnostics,
            });
        }

        density = next_density;
        fock = next_fock;
        total_energy = next_total_energy;
    }

    let last = diagnostics
        .last()
        .expect("positive max_iterations produces a diagnostic");
    Err(RestrictedHfError::NotConverged {
        iterations: spec.max_iterations,
        energy_change: last.energy_change.get(),
        density_rms: last.density_rms,
    })
}

fn validate(
    problem: &RestrictedHfProblem,
    spec: &RestrictedHfSpec,
) -> Result<(), RestrictedHfError> {
    let dimension = problem.overlap.dimension();
    if dimension == 0 {
        return Err(RestrictedHfError::EmptyBasis);
    }
    if problem.overlap.axis() != Axis::GlobalBasis {
        return Err(RestrictedHfError::MatrixAxis {
            matrix: "overlap",
            actual: problem.overlap.axis(),
        });
    }
    if problem.one_electron.axis() != Axis::GlobalBasis {
        return Err(RestrictedHfError::MatrixAxis {
            matrix: "one-electron Hamiltonian",
            actual: problem.one_electron.axis(),
        });
    }
    if problem.one_electron.dimension() != dimension {
        return Err(RestrictedHfError::MatrixDimension {
            one_electron: problem.one_electron.dimension(),
            overlap: dimension,
        });
    }
    if problem.electron_count % 2 != 0 {
        return Err(RestrictedHfError::OddElectronCount(problem.electron_count));
    }
    let capacity = 2 * dimension;
    if problem.electron_count > capacity {
        return Err(RestrictedHfError::ElectronCountExceedsBasis {
            electrons: problem.electron_count,
            capacity,
        });
    }
    let expected_eri = dimension
        .checked_pow(4)
        .expect("an allocated square AO matrix has a representable fourth power dimension");
    if problem.chemist_eri.len() != expected_eri {
        return Err(RestrictedHfError::EriLength {
            dimension,
            expected: expected_eri,
            actual: problem.chemist_eri.len(),
        });
    }
    if let Some(index) = problem
        .chemist_eri
        .iter()
        .position(|value| !value.is_finite())
    {
        return Err(RestrictedHfError::NonFiniteEri { index });
    }
    if !problem.nuclear_repulsion.get().is_finite() {
        return Err(RestrictedHfError::NonFiniteNuclearRepulsion(
            problem.nuclear_repulsion.get(),
        ));
    }
    if spec.max_iterations == 0 {
        return Err(RestrictedHfError::ZeroIterations);
    }
    if !spec.energy_tolerance.get().is_finite() || spec.energy_tolerance.get() < 0.0 {
        return Err(RestrictedHfError::InvalidEnergyTolerance(
            spec.energy_tolerance.get(),
        ));
    }
    if !spec.density_tolerance.is_finite() || spec.density_tolerance < 0.0 {
        return Err(RestrictedHfError::InvalidDensityTolerance(
            spec.density_tolerance,
        ));
    }
    if !spec.density_mixing.is_finite() || spec.density_mixing <= 0.0 || spec.density_mixing > 1.0 {
        return Err(RestrictedHfError::InvalidDensityMixing(spec.density_mixing));
    }
    if !spec.overlap_threshold.is_finite() || spec.overlap_threshold < 0.0 {
        return Err(RestrictedHfError::InvalidOverlapThreshold(
            spec.overlap_threshold,
        ));
    }
    Ok(())
}

fn ensure_occupied_capacity(occupied: usize, retained: usize) -> Result<(), RestrictedHfError> {
    if occupied > retained {
        return Err(RestrictedHfError::InsufficientOrbitals { occupied, retained });
    }
    Ok(())
}

fn density_from_orbitals(
    coefficients: &DenseEigenvectors,
    occupied: usize,
) -> Result<DenseHermitianMatrix, TensorError> {
    DenseHermitianMatrix::from_upper_triangle(coefficients.rows(), Axis::GlobalBasis, |mu, nu| {
        let mut value = Complex64::new(0.0, 0.0);
        for orbital in 0..occupied {
            value += 2.0 * coefficients.at(mu, orbital) * coefficients.at(nu, orbital).conj();
        }
        value
    })
}

fn mix_density(
    current: &DenseHermitianMatrix,
    solved: &DenseHermitianMatrix,
    alpha: f64,
) -> Result<DenseHermitianMatrix, TensorError> {
    DenseHermitianMatrix::from_upper_triangle(current.dimension(), Axis::GlobalBasis, |mu, nu| {
        (1.0 - alpha) * current.at(mu, nu) + alpha * solved.at(mu, nu)
    })
}

fn eri_index(dimension: usize, mu: usize, nu: usize, lambda: usize, sigma: usize) -> usize {
    (((mu * dimension + nu) * dimension + lambda) * dimension) + sigma
}

fn build_fock(
    problem: &RestrictedHfProblem,
    density: &DenseHermitianMatrix,
) -> Result<DenseHermitianMatrix, TensorError> {
    let dimension = density.dimension();
    let mut values = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for mu in 0..dimension {
        for nu in 0..dimension {
            let mut coulomb = Complex64::new(0.0, 0.0);
            let mut exchange = Complex64::new(0.0, 0.0);
            for lambda in 0..dimension {
                for sigma in 0..dimension {
                    let p = density.at(sigma, lambda);
                    coulomb += p * problem.chemist_eri[eri_index(dimension, mu, nu, lambda, sigma)];
                    exchange +=
                        p * problem.chemist_eri[eri_index(dimension, mu, lambda, nu, sigma)];
                }
            }
            values[mu * dimension + nu] =
                problem.one_electron.at(mu, nu) + coulomb - 0.5 * exchange;
        }
    }
    DenseHermitianMatrix::from_host_row_major(dimension, Axis::GlobalBasis, values)
}

fn electronic_energy(
    problem: &RestrictedHfProblem,
    density: &DenseHermitianMatrix,
    fock: &DenseHermitianMatrix,
) -> Hartree {
    let dimension = density.dimension();
    let mut trace = Complex64::new(0.0, 0.0);
    for mu in 0..dimension {
        for nu in 0..dimension {
            trace += density.at(nu, mu) * (problem.one_electron.at(mu, nu) + fock.at(mu, nu));
        }
    }
    Hartree(0.5 * trace.re)
}

fn density_rms_difference(left: &DenseHermitianMatrix, right: &DenseHermitianMatrix) -> f64 {
    let dimension = left.dimension();
    let mut squared_norm = 0.0;
    for mu in 0..dimension {
        for nu in 0..dimension {
            squared_norm += (left.at(mu, nu) - right.at(mu, nu)).norm_sqr();
        }
    }
    (squared_norm / (dimension * dimension) as f64).sqrt()
}
