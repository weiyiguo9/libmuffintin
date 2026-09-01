//! Gamma-only spinor-first valence Hartree--Fock SCF.

use muffintin_core::{Hartree, InverseBohr};
use muffintin_coulomb::{CoulombRequest, HartreeError, WeinertHartreeSpec};
use muffintin_dft::{
    BandState, CheckpointBandSolution, CheckpointKPointSolution, DensityError, DensityMixer,
    ElectrostaticSpec, MaterialKernelError, MixingError, OccupationError, RegionalDensity,
    RegionalElectrostaticError, RegionalError, RegionalPotential, ScfConfig, ScfConfigError,
    ScfKReduction, ScfLoop, ScfMixing, ScfOccupations, ScfPhysics, ScfRelativity, electron_count,
    evaluate_regional_electrostatics,
};
use muffintin_operators::{
    OperatorError, lift_band_hermitian_feedback, solve_generalized_hermitian,
};
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError, einsum};
use num_complex::Complex64;
use thiserror::Error;

use crate::{
    CheckpointPhysics, CheckpointPhysicsError, GammaExchangeTreatment, IsdfExchangeError,
    IsdfExchangeResult, IsdfExchangeSpec, SpinorMpbError, SpinorMpbSelection, SpinorMpbSpec,
    build_spinor_mpb, build_spinor_mpb_exchange,
};

const SPECTRAL_REFINEMENT_PASSES: usize = 16;
const IDENTITY_TOLERANCE: f64 = 1.0e-8;
const ELECTRON_COUNT_TOLERANCE: f64 = 1.0e-8;

/// Exact MPB controls and bounded iteration controls for the A1 driver.
#[derive(Clone, Debug, PartialEq)]
pub struct GammaValenceHfSpec {
    pub config: ScfConfig,
    pub product_l_max: u32,
    pub product_g_max: InverseBohr,
    pub overlap_tolerance: f64,
    pub coulomb: CoulombRequest,
    pub max_fock_iterations: usize,
    /// Regional-density fixed-point tolerance inside one fixed local potential.
    pub fock_density_tolerance: f64,
    /// Weight of the freshly rebuilt band-space exchange in the next Fock solve.
    pub fock_mixing: f64,
}

/// One completed outer Hartree-density iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct GammaValenceHfIterationDiagnostic {
    pub iteration: usize,
    pub fock_iterations: usize,
    pub exchange_rebuilds: usize,
    pub exchange_energy: Hartree,
    pub maximum_antihermitian_residual: f64,
    pub fock_fixed_point_residual: f64,
    pub regional_density_rms: f64,
    pub density_electron_count: f64,
    pub total_energy: Hartree,
    pub energy_change: Option<Hartree>,
    /// `E_x - 1/2 Tr(D K)`.
    pub exchange_energy_identity_residual: f64,
    /// `sum(f epsilon) - Tr(D H0) - Tr(D K)`.
    pub eigenvalue_identity_residual: f64,
    /// Difference between direct-functional and eigenvalue-corrected HF energy.
    pub total_energy_identity_residual: f64,
    /// Maximum element of `C^H (S C K C^H S) C - K`.
    pub lifting_identity_residual: f64,
    /// First global H0+lifted-K solve versus `diag(epsilon0)+K`.
    pub first_global_solve_identity_residual: Option<f64>,
    /// Independent first one-shot rebuild versus the driver first rebuild.
    pub first_one_shot_parity_residual: Option<f64>,
}

/// Converged Gamma valence-only spinor HF state.
#[derive(Clone, Debug)]
pub struct GammaValenceHfResult {
    pub density: RegionalDensity,
    pub potential: RegionalPotential,
    pub bands: CheckpointBandSolution,
    /// Per-k fractional occupations; A1 returns one row while preserving the
    /// mesh-shaped boundary needed by a later regular-k engine.
    pub occupations: Vec<Vec<f64>>,
    pub orbital_energies: Vec<Vec<Hartree>>,
    pub exchange_energy: Hartree,
    pub maximum_antihermitian_residual: f64,
    pub fock_fixed_point_residual: f64,
    pub regional_density_rms: f64,
    pub total_energy: Hartree,
    pub exchange_rebuilds: usize,
    pub first_one_shot_exchange: IsdfExchangeResult,
    pub diagnostics: Vec<GammaValenceHfIterationDiagnostic>,
}

/// Invalid A1 controls or a failed bounded HF solve.
#[derive(Debug, Error)]
pub enum GammaValenceHfError {
    #[error("Gamma valence HF requires a 1x1x1 full k mesh with zero shift")]
    GammaMesh,
    #[error("Gamma valence HF requires ScfRelativity::SpinorFirstVariation")]
    SpinorFirstVariation,
    #[error("Gamma valence HF is valence-only and rejects every occupied core state")]
    CoreStates,
    #[error("Gamma valence HF max_fock_iterations must be at least two")]
    FockIterations,
    #[error("Gamma valence HF fock_density_tolerance must be finite and positive")]
    FockTolerance,
    #[error("Gamma valence HF fock_mixing must be finite and in (0, 1]")]
    FockMixing,
    #[error("spectral radial-basis refinement did not settle after {passes} passes")]
    SpectralRefinement { passes: usize },
    #[error(
        "fixed-local-potential Fock iteration {outer_iteration} did not converge in {iterations} rebuilds (residual {residual})"
    )]
    FockNotConverged {
        outer_iteration: usize,
        iterations: usize,
        residual: f64,
    },
    #[error(
        "Gamma valence HF did not converge in {iterations} outer iterations (energy change {energy_change} Ha, density RMS {density_rms})"
    )]
    NotConverged {
        iterations: usize,
        energy_change: f64,
        density_rms: f64,
    },
    #[error(transparent)]
    Config(#[from] ScfConfigError),
    #[error(transparent)]
    Checkpoint(#[from] CheckpointPhysicsError),
    #[error(transparent)]
    Kernel(#[from] MaterialKernelError),
    #[error(transparent)]
    Electrostatic(#[from] RegionalElectrostaticError),
    #[error(transparent)]
    Hartree(#[from] HartreeError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Density(#[from] DensityError),
    #[error(transparent)]
    Occupation(#[from] OccupationError),
    #[error(transparent)]
    Mixing(#[from] MixingError),
    #[error(transparent)]
    Mpb(#[from] SpinorMpbError),
    #[error(transparent)]
    Exchange(#[from] IsdfExchangeError),
    #[error("exchange band matrix {actual} is not in expected k order {expected}")]
    ExchangeKIndex { expected: usize, actual: usize },
    #[error("Gamma valence HF gate {gate} has residual {residual}, above {tolerance}")]
    Gate {
        gate: &'static str,
        residual: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

/// Run the production A1 molecule-in-box route.
///
/// This fixes Gamma, one k point, full-spinor first variation, finite Coulomb
/// body, and an empty core sector. Every fixed-potential Fock step rebuilds
/// all VV MPB columns from the current live orbitals. Every outer density step
/// rematerializes the radial basis, so band feedback is never carried between
/// incompatible local H0/S frames.
pub fn run_gamma_valence_hf(
    physics: &mut CheckpointPhysics,
    spec: &GammaValenceHfSpec,
) -> Result<GammaValenceHfResult, GammaValenceHfError> {
    validate_spec(spec)?;
    let _ = ScfLoop::new(spec.config.clone(), None)?;
    let mut mixer = density_mixer(spec.config.mixing)?;
    let mut density = <muffintin_dft::MaterialKernel as ScfPhysics>::initial_density(
        &mut physics.kernel,
        &spec.config,
    )?;
    let mut previous_total = None;
    let mut diagnostics = Vec::with_capacity(spec.config.convergence.max_iterations);
    let mut total_exchange_rebuilds = 0;
    let mut first_one_shot_exchange = None;

    for outer_iteration in 1..=spec.config.convergence.max_iterations {
        let electrostatic = evaluate_regional_electrostatics(
            density.charge(),
            &ElectrostaticSpec::new(
                WeinertHartreeSpec::electronic(4)?,
                physics.nuclear_charges().to_vec(),
            )?,
        )?;
        let zero = electrostatic.potential.zero_like();
        let potential = RegionalPotential::new(
            electrostatic.potential.clone(),
            [zero.clone(), zero.clone(), zero],
        )?;
        let mut one_particle = physics
            .kernel
            .materialize_checkpoint_one_particle(&potential, &spec.config.basis)?;
        let mut bands = physics.kernel.solve_points(
            one_particle.potential(),
            one_particle.basis(),
            &[[0.0; 3]],
            ScfRelativity::SpinorFirstVariation,
        )?;
        let mut occupation = solve_occupations(
            bands.states(),
            spec.config.electron_count,
            spec.config.occupations,
        )?;
        for pass in 1..=SPECTRAL_REFINEMENT_PASSES {
            let Some(refined) = physics.kernel.refine_spectral_basis(
                &spec.config.basis,
                &one_particle,
                &bands,
                &occupation.values,
                occupation.chemical_potential,
                ScfRelativity::SpinorFirstVariation,
            )?
            else {
                break;
            };
            if pass == SPECTRAL_REFINEMENT_PASSES {
                return Err(GammaValenceHfError::SpectralRefinement { passes: pass });
            }
            one_particle = refined;
            bands = physics.kernel.solve_points(
                one_particle.potential(),
                one_particle.basis(),
                &[[0.0; 3]],
                ScfRelativity::SpinorFirstVariation,
            )?;
            occupation = solve_occupations(
                bands.states(),
                spec.config.electron_count,
                spec.config.occupations,
            )?;
        }

        let fixed = solve_fixed_potential(physics, spec, bands, outer_iteration)?;
        total_exchange_rebuilds += fixed.exchange_rebuilds;
        if let Some(first) = fixed.first_one_shot_exchange.clone() {
            first_one_shot_exchange = Some(first);
        }
        let output_density = physics
            .kernel
            .synthesize_bands(&fixed.bands, &fixed.occupation.values)?;
        let density_rms = density.difference_rms(&output_density)?;
        let output_electron_count = electron_count(&output_density)?;
        let energy = energy_diagnostic(
            &fixed.bands,
            &fixed.occupation,
            &fixed.exchange,
            &electrostatic,
        )?;
        let energy_change = previous_total
            .map(|previous: Hartree| Hartree((energy.total.get() - previous.get()).abs()));
        require_gate(
            "density electron count",
            (output_electron_count - spec.config.electron_count).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        require_gate(
            "exchange energy identity",
            energy.exchange_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        require_gate(
            "valence eigenvalue identity",
            energy.eigenvalue_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        require_gate(
            "HF total-energy identity",
            energy.total_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        diagnostics.push(GammaValenceHfIterationDiagnostic {
            iteration: outer_iteration,
            fock_iterations: fixed.fock_iterations,
            exchange_rebuilds: fixed.exchange_rebuilds,
            exchange_energy: fixed.exchange.exchange_energy,
            maximum_antihermitian_residual: fixed.exchange.maximum_antihermitian_residual,
            fock_fixed_point_residual: fixed.fixed_point_residual,
            regional_density_rms: density_rms,
            density_electron_count: output_electron_count,
            total_energy: energy.total,
            energy_change,
            exchange_energy_identity_residual: energy.exchange_identity_residual,
            eigenvalue_identity_residual: energy.eigenvalue_identity_residual,
            total_energy_identity_residual: energy.total_identity_residual,
            lifting_identity_residual: fixed.lifting_identity_residual,
            first_global_solve_identity_residual: fixed.first_global_solve_identity_residual,
            first_one_shot_parity_residual: fixed.first_one_shot_parity_residual,
        });
        let converged = energy_change
            .is_some_and(|change| change.get() <= spec.config.convergence.energy_tolerance.get())
            && density_rms <= spec.config.convergence.density_tolerance
            && fixed.fixed_point_residual <= spec.fock_density_tolerance;
        if converged {
            let orbital_energies = spinor_energies(&fixed.bands)?;
            let occupations = occupation_rows(&fixed.occupation.values, &fixed.bands)?;
            return Ok(GammaValenceHfResult {
                density: output_density,
                potential,
                bands: fixed.bands,
                occupations,
                orbital_energies,
                exchange_energy: fixed.exchange.exchange_energy,
                maximum_antihermitian_residual: fixed.exchange.maximum_antihermitian_residual,
                fock_fixed_point_residual: fixed.fixed_point_residual,
                regional_density_rms: density_rms,
                total_energy: energy.total,
                exchange_rebuilds: total_exchange_rebuilds,
                first_one_shot_exchange: first_one_shot_exchange
                    .expect("first outer iteration always records the one-shot oracle"),
                diagnostics,
            });
        }
        previous_total = Some(energy.total);
        density = mixer.mix(&density, &output_density)?.density;
    }

    let last = diagnostics
        .last()
        .expect("positive validated outer iteration count");
    Err(GammaValenceHfError::NotConverged {
        iterations: diagnostics.len(),
        energy_change: last
            .energy_change
            .map(Hartree::get)
            .unwrap_or(f64::INFINITY),
        density_rms: last.regional_density_rms,
    })
}

#[derive(Clone)]
struct OccupationSolution {
    chemical_potential: Hartree,
    values: Vec<f64>,
    band_energy: Hartree,
    correction: Hartree,
}

fn solve_occupations(
    states: &[BandState],
    electron_count: f64,
    spec: ScfOccupations,
) -> Result<OccupationSolution, OccupationError> {
    Ok(match spec {
        ScfOccupations::FermiDirac { temperature } => {
            let solved = muffintin_dft::solve_fermi_dirac(
                states,
                electron_count,
                temperature,
                1.0e-12,
                256,
            )?;
            OccupationSolution {
                chemical_potential: solved.chemical_potential,
                values: solved.occupations,
                band_energy: solved.band_energy,
                correction: solved.minus_temperature_entropy,
            }
        }
        ScfOccupations::Gaussian { width } => {
            let solved =
                muffintin_dft::solve_gaussian(states, electron_count, width, 1.0e-12, 256)?;
            OccupationSolution {
                chemical_potential: solved.chemical_potential,
                values: solved.occupations,
                band_energy: solved.band_energy,
                correction: solved.smearing_correction,
            }
        }
    })
}

struct FixedPotentialResult {
    bands: CheckpointBandSolution,
    occupation: OccupationSolution,
    exchange: IsdfExchangeResult,
    fock_iterations: usize,
    exchange_rebuilds: usize,
    fixed_point_residual: f64,
    lifting_identity_residual: f64,
    first_global_solve_identity_residual: Option<f64>,
    first_one_shot_parity_residual: Option<f64>,
    first_one_shot_exchange: Option<IsdfExchangeResult>,
}

fn solve_fixed_potential(
    physics: &CheckpointPhysics,
    spec: &GammaValenceHfSpec,
    mut bands: CheckpointBandSolution,
    outer_iteration: usize,
) -> Result<FixedPotentialResult, GammaValenceHfError> {
    let mut rebuilds = 0;
    let mut first_one_shot_exchange = None;
    let mut first_one_shot_parity_residual = None;
    let mut first_global_solve_identity_residual = None;
    let mut last_residual = f64::INFINITY;
    let mut previous_feedback = None;
    for fock_iteration in 1..=spec.max_fock_iterations {
        let occupation = solve_occupations(
            bands.states(),
            spec.config.electron_count,
            spec.config.occupations,
        )?;
        let occupation_rows = occupation_rows(&occupation.values, &bands)?;
        if outer_iteration == 1 && fock_iteration == 1 {
            let oracle = rebuild_exchange(physics, spec, &bands, &occupation_rows)?;
            rebuilds += 1;
            first_one_shot_exchange = Some(oracle.clone());
            let driver = rebuild_exchange(physics, spec, &bands, &occupation_rows)?;
            rebuilds += 1;
            first_one_shot_parity_residual =
                Some(exchange_difference(&oracle, &driver));
            require_gate(
                "first one-shot parity",
                first_one_shot_parity_residual.expect("just recorded"),
                IDENTITY_TOLERANCE,
            )?;
            let feedback = exchange_feedback(&driver)?;
            let lifting_identity_residual = lifting_identity(&bands, &feedback)?;
            require_gate(
                "band-feedback lifting",
                lifting_identity_residual,
                IDENTITY_TOLERANCE,
            )?;
            let solved = bands.solve_spinor_feedback(&feedback)?;
            let solve_identity = first_global_solve_identity(&bands, &feedback, &solved)?;
            require_gate(
                "first global generalized solve",
                solve_identity,
                IDENTITY_TOLERANCE,
            )?;
            first_global_solve_identity_residual = Some(solve_identity);
            let solved_occupation = solve_occupations(
                solved.states(),
                spec.config.electron_count,
                spec.config.occupations,
            )?;
            last_residual = fixed_point_density_residual(
                physics,
                &bands,
                &occupation.values,
                &solved,
                &solved_occupation.values,
            )?;
            bands = solved;
            previous_feedback = Some(feedback);
            if fock_iteration == spec.max_fock_iterations {
                return Err(GammaValenceHfError::FockNotConverged {
                    outer_iteration,
                    iterations: fock_iteration,
                    residual: last_residual,
                });
            }
            continue;
        }

        let rebuilt = rebuild_exchange(physics, spec, &bands, &occupation_rows)?;
        rebuilds += 1;
        let fresh_feedback = exchange_feedback(&rebuilt)?;
        let feedback_fixed_residual = previous_feedback
            .as_ref()
            .map(|previous| feedback_difference(previous, &fresh_feedback))
            .transpose()?
            .unwrap_or(f64::INFINITY);
        let feedback = match &previous_feedback {
            Some(previous) => mix_feedback(previous, &fresh_feedback, spec.fock_mixing)?,
            None => fresh_feedback,
        };
        let lifting_identity_residual = lifting_identity(&bands, &feedback)?;
        require_gate(
            "band-feedback lifting",
            lifting_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        let solved = bands.solve_spinor_feedback(&feedback)?;
        let solved_occupation = solve_occupations(
            solved.states(),
            spec.config.electron_count,
            spec.config.occupations,
        )?;
        last_residual = fixed_point_density_residual(
            physics,
            &bands,
            &occupation.values,
            &solved,
            &solved_occupation.values,
        )?;
        if last_residual <= spec.fock_density_tolerance
            && feedback_fixed_residual <= IDENTITY_TOLERANCE
        {
            return Ok(FixedPotentialResult {
                bands,
                occupation,
                exchange: rebuilt,
                fock_iterations: fock_iteration,
                exchange_rebuilds: rebuilds,
                fixed_point_residual: last_residual,
                lifting_identity_residual,
                first_global_solve_identity_residual,
                first_one_shot_parity_residual,
                first_one_shot_exchange,
            });
        }
        bands = solved;
        previous_feedback = Some(feedback);
    }
    Err(GammaValenceHfError::FockNotConverged {
        outer_iteration,
        iterations: spec.max_fock_iterations,
        residual: last_residual,
    })
}

fn rebuild_exchange(
    physics: &CheckpointPhysics,
    spec: &GammaValenceHfSpec,
    bands: &CheckpointBandSolution,
    occupations: &[Vec<f64>],
) -> Result<IsdfExchangeResult, GammaValenceHfError> {
    let input = physics.spinor_product_input_from_bands(bands, &[[0.0; 3]], [0.0; 3])?;
    let n_k = input.pair_columns.n_k;
    let n_orb = input.pair_columns.n_orb;
    let selections = (0..n_k)
        .flat_map(|k| {
            (0..n_orb).flat_map(move |left_band| {
                (0..n_orb).map(move |right_band| SpinorMpbSelection {
                    k,
                    left_band,
                    right_band,
                })
            })
        })
        .collect();
    let mpb = build_spinor_mpb(
        &input,
        &SpinorMpbSpec {
            product_l_max: spec.product_l_max,
            product_g_max: spec.product_g_max,
            overlap_tolerance: spec.overlap_tolerance,
            selections,
        },
    )?;
    Ok(build_spinor_mpb_exchange(
        std::slice::from_ref(&input),
        std::slice::from_ref(&mpb),
        &spec.coulomb,
        &IsdfExchangeSpec {
            k_weights: vec![1.0],
            occupations: occupations.to_vec(),
            gamma: GammaExchangeTreatment::FiniteBody,
        },
    )?)
}

fn exchange_feedback(
    exchange: &IsdfExchangeResult,
) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
    exchange
        .band_matrices
        .iter()
        .enumerate()
        .map(|(k, matrix)| {
            if matrix.k_index() != k {
                return Err(GammaValenceHfError::ExchangeKIndex {
                    expected: k,
                    actual: matrix.k_index(),
                });
            }
            Ok(DenseHermitianMatrix::from_upper_triangle(
                matrix.n_bands(),
                Axis::Band,
                |row, column| {
                    matrix
                        .element(row, column)
                        .expect("validated band matrix index")
                },
            )?)
        })
        .collect()
}

fn mix_feedback(
    previous: &[DenseHermitianMatrix],
    fresh: &[DenseHermitianMatrix],
    alpha: f64,
) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
    if previous.len() != fresh.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: previous.len(),
            actual: fresh.len(),
        });
    }
    previous
        .iter()
        .zip(fresh)
        .map(|(previous, fresh)| {
            if previous.dimension() != fresh.dimension() {
                return Err(GammaValenceHfError::ExchangeKIndex {
                    expected: previous.dimension(),
                    actual: fresh.dimension(),
                });
            }
            Ok(DenseHermitianMatrix::from_upper_triangle(
                fresh.dimension(),
                Axis::Band,
                |row, column| {
                    (1.0 - alpha) * previous.at(row, column)
                        + alpha * fresh.at(row, column)
                },
            )?)
        })
        .collect()
}

fn feedback_difference(
    left: &[DenseHermitianMatrix],
    right: &[DenseHermitianMatrix],
) -> Result<f64, GammaValenceHfError> {
    if left.len() != right.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: left.len(),
            actual: right.len(),
        });
    }
    let mut maximum = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        if left.dimension() != right.dimension() {
            return Err(GammaValenceHfError::ExchangeKIndex {
                expected: left.dimension(),
                actual: right.dimension(),
            });
        }
        for row in 0..left.dimension() {
            for column in 0..left.dimension() {
                maximum = maximum.max((left.at(row, column) - right.at(row, column)).norm());
            }
        }
    }
    Ok(maximum)
}

fn fixed_point_density_residual(
    physics: &CheckpointPhysics,
    current: &CheckpointBandSolution,
    current_occupations: &[f64],
    solved: &CheckpointBandSolution,
    solved_occupations: &[f64],
) -> Result<f64, GammaValenceHfError> {
    let current_density = physics
        .kernel
        .synthesize_bands(current, current_occupations)?;
    let solved_density = physics
        .kernel
        .synthesize_bands(solved, solved_occupations)?;
    Ok(current_density.difference_rms(&solved_density)?)
}

fn lifting_identity(
    bands: &CheckpointBandSolution,
    feedback: &[DenseHermitianMatrix],
) -> Result<f64, GammaValenceHfError> {
    let mut maximum = 0.0_f64;
    for (point, expected) in bands.points().iter().zip(feedback) {
        let CheckpointKPointSolution::Spinor {
            eigenproblem,
            solution,
            occupations: state_range,
            ..
        } = &point.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let lifted =
            lift_band_hermitian_feedback(&eigenproblem.overlap, &solution.eigenvectors, expected)?;
        let conjugate = solution.eigenvectors.as_tensor().conjugate();
        let projected = DenseHermitianMatrix::from_tensor(einsum(
            "ia,ij,jb->ab",
            &[
                &conjugate,
                lifted.as_tensor(),
                solution.eigenvectors.as_tensor(),
            ],
        )?)?;
        for row in 0..expected.dimension() {
            for column in 0..expected.dimension() {
                maximum =
                    maximum.max((projected.at(row, column) - expected.at(row, column)).norm());
            }
        }
    }
    Ok(maximum)
}

fn first_global_solve_identity(
    source: &CheckpointBandSolution,
    feedback: &[DenseHermitianMatrix],
    solved: &CheckpointBandSolution,
) -> Result<f64, GammaValenceHfError> {
    let mut maximum = 0.0_f64;
    for ((source, feedback), solved) in source.points().iter().zip(feedback).zip(solved.points()) {
        let CheckpointKPointSolution::Spinor {
            solution: source_solution,
            ..
        } = &source.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let CheckpointKPointSolution::Spinor {
            solution: solved_solution,
            ..
        } = &solved.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let dimension = source_solution.eigenvalues.len();
        let band_fock = DenseHermitianMatrix::from_upper_triangle(
            dimension,
            Axis::GlobalBasis,
            |row, column| {
                feedback.at(row, column)
                    + if row == column {
                        Complex64::new(source_solution.eigenvalues[row].get(), 0.0)
                    } else {
                        Complex64::default()
                    }
            },
        )?;
        let identity = DenseHermitianMatrix::from_upper_triangle(
            dimension,
            Axis::GlobalBasis,
            |row, column| {
                if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::default()
                }
            },
        )?;
        let band_solution = solve_generalized_hermitian(&band_fock, &identity, 0.0)?;
        for (&left, &right) in band_solution
            .eigenvalues
            .iter()
            .zip(&solved_solution.eigenvalues)
        {
            maximum = maximum.max((left.get() - right.get()).abs());
        }
    }
    Ok(maximum)
}

struct EnergyDiagnostic {
    total: Hartree,
    exchange_identity_residual: f64,
    eigenvalue_identity_residual: f64,
    total_identity_residual: f64,
}

fn energy_diagnostic(
    bands: &CheckpointBandSolution,
    occupation: &OccupationSolution,
    exchange: &IsdfExchangeResult,
    electrostatic: &muffintin_dft::RegionalElectrostaticResult,
) -> Result<EnergyDiagnostic, GammaValenceHfError> {
    let rows = occupation_rows(&occupation.values, bands)?;
    let mut h0_expectation = 0.0;
    let mut exchange_trace = 0.0;
    for (k, point) in bands.points().iter().enumerate() {
        let CheckpointKPointSolution::Spinor {
            eigenproblem,
            solution,
            occupations: state_range,
            ..
        } = &point.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let conjugate = solution.eigenvectors.as_tensor().conjugate();
        let projected_h0 = DenseHermitianMatrix::from_tensor(einsum(
            "ia,ij,jb->ab",
            &[
                &conjugate,
                eigenproblem.hamiltonian.as_tensor(),
                solution.eigenvectors.as_tensor(),
            ],
        )?)?;
        let matrix = &exchange.band_matrices[k];
        for (band, &value) in rows[k].iter().enumerate() {
            let weight = bands.states()[state_range.start + band].k_weight;
            h0_expectation += weight * value * projected_h0.at(band, band).re;
            exchange_trace += weight
                * value
                * matrix
                    .element(band, band)
                    .expect("validated exchange band index")
                    .re;
        }
    }
    let exchange_identity_residual = (exchange.exchange_energy.get() - 0.5 * exchange_trace).abs();
    let eigenvalue_identity_residual =
        (occupation.band_energy.get() - h0_expectation - exchange_trace).abs();
    let direct = h0_expectation - electrostatic.electron_hartree.get()
        + electrostatic.nuclear_nuclear.get()
        + exchange.exchange_energy.get()
        + occupation.correction.get();
    let eigenvalue = occupation.band_energy.get() - electrostatic.electron_hartree.get()
        + electrostatic.nuclear_nuclear.get()
        - exchange.exchange_energy.get()
        + occupation.correction.get();
    Ok(EnergyDiagnostic {
        total: Hartree(direct),
        exchange_identity_residual,
        eigenvalue_identity_residual,
        total_identity_residual: (direct - eigenvalue).abs(),
    })
}

fn occupation_rows(
    flat: &[f64],
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<f64>>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Spinor { occupations, .. }
                if occupations.end <= flat.len() =>
            {
                Ok(flat[occupations.clone()].to_vec())
            }
            CheckpointKPointSolution::Spinor { .. }
            | CheckpointKPointSolution::Collinear { .. } => {
                Err(GammaValenceHfError::SpinorFirstVariation)
            }
        })
        .collect()
}

fn spinor_energies(
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<Hartree>>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Spinor { solution, .. } => Ok(solution.eigenvalues.clone()),
            CheckpointKPointSolution::Collinear { .. } => {
                Err(GammaValenceHfError::SpinorFirstVariation)
            }
        })
        .collect()
}

fn exchange_difference(left: &IsdfExchangeResult, right: &IsdfExchangeResult) -> f64 {
    let mut maximum = (left.exchange_energy.get() - right.exchange_energy.get()).abs();
    maximum = maximum
        .max((left.maximum_antihermitian_residual - right.maximum_antihermitian_residual).abs());
    for (left, right) in left.band_matrices.iter().zip(&right.band_matrices) {
        for (&left, &right) in left.values().iter().zip(right.values()) {
            maximum = maximum.max((left - right).norm());
        }
    }
    maximum
}

fn density_mixer(spec: ScfMixing) -> Result<DensityMixer, MixingError> {
    match spec {
        ScfMixing::Linear { alpha } => DensityMixer::linear(alpha),
        ScfMixing::Broyden2 { alpha, history } => DensityMixer::broyden2(alpha, history),
        ScfMixing::PulayAnderson { alpha, history } => DensityMixer::pulay_anderson(alpha, history),
    }
}

fn validate_spec(spec: &GammaValenceHfSpec) -> Result<(), GammaValenceHfError> {
    let mesh = spec.config.k_mesh;
    if mesh.divisions != [1, 1, 1]
        || mesh.shift != [0.0; 3]
        || mesh.reduction != ScfKReduction::Full
    {
        return Err(GammaValenceHfError::GammaMesh);
    }
    if spec.config.relativity != ScfRelativity::SpinorFirstVariation {
        return Err(GammaValenceHfError::SpinorFirstVariation);
    }
    if spec
        .config
        .core_sites
        .iter()
        .any(|site| !site.states.is_empty())
    {
        return Err(GammaValenceHfError::CoreStates);
    }
    if spec.max_fock_iterations < 2 {
        return Err(GammaValenceHfError::FockIterations);
    }
    if !spec.fock_density_tolerance.is_finite() || spec.fock_density_tolerance <= 0.0 {
        return Err(GammaValenceHfError::FockTolerance);
    }
    if !spec.fock_mixing.is_finite() || spec.fock_mixing <= 0.0 || spec.fock_mixing > 1.0 {
        return Err(GammaValenceHfError::FockMixing);
    }
    Ok(())
}

fn require_gate(
    gate: &'static str,
    residual: f64,
    tolerance: f64,
) -> Result<(), GammaValenceHfError> {
    if residual.is_finite() && residual <= tolerance {
        Ok(())
    } else {
        Err(GammaValenceHfError::Gate {
            gate,
            residual,
            tolerance,
        })
    }
}
