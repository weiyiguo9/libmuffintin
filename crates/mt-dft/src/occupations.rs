//! Finite-temperature occupations on a supplied regular-k spectrum.

use muffintin_core::Hartree;
use thiserror::Error;

/// One explicitly weighted one-particle eigenstate.
///
/// `k_weight * degeneracy` is the state's electron capacity. Use degeneracy two for a nonmagnetic scalar band whose spin partner is implicit, and one when spin or spinor states are enumerated explicitly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandState {
    pub energy: Hartree,
    pub k_weight: f64,
    pub degeneracy: u32,
}

impl BandState {
    pub const fn new(energy: Hartree, k_weight: f64, degeneracy: u32) -> Self {
        Self {
            energy,
            k_weight,
            degeneracy,
        }
    }

    fn capacity(self) -> f64 {
        self.k_weight * f64::from(self.degeneracy)
    }
}

/// Fermi--Dirac occupations and their finite-temperature energy terms.
#[derive(Clone, Debug, PartialEq)]
pub struct FermiDiracResult {
    pub chemical_potential: Hartree,
    /// Fractional occupations in the input state order, without k weights or degeneracy folded in.
    pub occupations: Vec<f64>,
    pub electron_count: f64,
    pub band_energy: Hartree,
    /// Dimensionless independent-particle entropy (Boltzmann constant set to one in the Hartree temperature convention).
    pub entropy: f64,
    pub minus_temperature_entropy: Hartree,
    pub iterations: usize,
}

/// Gaussian-broadened occupations and weighted band energy.
///
/// Gaussian broadening is a numerical Brillouin-zone integration scheme, not a thermal ensemble. This result therefore has no entropy or `-TS` field.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianResult {
    pub chemical_potential: Hartree,
    /// Fractional occupations in the input state order, without k weights or degeneracy folded in.
    pub occupations: Vec<f64>,
    pub electron_count: f64,
    pub band_energy: Hartree,
    /// Variational broadening correction `-sigma * sum(weight * degeneracy * normal_pdf)`.
    ///
    /// This is the generalized-entropy term associated with Gaussian occupation broadening, not a physical finite-temperature entropy.
    pub smearing_correction: Hartree,
    pub iterations: usize,
}

/// Invalid spectrum or failed finite-temperature chemical-potential solve.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum OccupationError {
    #[error("the supplied band spectrum is empty")]
    EmptySpectrum,
    #[error("state {index} has non-finite energy {energy} Ha")]
    NonFiniteEnergy { index: usize, energy: f64 },
    #[error("state {index} has invalid k weight {weight}; weights must be finite and positive")]
    InvalidWeight { index: usize, weight: f64 },
    #[error("state {index} has zero degeneracy")]
    ZeroDegeneracy { index: usize },
    #[error("the represented electron capacity is not finite")]
    NonFiniteCapacity,
    #[error("electron count must be finite, got {0}")]
    NonFiniteElectronCount(f64),
    #[error(
        "requested electron count {requested} is outside the finite-chemical-potential domain (0, {capacity})"
    )]
    ElectronCountOutsideFiniteDomain { requested: f64, capacity: f64 },
    #[error("electronic temperature must be finite and positive, got {0} Ha")]
    InvalidTemperature(f64),
    #[error("Gaussian standard deviation must be finite and positive, got {0} Ha")]
    InvalidGaussianWidth(f64),
    #[error("electron-count tolerance must be finite and positive, got {0}")]
    InvalidTolerance(f64),
    #[error("maximum iteration count must be positive")]
    ZeroMaxIterations,
    #[error("chemical potential must be finite, got {0} Ha")]
    NonFiniteChemicalPotential(f64),
    #[error("could not bracket the requested electron count with finite chemical potentials")]
    BracketingFailed,
    #[error(
        "chemical-potential solve did not converge after {iterations} iterations (electron residual {electron_residual})"
    )]
    NotConverged {
        iterations: usize,
        electron_residual: f64,
    },
    #[error("the weighted band-energy or entropy sum is not finite")]
    NonFiniteThermodynamics,
}

/// Evaluate one finite-temperature Fermi--Dirac occupation.
pub fn fermi_dirac(
    energy: Hartree,
    chemical_potential: Hartree,
    temperature: Hartree,
) -> Result<f64, OccupationError> {
    validate_energy_and_chemical_potential(energy, chemical_potential)?;
    validate_temperature(temperature)?;
    Ok(logistic((energy - chemical_potential) / temperature))
}

/// Evaluate a Gaussian-broadened occupation with standard deviation `width`.
pub fn gaussian_occupation(
    energy: Hartree,
    chemical_potential: Hartree,
    width: Hartree,
) -> Result<f64, OccupationError> {
    validate_energy_and_chemical_potential(energy, chemical_potential)?;
    validate_gaussian_width(width)?;
    Ok(gaussian_tail((energy - chemical_potential) / width))
}

/// Convert a temperature-like width to the Gaussian standard deviation whose slope at the chemical potential matches Fermi--Dirac broadening.
///
/// The factor is `4 / sqrt(2 pi)`, matching the SPEX `KSUM` input convention.
pub fn gaussian_width_matching_fermi_dirac_temperature(
    temperature: Hartree,
) -> Result<Hartree, OccupationError> {
    validate_temperature(temperature)?;
    let width = temperature * (4.0 / (2.0 * std::f64::consts::PI).sqrt());
    validate_gaussian_width(width)?;
    Ok(width)
}

/// Solve the weighted electron-count equation at finite temperature.
pub fn solve_fermi_dirac(
    states: &[BandState],
    requested_electrons: f64,
    temperature: Hartree,
    electron_tolerance: f64,
    max_iterations: usize,
) -> Result<FermiDiracResult, OccupationError> {
    validate_temperature(temperature)?;
    let solution = solve_smearing(
        states,
        requested_electrons,
        temperature.get(),
        electron_tolerance,
        max_iterations,
        logistic,
    )?;
    fermi_dirac_thermodynamics(states, solution, temperature)
}

/// Solve the weighted electron-count equation with Gaussian broadening.
pub fn solve_gaussian(
    states: &[BandState],
    requested_electrons: f64,
    width: Hartree,
    electron_tolerance: f64,
    max_iterations: usize,
) -> Result<GaussianResult, OccupationError> {
    validate_gaussian_width(width)?;
    let solution = solve_smearing(
        states,
        requested_electrons,
        width.get(),
        electron_tolerance,
        max_iterations,
        gaussian_tail,
    )?;
    let mut generalized_entropy = 0.0;
    for &state in states {
        let scaled = (state.energy.get() - solution.chemical_potential) / width.get();
        generalized_entropy += state.capacity() * normal_pdf(scaled);
    }
    let smearing_correction = -width.get() * generalized_entropy;
    if !smearing_correction.is_finite() {
        return Err(OccupationError::NonFiniteThermodynamics);
    }
    Ok(GaussianResult {
        chemical_potential: Hartree(solution.chemical_potential),
        occupations: solution.occupations,
        electron_count: solution.electron_count,
        band_energy: Hartree(solution.band_energy),
        smearing_correction: Hartree(smearing_correction),
        iterations: solution.iterations,
    })
}

fn validate_temperature(temperature: Hartree) -> Result<(), OccupationError> {
    if !temperature.get().is_finite() || temperature.get() <= 0.0 {
        return Err(OccupationError::InvalidTemperature(temperature.get()));
    }
    Ok(())
}

fn validate_gaussian_width(width: Hartree) -> Result<(), OccupationError> {
    if !width.get().is_finite() || width.get() <= 0.0 {
        return Err(OccupationError::InvalidGaussianWidth(width.get()));
    }
    Ok(())
}

fn validate_energy_and_chemical_potential(
    energy: Hartree,
    chemical_potential: Hartree,
) -> Result<(), OccupationError> {
    if !energy.get().is_finite() {
        return Err(OccupationError::NonFiniteEnergy {
            index: 0,
            energy: energy.get(),
        });
    }
    if !chemical_potential.get().is_finite() {
        return Err(OccupationError::NonFiniteChemicalPotential(
            chemical_potential.get(),
        ));
    }
    Ok(())
}

fn logistic(scaled_energy: f64) -> f64 {
    if scaled_energy >= 0.0 {
        let tail = (-scaled_energy).exp();
        tail / (1.0 + tail)
    } else {
        1.0 / (1.0 + scaled_energy.exp())
    }
}

fn gaussian_tail(scaled_energy: f64) -> f64 {
    let tail = 0.5 * libm::erfc(scaled_energy.abs() / std::f64::consts::SQRT_2);
    if scaled_energy >= 0.0 {
        tail
    } else {
        1.0 - tail
    }
}

fn normal_pdf(scaled_energy: f64) -> f64 {
    (-0.5 * scaled_energy * scaled_energy).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[derive(Debug)]
struct SpectrumSummary {
    capacity: f64,
    minimum_energy: f64,
    maximum_energy: f64,
}

#[derive(Debug)]
struct SmearingSolution {
    chemical_potential: f64,
    occupations: Vec<f64>,
    electron_count: f64,
    band_energy: f64,
    iterations: usize,
}

fn solve_smearing(
    states: &[BandState],
    requested_electrons: f64,
    scale: f64,
    electron_tolerance: f64,
    max_iterations: usize,
    kernel: fn(f64) -> f64,
) -> Result<SmearingSolution, OccupationError> {
    if !electron_tolerance.is_finite() || electron_tolerance <= 0.0 {
        return Err(OccupationError::InvalidTolerance(electron_tolerance));
    }
    if max_iterations == 0 {
        return Err(OccupationError::ZeroMaxIterations);
    }
    if !requested_electrons.is_finite() {
        return Err(OccupationError::NonFiniteElectronCount(requested_electrons));
    }
    let summary = validate_spectrum(states)?;
    if requested_electrons <= 0.0 || requested_electrons >= summary.capacity {
        return Err(OccupationError::ElectronCountOutsideFiniteDomain {
            requested: requested_electrons,
            capacity: summary.capacity,
        });
    }
    let (mut lower, mut upper) = bracket(
        states,
        requested_electrons,
        scale,
        summary.minimum_energy,
        summary.maximum_energy,
        kernel,
    )?;
    let mut electron_residual = f64::INFINITY;
    let mut completed_iterations = 0;
    for iteration in 1..=max_iterations {
        completed_iterations = iteration;
        let chemical_potential = lower * 0.5 + upper * 0.5;
        let stalled = chemical_potential == lower || chemical_potential == upper;
        let count = electron_count(states, chemical_potential, scale, kernel);
        electron_residual = count - requested_electrons;
        if electron_residual.abs() <= electron_tolerance {
            let occupations = occupations(states, chemical_potential, scale, kernel);
            return summarize_solution(states, occupations, chemical_potential, iteration);
        }
        if stalled {
            break;
        }
        if electron_residual < 0.0 {
            lower = chemical_potential;
        } else {
            upper = chemical_potential;
        }
    }
    Err(OccupationError::NotConverged {
        iterations: completed_iterations,
        electron_residual,
    })
}

fn validate_spectrum(states: &[BandState]) -> Result<SpectrumSummary, OccupationError> {
    if states.is_empty() {
        return Err(OccupationError::EmptySpectrum);
    }
    let mut capacity = 0.0;
    let mut minimum_energy = f64::INFINITY;
    let mut maximum_energy = f64::NEG_INFINITY;
    for (index, &state) in states.iter().enumerate() {
        let energy = state.energy.get();
        if !energy.is_finite() {
            return Err(OccupationError::NonFiniteEnergy { index, energy });
        }
        if !state.k_weight.is_finite() || state.k_weight <= 0.0 {
            return Err(OccupationError::InvalidWeight {
                index,
                weight: state.k_weight,
            });
        }
        if state.degeneracy == 0 {
            return Err(OccupationError::ZeroDegeneracy { index });
        }
        capacity += state.capacity();
        minimum_energy = minimum_energy.min(energy);
        maximum_energy = maximum_energy.max(energy);
    }
    if !capacity.is_finite() {
        return Err(OccupationError::NonFiniteCapacity);
    }
    Ok(SpectrumSummary {
        capacity,
        minimum_energy,
        maximum_energy,
    })
}

fn occupations(
    states: &[BandState],
    chemical_potential: f64,
    scale: f64,
    kernel: fn(f64) -> f64,
) -> Vec<f64> {
    states
        .iter()
        .map(|state| kernel((state.energy.get() - chemical_potential) / scale))
        .collect()
}

fn electron_count(
    states: &[BandState],
    chemical_potential: f64,
    scale: f64,
    kernel: fn(f64) -> f64,
) -> f64 {
    states
        .iter()
        .map(|state| state.capacity() * kernel((state.energy.get() - chemical_potential) / scale))
        .sum()
}

fn bracket(
    states: &[BandState],
    requested_electrons: f64,
    scale: f64,
    minimum_energy: f64,
    maximum_energy: f64,
    kernel: fn(f64) -> f64,
) -> Result<(f64, f64), OccupationError> {
    let span = maximum_energy - minimum_energy;
    let mut step = span.max(scale).max(1.0);
    if !step.is_finite() {
        return Err(OccupationError::BracketingFailed);
    }
    let mut lower = minimum_energy - step;
    let mut upper = maximum_energy + step;
    for _ in 0..1024 {
        if lower.is_finite()
            && electron_count(states, lower, scale, kernel) <= requested_electrons
            && upper.is_finite()
            && electron_count(states, upper, scale, kernel) >= requested_electrons
        {
            return Ok((lower, upper));
        }
        step *= 2.0;
        lower = minimum_energy - step;
        upper = maximum_energy + step;
        if !step.is_finite() {
            break;
        }
    }
    Err(OccupationError::BracketingFailed)
}

fn summarize_solution(
    states: &[BandState],
    occupations: Vec<f64>,
    chemical_potential: f64,
    iterations: usize,
) -> Result<SmearingSolution, OccupationError> {
    let mut electron_count = 0.0;
    let mut band_energy = 0.0;
    for (&state, &occupation) in states.iter().zip(&occupations) {
        let weighted_occupation = state.capacity() * occupation;
        electron_count += weighted_occupation;
        band_energy += weighted_occupation * state.energy.get();
    }
    if !electron_count.is_finite() || !band_energy.is_finite() {
        return Err(OccupationError::NonFiniteThermodynamics);
    }
    Ok(SmearingSolution {
        chemical_potential,
        occupations,
        electron_count,
        band_energy,
        iterations,
    })
}

fn fermi_dirac_thermodynamics(
    states: &[BandState],
    solution: SmearingSolution,
    temperature: Hartree,
) -> Result<FermiDiracResult, OccupationError> {
    let mut entropy = 0.0;
    for (&state, &occupation) in states.iter().zip(&solution.occupations) {
        if occupation > 0.0 && occupation < 1.0 {
            entropy -= state.capacity()
                * (occupation * occupation.ln() + (1.0 - occupation) * (-occupation).ln_1p());
        }
    }
    let minus_temperature_entropy = -temperature.get() * entropy;
    if !entropy.is_finite() || !minus_temperature_entropy.is_finite() {
        return Err(OccupationError::NonFiniteThermodynamics);
    }
    Ok(FermiDiracResult {
        chemical_potential: Hartree(solution.chemical_potential),
        occupations: solution.occupations,
        electron_count: solution.electron_count,
        band_energy: Hartree(solution.band_energy),
        entropy,
        minus_temperature_entropy: Hartree(minus_temperature_entropy),
        iterations: solution.iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(energy: f64, weight: f64, degeneracy: u32) -> BandState {
        BandState::new(Hartree(energy), weight, degeneracy)
    }

    #[test]
    fn logistic_is_stable_in_both_tails_and_half_filled_at_mu() {
        let temperature = Hartree(0.01);
        assert_eq!(
            fermi_dirac(Hartree(-100.0), Hartree(0.0), temperature).unwrap(),
            1.0
        );
        assert_eq!(
            fermi_dirac(Hartree(100.0), Hartree(0.0), temperature).unwrap(),
            0.0
        );
        assert_eq!(
            fermi_dirac(Hartree(0.0), Hartree(0.0), temperature).unwrap(),
            0.5
        );
    }

    #[test]
    fn gaussian_uses_the_spex_standard_deviation_convention() {
        let width = Hartree(0.2);
        let at_mu = gaussian_occupation(Hartree(0.0), Hartree(0.0), width).unwrap();
        let above = gaussian_occupation(Hartree(0.2), Hartree(0.0), width).unwrap();
        let below = gaussian_occupation(Hartree(-0.2), Hartree(0.0), width).unwrap();
        assert_eq!(at_mu, 0.5);
        assert!((above - 0.158_655_253_931_457_07).abs() < 2.0e-16);
        assert!((below - 0.841_344_746_068_542_9).abs() < 2.0e-16);
        assert!((above + below - 1.0).abs() < 2.0e-16);
        assert_eq!(
            gaussian_occupation(Hartree(20.0), Hartree(0.0), width).unwrap(),
            0.0
        );
        assert_eq!(
            gaussian_occupation(Hartree(-20.0), Hartree(0.0), width).unwrap(),
            1.0
        );
    }

    #[test]
    fn slope_matched_width_reproduces_spex_reference_values() {
        let temperature = Hartree(0.1);
        let width = gaussian_width_matching_fermi_dirac_temperature(temperature).unwrap();
        let expected_width = temperature.get() * 4.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((width.get() - expected_width).abs() < 1.0e-16);
        for (multiple, reference) in [(1.0, 0.2654), (2.0, 0.1050), (3.0, 0.0301)] {
            let value =
                gaussian_occupation(Hartree(multiple * temperature.get()), Hartree(0.0), width)
                    .unwrap();
            assert!((value - reference).abs() < 5.0e-5);
        }
    }

    #[test]
    fn symmetric_two_level_problem_has_zero_chemical_potential() {
        let states = [state(-1.0, 1.0, 1), state(1.0, 1.0, 1)];
        let result = solve_fermi_dirac(&states, 1.0, Hartree(0.2), 1.0e-14, 200).unwrap();
        assert!(result.chemical_potential.get().abs() < 1.0e-14);
        assert!((result.electron_count - 1.0).abs() < 1.0e-14);
        assert!((result.occupations[0] + result.occupations[1] - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn midpoint_rounding_still_accepts_an_exact_solution() {
        let states = [state(1.0e16, 1.0, 2)];
        let fermi = solve_fermi_dirac(&states, 1.0, Hartree(0.1), 1.0e-14, 64).unwrap();
        let gaussian = solve_gaussian(&states, 1.0, Hartree(0.1), 1.0e-14, 64).unwrap();
        for (chemical_potential, occupation, electron_count) in [
            (
                fermi.chemical_potential,
                fermi.occupations[0],
                fermi.electron_count,
            ),
            (
                gaussian.chemical_potential,
                gaussian.occupations[0],
                gaussian.electron_count,
            ),
        ] {
            assert_eq!(chemical_potential, Hartree(1.0e16));
            assert_eq!(occupation, 0.5);
            assert_eq!(electron_count, 1.0);
        }
    }

    #[test]
    fn unequal_weights_and_explicit_degeneracy_conserve_electrons() {
        let states = [
            state(-0.7, 0.25, 2),
            state(-0.1, 0.75, 2),
            state(0.4, 0.25, 2),
            state(1.2, 0.75, 2),
        ];
        let result = solve_fermi_dirac(&states, 2.3, Hartree(0.05), 1.0e-13, 256).unwrap();
        assert!((result.electron_count - 2.3).abs() <= 1.0e-13);
        assert!(
            result
                .occupations
                .iter()
                .all(|&value| (0.0..=1.0).contains(&value))
        );
    }

    #[test]
    fn gaussian_solver_conserves_weighted_electrons_and_reports_its_correction() {
        let states = [state(0.0, 1.0, 2)];
        let width = Hartree(0.125);
        let result = solve_gaussian(&states, 1.0, width, 1.0e-14, 64).unwrap();
        let expected_correction = -2.0 * width.get() / (2.0 * std::f64::consts::PI).sqrt();
        assert!(result.chemical_potential.get().abs() < 1.0e-14);
        assert_eq!(result.occupations, vec![0.5]);
        assert!((result.electron_count - 1.0).abs() < 1.0e-14);
        assert_eq!(result.band_energy, Hartree(0.0));
        assert!((result.smearing_correction.get() - expected_correction).abs() < 1.0e-14);
    }

    #[test]
    fn gaussian_correction_generates_the_reported_occupation() {
        let width = 0.17;
        let chemical_potential = -0.08;
        let energy = 0.21;
        let generalized_grand_potential = |trial_energy: f64| {
            let scaled = (trial_energy - chemical_potential) / width;
            (trial_energy - chemical_potential) * gaussian_tail(scaled) - width * normal_pdf(scaled)
        };
        let step = 1.0e-6;
        let derivative = (generalized_grand_potential(energy + step)
            - generalized_grand_potential(energy - step))
            / (2.0 * step);
        let occupation = gaussian_tail((energy - chemical_potential) / width);
        assert!((derivative - occupation).abs() < 2.0e-11);
    }

    #[test]
    fn gaussian_solution_is_covariant_under_a_uniform_energy_shift() {
        let original = [
            state(-0.7, 0.25, 2),
            state(0.1, 0.75, 2),
            state(0.8, 0.5, 1),
        ];
        let shifted = [state(0.8, 0.25, 2), state(1.6, 0.75, 2), state(2.3, 0.5, 1)];
        let left = solve_gaussian(&original, 1.4, Hartree(0.08), 1.0e-13, 256).unwrap();
        let right = solve_gaussian(&shifted, 1.4, Hartree(0.08), 1.0e-13, 256).unwrap();
        assert!((left.electron_count - 1.4).abs() <= 1.0e-13);
        assert!(
            (right.chemical_potential.get() - left.chemical_potential.get() - 1.5).abs() < 1.0e-12
        );
        for (&a, &b) in left.occupations.iter().zip(&right.occupations) {
            assert!((a - b).abs() < 1.0e-12);
        }
        assert!((right.band_energy.get() - left.band_energy.get() - 2.1).abs() < 1.0e-12);
        assert!((right.smearing_correction.get() - left.smearing_correction.get()).abs() < 1.0e-12);
    }

    #[test]
    fn entropy_and_free_energy_term_have_the_variational_sign() {
        let states = [state(0.0, 1.0, 2)];
        let temperature = Hartree(0.125);
        let result = solve_fermi_dirac(&states, 1.0, temperature, 1.0e-14, 64).unwrap();
        assert!((result.entropy - 2.0 * std::f64::consts::LN_2).abs() < 1.0e-14);
        assert!(result.minus_temperature_entropy.get() < 0.0);
        assert!(
            (result.minus_temperature_entropy.get() + temperature.get() * result.entropy).abs()
                < 1.0e-14
        );
    }

    #[test]
    fn uniform_energy_shift_moves_only_mu_and_band_energy() {
        let original = [state(-0.3, 0.5, 2), state(0.8, 0.5, 2)];
        let shifted = [state(1.7, 0.5, 2), state(2.8, 0.5, 2)];
        let left = solve_fermi_dirac(&original, 1.3, Hartree(0.07), 1.0e-13, 256).unwrap();
        let right = solve_fermi_dirac(&shifted, 1.3, Hartree(0.07), 1.0e-13, 256).unwrap();
        assert!(
            (right.chemical_potential.get() - left.chemical_potential.get() - 2.0).abs() < 1.0e-12
        );
        for (&a, &b) in left.occupations.iter().zip(&right.occupations) {
            assert!((a - b).abs() < 1.0e-12);
        }
        assert!((right.band_energy.get() - left.band_energy.get() - 2.6).abs() < 1.0e-12);
        assert!((right.entropy - left.entropy).abs() < 1.0e-12);
    }

    #[test]
    fn exact_empty_and_full_counts_require_infinite_chemical_potential() {
        let states = [state(-0.2, 0.5, 2), state(0.4, 0.5, 2)];
        assert!(matches!(
            solve_fermi_dirac(&states, 0.0, Hartree(0.01), 1.0e-12, 64),
            Err(OccupationError::ElectronCountOutsideFiniteDomain { .. })
        ));
        assert!(matches!(
            solve_fermi_dirac(&states, 2.0, Hartree(0.01), 1.0e-12, 64),
            Err(OccupationError::ElectronCountOutsideFiniteDomain { .. })
        ));
        assert!(matches!(
            solve_gaussian(&states, 0.0, Hartree(0.01), 1.0e-12, 64),
            Err(OccupationError::ElectronCountOutsideFiniteDomain { .. })
        ));
    }

    #[test]
    fn invalid_inputs_are_rejected_at_the_public_boundary() {
        let valid = [state(0.0, 1.0, 1)];
        assert_eq!(
            solve_fermi_dirac(&[], 0.0, Hartree(0.1), 1.0e-12, 10),
            Err(OccupationError::EmptySpectrum)
        );
        assert!(matches!(
            solve_fermi_dirac(&[state(f64::NAN, 1.0, 1)], 0.5, Hartree(0.1), 1.0e-12, 10),
            Err(OccupationError::NonFiniteEnergy { .. })
        ));
        assert!(matches!(
            solve_fermi_dirac(&[state(0.0, 0.0, 1)], 0.5, Hartree(0.1), 1.0e-12, 10),
            Err(OccupationError::InvalidWeight { .. })
        ));
        assert!(matches!(
            solve_fermi_dirac(&[state(0.0, 1.0, 0)], 0.5, Hartree(0.1), 1.0e-12, 10),
            Err(OccupationError::ZeroDegeneracy { .. })
        ));
        assert!(matches!(
            solve_fermi_dirac(&valid, 0.5, Hartree(0.0), 1.0e-12, 10),
            Err(OccupationError::InvalidTemperature(_))
        ));
        assert!(matches!(
            solve_gaussian(&valid, 0.5, Hartree(0.0), 1.0e-12, 10),
            Err(OccupationError::InvalidGaussianWidth(_))
        ));
        assert!(matches!(
            gaussian_width_matching_fermi_dirac_temperature(Hartree(f64::MAX)),
            Err(OccupationError::InvalidGaussianWidth(value)) if value.is_infinite()
        ));
        assert!(matches!(
            solve_fermi_dirac(&valid, 0.5, Hartree(0.1), 0.0, 10),
            Err(OccupationError::InvalidTolerance(_))
        ));
        assert_eq!(
            solve_fermi_dirac(&valid, 0.5, Hartree(0.1), 1.0e-12, 0),
            Err(OccupationError::ZeroMaxIterations)
        );
        assert!(matches!(
            solve_fermi_dirac(&valid, 1.1, Hartree(0.1), 1.0e-12, 10),
            Err(OccupationError::ElectronCountOutsideFiniteDomain { .. })
        ));
    }
}
