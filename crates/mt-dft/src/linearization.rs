//! Current-potential linearization-energy generators and their provenance.

use muffintin_core::{Bohr, ExponentialMesh, Hartree, InverseBohr, Kappa, KappaError};
use muffintin_sphere::{
    CoreBracketSearch, CoreDiracSolution, CoreState, DiracError, EnergyBracket, RadialEquation,
    RadialError, RadialSolver, isolate_core_dirac_bracket,
};
use thiserror::Error;

use crate::BandState;

/// Stable method names recorded in recipes and SCF provenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearizationEnergyGenerator {
    Explicit,
    Atomic,
    BandCenter,
    LogDerivative,
    BandCog,
    FermiOffset,
    FrozenCheckpoint,
}

/// Method-specific evidence retained with one generated energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinearizationEnergyDiagnostic {
    Stored,
    Atomic {
        state: CoreState,
        bracket: EnergyBracket,
    },
    BandCenter {
        angular_momentum: u32,
        bottom: Hartree,
        top: Hartree,
    },
    LogDerivative {
        principal_quantum_number: u32,
        angular_momentum: u32,
        target: InverseBohr,
        nodes: u32,
    },
    BandCog {
        samples: usize,
        total_weight: f64,
    },
    FermiOffset {
        fermi_energy: Hartree,
        offset: Hartree,
    },
}

/// One resolved value with enough information for deterministic provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneratedLinearizationEnergy {
    pub generator: LinearizationEnergyGenerator,
    pub seed: Option<Hartree>,
    pub energy: Hartree,
    pub diagnostic: LinearizationEnergyDiagnostic,
}

/// Bound-state identity and current-potential scale for the `atomic` generator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtomicEnergyRequest {
    pub state: CoreState,
    pub nuclear_charge: f64,
    pub muffin_tin_radius: Bohr,
    pub intervals: usize,
    /// Energy of this state under a nearby potential, if one is known.
    ///
    /// A caller that re-solves the same state while its potential moves, such
    /// as a free-atom SCF loop, can supply the previous energy. The search
    /// then scans a narrow window around it first and falls back to the full
    /// window when that window holds no node-compatible bracket, so the seed
    /// cannot change which state is returned.
    pub seed: Option<Hartree>,
}

impl AtomicEnergyRequest {
    pub const fn new(state: CoreState, nuclear_charge: f64, muffin_tin_radius: Bohr) -> Self {
        Self {
            state,
            nuclear_charge,
            muffin_tin_radius,
            intervals: 512,
            seed: None,
        }
    }

    pub const fn with_intervals(mut self, intervals: usize) -> Self {
        self.intervals = intervals;
        self
    }

    pub const fn with_seed(mut self, seed: Hartree) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Half-width of the seeded window as a fraction of the seed magnitude.
const SEEDED_WINDOW_FRACTION: f64 = 0.1;
/// Absolute half-width floor in Hartree, so a shallow state still moves freely.
const SEEDED_WINDOW_FLOOR_HARTREE: f64 = 1.0;
/// Scan intervals inside the seeded window.
const SEEDED_WINDOW_INTERVALS: usize = 256;

/// One nonnegative contribution to an occupied projected density of states.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PdosEnergySample {
    pub state: BandState,
    /// Fractional occupation of `state`, before k weight and degeneracy.
    pub occupation: f64,
    /// Nonnegative projection onto this radial channel.
    pub projection: f64,
}

impl PdosEnergySample {
    pub const fn new(state: BandState, occupation: f64, projection: f64) -> Self {
        Self {
            state,
            occupation,
            projection,
        }
    }
}

/// Preserve a user-supplied energy exactly.
pub fn generate_explicit_energy(
    energy: Hartree,
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    stored_energy(LinearizationEnergyGenerator::Explicit, energy)
}

/// Preserve a checkpoint energy exactly while retaining its distinct provenance.
pub fn generate_frozen_checkpoint_energy(
    energy: Hartree,
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    stored_energy(LinearizationEnergyGenerator::FrozenCheckpoint, energy)
}

/// Solve a signed-kappa bound state on an extended current spherical potential.
pub fn generate_atomic_energy(
    mesh: &ExponentialMesh,
    potential: &[f64],
    request: AtomicEnergyRequest,
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    let solved = solve_atomic_bound_state(mesh, potential, request)?;
    Ok(GeneratedLinearizationEnergy {
        generator: LinearizationEnergyGenerator::Atomic,
        seed: None,
        energy: solved.solution.energy,
        diagnostic: LinearizationEnergyDiagnostic::Atomic {
            state: request.state,
            bracket: solved.bracket,
        },
    })
}

/// Narrow scan window around `seed`, clamped inside the full `window`.
///
/// Returns `None` when the clamped window is empty or degenerate, in which case
/// the caller uses the full window.
fn seeded_window(seed: Hartree, window: EnergyBracket) -> Option<EnergyBracket> {
    let (full_lower, full_upper) = window.values();
    let half_width = (SEEDED_WINDOW_FRACTION * seed.get().abs()).max(SEEDED_WINDOW_FLOOR_HARTREE);
    if !half_width.is_finite() || half_width <= 0.0 {
        return None;
    }
    let lower = (seed.get() - half_width).max(full_lower);
    let upper = (seed.get() + half_width).min(full_upper);
    EnergyBracket::from_values(lower, upper).ok()
}

pub(crate) struct SolvedAtomicBoundState {
    pub(crate) solution: CoreDiracSolution,
    pub(crate) bracket: EnergyBracket,
}

pub(crate) fn solve_atomic_bound_state(
    mesh: &ExponentialMesh,
    potential: &[f64],
    request: AtomicEnergyRequest,
) -> Result<SolvedAtomicBoundState, LinearizationEnergyError> {
    if !request.nuclear_charge.is_finite() || request.nuclear_charge <= 0.0 {
        return Err(LinearizationEnergyError::InvalidNuclearCharge(
            request.nuclear_charge,
        ));
    }
    let Some(&continuum) = potential.last() else {
        return Err(LinearizationEnergyError::EmptyAtomicPotential);
    };
    let charge_squared = request.nuclear_charge * request.nuclear_charge;
    let atomic_scale = (charge_squared / f64::from(request.state.n).powi(2)).max(1.0);
    let energy_window = EnergyBracket::from_values(
        continuum - 2.0 * atomic_scale,
        continuum - 1.0e-8 * atomic_scale,
    )
    .map_err(|source| LinearizationEnergyError::Atomic {
        state: request.state,
        source,
    })?;
    let seeded = request
        .seed
        .and_then(|seed| seeded_window(seed, energy_window))
        .and_then(|window| {
            isolate_core_dirac_bracket(
                mesh,
                potential,
                CoreBracketSearch::new(
                    request.state,
                    request.nuclear_charge,
                    request.muffin_tin_radius,
                    window,
                )
                .with_intervals(SEEDED_WINDOW_INTERVALS),
            )
            .ok()
        });
    let isolated = match seeded {
        Some(isolated) => isolated,
        None => isolate_core_dirac_bracket(
            mesh,
            potential,
            CoreBracketSearch::new(
                request.state,
                request.nuclear_charge,
                request.muffin_tin_radius,
                energy_window,
            )
            .with_intervals(request.intervals),
        )
        .map_err(|source| LinearizationEnergyError::Atomic {
            state: request.state,
            source,
        })?,
    };
    Ok(SolvedAtomicBoundState {
        solution: isolated.solution,
        bracket: isolated.bracket,
    })
}

/// Generate an Elk-style scalar band center from the current spherical potential.
pub fn generate_band_center_energy(
    mesh: &ExponentialMesh,
    potential: &[f64],
    equation: RadialEquation,
    angular_momentum: u32,
    seed: Hartree,
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    let result = RadialSolver::new(mesh, potential, equation)
        .and_then(|solver| solver.band_center(angular_momentum, seed))
        .map_err(|source| LinearizationEnergyError::BandCenter {
            angular_momentum,
            seed,
            source,
        })?;
    Ok(GeneratedLinearizationEnergy {
        generator: LinearizationEnergyGenerator::BandCenter,
        seed: Some(seed),
        energy: result.energy,
        diagnostic: LinearizationEnergyDiagnostic::BandCenter {
            angular_momentum,
            bottom: result.bottom,
            top: result.top,
        },
    })
}

/// Generate the selected radial branch at a prescribed `u'(R) / u(R)` in inverse bohr.
pub fn generate_log_derivative_energy(
    mesh: &ExponentialMesh,
    potential: &[f64],
    equation: RadialEquation,
    principal_quantum_number: u32,
    angular_momentum: u32,
    seed: Hartree,
    target: InverseBohr,
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    let result = RadialSolver::new(mesh, potential, equation)
        .and_then(|solver| {
            solver.energy_at_log_derivative(
                principal_quantum_number,
                angular_momentum,
                seed,
                target,
            )
        })
        .map_err(|source| LinearizationEnergyError::LogDerivative {
            principal_quantum_number,
            angular_momentum,
            target,
            seed,
            source,
        })?;
    Ok(GeneratedLinearizationEnergy {
        generator: LinearizationEnergyGenerator::LogDerivative,
        seed: Some(seed),
        energy: result.energy,
        diagnostic: LinearizationEnergyDiagnostic::LogDerivative {
            principal_quantum_number,
            angular_momentum,
            target,
            nodes: result.nodes,
        },
    })
}

/// Compute the occupied projected-DOS center of gravity.
pub fn generate_band_cog_energy(
    samples: &[PdosEnergySample],
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    if samples.is_empty() {
        return Err(LinearizationEnergyError::EmptyPdos);
    }
    let mut total_weight = 0.0;
    let mut first_moment = 0.0;
    for (index, sample) in samples.iter().enumerate() {
        if !sample.state.energy.get().is_finite() {
            return Err(LinearizationEnergyError::NonFinitePdosEnergy {
                index,
                energy: sample.state.energy.get(),
            });
        }
        if !sample.state.k_weight.is_finite() || sample.state.k_weight <= 0.0 {
            return Err(LinearizationEnergyError::InvalidPdosKWeight {
                index,
                weight: sample.state.k_weight,
            });
        }
        if sample.state.degeneracy == 0 {
            return Err(LinearizationEnergyError::ZeroPdosDegeneracy { index });
        }
        if !sample.occupation.is_finite() || !(0.0..=1.0).contains(&sample.occupation) {
            return Err(LinearizationEnergyError::InvalidPdosOccupation {
                index,
                occupation: sample.occupation,
            });
        }
        if !sample.projection.is_finite() || sample.projection < 0.0 {
            return Err(LinearizationEnergyError::InvalidPdosProjection {
                index,
                projection: sample.projection,
            });
        }
        let weight = sample.state.k_weight
            * f64::from(sample.state.degeneracy)
            * sample.occupation
            * sample.projection;
        total_weight += weight;
        first_moment += weight * sample.state.energy.get();
    }
    if !total_weight.is_finite() || total_weight <= 0.0 || !first_moment.is_finite() {
        return Err(LinearizationEnergyError::InvalidPdosNormalization {
            total_weight,
            first_moment,
        });
    }
    let energy = Hartree(first_moment / total_weight);
    if !energy.get().is_finite() {
        return Err(LinearizationEnergyError::NonFiniteGeneratedEnergy);
    }
    Ok(GeneratedLinearizationEnergy {
        generator: LinearizationEnergyGenerator::BandCog,
        seed: None,
        energy,
        diagnostic: LinearizationEnergyDiagnostic::BandCog {
            samples: samples.len(),
            total_weight,
        },
    })
}

/// Shift the current Fermi energy by an explicit Hartree offset.
pub fn generate_fermi_offset_energy(
    fermi_energy: Hartree,
    offset: Hartree,
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    if !fermi_energy.get().is_finite() {
        return Err(LinearizationEnergyError::NonFiniteFermiEnergy(
            fermi_energy.get(),
        ));
    }
    if !offset.get().is_finite() {
        return Err(LinearizationEnergyError::NonFiniteFermiOffset(offset.get()));
    }
    let energy = Hartree(fermi_energy.get() + offset.get());
    if !energy.get().is_finite() {
        return Err(LinearizationEnergyError::NonFiniteGeneratedEnergy);
    }
    Ok(GeneratedLinearizationEnergy {
        generator: LinearizationEnergyGenerator::FermiOffset,
        seed: None,
        energy,
        diagnostic: LinearizationEnergyDiagnostic::FermiOffset {
            fermi_energy,
            offset,
        },
    })
}

/// Average signed-kappa energies with their exact magnetic degeneracies.
pub fn kappa_degeneracy_average(
    angular_momentum: u32,
    energies: &[(Kappa, Hartree)],
) -> Result<Hartree, LinearizationEnergyError> {
    let l = i32::try_from(angular_momentum)
        .map_err(|_| LinearizationEnergyError::AngularMomentumOverflow)?;
    let aligned = Kappa::new(
        l.checked_add(1)
            .and_then(i32::checked_neg)
            .ok_or(LinearizationEnergyError::AngularMomentumOverflow)?,
    )?;
    let anti_aligned = (l != 0).then(|| Kappa::new(l)).transpose()?;
    let required = usize::from(anti_aligned.is_some()) + 1;
    if energies.len() != required {
        return Err(LinearizationEnergyError::KappaPartnerSet { angular_momentum });
    }
    let mut weighted = 0.0;
    let mut degeneracy = 0_u32;
    for expected in std::iter::once(aligned).chain(anti_aligned) {
        let Some((_, energy)) = energies.iter().find(|(kappa, _)| *kappa == expected) else {
            return Err(LinearizationEnergyError::KappaPartnerSet { angular_momentum });
        };
        if !energy.get().is_finite() {
            return Err(LinearizationEnergyError::NonFiniteKappaEnergy {
                kappa: expected.get(),
                energy: energy.get(),
            });
        }
        weighted += f64::from(expected.degeneracy()) * energy.get();
        degeneracy += expected.degeneracy();
    }
    let energy = Hartree(weighted / f64::from(degeneracy));
    if !energy.get().is_finite() {
        return Err(LinearizationEnergyError::NonFiniteGeneratedEnergy);
    }
    Ok(energy)
}

fn stored_energy(
    generator: LinearizationEnergyGenerator,
    energy: Hartree,
) -> Result<GeneratedLinearizationEnergy, LinearizationEnergyError> {
    if !energy.get().is_finite() {
        return Err(LinearizationEnergyError::NonFiniteStoredEnergy {
            generator,
            energy: energy.get(),
        });
    }
    Ok(GeneratedLinearizationEnergy {
        generator,
        seed: Some(energy),
        energy,
        diagnostic: LinearizationEnergyDiagnostic::Stored,
    })
}

/// Diagnosable generator failures; no method silently falls back to its seed.
#[derive(Debug, Error, PartialEq)]
pub enum LinearizationEnergyError {
    #[error("{generator:?} energy is not finite: {energy} Ha")]
    NonFiniteStoredEnergy {
        generator: LinearizationEnergyGenerator,
        energy: f64,
    },
    #[error("atomic generator requires a finite positive nuclear charge, got {0}")]
    InvalidNuclearCharge(f64),
    #[error("atomic generator received an empty extended potential")]
    EmptyAtomicPotential,
    #[error("the projected DOS is empty")]
    EmptyPdos,
    #[error("projected-DOS sample {index} has non-finite energy {energy} Ha")]
    NonFinitePdosEnergy { index: usize, energy: f64 },
    #[error("projected-DOS sample {index} has invalid k weight {weight}")]
    InvalidPdosKWeight { index: usize, weight: f64 },
    #[error("projected-DOS sample {index} has zero degeneracy")]
    ZeroPdosDegeneracy { index: usize },
    #[error("projected-DOS sample {index} has invalid occupation {occupation}")]
    InvalidPdosOccupation { index: usize, occupation: f64 },
    #[error("projected-DOS sample {index} has invalid projection {projection}")]
    InvalidPdosProjection { index: usize, projection: f64 },
    #[error(
        "projected-DOS moments are invalid: total weight {total_weight}, first moment {first_moment}"
    )]
    InvalidPdosNormalization {
        total_weight: f64,
        first_moment: f64,
    },
    #[error("Fermi energy is not finite: {0} Ha")]
    NonFiniteFermiEnergy(f64),
    #[error("Fermi offset is not finite: {0} Ha")]
    NonFiniteFermiOffset(f64),
    #[error("generated energy overflowed")]
    NonFiniteGeneratedEnergy,
    #[error("angular momentum does not fit the signed-kappa representation")]
    AngularMomentumOverflow,
    #[error("energies do not contain exactly the signed-kappa partners for l={angular_momentum}")]
    KappaPartnerSet { angular_momentum: u32 },
    #[error("kappa={kappa} has non-finite energy {energy} Ha")]
    NonFiniteKappaEnergy { kappa: i32, energy: f64 },
    #[error("atomic generator failed for n={}, kappa={}", state.n, state.kappa.get())]
    Atomic {
        state: CoreState,
        #[source]
        source: DiracError,
    },
    #[error("band-center generator failed for l={angular_momentum} from seed {seed}")]
    BandCenter {
        angular_momentum: u32,
        seed: Hartree,
        #[source]
        source: RadialError,
    },
    #[error(
        "log-derivative generator failed for n={principal_quantum_number}, l={angular_momentum}, \
         target={target} from seed {seed}"
    )]
    LogDerivative {
        principal_quantum_number: u32,
        angular_momentum: u32,
        target: InverseBohr,
        seed: Hartree,
        #[source]
        source: RadialError,
    },
    #[error(transparent)]
    Kappa(#[from] KappaError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_core::Kappa;

    fn mesh(first: f64, last: f64, increment: f64) -> ExponentialMesh {
        let number = ((last / first).ln() / increment).ceil() as usize + 1;
        ExponentialMesh::new(Bohr(first), increment, number).unwrap()
    }

    #[test]
    fn stored_spectral_and_fermi_generators_preserve_distinct_semantics() {
        let explicit = generate_explicit_energy(Hartree(-0.4)).unwrap();
        let frozen = generate_frozen_checkpoint_energy(Hartree(-0.4)).unwrap();
        assert_eq!(explicit.energy, frozen.energy);
        assert_eq!(explicit.generator, LinearizationEnergyGenerator::Explicit);
        assert_eq!(
            frozen.generator,
            LinearizationEnergyGenerator::FrozenCheckpoint
        );

        let cog = generate_band_cog_energy(&[
            PdosEnergySample::new(BandState::new(Hartree(-0.5), 0.25, 2), 1.0, 0.5),
            PdosEnergySample::new(BandState::new(Hartree(0.5), 0.75, 1), 0.5, 2.0),
        ])
        .unwrap();
        assert!((cog.energy.get() - 0.25).abs() < 1.0e-15);

        let shifted = generate_fermi_offset_energy(Hartree(-0.05), Hartree(-0.1)).unwrap();
        assert!((shifted.energy.get() + 0.15).abs() < 1.0e-15);

        let averaged = kappa_degeneracy_average(
            2,
            &[
                (Kappa::new(-3).unwrap(), Hartree(-0.4)),
                (Kappa::new(2).unwrap(), Hartree(-0.1)),
            ],
        )
        .unwrap();
        assert!((averaged.get() + 0.28).abs() < 1.0e-15);
    }

    #[test]
    fn scalar_current_potential_generators_return_method_provenance() {
        let mesh = mesh(1.0e-7, 5.0, 0.002);
        let potential = vec![-0.35; mesh.len()];
        let center = generate_band_center_energy(
            &mesh,
            &potential,
            RadialEquation::Schroedinger,
            0,
            Hartree(-0.2),
        )
        .unwrap();
        assert_eq!(center.generator, LinearizationEnergyGenerator::BandCenter);
        assert_eq!(center.seed, Some(Hartree(-0.2)));
        assert!(matches!(
            center.diagnostic,
            LinearizationEnergyDiagnostic::BandCenter { .. }
        ));

        let logarithmic = generate_log_derivative_energy(
            &mesh,
            &potential,
            RadialEquation::Schroedinger,
            1,
            0,
            Hartree(-0.2),
            InverseBohr(-1.0),
        )
        .unwrap();
        let solution = RadialSolver::new(&mesh, &potential, RadialEquation::Schroedinger)
            .unwrap()
            .solve(0, logarithmic.energy)
            .unwrap();
        assert!((solution.boundary.log_derivative.unwrap().get() + 1.0).abs() < 2.0e-8);
        assert!(matches!(
            logarithmic.diagnostic,
            LinearizationEnergyDiagnostic::LogDerivative { nodes: 0, .. }
        ));
    }

    #[test]
    fn atomic_generator_selects_the_signed_kappa_bound_state() {
        let mesh = mesh(1.0e-7, 40.0, 0.002);
        let potential = mesh
            .radii()
            .iter()
            .map(|radius| -1.0 / radius.get())
            .collect::<Vec<_>>();
        let muffin_tin_radius = *mesh
            .radii()
            .iter()
            .min_by(|left, right| {
                (left.get() - 6.0)
                    .abs()
                    .total_cmp(&(right.get() - 6.0).abs())
            })
            .unwrap();
        let state = CoreState::new(1, Kappa::new(-1).unwrap()).unwrap();
        let generated = generate_atomic_energy(
            &mesh,
            &potential,
            AtomicEnergyRequest::new(state, 1.0, muffin_tin_radius).with_intervals(48),
        )
        .unwrap();
        assert_eq!(generated.generator, LinearizationEnergyGenerator::Atomic);
        assert!((generated.energy.get() + 0.5).abs() < 2.0e-5);
        assert!(matches!(
            generated.diagnostic,
            LinearizationEnergyDiagnostic::Atomic { state: actual, .. } if actual == state
        ));
    }
}
