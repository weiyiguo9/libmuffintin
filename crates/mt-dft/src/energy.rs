//! Variational Kohn--Sham energy bookkeeping.

use muffintin_core::Hartree;
use thiserror::Error;

/// Energy correction paired with the caller's occupation functional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OccupationEnergy {
    FermiDirac {
        minus_temperature_entropy: Hartree,
    },
    Gaussian {
        smearing_correction: Hartree,
    },
    /// Caller-owned occupation method with an explicitly evaluated correction.
    External {
        correction: Hartree,
    },
}

impl OccupationEnergy {
    pub const fn correction(self) -> Hartree {
        match self {
            Self::FermiDirac {
                minus_temperature_entropy,
            } => minus_temperature_entropy,
            Self::Gaussian {
                smearing_correction,
            } => smearing_correction,
            Self::External { correction } => correction,
        }
    }
}

/// Terms in the full-potential total-energy expression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfEnergy {
    pub band: Hartree,
    pub core_eigenvalues: Hartree,
    pub madelung: Hartree,
    pub coulomb: Hartree,
    pub exchange_correlation: Hartree,
    pub exchange_correlation_potential: Hartree,
    pub occupation: OccupationEnergy,
    pub total: Hartree,
}

/// Method-neutral scalar inputs whose counting is owned by the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TotalEnergyInput {
    pub band_energy: Hartree,
    pub core_eigenvalue_sum: Hartree,
    pub occupation_correction: Hartree,
}

/// Total energy plus the two neutral convergence observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TotalEnergyEvaluation {
    pub energy: ScfEnergy,
    pub density_rms: f64,
    pub energy_change: Option<Hartree>,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum EnergyError {
    #[error("SCF energy term {term} is not finite: {value} Ha")]
    NonFinite { term: &'static str, value: f64 },
}

/// Invalid method-neutral energy input or incompatible regional density.
#[derive(Debug, Error)]
pub enum TotalEnergyError {
    #[error("previous total energy is not finite: {0} Ha")]
    NonFinitePreviousTotal(f64),
    #[error(transparent)]
    Energy(#[from] EnergyError),
    #[error(transparent)]
    Regional(#[from] crate::RegionalError),
}

/// Combine band, electrostatic, XC, and occupation terms once each.
pub fn assemble_scf_energy(
    band: Hartree,
    core_eigenvalues: Hartree,
    madelung: Hartree,
    coulomb: Hartree,
    exchange_correlation: Hartree,
    exchange_correlation_potential: Hartree,
    occupation: OccupationEnergy,
) -> Result<ScfEnergy, EnergyError> {
    for (term, value) in [
        ("band", band.get()),
        ("core_eigenvalues", core_eigenvalues.get()),
        ("madelung", madelung.get()),
        ("coulomb", coulomb.get()),
        ("exchange_correlation", exchange_correlation.get()),
        (
            "exchange_correlation_potential",
            exchange_correlation_potential.get(),
        ),
        ("occupation", occupation.correction().get()),
    ] {
        if !value.is_finite() {
            return Err(EnergyError::NonFinite { term, value });
        }
    }
    let total = band.get()
        + core_eigenvalues.get()
        + 0.5 * (madelung.get() - coulomb.get())
        + exchange_correlation.get()
        - exchange_correlation_potential.get()
        + occupation.correction().get();
    if !total.is_finite() {
        return Err(EnergyError::NonFinite {
            term: "total",
            value: total,
        });
    }
    Ok(ScfEnergy {
        band,
        core_eigenvalues,
        madelung,
        coulomb,
        exchange_correlation,
        exchange_correlation_potential,
        occupation,
        total: Hartree(total),
    })
}

/// Evaluate the full-potential total energy without owning occupations or bands.
pub fn evaluate_total_energy(
    potential: &crate::ScfPotentialBuild,
    output_density: &crate::RegionalDensity,
    input: TotalEnergyInput,
    previous_total: Option<Hartree>,
) -> Result<TotalEnergyEvaluation, TotalEnergyError> {
    if let Some(previous) = previous_total {
        if !previous.get().is_finite() {
            return Err(TotalEnergyError::NonFinitePreviousTotal(previous.get()));
        }
    }
    let terms = potential.energy_terms;
    let energy = assemble_scf_energy(
        input.band_energy,
        input.core_eigenvalue_sum,
        terms.madelung,
        terms.coulomb,
        terms.exchange_correlation,
        terms.exchange_correlation_potential,
        OccupationEnergy::External {
            correction: input.occupation_correction,
        },
    )?;
    let density_rms = potential.source_density().difference_rms(output_density)?;
    let energy_change =
        previous_total.map(|previous| Hartree((energy.total.get() - previous.get()).abs()));
    Ok(TotalEnergyEvaluation {
        energy,
        density_rms,
        energy_change,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupation_functional_enters_exactly_once() {
        let common = |occupation| {
            assemble_scf_energy(
                Hartree(-3.0),
                Hartree(-5.0),
                Hartree(1.2),
                Hartree(0.4),
                Hartree(-0.8),
                Hartree(-1.1),
                occupation,
            )
            .unwrap()
        };
        let fd = common(OccupationEnergy::FermiDirac {
            minus_temperature_entropy: Hartree(-0.03),
        });
        let gaussian = common(OccupationEnergy::Gaussian {
            smearing_correction: Hartree(-0.07),
        });
        assert!((fd.total.get() - gaussian.total.get() - 0.04).abs() < 1.0e-15);
    }

    #[test]
    fn non_finite_terms_are_rejected() {
        assert!(matches!(
            assemble_scf_energy(
                Hartree(f64::NAN),
                Hartree(0.0),
                Hartree(0.0),
                Hartree(0.0),
                Hartree(0.0),
                Hartree(0.0),
                OccupationEnergy::Gaussian {
                    smearing_correction: Hartree(0.0)
                }
            ),
            Err(EnergyError::NonFinite { term: "band", value }) if value.is_nan()
        ));
    }
}
