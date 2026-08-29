//! Variational Kohn--Sham energy bookkeeping.

use muffintin_core::Hartree;
use thiserror::Error;

/// Smearing contribution paired with the selected occupation functional.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OccupationEnergy {
    FermiDirac { minus_temperature_entropy: Hartree },
    Gaussian { smearing_correction: Hartree },
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

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum EnergyError {
    #[error("SCF energy term {term} is not finite: {value} Ha")]
    NonFinite { term: &'static str, value: f64 },
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
