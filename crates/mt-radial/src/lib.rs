//! Radial Schrödinger, scalar-relativistic, and spherical core-Dirac solvers.
//!
//! Energies are Hartree throughout.  Valence functions use the LAPW convention
//! `p(r) = r u(r)`.  For Koelling--Harmon and four-component Dirac solutions,
//! the public small component is the physical radial component `Q`; internally
//! the differential equations evolve `c Q`, as SPEX does.

#![forbid(unsafe_code)]

mod core_dirac;
mod integrals;
mod valence;

pub use core_dirac::{
    CoreDiracSolution, CoreDiracSpec, CoreState, DiracAngularContract, EnergyBracket, Kappa,
    KappaError, RelativisticRole, ValenceDiracSolution, ValenceDiracSpec, solve_core_dirac,
    solve_valence_dirac,
};
pub use integrals::{RadialComponents, RadialIntegralError, RadialIntegralKernel, radial_integral};
pub use valence::{
    BoundaryData, EnergyDerivative, LinearizedRadialSolution, LocalOrbital,
    LocalOrbitalCoefficients, RadialEquation, RadialError, RadialSolution, RadialSolver,
    SPEX_SPEED_OF_LIGHT,
};
