//! Radial Schrödinger, scalar-relativistic, and spherical core-Dirac solvers.
//!
//! Energies are Hartree throughout.  Valence functions use the LAPW convention
//! `p(r) = r u(r)`.  For Koelling--Harmon and four-component Dirac solutions,
//! the public small component is the physical radial component `Q`; internally
//! the differential equations evolve `c Q`, as SPEX does.

#![forbid(unsafe_code)]

/// Implementation diagnostic for RK4 state growth, not a public convention.
///
/// It fires before squaring can overflow (`sqrt(f64::MAX) ~= 1.34e154`) and
/// leaves headroom for the mesh-weighted norm.
const MAX_RADIAL_AMPLITUDE: f64 = 1.0e150;

mod core_dirac;
mod core_potential;
mod integrals;
mod spin_orbit;
mod valence;

pub use core_dirac::{
    CoreBracketSearch, CoreDiracSolution, CoreDiracSpec, CoreState, DiracBoundaryTrace,
    DiracEnergyDerivative, DiracError, DiracLocalOrbital, DiracSecondEnergyDerivative,
    EnergyBracket, RelativisticRole, ValenceDiracSolution, ValenceDiracSpec,
    isolate_core_dirac_bracket, solve_core_dirac, solve_valence_dirac,
};
pub use core_potential::{
    CenteredSphericalFourierMode, CorePotentialContinuationError, CorePotentialContinuationSpec,
    ExtendedCorePotential, continue_core_spherical_potential, join_core_spherical_potential,
};
pub use integrals::{RadialComponents, RadialIntegralError, RadialIntegralKernel, radial_integral};
pub use spin_orbit::{
    SpexSpinOrbitPotential, SpinOrbitRadialError, SpinOrbitRadialShell,
    spex_spin_orbit_radial_shell,
};
pub use valence::{
    BandCenter, BandEdge, BoundaryData, EnergyDerivative, LinearizedRadialSolution, LocalOrbital,
    LocalOrbitalCoefficients, LogDerivativeEnergy, RadialEquation, RadialError, RadialSolution,
    RadialSolver, SPEX_SPEED_OF_LIGHT, SecondEnergyDerivative,
};
