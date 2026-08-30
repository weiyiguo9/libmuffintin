//! Representation-neutral finite-$q$ Weinert/SPEX Coulomb operator.
//!
//! Production code consumes [`muffintin_prodbasis::CompiledAuxiliaryBasis`]
//! plus, for interpolation points, [`SampledAuxiliaryFunctions`] carrying
//! parent-grid $\zeta^q$ samples. Mixed-product and interpolation-point
//! payloads share one muffin-tin radial $\times Y_{LM}$ plus interstitial
//! $|q+G|$ charge expansion. There is no production $1/r$ quadrature; the
//! direct Ewald kernel in [`ewald`] is an independent toy oracle.
//!
//! Conventions follow SPEX `coulombmatrix.f` (local trees A/B, SHA prefix
//! `6ea02fd7`): multipole moments, MT-MT, MT-PW, PW-PW, Andersen Ewald
//! structure constants, Gamma block Taylor terms, and spherical averaging.
//! There is no live SPEX $V^q$ dump in-tree.

#![forbid(unsafe_code)]

mod assemble;
mod error;
pub mod ewald;
mod expansion;
mod hartree;
mod math;
mod moments;
mod operator;
mod primitive;
mod spec;
mod structure;

pub use assemble::{assemble_coulomb, assemble_point_charge_oracle, assemble_sampled_coulomb};
pub use error::CoulombError;
pub use ewald::{
    EwaldConvergence, EwaldScan, EwaldSummation, converged_ewald_point_kernel, erfc,
    ewald_point_kernel,
};
pub use expansion::{SampledAuxiliaryFunctions, SampledPointSupport};
pub use hartree::{
    ComplexHartree, HartreeError, HartreeGauge, InterstitialHartreePotential,
    MuffinTinChargeDensity, MuffinTinHartreePotential, PeriodicChargeTreatment,
    RawElectrostaticPotential, RawHartreePotential, RawNuclearPotential, WeinertChargeDensity,
    WeinertHartreeSpec, solve_periodic_nuclear_potential, solve_weinert_hartree,
};
pub use math::weinert_gmat;
pub use moments::{
    bessel_overlap, bessel_weinert_integral, multipole_moment, second_moment,
    sphbessel_pw_integral, spherical_bessel_moment,
};
pub use operator::{AuxiliaryKind, CoulombOperator, GammaHead};
pub use primitive::{intra_sphere_poisson, radial_primitive};
pub use spec::{CoulombRequest, DEFAULT_LEXP, InterpolationProjection};
pub use structure::{
    StructureConstants, brute_force_structure_constant, spex_real_g, structure_constants,
};
