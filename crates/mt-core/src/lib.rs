//! Convention-bearing numerical primitives shared by all muffin-tin methods.
//!
//! The crate deliberately owns conventions that would otherwise be implicit:
//! Hartree atomic units, Condon--Shortley spherical harmonics, SPEX Gaunt
//! coefficients, the SPEX radial quadrature, reciprocal-vector cutoffs, and
//! interstitial step-function Fourier coefficients.

#![forbid(unsafe_code)]

pub mod bessel;
pub mod conventions;
pub mod gaunt;
pub mod harmonics;
pub mod mesh;
pub mod reciprocal;
pub mod step_function;
pub mod units;

pub use bessel::{
    BesselError, spherical_bessel_j, spherical_bessel_j_derivative, spherical_bessel_y,
    spherical_bessel_y_derivative,
};
pub use conventions::{KineticOperatorConvention, spherical_value_from_y00_coefficient};
pub use gaunt::{gaunt, real_gaunt, wigner_3j};
pub use harmonics::{
    Lm, LmError, complex_spherical_harmonic, complex_spherical_harmonics, lm_count, lm_from_index,
    lm_index, real_spherical_harmonic, real_spherical_harmonics,
};
pub use mesh::{ExponentialMesh, MeshError, OriginContribution, QuadratureWeights};
pub use reciprocal::{GVector, LatticeError, ReciprocalLattice};
pub use step_function::{
    InterstitialGeometry, Sphere, StepFunctionError, sphere_form_factor,
    step_function_coefficient_cutoff,
};
pub use units::{Bohr, Hartree, InverseBohr, VolumeBohr3};

/// Internal energy convention: Hartree atomic units, so `T = -1/2 * laplacian`.
pub const KINETIC_ENERGY_FACTOR: f64 = 0.5;
