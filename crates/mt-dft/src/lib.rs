//! Basis-neutral density-functional and self-consistency primitives.

#![forbid(unsafe_code)]

mod occupations;

pub use occupations::{
    BandState, FermiDiracResult, GaussianResult, OccupationError, fermi_dirac, gaussian_occupation,
    gaussian_width_matching_fermi_dirac_temperature, solve_fermi_dirac, solve_gaussian,
};
