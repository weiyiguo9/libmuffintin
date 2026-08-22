//! Basis-neutral density-functional and self-consistency primitives.

#![forbid(unsafe_code)]

mod core_potential;
mod density;
mod energy;
mod hartree;
mod mixing;
mod occupations;
mod regional;
mod tetrahedron;
mod xc;
mod xc_field;

pub use core_potential::{
    BuiltExtendedCorePotential, CorePotentialBuildError, CorePotentialBuildSpec, CorePotentialJoin,
    build_extended_core_potentials, build_extended_snapshot_core_potentials,
};
pub use density::{
    CollinearKPoint, DensityError, FullSpinorDensitySiteBasis, FullSpinorKPoint,
    FullSpinorRegionalDensity, ScalarSiteBasis, add_core_density, core_shell_density,
    correct_electron_count, electron_count, synthesize_collinear_valence_density,
    synthesize_full_spinor_valence_density,
};
pub use energy::{EnergyError, OccupationEnergy, ScfEnergy, assemble_scf_energy};
pub use hartree::{
    ElectrostaticSpec, RegionalElectrostaticError, RegionalElectrostaticResult,
    evaluate_regional_electrostatics,
};
pub use mixing::{DensityMixer, MixRecord, MixingError};
pub use occupations::{
    BandState, FermiDiracResult, GaussianResult, OccupationError, fermi_dirac, gaussian_occupation,
    gaussian_width_matching_fermi_dirac_temperature, solve_fermi_dirac, solve_gaussian,
};
pub use regional::{
    InterstitialField, MuffinTinField, RegionalDensity, RegionalError, RegionalPotential,
    RegionalScalarField,
};
pub use tetrahedron::{
    RegularSpectrum, TetrahedronDosBins, TetrahedronError, tetrahedron_dos_bins,
};
pub use xc::{DensityJet2, XcError, XcFunctional, XcPoint, evaluate_xc_point};
pub use xc_field::{RegionalXcError, RegionalXcResult, XcFieldSpec, evaluate_regional_xc};
