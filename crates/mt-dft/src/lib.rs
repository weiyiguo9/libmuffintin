//! Basis-neutral density-functional and self-consistency primitives.

#![forbid(unsafe_code)]

mod atomic_configuration;
mod core_density;
mod core_potential;
mod density;
mod energy;
mod hartree;
mod mixing;
mod occupations;
mod regional;
mod scalar;
mod scf;
mod soc;
mod spinor;
mod spinor_builder;
mod tetrahedron;
mod xc;
mod xc_field;

pub use atomic_configuration::{
    AtomicChannelTreatment, AtomicElectronicConfiguration, AtomicNumber, AtomicOccupation,
    RelativisticOrbital, fleur_default_atomic_configuration,
};
pub use core_density::{
    BuiltRegionalCoreContribution, CoreDensityDiagnostics, CoreDensityError,
    CoreShellDensityDiagnostic, CoreSpinPartition, PseudochargeBoundaryDiagnostic,
    PseudochargeZeroModeAdjustment, RegionalCoreShellInput, build_regional_core_contribution,
};
pub use core_potential::{
    BuiltExtendedCorePotential, CorePotentialBuildError, CorePotentialBuildSpec, CorePotentialJoin,
    build_extended_core_potentials, build_extended_snapshot_core_potentials,
};
pub use density::{
    CollinearKPoint, DensityError, FullSpinorDensitySiteBasis, FullSpinorKPoint, ScalarSiteBasis,
    add_core_density, core_shell_density, correct_electron_count, electron_count,
    synthesize_collinear_valence_density, synthesize_full_spinor_valence_density,
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
pub use scalar::{
    BuiltScalarLocalOrbital, ScalarBuilderError, ScalarIterationBasis, ScalarLocalOrbitalOrigin,
    ScalarLocalOrbitalRequest, ScalarRadialSite, ScalarSiteInput, SolvedScalarKPoint,
    build_collinear_scalar_iteration_bases, build_scalar_iteration_basis,
    solve_collinear_scalar_k_point, solve_scalar_k_point, solve_scalar_second_variation,
};
pub use scf::{
    BandPathPoint, BandPathPointResult, BandPathRequest, BandPathResult, CoreContribution,
    DosRequest, DosResult, ScfBasis, ScfConfig, ScfConfigError, ScfConvergence, ScfCoreSite,
    ScfCoreState, ScfEnergyContext, ScfEnergyTerms, ScfError, ScfExchangeCorrelation,
    ScfIterationDiagnostic, ScfKMesh, ScfLocalOrbital, ScfLocalOrbitalKind, ScfMixing,
    ScfOccupations, ScfPhysics, ScfRelativisticLocalOrbital, ScfRelativity, ScfState,
    run_band_path, run_dos, run_scf,
};
pub use soc::{
    FirstVariationRoute, FirstVariationSubspace, FirstVariationWindow,
    SecondVariationBandDiagnostic, SecondVariationError, SecondVariationResult, SourceBandWeight,
    solve_soc_second_variation,
};
pub use spinor::{
    FullSpinorSiteInput, LocalPauliPotential, RelativisticSpinorRoute,
    SolvedFullSpinorFirstVariation, SpinorFirstVariationError, build_full_spinor_site_blocks,
    solve_full_spinor_first_variation,
};
pub use spinor_builder::{
    BuiltSpinorLocalOrbital, SpinorBuilderError, SpinorIterationBasis, SpinorLocalOrbitalOrigin,
    SpinorLocalOrbitalRequest, SpinorRadialSite, SpinorSiteInput, build_spinor_iteration_basis,
    solve_spinor_k_point,
};
pub use tetrahedron::{
    RegularSpectrum, TetrahedronDosBins, TetrahedronError, tetrahedron_dos_bins,
};
pub use xc::{DensityJet2, XcError, XcFunctional, XcPoint, evaluate_xc_point};
pub use xc_field::{
    NoncollinearXcRoute, RegionalXcError, RegionalXcResult, XcFieldSpec, evaluate_regional_xc,
};
