//! Basis-neutral density-functional and self-consistency primitives.

#![forbid(unsafe_code)]

mod atomic_configuration;
mod atomic_superposition;
mod core_density;
mod core_potential;
mod density;
mod energy;
mod free_atom;
mod hartree;
mod linearization;
mod material_kernel;
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
pub use atomic_superposition::{
    AtomicSuperpositionChargeClosure, AtomicSuperpositionDensity, AtomicSuperpositionError,
    AtomicSuperpositionSite, AtomicSuperpositionSpec, build_atomic_superposition_density,
};
pub use core_density::{
    BuiltRegionalCoreContribution, CoreDensityDiagnostics, CoreDensityError,
    CoreShellDensityDiagnostic, CoreSpinPartition, PseudochargeBoundaryDiagnostic,
    PseudochargeZeroModeAdjustment, RegionalCoreShellInput, build_regional_core_contribution,
};
pub use core_potential::{
    BuiltExtendedCorePotential, CorePotentialBuildError, CorePotentialBuildSpec, CorePotentialJoin,
    build_extended_core_potentials, build_extended_checkpoint_core_potentials,
};
pub use density::{
    CollinearKPoint, DensityError, FullSpinorDensitySiteBasis, FullSpinorKPoint, ScalarSiteBasis,
    add_core_density, core_shell_density, correct_electron_count, electron_count,
    physical_site_band_projections, synthesize_collinear_valence_density,
    synthesize_full_spinor_valence_density,
};
pub use energy::{EnergyError, OccupationEnergy, ScfEnergy, assemble_scf_energy};
pub use free_atom::{
    FreeAtomOrbital, FreeAtomScfError, FreeAtomScfSpec, FreeAtomState, run_free_atom_lda,
};
pub use hartree::{
    ScfPotentialBuild, ScfPotentialBuildError, build_scf_potential,
    ElectrostaticSpec, RegionalElectrostaticError, RegionalElectrostaticResult,
    evaluate_regional_electrostatics,
};
pub use linearization::{
    AtomicEnergyRequest, GeneratedLinearizationEnergy, LinearizationEnergyDiagnostic,
    LinearizationEnergyError, LinearizationEnergyGenerator, PdosEnergySample,
    generate_atomic_energy, generate_band_center_energy, generate_band_cog_energy,
    generate_explicit_energy, generate_fermi_offset_energy, generate_frozen_checkpoint_energy,
    generate_log_derivative_energy, kappa_degeneracy_average,
};
pub use material_kernel::{
    CheckpointBandSolution, CheckpointKPoint, CheckpointKPointSolution, CheckpointOneParticle,
    CheckpointSite, CheckpointSpin, MaterialKernel, MaterialKernelError, RadialRoute,
    SpexBoundSpinorChannel, SpexSpinorMaterialBinding, g_vector, production_density_layout,
    regular_k_points,
};
pub use mixing::{DensityMixer, MixAlgebraQuantity, MixRecord, MixStatus, MixStep, MixingError};
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
    ChannelKappaError, channel_kappas, channel_l, channel_n, scalar_component_energy,
    spin_resolved_energy, spinor_kappas_for_l,
    BandPathPoint, BandPathPointResult, BandPathRequest, BandPathResult, ContinueStep,
    ConvergenceDecision, CoreContribution, CoreStep, DosRequest, DosResult, EnergyRecord,
    LapwDensityAssembly, LapwSolution, OccupationStep, RegionalDensityStep, RegionalPotentialStep,
    ScfBasis, ScfChannelIdentity, ScfChannelProvenance, ScfChannelRecipe,
    ScfChannelTreatment, ScfConfig, ScfConfigError, ScfConvergence, ScfCoreSite, ScfCoreState,
    ScfEnergyContext, ScfEnergyTerms, ScfError, ScfExchangeCorrelation, ScfIterationDiagnostic,
    ScfKMesh, ScfLoop, ScfMixing, ScfOccupations, ScfPhysics, ScfRelativity,
    ScfResolvedChannelEnergy, ScfState, run_band_path, run_dos, run_scf,
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
    BuiltSpinorLocalOrbital, SpinorBuilderError, SpinorIterationBasis, SpinorLinearizationEnergy,
    SpinorLocalOrbitalOrigin, SpinorLocalOrbitalRequest, SpinorRadialSite, SpinorSiteInput,
    build_spinor_iteration_basis, solve_spinor_k_point,
};
pub use tetrahedron::{
    RegularSpectrum, TetrahedronDosBins, TetrahedronError, tetrahedron_dos_bins,
};
pub use xc::{DensityJet2, XcError, XcFunctional, XcPoint, evaluate_xc_point};
pub use xc_field::{
    xc_spec_for_density,
    NoncollinearXcRoute, RegionalXcError, RegionalXcResult, XcFieldSpec, evaluate_regional_xc,
};
