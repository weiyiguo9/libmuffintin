//! Versioned workflow input, preparation, and the unified libmuffintin runtime boundary.

#![forbid(unsafe_code)]

#[cfg(test)]
extern crate self as muffintin;

mod channel_recipe;
mod channel_token;
mod checkpoint_physics;
mod dft_scf;
mod error;
mod hf_scf;
mod input;
mod isdf_exchange;
mod mldump_header;
mod mldump_write;
mod q_mesh;
mod runner;
mod scalar_coqui_cholesky;
mod scalar_coulomb;
mod scalar_mldump;
mod scalar_mpb;
mod scalar_product;
mod scalar_thc;
mod single_dft_scf;
mod site_coords;
mod spinor_coulomb;
mod spinor_exchange_mpb;
mod spinor_mldump;
mod spinor_mpb;
mod spinor_product;
mod spinor_sector_exchange;
mod spinor_thc;
mod thc_grid;

pub use channel_recipe::{
    ChannelRecipeArtifact, ChannelRecipeError, CompiledChannelRecipe, CompiledSiteRecipe,
    ExternalChannelRecipe, RecipeSite, channel_recipe_to_toml, compile_channel_recipe,
    parse_channel_recipe_toml,
};
pub use channel_token::{
    ChannelEnergyGenerator, ChannelIdentity, ChannelProvenance, ChannelRecipeRecord, ChannelScope,
    ChannelTokenContext, ChannelTokenError, ChannelTreatment, ParsedChannelToken,
    parse_channel_token,
};
pub use checkpoint_physics::{
    AtomicStart, AtomicStartError, AtomicStartRequest, CheckpointPhysics, CheckpointPhysicsError,
    MaterialKernelError, RegionalFieldLayout, RegionalFieldLayoutError, Structure,
    checkpoint_v2_from_regional_state, checkpoint_v2_from_state, materialize_atomic_start,
};
pub use dft_scf::{
    DftConvergenceDecision, DftCoreStep, DftEnergyRecord, DftLapwDensityAssembly, DftLapwSolution,
    DftOccupations, DftRegionalDensity, DftRegionalFourier, DftRegionalPotentialStep, DftScfError,
    DftScfPlan, DftScfResult, DftScfSession, density_fourier, potential_fourier, prepare_dft_scf,
    run_dft_scf,
};
pub use error::{InputError, InputValidationError};
pub use hf_scf::{
    GammaValenceHfError, GammaValenceHfIterationDiagnostic, GammaValenceHfResult,
    GammaValenceHfSpec, ValenceHfError, ValenceHfIterationDiagnostic, ValenceHfResult,
    ValenceHfSpec, run_gamma_valence_hf, run_valence_hf,
};
pub use input::{
    BandPathPoint, Basis, BasisEnvelope, BasisEnvelopeKind, Convergence, EnergyWindow,
    ExchangeCorrelation, INPUT_FORMAT, INPUT_VERSION, Input, KMesh, Mixing, NoncollinearXcRoute,
    Occupations, Relativity, Symmetry, Task, TaskKind, Workflow, input_to_toml, parse_input_toml,
};
pub use isdf_exchange::{
    GammaExchangeTreatment, IsdfExchangeBandMatrix, IsdfExchangeError, IsdfExchangeResult,
    IsdfExchangeSpec, build_scalar_isdf_exchange, build_spinor_isdf_exchange,
    build_spinor_mpb_exchange,
};
pub use muffintin_dft::{ScalarRadialSamples, ScfKSamplingProvenance};
pub use muffintin_prodbasis::thc::RankPolicy;
pub use runner::{
    PreparedSource, PreparedTask, PreparedWorkflow, TaskResult, WorkflowResult,
    execute_prepared_with, load_input_path, prepare_input, prepare_input_with_recipes,
};
pub use scalar_coqui_cholesky::{
    ScalarCoquiCholeskyError, ScalarCoquiCholeskySpec, write_scalar_coqui_cholesky,
};
pub use scalar_coulomb::{
    SCALAR_COULOMB_EXACTNESS_FLOOR, ScalarCoulombDiscrepancy, ScalarCoulombError,
    ScalarCoulombPairDiagnostic, ScalarCoulombPairMatch, ScalarCoulombQRecord, ScalarCoulombResult,
    ScalarCoulombSpec, build_scalar_coulomb,
};
pub use scalar_mldump::{ScalarMldumpError, write_scalar_mldump};
pub use scalar_mpb::{
    SCALAR_MPB_NSPIN, ScalarMpbError, ScalarMpbPairVertex, ScalarMpbResult, ScalarMpbSelection,
    ScalarMpbSpec, build_scalar_mpb,
};
pub use scalar_product::{
    SCALAR_RADIAL_LO0, SCALAR_RADIAL_U, SCALAR_RADIAL_UDOT, ScalarBandWindow, ScalarFrozenOrbitals,
    ScalarKMinusQ, ScalarProductInput, ScalarSpinChannel,
};
pub use scalar_thc::{
    ScalarOrbitalSamples, ScalarThcError, ScalarThcResult, ScalarThcSpec, build_scalar_thc,
    sample_scalar_orbitals,
};
pub use single_dft_scf::{SingleDftScfConfigError, single_dft_scf_config};
pub use spinor_coulomb::{
    SPINOR_COULOMB_EXACTNESS_FLOOR, SpinorCoulombDiscrepancy, SpinorCoulombError,
    SpinorCoulombPairDiagnostic, SpinorCoulombPairMatch, SpinorCoulombQRecord, SpinorCoulombResult,
    SpinorCoulombSpec, build_spinor_coulomb,
};
pub use spinor_exchange_mpb::{
    SpinorExchangeMpbDiagnostics, SpinorExchangeMpbError, SpinorExchangeMpbPairVertex,
    SpinorExchangeMpbResult, SpinorExchangeMpbSector, SpinorExchangeMpbSpec,
    SpinorGammaConstantModeDiagnostic, build_spinor_exchange_mpb,
};
pub use spinor_mldump::{SpinorMldumpError, write_spinor_mldump};
pub use spinor_mpb::{
    SPINOR_MPB_NSPIN, SpinorMpbError, SpinorMpbPairVertex, SpinorMpbResult, SpinorMpbSelection,
    SpinorMpbSpec, build_spinor_mpb,
};
pub use spinor_product::{
    SPINOR_RADIAL_LO0, SPINOR_RADIAL_P, SPINOR_RADIAL_PDOT, SpinorBandWindow, SpinorCoreInputError,
    SpinorCoreOrbital, SpinorCoreTable, SpinorFrozenOrbitals, SpinorKMinusQ, SpinorProductInput,
};
pub use spinor_sector_exchange::{
    CoreShellSpillDiagnostic, FrozenExchangeSector, FrozenSpinorSectorExchange,
    FrozenSpinorSectorExchangeError, SectorOccupations, SectorRadialComparison,
    SectorRadialComparisonSpec, build_frozen_spinor_sector_exchange,
    compare_frozen_sector_radial,
};
pub use spinor_thc::{SpinorThcError, SpinorThcResult, SpinorThcSpec, build_spinor_thc};
pub use thc_grid::{
    NaturalThcGridError, NaturalThcGridSpec, ThcCandidates, ThcEngine, ThcGridError, ThcParentGrid,
    ThcPoint, ThcQRecord, ThcRegion, build_natural_thc_parent_grid,
};
