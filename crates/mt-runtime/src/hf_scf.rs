//! Full-regular-BZ spinor-first valence Hartree--Fock SCF.

use muffintin_core::{FourierLayout, Hartree, InverseBohr};
use muffintin_coulomb::{
    CoreCoreFockError, CoreCoreFockShell, CoulombOperator, CoulombRequest, HartreeError,
    WeinertHartreeSpec, assemble_coulomb, core_core_fock_actions,
};
use muffintin_dft::{
    BandState, CheckpointBandSolution, CheckpointKPointSolution, CoreDensityError,
    CoreFixedPotentialResult, CoreFixedPotentialSpec, CoreLocalOneBodyError, CoreLocalOneBodyTrace,
    CoreShellOccupations, CoreShellOrbitals, DensityError, DensityMixer, ElectrostaticSpec,
    MaterialKernelError, MixingError, OccupationError, RegionalDensity, RegionalElectrostaticError,
    RegionalError, RegionalPotential, ScfConfig, ScfConfigError, ScfKReduction, ScfLoop, ScfMixing,
    ScfOccupations, ScfPhysics, ScfRelativity, build_regional_core_contribution_from_sidecar,
    core_local_one_body_trace, electron_count, evaluate_regional_electrostatics,
};
use muffintin_operators::{
    Collinear, OperatorError, lift_band_hermitian_feedback, solve_generalized_hermitian,
};
use muffintin_tensor::{Axis, ComplexTensor, DenseHermitianMatrix, TensorError, einsum};
use num_complex::Complex64;
use thiserror::Error;

use crate::q_mesh::{CanonicalQMapError, canonical_q_points};
use crate::scalar_mpb::{ScalarMpbSelection, ScalarMpbSpec, build_scalar_mpb};
use crate::spinor_exchange_mpb::{
    SpinorExchangeMpbBasis, build_spinor_exchange_feedback_from_basis,
    compile_spinor_exchange_mpb_basis,
};
use crate::spinor_mpb::{SpinorMpbBasis, build_spinor_mpb_from_basis, compile_spinor_mpb_basis};
use crate::spinor_sector_exchange::{
    FrozenSpinorValenceFeedbackExchange, build_cached_spinor_valence_feedback_exchange,
    complete_frozen_spinor_sector_exchange,
};
use crate::{
    CheckpointPhysics, CheckpointPhysicsError, CoreValenceComparisonSpec,
    CoreValenceDeltaDiagnostic, FrozenCoreValenceComparison, FrozenCoreValenceError,
    FrozenSpinorSectorExchange, FrozenSpinorSectorExchangeError, GammaExchangeTreatment,
    IsdfExchangeError, IsdfExchangeResult, IsdfExchangeSpec, SecondVariationMpbSelection,
    SecondVariationMpbSpec, SectorOccupations, SpinorCoreInputError, SpinorExchangeMpbError,
    SpinorExchangeMpbResult, SpinorExchangeMpbSpec, SpinorMpbError, SpinorMpbSelection,
    SpinorMpbSpec, SpinorProductInput, build_frozen_core_valence_exchange,
    build_frozen_site_valence_densities, build_scalar_mpb_exchange, build_second_variation_mpb,
    build_second_variation_mpb_exchange, build_spinor_exchange_mpb, build_spinor_mpb,
    build_spinor_mpb_exchange, compare_frozen_core_valence, relax_frozen_core_at_fixed_potential,
};

const SPECTRAL_REFINEMENT_PASSES: usize = 16;
const IDENTITY_TOLERANCE: f64 = 2.0e-8;
const ELECTRON_COUNT_TOLERANCE: f64 = 1.0e-8;

/// Mixing algorithm for the fixed-potential global exchange feedback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FockMixing {
    Linear {
        alpha: f64,
    },
    PulayAnderson {
        alpha: f64,
        history: usize,
    },
    CommutatorDiis {
        history: usize,
    },
    QuasiNewtonDiis {
        history: usize,
        level_shift: Hartree,
    },
}

impl FockMixing {
    const fn history(self) -> Option<usize> {
        match self {
            Self::Linear { .. } => None,
            Self::PulayAnderson { history, .. }
            | Self::CommutatorDiis { history }
            | Self::QuasiNewtonDiis { history, .. } => Some(history),
        }
    }
}

/// Exact MPB controls and bounded iteration controls for the full-BZ
/// spinor-first valence HF engine.
#[derive(Clone, Debug, PartialEq)]
pub struct GammaValenceHfSpec {
    pub config: ScfConfig,
    pub product_l_max: u32,
    pub product_g_max: InverseBohr,
    pub overlap_tolerance: f64,
    pub coulomb: CoulombRequest,
    pub max_fock_iterations: usize,
    /// Regional-density fixed-point tolerance inside one fixed local potential.
    pub fock_density_tolerance: f64,
    /// Maximum global exchange-feedback matrix-element residual in Hartree.
    pub fock_feedback_tolerance: Hartree,
    /// Mixer for the freshly rebuilt global exchange feedback.
    pub fock_mixing: FockMixing,
}

pub type ValenceHfSpec = GammaValenceHfSpec;

/// One completed outer Hartree-density iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct GammaValenceHfIterationDiagnostic {
    pub iteration: usize,
    pub fock_iterations: usize,
    pub exchange_rebuilds: usize,
    pub exchange_energy: Hartree,
    pub maximum_antihermitian_residual: f64,
    pub fock_fixed_point_residual: f64,
    pub fock_feedback_residual: Hartree,
    pub regional_density_rms: f64,
    pub density_electron_count: f64,
    pub total_energy: Hartree,
    pub energy_change: Option<Hartree>,
    /// `E_x - 1/2 Tr(D K)`.
    pub exchange_energy_identity_residual: f64,
    /// `sum(f epsilon) - Tr(D H0) - Tr(D K)`.
    pub eigenvalue_identity_residual: f64,
    /// Difference between direct-functional and eigenvalue-corrected HF energy.
    pub total_energy_identity_residual: f64,
    /// Maximum element of `C^H (S C K C^H S) C - K`.
    pub lifting_identity_residual: f64,
    /// First global H0+lifted-K solve versus `diag(epsilon0)+K`.
    pub first_global_solve_identity_residual: Option<f64>,
}

pub type ValenceHfIterationDiagnostic = GammaValenceHfIterationDiagnostic;

/// Converged finite-full-BZ-mesh valence-only spinor HF state.
#[derive(Clone, Debug)]
pub struct GammaValenceHfResult {
    pub density: RegionalDensity,
    pub potential: RegionalPotential,
    pub bands: CheckpointBandSolution,
    /// Per-k fractional occupations; A1 returns one row while preserving the
    /// mesh-shaped boundary needed by a later regular-k engine.
    pub occupations: Vec<Vec<f64>>,
    pub orbital_energies: Vec<Vec<Hartree>>,
    pub exchange_energy: Hartree,
    pub maximum_antihermitian_residual: f64,
    pub fock_fixed_point_residual: f64,
    pub fock_feedback_residual: Hartree,
    pub regional_density_rms: f64,
    pub total_energy: Hartree,
    pub exchange_rebuilds: usize,
    pub first_one_shot_exchange: IsdfExchangeResult,
    pub k_fractional: Vec<[f64; 3]>,
    pub q_fractional: Vec<[f64; 3]>,
    pub k_weights: Vec<f64>,
    pub diagnostics: Vec<GammaValenceHfIterationDiagnostic>,
}

pub type ValenceHfResult = GammaValenceHfResult;

/// Exact-MPB controls for scalar KH HF followed by conventional SOC second variation.
#[derive(Clone, Debug, PartialEq)]
pub struct KhSocValenceHfSpec {
    pub config: ScfConfig,
    pub product_l_max: u32,
    pub product_g_max: InverseBohr,
    pub overlap_tolerance: f64,
    pub coulomb: CoulombRequest,
    pub gamma: GammaExchangeTreatment,
    pub max_fock_iterations: usize,
    pub fock_density_tolerance: f64,
    pub fock_feedback_tolerance: Hartree,
    /// Scalar KH currently accepts linear or Pulay–Anderson feedback mixing.
    pub fock_mixing: FockMixing,
    pub core_treatment: KhSocCoreTreatment,
}

/// Treatment of core states in the scalar KH plus SOC HF route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KhSocCoreTreatment {
    ValenceOnly,
    Frozen,
}

/// One outer scalar-KH HF iteration preceding SOC second variation.
#[derive(Clone, Debug, PartialEq)]
pub struct KhSocValenceHfIterationDiagnostic {
    pub iteration: usize,
    pub fock_iterations: usize,
    pub exchange_rebuilds: usize,
    pub exchange_energy: Hartree,
    pub fock_fixed_point_residual: f64,
    pub fock_feedback_residual: Hartree,
    pub valence_density_rms: f64,
    pub regional_density_rms: f64,
    pub total_energy: Hartree,
    pub energy_change: Option<Hartree>,
}

/// Converged scalar KH HF state and its one-shot SOC second variation.
#[derive(Clone, Debug)]
pub struct KhSocValenceHfResult {
    pub valence_density: RegionalDensity,
    pub core_density: RegionalDensity,
    pub total_density: RegionalDensity,
    pub potential: RegionalPotential,
    pub scalar_bands: CheckpointBandSolution,
    pub bands: CheckpointBandSolution,
    pub occupations: Vec<Vec<f64>>,
    pub orbital_energies: Vec<Vec<Hartree>>,
    pub scalar_exchange: Collinear<IsdfExchangeResult>,
    pub second_variation_exchange: IsdfExchangeResult,
    pub core_orbitals: Vec<CoreShellOrbitals>,
    pub core_h0_trace: Hartree,
    pub core_core_exchange: Hartree,
    pub core_valence_exchange: Hartree,
    pub total_energy: Hartree,
    pub fock_fixed_point_residual: f64,
    pub fock_feedback_residual: Hartree,
    pub valence_density_rms: f64,
    pub regional_density_rms: f64,
    pub second_variation_density_rms: f64,
    pub exchange_energy_change: Hartree,
    pub exchange_rebuilds: usize,
    pub k_fractional: Vec<[f64; 3]>,
    pub q_fractional: Vec<[f64; 3]>,
    pub k_weights: Vec<f64>,
    pub diagnostics: Vec<KhSocValenceHfIterationDiagnostic>,
}

#[derive(Debug, Error)]
pub enum KhSocValenceHfError {
    #[error("KH+SOC valence HF requires ScfRelativity::SocSecondVariation")]
    Relativity,
    #[error("KH+SOC valence HF currently requires linear or Pulay–Anderson Fock mixing")]
    FockMixing,
    #[error("KH+SOC core treatment is inconsistent with the configured occupied core states")]
    CoreTreatment,
    #[error(transparent)]
    Hf(#[from] GammaValenceHfError),
}

/// Exact-MPB and bounded inner-loop controls for relaxed-core spinor HF.
///
/// `config.electron_count` is the total electron count. The driver subtracts
/// the requested occupied core charge exactly once before solving valence
/// occupations.
#[derive(Clone, Debug, PartialEq)]
pub struct RelaxedCoreHfSpec {
    pub config: ScfConfig,
    pub product_l_max: u32,
    pub product_g_max: InverseBohr,
    pub overlap_tolerance: f64,
    pub coulomb: CoulombRequest,
    pub gamma: GammaExchangeTreatment,
    pub max_fock_iterations: usize,
    pub fock_density_tolerance: f64,
    /// Maximum global exchange-feedback matrix-element residual in Hartree.
    pub fock_feedback_tolerance: Hartree,
    pub fock_mixing: FockMixing,
    pub core: CoreFixedPotentialSpec,
    /// Numerical MPB/radial and CV/VC trace tolerance.
    pub sector_numerical_tolerance: Hartree,
    /// Dimensionless maximum permitted core-shell norm outside the muffin tin.
    pub maximum_core_shell_spill: f64,
}

/// One completed relaxed-core outer Hartree-density iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct RelaxedCoreHfIterationDiagnostic {
    pub iteration: usize,
    pub fock_iterations: usize,
    pub exchange_rebuilds: usize,
    pub trace_vv: Hartree,
    pub trace_cv: Hartree,
    pub trace_vc: Hartree,
    pub trace_cc: Hartree,
    pub exchange_vv: Hartree,
    pub exchange_cv: Hartree,
    pub exchange_cc: Hartree,
    pub exchange_total: Hartree,
    pub cv_vc_trace_mismatch: Hartree,
    /// The trace of the only feedback entering the valence Fock equation.
    pub valence_feedback_vv_cv_trace: Hartree,
    pub maximum_antihermitian_residual: f64,
    pub fock_fixed_point_residual: f64,
    pub fock_feedback_residual: Hartree,
    pub valence_density_rms: f64,
    pub total_density_rms: f64,
    pub valence_electron_count: f64,
    pub core_electron_count: f64,
    pub total_electron_count: f64,
    pub core_inner_iterations: Vec<usize>,
    pub core_maximum_energy_change: Hartree,
    pub core_maximum_radial_residual: f64,
    pub core_h0_trace: Hartree,
    pub total_energy: Hartree,
    pub energy_change: Option<Hartree>,
    /// `sum(f epsilon_v) - Tr(D_v H0) - T_vv - T_cv`.
    pub valence_eigenvalue_identity_residual: f64,
    pub lifting_identity_residual: f64,
    pub first_global_solve_identity_residual: Option<f64>,
    /// Difference between the previous core density and its fresh replacement.
    pub fresh_core_replacement_rms: f64,
    /// Per-core exact-minus-spherical VC expectations from the final rebuilt frame.
    pub delta_c: Vec<CoreValenceDeltaDiagnostic>,
    pub weighted_delta_closure_residual: Hartree,
}

/// Converged relaxed-core spinor HF state on one explicit full regular mesh.
#[derive(Clone, Debug)]
pub struct RelaxedCoreHfResult {
    pub valence_density: RegionalDensity,
    pub core_density: RegionalDensity,
    pub total_density: RegionalDensity,
    pub potential: RegionalPotential,
    pub bands: CheckpointBandSolution,
    pub occupations: Vec<Vec<f64>>,
    pub orbital_energies: Vec<Vec<Hartree>>,
    pub core_orbitals: Vec<CoreShellOrbitals>,
    pub core_one_body_traces: Vec<CoreLocalOneBodyTrace>,
    /// Complete canonical q slice rebuilt from the converged band/core frame.
    ///
    /// This is the same frozen input slice sealed into [`Self::sector_exchange`],
    /// not the bootstrap frame or an earlier Fock iteration.
    pub final_exchange_inputs: Vec<SpinorProductInput>,
    pub sector_exchange: FrozenSpinorSectorExchange,
    pub core_valence_comparison: FrozenCoreValenceComparison,
    pub core_h0_trace: Hartree,
    pub total_energy: Hartree,
    pub maximum_antihermitian_residual: f64,
    pub fock_fixed_point_residual: f64,
    pub fock_feedback_residual: Hartree,
    pub valence_density_rms: f64,
    pub total_density_rms: f64,
    pub exchange_rebuilds: usize,
    pub k_fractional: Vec<[f64; 3]>,
    pub q_fractional: Vec<[f64; 3]>,
    pub k_weights: Vec<f64>,
    pub diagnostics: Vec<RelaxedCoreHfIterationDiagnostic>,
}

/// Invalid relaxed-core controls or a failed bounded HF solve.
#[derive(Debug, Error)]
pub enum RelaxedCoreHfError {
    #[error("Gamma relaxed-core HF requires a 1x1x1 full k mesh with zero shift")]
    GammaMesh,
    #[error("Gamma relaxed-core HF requires explicit FiniteBody exchange")]
    GammaFiniteBody,
    #[error("relaxed-core HF requires an explicit full regular Brillouin-zone mesh")]
    SymmetryReduction,
    #[error("relaxed-core HF regular k mesh cannot define its canonical q topology")]
    QTopology,
    #[error("relaxed-core HF requires ScfRelativity::SpinorFirstVariation")]
    SpinorFirstVariation,
    #[error("relaxed-core HF requires at least one occupied core state")]
    CoreStates,
    #[error("relaxed-core HF total electron count must exceed its core electron count")]
    ValenceElectronCount,
    #[error("relaxed-core HF max_fock_iterations must be at least two")]
    FockIterations,
    #[error("relaxed-core HF fock_density_tolerance must be finite and positive")]
    FockTolerance,
    #[error("relaxed-core HF fock_feedback_tolerance must be finite and positive")]
    FockFeedbackTolerance,
    #[error(
        "relaxed-core HF Fock mixer requires alpha in (0, 1], nonlinear history >= 2, and a finite nonnegative level shift"
    )]
    FockMixing,
    #[error("relaxed-core HF sector numerical tolerance must be finite and nonnegative")]
    SectorTolerance,
    #[error("relaxed-core HF maximum core-shell spill must be finite and nonnegative")]
    CoreSpillTolerance,
    #[error("spectral radial-basis refinement did not settle after {passes} passes")]
    SpectralRefinement { passes: usize },
    #[error(
        "fixed-local-potential Fock iteration {outer_iteration} did not converge in {iterations} rebuilds (density residual {density_residual}, fresh-feedback residual {feedback_residual})"
    )]
    FockNotConverged {
        outer_iteration: usize,
        iterations: usize,
        density_residual: f64,
        feedback_residual: f64,
    },
    #[error(
        "relaxed-core HF did not converge in {iterations} outer iterations (energy change {energy_change} Ha, density RMS {density_rms})"
    )]
    NotConverged {
        iterations: usize,
        energy_change: f64,
        density_rms: f64,
    },
    #[error(transparent)]
    Config(#[from] ScfConfigError),
    #[error(transparent)]
    Checkpoint(#[from] CheckpointPhysicsError),
    #[error(transparent)]
    Kernel(#[from] MaterialKernelError),
    #[error(transparent)]
    Electrostatic(#[from] RegionalElectrostaticError),
    #[error(transparent)]
    Hartree(#[from] HartreeError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Density(#[from] DensityError),
    #[error(transparent)]
    Occupation(#[from] OccupationError),
    #[error(transparent)]
    Mixing(#[from] MixingError),
    #[error(transparent)]
    Mpb(#[from] SpinorMpbError),
    #[error(transparent)]
    CoreMpb(#[from] SpinorExchangeMpbError),
    #[error(transparent)]
    CoreInput(#[from] SpinorCoreInputError),
    #[error(transparent)]
    Sector(#[from] FrozenSpinorSectorExchangeError),
    #[error(transparent)]
    CoreValence(#[from] FrozenCoreValenceError),
    #[error(transparent)]
    CoreDensity(#[from] CoreDensityError),
    #[error(transparent)]
    CoreOneBody(#[from] CoreLocalOneBodyError),
    #[error("relaxed-core HF exchange band matrix {actual} is not in expected k order {expected}")]
    ExchangeKIndex { expected: usize, actual: usize },
    #[error("relaxed-core HF gate {gate} has residual {residual}, above {tolerance}")]
    Gate {
        gate: &'static str,
        residual: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    ValencePath(#[from] GammaValenceHfError),
}

/// Invalid valence-HF controls or a failed bounded HF solve.
#[derive(Debug, Error)]
pub enum GammaValenceHfError {
    #[error("Gamma valence HF requires a 1x1x1 full k mesh with zero shift")]
    GammaMesh,
    #[error("valence HF requires an explicit full regular Brillouin-zone mesh")]
    SymmetryReduction,
    #[error("valence HF regular k mesh cannot define its canonical q topology")]
    QTopology,
    #[error("valence HF requires ScfRelativity::SpinorFirstVariation")]
    SpinorFirstVariation,
    #[error("valence HF rejects every occupied core state")]
    CoreStates,
    #[error("valence HF max_fock_iterations must be at least two")]
    FockIterations,
    #[error("valence HF fock_density_tolerance must be finite and positive")]
    FockTolerance,
    #[error("valence HF fock_feedback_tolerance must be finite and positive")]
    FockFeedbackTolerance,
    #[error(
        "valence HF Fock mixer requires alpha in (0, 1], nonlinear history >= 2, and a finite nonnegative level shift"
    )]
    FockMixing,
    #[error("valence HF nonlinear feedback algebra produced a non-finite value")]
    FockMixingAlgebra,
    #[error("spectral radial-basis refinement did not settle after {passes} passes")]
    SpectralRefinement { passes: usize },
    #[error(
        "fixed-local-potential Fock iteration {outer_iteration} did not converge in {iterations} rebuilds (density residual {density_residual}, fresh-feedback residual {feedback_residual})"
    )]
    FockNotConverged {
        outer_iteration: usize,
        iterations: usize,
        density_residual: f64,
        feedback_residual: f64,
    },
    #[error(
        "valence HF did not converge in {iterations} outer iterations (energy change {energy_change} Ha, density RMS {density_rms})"
    )]
    NotConverged {
        iterations: usize,
        energy_change: f64,
        density_rms: f64,
    },
    #[error(transparent)]
    Config(#[from] ScfConfigError),
    #[error(transparent)]
    Checkpoint(#[from] CheckpointPhysicsError),
    #[error(transparent)]
    Kernel(#[from] MaterialKernelError),
    #[error(transparent)]
    Electrostatic(#[from] RegionalElectrostaticError),
    #[error(transparent)]
    Hartree(#[from] HartreeError),
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error(transparent)]
    Density(#[from] DensityError),
    #[error(transparent)]
    Occupation(#[from] OccupationError),
    #[error(transparent)]
    Mixing(#[from] MixingError),
    #[error(transparent)]
    CoreOneBody(#[from] CoreLocalOneBodyError),
    #[error(transparent)]
    CoreCore(#[from] CoreCoreFockError),
    #[error(transparent)]
    Mpb(#[from] SpinorMpbError),
    #[error(transparent)]
    ScalarMpb(#[from] crate::ScalarMpbError),
    #[error(transparent)]
    Exchange(#[from] IsdfExchangeError),
    #[error("exchange band matrix {actual} is not in expected k order {expected}")]
    ExchangeKIndex { expected: usize, actual: usize },
    #[error("valence HF gate {gate} has residual {residual}, above {tolerance}")]
    Gate {
        gate: &'static str,
        residual: f64,
        tolerance: f64,
    },
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

pub type ValenceHfError = GammaValenceHfError;

/// Run the strict A1 molecule-in-box wrapper through the generic mesh engine.
pub fn run_gamma_valence_hf(
    physics: &mut CheckpointPhysics,
    spec: &GammaValenceHfSpec,
) -> Result<GammaValenceHfResult, GammaValenceHfError> {
    validate_gamma_spec(spec)?;
    run_valence_hf(physics, spec)
}

/// Run valence-only spinor HF on one explicit full regular BZ mesh.
///
/// Every Fock rebuild consumes one live frame containing all physical shifted
/// k points and constructs the complete unshifted canonical q slice from that
/// same frame. No vertex or Coulomb record survives an orbital update. Every
/// outer density step rematerializes the radial basis, so feedback is never
/// carried between incompatible H0/S frames.
pub fn run_valence_hf(
    physics: &mut CheckpointPhysics,
    spec: &ValenceHfSpec,
) -> Result<ValenceHfResult, ValenceHfError> {
    validate_spec(spec)?;
    let _ = ScfLoop::new(spec.config.clone(), None)?;
    let k_fractional = muffintin_dft::regular_k_points(spec.config.k_mesh)?;
    let q_fractional = canonical_q_points(&k_fractional).map_err(q_topology_error)?;
    let mut mixer = density_mixer(spec.config.mixing)?;
    let mut density = <muffintin_dft::MaterialKernel as ScfPhysics>::initial_density(
        &mut physics.kernel,
        &spec.config,
    )?;
    let mut previous_total = None;
    let mut diagnostics = Vec::with_capacity(spec.config.convergence.max_iterations);
    let mut total_exchange_rebuilds = 0;
    let mut first_one_shot_exchange = None;

    for outer_iteration in 1..=spec.config.convergence.max_iterations {
        let electrostatic = evaluate_regional_electrostatics(
            density.charge(),
            &ElectrostaticSpec::new(
                WeinertHartreeSpec::electronic(4)?,
                physics.nuclear_charges().to_vec(),
            )?,
        )?;
        let zero = electrostatic.potential.zero_like();
        let potential = RegionalPotential::new(
            electrostatic.potential.clone(),
            [zero.clone(), zero.clone(), zero],
        )?;
        let (bands, _occupation) = solve_h0_bands(
            physics,
            &spec.config,
            &potential,
            &k_fractional,
            spec.config.electron_count,
            density.charge().interstitial().layout(),
            ScfRelativity::SpinorFirstVariation,
        )?;

        let fixed = solve_fixed_potential(
            physics,
            spec,
            bands,
            outer_iteration,
            &k_fractional,
            &q_fractional,
        )?;
        total_exchange_rebuilds += fixed.exchange_rebuilds;
        if let Some(first) = fixed.first_one_shot_exchange.clone() {
            first_one_shot_exchange = Some(first);
        }
        let output_density = physics
            .kernel
            .synthesize_bands(&fixed.bands, &fixed.occupation.values)?;
        let density_rms = density.difference_rms(&output_density)?;
        let output_electron_count = electron_count(&output_density)?;
        let energy = energy_diagnostic(
            &fixed.bands,
            &fixed.occupation,
            &fixed.exchange,
            &electrostatic,
        )?;
        let energy_change = previous_total
            .map(|previous: Hartree| Hartree((energy.total.get() - previous.get()).abs()));
        require_gate(
            "density electron count",
            (output_electron_count - spec.config.electron_count).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        require_gate(
            "exchange energy identity",
            energy.exchange_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        require_gate(
            "valence eigenvalue identity",
            energy.eigenvalue_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        require_gate(
            "HF total-energy identity",
            energy.total_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        diagnostics.push(GammaValenceHfIterationDiagnostic {
            iteration: outer_iteration,
            fock_iterations: fixed.fock_iterations,
            exchange_rebuilds: fixed.exchange_rebuilds,
            exchange_energy: fixed.exchange.exchange_energy,
            maximum_antihermitian_residual: fixed.exchange.maximum_antihermitian_residual,
            fock_fixed_point_residual: fixed.fixed_point_residual,
            fock_feedback_residual: Hartree(fixed.feedback_residual),
            regional_density_rms: density_rms,
            density_electron_count: output_electron_count,
            total_energy: energy.total,
            energy_change,
            exchange_energy_identity_residual: energy.exchange_identity_residual,
            eigenvalue_identity_residual: energy.eigenvalue_identity_residual,
            total_energy_identity_residual: energy.total_identity_residual,
            lifting_identity_residual: fixed.lifting_identity_residual,
            first_global_solve_identity_residual: fixed.first_global_solve_identity_residual,
        });
        let converged = energy_change
            .is_some_and(|change| change.get() <= spec.config.convergence.energy_tolerance.get())
            && density_rms <= spec.config.convergence.density_tolerance
            && fixed.fixed_point_residual <= spec.fock_density_tolerance;
        if converged {
            let orbital_energies = spinor_energies(&fixed.bands)?;
            let occupations = occupation_rows(&fixed.occupation.values, &fixed.bands)?;
            let k_weights = k_weights(&fixed.bands)?;
            return Ok(GammaValenceHfResult {
                density: output_density,
                potential,
                bands: fixed.bands,
                occupations,
                orbital_energies,
                exchange_energy: fixed.exchange.exchange_energy,
                maximum_antihermitian_residual: fixed.exchange.maximum_antihermitian_residual,
                fock_fixed_point_residual: fixed.fixed_point_residual,
                fock_feedback_residual: Hartree(fixed.feedback_residual),
                regional_density_rms: density_rms,
                total_energy: energy.total,
                exchange_rebuilds: total_exchange_rebuilds,
                first_one_shot_exchange: first_one_shot_exchange
                    .expect("first outer iteration always records the one-shot oracle"),
                k_fractional,
                q_fractional,
                k_weights,
                diagnostics,
            });
        }
        previous_total = Some(energy.total);
        density = mixer.mix(&density, &output_density)?.density;
    }

    let last = diagnostics
        .last()
        .expect("positive validated outer iteration count");
    Err(GammaValenceHfError::NotConverged {
        iterations: diagnostics.len(),
        energy_change: last
            .energy_change
            .map(Hartree::get)
            .unwrap_or(f64::INFINITY),
        density_rms: last.regional_density_rms,
    })
}

/// Run scalar Koelling–Harmon HF to convergence, then apply conventional SOC
/// second variation to that converged nonlocal Fock spectrum.
pub fn run_kh_soc_valence_hf(
    physics: &mut CheckpointPhysics,
    spec: &KhSocValenceHfSpec,
) -> Result<KhSocValenceHfResult, KhSocValenceHfError> {
    let ScfRelativity::SocSecondVariation { window } = spec.config.relativity else {
        return Err(KhSocValenceHfError::Relativity);
    };
    if !matches!(
        spec.fock_mixing,
        FockMixing::Linear { .. } | FockMixing::PulayAnderson { .. }
    ) {
        return Err(KhSocValenceHfError::FockMixing);
    }
    let valence_electrons = validate_kh_soc_spec(spec)?;
    Ok(run_kh_soc_valence_hf_inner(
        physics,
        spec,
        window,
        valence_electrons,
    )?)
}

fn run_kh_soc_valence_hf_inner(
    physics: &mut CheckpointPhysics,
    spec: &KhSocValenceHfSpec,
    window: muffintin_dft::FirstVariationWindow,
    valence_electrons: f64,
) -> Result<KhSocValenceHfResult, GammaValenceHfError> {
    let _ = ScfLoop::new(spec.config.clone(), None)?;
    let k_fractional = muffintin_dft::regular_k_points(spec.config.k_mesh)?;
    let q_fractional = canonical_q_points(&k_fractional).map_err(q_topology_error)?;
    let mut mixer = density_mixer(spec.config.mixing)?;
    let initial = physics.kernel.initial_density_components(&spec.config)?;
    let mut valence_density = initial.valence;
    let mut core_density = initial.core;
    let mut total_density = initial.total;
    let frozen_core = if spec.core_treatment == KhSocCoreTreatment::Frozen {
        let core = physics
            .kernel
            .frozen_checkpoint_core(&total_density, &spec.config)?;
        core_density = core.density;
        total_density = sum_kh_density(&valence_density, &core_density)?;
        Some(core.orbitals)
    } else {
        None
    };
    let mut previous_total = None;
    let mut diagnostics = Vec::with_capacity(spec.config.convergence.max_iterations);
    let mut total_exchange_rebuilds = 0;

    for outer_iteration in 1..=spec.config.convergence.max_iterations {
        let electrostatic = evaluate_regional_electrostatics(
            total_density.charge(),
            &ElectrostaticSpec::new(
                WeinertHartreeSpec::electronic(4)?,
                physics.nuclear_charges().to_vec(),
            )?,
        )?;
        let zero = electrostatic.potential.zero_like();
        let potential = RegionalPotential::new(
            electrostatic.potential.clone(),
            [zero.clone(), zero.clone(), zero],
        )?;
        let (bands, _) = solve_h0_bands(
            physics,
            &spec.config,
            &potential,
            &k_fractional,
            valence_electrons,
            total_density.charge().interstitial().layout(),
            ScfRelativity::Scalar,
        )?;
        let core_sidecars = frozen_core.as_deref().unwrap_or(&[]);
        let fixed = solve_scalar_fixed_potential(
            physics,
            spec,
            bands,
            valence_electrons,
            outer_iteration,
            &k_fractional,
            &q_fractional,
            core_sidecars,
        )?;
        total_exchange_rebuilds += fixed.exchange_rebuilds;
        let valence_output = physics
            .kernel
            .synthesize_bands(&fixed.bands, &fixed.occupation.values)?;
        let total_output = sum_kh_density(&valence_output, &core_density)?;
        let valence_density_rms = valence_density.difference_rms(&valence_output)?;
        let density_rms = total_density.difference_rms(&total_output)?;
        let core_terms = frozen_core_energy_terms(core_sidecars)?;
        let energy = kh_soc_energy_diagnostic(
            &fixed.bands,
            &fixed.occupation,
            &fixed.exchange,
            &fixed.core_feedback,
            &electrostatic,
            core_terms,
        )?;
        require_gate(
            "scalar valence electron count",
            (electron_count(&valence_output)? - valence_electrons).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        require_gate(
            "scalar total electron count",
            (electron_count(&total_output)? - spec.config.electron_count).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        require_gate(
            "scalar exchange energy identity",
            energy.exchange_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        require_gate(
            "scalar eigenvalue identity",
            energy.eigenvalue_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        require_gate(
            "scalar HF total-energy identity",
            energy.total_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        let energy_change = previous_total
            .map(|previous: Hartree| Hartree((energy.total.get() - previous.get()).abs()));
        diagnostics.push(KhSocValenceHfIterationDiagnostic {
            iteration: outer_iteration,
            fock_iterations: fixed.fock_iterations,
            exchange_rebuilds: fixed.exchange_rebuilds,
            exchange_energy: Hartree(
                fixed.exchange.up.exchange_energy.get() + fixed.exchange.down.exchange_energy.get(),
            ),
            fock_fixed_point_residual: fixed.fixed_point_residual,
            fock_feedback_residual: Hartree(fixed.feedback_residual),
            valence_density_rms,
            regional_density_rms: density_rms,
            total_energy: energy.total,
            energy_change,
        });
        let converged = energy_change
            .is_some_and(|change| change.get() <= spec.config.convergence.energy_tolerance.get())
            && density_rms <= spec.config.convergence.density_tolerance
            && fixed.fixed_point_residual <= spec.fock_density_tolerance;
        if converged {
            let bands = physics.kernel.apply_soc_second_variation(
                &potential,
                &fixed.bands,
                window,
                core_sidecars,
            )?;
            let occupation =
                solve_occupations(bands.states(), valence_electrons, spec.config.occupations)?;
            let second_variation_valence_density = physics
                .kernel
                .synthesize_bands(&bands, &occupation.values)?;
            let second_variation_density_rms =
                valence_output.difference_rms(&second_variation_valence_density)?;
            let second_variation_total_density =
                sum_kh_density(&second_variation_valence_density, &core_density)?;
            let occupations = second_variation_occupation_rows(&occupation.values, &bands)?;
            let second_variation_exchange = rebuild_second_variation_exchange(
                physics,
                spec,
                &bands,
                &occupations,
                &k_fractional,
                &q_fractional,
            )?;
            total_exchange_rebuilds += 1;
            let scalar_exchange_energy =
                fixed.exchange.up.exchange_energy.get() + fixed.exchange.down.exchange_energy.get();
            let exchange_energy_change = Hartree(
                (second_variation_exchange.exchange_energy.get() - scalar_exchange_energy).abs(),
            );
            let k_weights = scalar_k_weights(&fixed.bands)?;
            return Ok(KhSocValenceHfResult {
                valence_density: second_variation_valence_density,
                core_density,
                total_density: second_variation_total_density,
                potential,
                scalar_bands: fixed.bands,
                orbital_energies: second_variation_energies(&bands)?,
                bands,
                occupations,
                scalar_exchange: fixed.exchange,
                second_variation_exchange,
                core_orbitals: core_sidecars.to_vec(),
                core_h0_trace: core_terms.h0,
                core_core_exchange: Hartree(0.5 * core_terms.cc_trace.get()),
                core_valence_exchange: energy.core_valence_exchange,
                total_energy: energy.total,
                fock_fixed_point_residual: fixed.fixed_point_residual,
                fock_feedback_residual: Hartree(fixed.feedback_residual),
                valence_density_rms,
                regional_density_rms: density_rms,
                second_variation_density_rms,
                exchange_energy_change,
                exchange_rebuilds: total_exchange_rebuilds,
                k_fractional,
                q_fractional,
                k_weights,
                diagnostics,
            });
        }
        previous_total = Some(energy.total);
        let mixed_total = mixer.mix(&total_density, &total_output)?.density;
        valence_density = mixed_total.difference(&core_density)?;
        total_density = mixed_total;
    }
    let last = diagnostics
        .last()
        .expect("positive validated outer iteration count");
    Err(GammaValenceHfError::NotConverged {
        iterations: diagnostics.len(),
        energy_change: last
            .energy_change
            .map(Hartree::get)
            .unwrap_or(f64::INFINITY),
        density_rms: last.regional_density_rms,
    })
}

/// Run the strict Gamma molecule-in-box wrapper for KH+SOC valence HF.
pub fn run_gamma_kh_soc_valence_hf(
    physics: &mut CheckpointPhysics,
    spec: &KhSocValenceHfSpec,
) -> Result<KhSocValenceHfResult, KhSocValenceHfError> {
    let mesh = spec.config.k_mesh;
    if mesh.divisions != [1, 1, 1]
        || mesh.shift != [0.0; 3]
        || mesh.reduction != ScfKReduction::Full
    {
        return Err(KhSocValenceHfError::Hf(GammaValenceHfError::GammaMesh));
    }
    run_kh_soc_valence_hf(physics, spec)
}

/// Run the strict molecule-in-box setup through the generic relaxed-core mesh loop.
pub fn run_gamma_relaxed_core_hf(
    physics: &mut CheckpointPhysics,
    spec: &RelaxedCoreHfSpec,
) -> Result<RelaxedCoreHfResult, RelaxedCoreHfError> {
    let mesh = spec.config.k_mesh;
    if mesh.divisions != [1, 1, 1]
        || mesh.shift != [0.0; 3]
        || mesh.reduction != ScfKReduction::Full
    {
        return Err(RelaxedCoreHfError::GammaMesh);
    }
    if spec.gamma != GammaExchangeTreatment::FiniteBody {
        return Err(RelaxedCoreHfError::GammaFiniteBody);
    }
    run_relaxed_core_hf(physics, spec)
}

/// Run unified relaxed-core spinor HF on one explicit full regular BZ mesh.
///
/// The outer state retains valence, core, and total densities separately.
/// The complete total density enters the configured mixer. Core orbitals are
/// still solved freshly at every outer step; after mixing, their fresh density
/// is subtracted from the mixed total density to recover the next valence input.
pub fn run_relaxed_core_hf(
    physics: &mut CheckpointPhysics,
    spec: &RelaxedCoreHfSpec,
) -> Result<RelaxedCoreHfResult, RelaxedCoreHfError> {
    let valence_electrons = validate_relaxed_core_spec(spec)?;
    let _ = ScfLoop::new(spec.config.clone(), None)?;
    let k_fractional = muffintin_dft::regular_k_points(spec.config.k_mesh)?;
    let q_fractional =
        canonical_q_points(&k_fractional).map_err(|_| RelaxedCoreHfError::QTopology)?;
    let mut mixer = density_mixer(spec.config.mixing)?;
    let initial = physics.kernel.initial_density_components(&spec.config)?;
    let mut valence_density = initial.valence;
    let mut core_density = initial.core;
    let mut total_density = initial.total;
    let expected_core_electrons = spec.config.electron_count - valence_electrons;
    let mut previous_total = None;
    let mut diagnostics = Vec::with_capacity(spec.config.convergence.max_iterations);
    let mut total_exchange_rebuilds = 0;

    for outer_iteration in 1..=spec.config.convergence.max_iterations {
        let electrostatic = evaluate_regional_electrostatics(
            total_density.charge(),
            &ElectrostaticSpec::new(
                WeinertHartreeSpec::electronic(4)?,
                physics.nuclear_charges().to_vec(),
            )?,
        )?;
        let zero = electrostatic.potential.zero_like();
        let potential = RegionalPotential::new(
            electrostatic.potential.clone(),
            [zero.clone(), zero.clone(), zero],
        )?;
        let (bands, provisional_occupation) = solve_h0_bands(
            physics,
            &spec.config,
            &potential,
            &k_fractional,
            valence_electrons,
            total_density.charge().interstitial().layout(),
            ScfRelativity::SpinorFirstVariation,
        )?;

        let bootstrap = physics.kernel.bootstrap_hf_core(
            &total_density,
            &electrostatic,
            &spec.config.core_sites,
        )?;
        let bootstrap_inputs = build_q_inputs_with_cores(
            physics,
            &bands,
            &k_fractional,
            &q_fractional,
            &bootstrap.orbitals,
        )?;
        let bootstrap_sector_occupations = sector_occupations(
            &bands,
            &provisional_occupation,
            &bootstrap_inputs,
            spec.gamma,
        )?;
        let frozen_valence =
            build_frozen_site_valence_densities(&bootstrap_inputs, &bootstrap_sector_occupations)?;
        let core_relaxations = bootstrap
            .orbitals
            .iter()
            .map(|sidecar| {
                relax_frozen_core_at_fixed_potential(&frozen_valence, sidecar, spec.core)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let fresh_sidecars = core_relaxations
            .iter()
            .map(|result| result.orbitals.clone())
            .collect::<Vec<_>>();
        let fresh_core_density = synthesize_core_density(&total_density, &fresh_sidecars)?;

        let fixed = solve_relaxed_fixed_potential(
            physics,
            spec,
            bands,
            valence_electrons,
            outer_iteration,
            &k_fractional,
            &q_fractional,
            &fresh_sidecars,
        )?;
        total_exchange_rebuilds += fixed.exchange_rebuilds;
        let valence_output = physics
            .kernel
            .synthesize_bands(&fixed.bands, &fixed.occupation.values)?;
        let total_output = sum_density(&valence_output, &fresh_core_density)?;
        let valence_density_rms = valence_density.difference_rms(&valence_output)?;
        let total_density_rms = total_density.difference_rms(&total_output)?;
        let fresh_core_replacement_rms = core_density.difference_rms(&fresh_core_density)?;
        let output_valence_electrons = electron_count(&valence_output)?;
        let output_core_electrons = electron_count(&fresh_core_density)?;
        let output_total_electrons = electron_count(&total_output)?;
        require_relaxed_gate(
            "valence electron count",
            (output_valence_electrons - valence_electrons).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        require_relaxed_gate(
            "core electron count",
            (output_core_electrons - expected_core_electrons).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        require_relaxed_gate(
            "total electron count",
            (output_total_electrons - spec.config.electron_count).abs(),
            ELECTRON_COUNT_TOLERANCE,
        )?;
        let final_valence_densities = build_frozen_site_valence_densities(
            &fixed.exchange.inputs,
            &fixed.exchange.occupations,
        )?;
        let final_core_valence = build_frozen_core_valence_exchange(
            &fixed.exchange.inputs,
            &fixed.exchange.core_mpb,
            &spec.coulomb,
            &fixed.exchange.occupations,
        )?;
        let core_valence_comparison = compare_frozen_core_valence(
            &final_core_valence,
            &final_valence_densities,
            CoreValenceComparisonSpec {
                numerical_tolerance: spec.sector_numerical_tolerance,
                maximum_shell_spill: spec.maximum_core_shell_spill,
            },
        )?;
        let core_one_body_traces = fresh_sidecars
            .iter()
            .map(core_local_one_body_trace)
            .collect::<Result<Vec<_>, _>>()?;
        let core_h0_trace = Hartree(
            core_one_body_traces
                .iter()
                .map(|trace| trace.total.get())
                .sum(),
        );
        let energy = relaxed_energy_diagnostic(
            &fixed.bands,
            &fixed.occupation,
            &fixed.exchange.exchange,
            &electrostatic,
            core_h0_trace,
        )?;
        require_relaxed_gate(
            "valence eigenvalue identity",
            energy.valence_eigenvalue_identity_residual,
            spec.fock_feedback_tolerance.get(),
        )?;
        let energy_change = previous_total
            .map(|previous: Hartree| Hartree((energy.total.get() - previous.get()).abs()));
        let converged = energy_change
            .is_some_and(|change| change.get() <= spec.config.convergence.energy_tolerance.get())
            && total_density_rms <= spec.config.convergence.density_tolerance
            && fixed.fixed_point_residual <= spec.fock_density_tolerance;

        let (core_maximum_energy_change, core_maximum_radial_residual) =
            maximum_core_residuals(&core_relaxations);
        let mut next_valence = None;
        let mut next_total = None;
        if !converged {
            let mixed_total = mixer.mix(&total_density, &total_output)?.density;
            let mixed_valence = mixed_total.difference(&fresh_core_density)?;
            next_valence = Some(mixed_valence);
            next_total = Some(mixed_total);
        }
        let exchange = &fixed.exchange.exchange;
        diagnostics.push(RelaxedCoreHfIterationDiagnostic {
            iteration: outer_iteration,
            fock_iterations: fixed.fock_iterations,
            exchange_rebuilds: fixed.exchange_rebuilds,
            trace_vv: exchange.vv.trace,
            trace_cv: exchange.cv.trace,
            trace_vc: exchange.vc.trace,
            trace_cc: exchange.cc.trace,
            exchange_vv: exchange.exchange_vv,
            exchange_cv: exchange.exchange_cv,
            exchange_cc: exchange.exchange_cc,
            exchange_total: exchange.exchange_total,
            cv_vc_trace_mismatch: exchange.cross_trace_mismatch,
            valence_feedback_vv_cv_trace: Hartree(
                exchange.vv.trace.get() + exchange.cv.trace.get(),
            ),
            maximum_antihermitian_residual: maximum_sector_antihermitian(exchange),
            fock_fixed_point_residual: fixed.fixed_point_residual,
            fock_feedback_residual: Hartree(fixed.feedback_residual),
            valence_density_rms,
            total_density_rms,
            valence_electron_count: output_valence_electrons,
            core_electron_count: output_core_electrons,
            total_electron_count: output_total_electrons,
            core_inner_iterations: core_relaxations
                .iter()
                .map(|result| result.diagnostics.len())
                .collect(),
            core_maximum_energy_change,
            core_maximum_radial_residual,
            core_h0_trace,
            total_energy: energy.total,
            energy_change,
            valence_eigenvalue_identity_residual: energy.valence_eigenvalue_identity_residual,
            lifting_identity_residual: fixed.lifting_identity_residual,
            first_global_solve_identity_residual: fixed.first_global_solve_identity_residual,
            fresh_core_replacement_rms,
            delta_c: core_valence_comparison.deltas.clone(),
            weighted_delta_closure_residual: core_valence_comparison
                .weighted_delta_closure_residual,
        });
        if converged {
            let orbital_energies = spinor_energies_relaxed(&fixed.bands)?;
            let occupations = occupation_rows_relaxed(&fixed.occupation.values, &fixed.bands)?;
            let k_weights = k_weights_relaxed(&fixed.bands)?;
            let maximum_antihermitian_residual = maximum_sector_antihermitian(exchange);
            return Ok(RelaxedCoreHfResult {
                valence_density: valence_output,
                core_density: fresh_core_density,
                total_density: total_output,
                potential,
                bands: fixed.bands,
                occupations,
                orbital_energies,
                core_orbitals: fresh_sidecars,
                core_one_body_traces,
                final_exchange_inputs: fixed.exchange.inputs,
                sector_exchange: fixed.exchange.exchange,
                core_valence_comparison,
                core_h0_trace,
                total_energy: energy.total,
                maximum_antihermitian_residual,
                fock_fixed_point_residual: fixed.fixed_point_residual,
                fock_feedback_residual: Hartree(fixed.feedback_residual),
                valence_density_rms,
                total_density_rms,
                exchange_rebuilds: total_exchange_rebuilds,
                k_fractional,
                q_fractional,
                k_weights,
                diagnostics,
            });
        }
        previous_total = Some(energy.total);
        valence_density = next_valence.expect("nonconverged step mixes valence density");
        core_density = fresh_core_density;
        total_density = next_total.expect("nonconverged step assembles total density");
    }

    let last = diagnostics
        .last()
        .expect("positive validated outer iteration count");
    Err(RelaxedCoreHfError::NotConverged {
        iterations: diagnostics.len(),
        energy_change: last
            .energy_change
            .map(Hartree::get)
            .unwrap_or(f64::INFINITY),
        density_rms: last.total_density_rms,
    })
}

#[derive(Clone)]
struct OccupationSolution {
    chemical_potential: Hartree,
    values: Vec<f64>,
    band_energy: Hartree,
    correction: Hartree,
}

fn solve_occupations(
    states: &[BandState],
    electron_count: f64,
    spec: ScfOccupations,
) -> Result<OccupationSolution, OccupationError> {
    Ok(match spec {
        ScfOccupations::FermiDirac { temperature } => {
            let solved = muffintin_dft::solve_fermi_dirac(
                states,
                electron_count,
                temperature,
                1.0e-12,
                256,
            )?;
            OccupationSolution {
                chemical_potential: solved.chemical_potential,
                values: solved.occupations,
                band_energy: solved.band_energy,
                correction: solved.minus_temperature_entropy,
            }
        }
        ScfOccupations::Gaussian { width } => {
            let solved =
                muffintin_dft::solve_gaussian(states, electron_count, width, 1.0e-12, 256)?;
            OccupationSolution {
                chemical_potential: solved.chemical_potential,
                values: solved.occupations,
                band_energy: solved.band_energy,
                correction: solved.smearing_correction,
            }
        }
    })
}

fn solve_h0_bands(
    physics: &mut CheckpointPhysics,
    config: &ScfConfig,
    potential: &RegionalPotential,
    k_fractional: &[[f64; 3]],
    electron_count: f64,
    density_layout: &FourierLayout,
    relativity: ScfRelativity,
) -> Result<(CheckpointBandSolution, OccupationSolution), GammaValenceHfError> {
    let mut one_particle = physics
        .kernel
        .materialize_checkpoint_one_particle(potential, &config.basis)?;
    let mut bands = physics.kernel.solve_points(
        one_particle.potential(),
        one_particle.basis(),
        k_fractional,
        relativity,
    )?;
    let mut occupation = solve_occupations(bands.states(), electron_count, config.occupations)?;
    for pass in 1..=SPECTRAL_REFINEMENT_PASSES {
        let Some(refined) = physics.kernel.refine_spectral_basis(
            &config.basis,
            &one_particle,
            &bands,
            &occupation.values,
            occupation.chemical_potential,
            relativity,
        )?
        else {
            break;
        };
        if pass == SPECTRAL_REFINEMENT_PASSES {
            return Err(GammaValenceHfError::SpectralRefinement { passes: pass });
        }
        one_particle = refined;
        bands = physics.kernel.solve_points(
            one_particle.potential(),
            one_particle.basis(),
            k_fractional,
            relativity,
        )?;
        occupation = solve_occupations(bands.states(), electron_count, config.occupations)?;
    }
    bands.set_density_layout(density_layout.clone());
    Ok((bands, occupation))
}

#[derive(Clone)]
struct RelaxedSectorFrame {
    inputs: Vec<SpinorProductInput>,
    core_mpb: Vec<SpinorExchangeMpbResult>,
    occupations: SectorOccupations,
    exchange: FrozenSpinorSectorExchange,
}

#[derive(Clone)]
struct RelaxedFeedbackFrame {
    inputs: Vec<SpinorProductInput>,
    occupations: SectorOccupations,
    exchange: FrozenSpinorValenceFeedbackExchange,
}

struct RelaxedExchangeCache {
    vv_bases: Vec<SpinorMpbBasis>,
    vv_operators: Vec<CoulombOperator>,
    core_bases: Vec<SpinorExchangeMpbBasis>,
    core_operators: Vec<CoulombOperator>,
}

struct RelaxedFixedPotentialResult {
    bands: CheckpointBandSolution,
    occupation: OccupationSolution,
    exchange: RelaxedSectorFrame,
    fock_iterations: usize,
    exchange_rebuilds: usize,
    fixed_point_residual: f64,
    feedback_residual: f64,
    lifting_identity_residual: f64,
    first_global_solve_identity_residual: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
fn solve_relaxed_fixed_potential(
    physics: &CheckpointPhysics,
    spec: &RelaxedCoreHfSpec,
    mut bands: CheckpointBandSolution,
    valence_electrons: f64,
    outer_iteration: usize,
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
    core_sidecars: &[CoreShellOrbitals],
) -> Result<RelaxedFixedPotentialResult, RelaxedCoreHfError> {
    let mut rebuilds = 0;
    let mut first_global_solve_identity_residual = None;
    let mut last_residual = f64::INFINITY;
    let mut last_feedback_residual = f64::INFINITY;
    let mut previous_global_feedback = None;
    let mut feedback_mixer = FeedbackMixer::new(spec.fock_mixing);
    // The next step's input bands and occupations are this step's solved ones,
    // so its input density is the density already synthesized here.
    let mut current_density = None;
    let mut exchange_cache = None;

    for fock_iteration in 1..=spec.max_fock_iterations {
        let occupation =
            solve_occupations(bands.states(), valence_electrons, spec.config.occupations)?;
        let occupation_rows = occupation_rows(&occupation.values, &bands)?;
        if outer_iteration == 1 && fock_iteration == 1 {
            let driver = rebuild_relaxed_feedback_frame(
                physics,
                spec,
                &bands,
                &occupation,
                k_fractional,
                q_fractional,
                core_sidecars,
                &mut exchange_cache,
            )?;
            rebuilds += 1;
            let band_feedback = relaxed_valence_feedback(&driver.exchange)?;
            let global_feedback = lift_global_feedback(&bands, &band_feedback)?;
            let lifting_identity_residual =
                lifting_identity(&bands, &band_feedback, &global_feedback)?;
            require_relaxed_gate(
                "band-feedback lifting",
                lifting_identity_residual,
                IDENTITY_TOLERANCE,
            )?;
            let solved = bands.solve_spinor_global_feedback(&global_feedback)?;
            let solve_identity = first_global_solve_identity(&bands, &band_feedback, &solved)?;
            require_relaxed_gate(
                "first global generalized solve",
                solve_identity,
                IDENTITY_TOLERANCE,
            )?;
            first_global_solve_identity_residual = Some(solve_identity);
            let solved_occupation =
                solve_occupations(solved.states(), valence_electrons, spec.config.occupations)?;
            let (residual, solved_density) = fixed_point_density_residual(
                physics,
                &bands,
                &occupation.values,
                current_density.take(),
                &solved,
                &solved_occupation.values,
            )?;
            last_residual = residual;
            current_density = Some(solved_density);
            bands = solved;
            previous_global_feedback = Some(global_feedback);
            if fock_iteration == spec.max_fock_iterations {
                return Err(RelaxedCoreHfError::FockNotConverged {
                    outer_iteration,
                    iterations: fock_iteration,
                    density_residual: last_residual,
                    feedback_residual: last_feedback_residual,
                });
            }
            continue;
        }

        let rebuilt = rebuild_relaxed_feedback_frame(
            physics,
            spec,
            &bands,
            &occupation,
            k_fractional,
            q_fractional,
            core_sidecars,
            &mut exchange_cache,
        )?;
        rebuilds += 1;
        let fresh_band_feedback = relaxed_valence_feedback(&rebuilt.exchange)?;
        let fresh_global_feedback = lift_global_feedback(&bands, &fresh_band_feedback)?;
        let feedback_fixed_residual = previous_global_feedback
            .as_ref()
            .map(|previous| global_feedback_difference(previous, &fresh_global_feedback))
            .transpose()?
            .unwrap_or(f64::INFINITY);
        last_feedback_residual = feedback_fixed_residual;
        let global_feedback = match &previous_global_feedback {
            Some(previous) => {
                feedback_mixer.mix(&bands, &occupation_rows, previous, &fresh_global_feedback)?
            }
            None => fresh_global_feedback.clone(),
        };
        let lifting_identity_residual =
            lifting_identity(&bands, &fresh_band_feedback, &fresh_global_feedback)?;
        require_relaxed_gate(
            "band-feedback lifting",
            lifting_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        let solved = bands.solve_spinor_global_feedback(&global_feedback)?;
        let solved_occupation =
            solve_occupations(solved.states(), valence_electrons, spec.config.occupations)?;
        let (residual, solved_density) = fixed_point_density_residual(
            physics,
            &bands,
            &occupation.values,
            current_density.take(),
            &solved,
            &solved_occupation.values,
        )?;
        last_residual = residual;
        current_density = Some(solved_density);
        if last_residual <= spec.fock_density_tolerance
            && feedback_fixed_residual <= spec.fock_feedback_tolerance.get()
        {
            let exchange = complete_relaxed_sector_frame(spec, rebuilt)?;
            return Ok(RelaxedFixedPotentialResult {
                bands,
                occupation,
                exchange,
                fock_iterations: fock_iteration,
                exchange_rebuilds: rebuilds,
                fixed_point_residual: last_residual,
                feedback_residual: feedback_fixed_residual,
                lifting_identity_residual,
                first_global_solve_identity_residual,
            });
        }
        bands = solved;
        previous_global_feedback = Some(global_feedback);
    }
    Err(RelaxedCoreHfError::FockNotConverged {
        outer_iteration,
        iterations: spec.max_fock_iterations,
        density_residual: last_residual,
        feedback_residual: last_feedback_residual,
    })
}

fn rebuild_relaxed_feedback_frame(
    physics: &CheckpointPhysics,
    spec: &RelaxedCoreHfSpec,
    bands: &CheckpointBandSolution,
    occupation: &OccupationSolution,
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
    core_sidecars: &[CoreShellOrbitals],
    cache: &mut Option<RelaxedExchangeCache>,
) -> Result<RelaxedFeedbackFrame, RelaxedCoreHfError> {
    let inputs =
        build_q_inputs_with_cores(physics, bands, k_fractional, q_fractional, core_sidecars)?;
    let first = inputs.first().ok_or(RelaxedCoreHfError::QTopology)?;
    let n_k = first.pair_columns.n_k;
    let n_orb = first.pair_columns.n_orb;
    let occupations = sector_occupations(bands, occupation, &inputs, spec.gamma)?;
    let occupied_bands = (0..n_orb)
        .filter(|&band| occupations.valence.iter().any(|row| row[band] != 0.0))
        .collect::<Vec<_>>();
    let mut selections = Vec::with_capacity(n_k * occupied_bands.len() * n_orb);
    for k in 0..n_k {
        for &left_band in &occupied_bands {
            for right_band in 0..n_orb {
                selections.push(SpinorMpbSelection {
                    k,
                    left_band,
                    right_band,
                });
            }
        }
    }
    let vv_spec = SpinorMpbSpec {
        product_l_max: spec.product_l_max,
        product_g_max: spec.product_g_max,
        overlap_tolerance: spec.overlap_tolerance,
        selections,
    };
    let core_spec = SpinorExchangeMpbSpec {
        product_l_max: spec.product_l_max,
        product_g_max: spec.product_g_max,
        overlap_tolerance: spec.overlap_tolerance,
    };
    if cache.is_none() {
        let vv_bases = inputs
            .iter()
            .map(|input| compile_spinor_mpb_basis(input, &vv_spec))
            .collect::<Result<Vec<_>, _>>()?;
        let vv_operators = vv_bases
            .iter()
            .map(|basis| assemble_coulomb(&basis.auxiliary, &spec.coulomb))
            .collect::<Result<Vec<_>, _>>()
            .map_err(FrozenSpinorSectorExchangeError::from)?;
        let core_bases = inputs
            .iter()
            .map(|input| compile_spinor_exchange_mpb_basis(input, &core_spec))
            .collect::<Result<Vec<_>, _>>()?;
        let core_operators = core_bases
            .iter()
            .map(|basis| assemble_coulomb(&basis.auxiliary, &spec.coulomb))
            .collect::<Result<Vec<_>, _>>()
            .map_err(FrozenSpinorSectorExchangeError::from)?;
        *cache = Some(RelaxedExchangeCache {
            vv_bases,
            vv_operators,
            core_bases,
            core_operators,
        });
    }
    let cache = cache
        .as_ref()
        .expect("the fixed-potential exchange cache was just initialized");
    let vv_mpb = inputs
        .iter()
        .zip(&cache.vv_bases)
        .map(|(input, basis)| build_spinor_mpb_from_basis(input, &vv_spec, basis))
        .collect::<Result<Vec<_>, _>>()?;
    let core_mpb = inputs
        .iter()
        .zip(&cache.core_bases)
        .map(|(input, basis)| build_spinor_exchange_feedback_from_basis(input, &core_spec, basis))
        .collect::<Result<Vec<_>, _>>()?;
    let exchange = build_cached_spinor_valence_feedback_exchange(
        &inputs,
        &vv_mpb,
        &cache.vv_operators,
        &occupied_bands,
        &core_mpb,
        &cache.core_operators,
        &spec.coulomb,
        &occupations,
    )?;
    Ok(RelaxedFeedbackFrame {
        inputs,
        occupations,
        exchange,
    })
}

fn complete_relaxed_sector_frame(
    spec: &RelaxedCoreHfSpec,
    feedback: RelaxedFeedbackFrame,
) -> Result<RelaxedSectorFrame, RelaxedCoreHfError> {
    let core_mpb = feedback
        .inputs
        .iter()
        .map(|input| {
            build_spinor_exchange_mpb(
                input,
                &SpinorExchangeMpbSpec {
                    product_l_max: spec.product_l_max,
                    product_g_max: spec.product_g_max,
                    overlap_tolerance: spec.overlap_tolerance,
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let exchange = complete_frozen_spinor_sector_exchange(
        &feedback.inputs,
        &core_mpb,
        &spec.coulomb,
        &feedback.occupations,
        feedback.exchange.vv,
    )?;
    Ok(RelaxedSectorFrame {
        inputs: feedback.inputs,
        core_mpb,
        occupations: feedback.occupations,
        exchange,
    })
}

fn build_q_inputs_with_cores(
    physics: &CheckpointPhysics,
    bands: &CheckpointBandSolution,
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
    core_sidecars: &[CoreShellOrbitals],
) -> Result<Vec<SpinorProductInput>, RelaxedCoreHfError> {
    q_fractional
        .iter()
        .map(|&q| {
            Ok(physics
                .spinor_product_input_from_bands(bands, k_fractional, q)?
                .replace_core_sidecars(core_sidecars)?)
        })
        .collect()
}

fn sector_occupations(
    bands: &CheckpointBandSolution,
    occupation: &OccupationSolution,
    inputs: &[SpinorProductInput],
    gamma: GammaExchangeTreatment,
) -> Result<SectorOccupations, RelaxedCoreHfError> {
    let first = inputs.first().ok_or(RelaxedCoreHfError::QTopology)?;
    Ok(SectorOccupations {
        k_weights: k_weights_relaxed(bands)?,
        valence: occupation_rows_relaxed(&occupation.values, bands)?,
        core: first
            .core
            .orbitals
            .iter()
            .map(|orbital| orbital.occupation)
            .collect(),
        gamma,
    })
}

fn relaxed_valence_feedback(
    exchange: &FrozenSpinorValenceFeedbackExchange,
) -> Result<Vec<DenseHermitianMatrix>, RelaxedCoreHfError> {
    if exchange.vv.target_matrices.len() != exchange.cv.target_matrices.len() {
        return Err(RelaxedCoreHfError::ExchangeKIndex {
            expected: exchange.vv.target_matrices.len(),
            actual: exchange.cv.target_matrices.len(),
        });
    }
    exchange
        .vv
        .target_matrices
        .iter()
        .zip(&exchange.cv.target_matrices)
        .enumerate()
        .map(|(k, (vv, cv))| {
            if vv.k_index() != k || cv.k_index() != k {
                return Err(RelaxedCoreHfError::ExchangeKIndex {
                    expected: k,
                    actual: if vv.k_index() != k {
                        vv.k_index()
                    } else {
                        cv.k_index()
                    },
                });
            }
            if vv.n_bands() != cv.n_bands() {
                return Err(RelaxedCoreHfError::ExchangeKIndex {
                    expected: vv.n_bands(),
                    actual: cv.n_bands(),
                });
            }
            Ok(DenseHermitianMatrix::from_upper_triangle(
                vv.n_bands(),
                Axis::Band,
                |row, column| {
                    vv.element(row, column).expect("validated VV band index")
                        + cv.element(row, column).expect("validated CV band index")
                },
            )?)
        })
        .collect()
}

struct FixedPotentialResult {
    bands: CheckpointBandSolution,
    occupation: OccupationSolution,
    exchange: IsdfExchangeResult,
    fock_iterations: usize,
    exchange_rebuilds: usize,
    fixed_point_residual: f64,
    feedback_residual: f64,
    lifting_identity_residual: f64,
    first_global_solve_identity_residual: Option<f64>,
    first_one_shot_exchange: Option<IsdfExchangeResult>,
}

struct ScalarFixedPotentialResult {
    bands: CheckpointBandSolution,
    occupation: OccupationSolution,
    exchange: Collinear<IsdfExchangeResult>,
    core_feedback: Vec<Collinear<DenseHermitianMatrix>>,
    fock_iterations: usize,
    exchange_rebuilds: usize,
    fixed_point_residual: f64,
    feedback_residual: f64,
}

#[allow(clippy::too_many_arguments)]
fn solve_scalar_fixed_potential(
    physics: &CheckpointPhysics,
    spec: &KhSocValenceHfSpec,
    mut bands: CheckpointBandSolution,
    valence_electrons: f64,
    outer_iteration: usize,
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
    core_sidecars: &[CoreShellOrbitals],
) -> Result<ScalarFixedPotentialResult, GammaValenceHfError> {
    let mut rebuilds = 0;
    let mut last_residual = f64::INFINITY;
    let mut last_feedback_residual = f64::INFINITY;
    let mut previous_global_feedback: Option<Collinear<Vec<DenseHermitianMatrix>>> = None;
    let mut up_mixer = FeedbackMixer::new(spec.fock_mixing);
    let mut down_mixer = FeedbackMixer::new(spec.fock_mixing);
    let mut current_density = None;
    let core_feedback = physics
        .kernel
        .scalar_static_core_exchange_feedback(&bands, core_sidecars)?;
    for fock_iteration in 1..=spec.max_fock_iterations {
        let occupation =
            solve_occupations(bands.states(), valence_electrons, spec.config.occupations)?;
        let occupation_rows = scalar_occupation_rows(&occupation.values, &bands)?;
        let rebuilt = rebuild_scalar_exchange(
            physics,
            spec,
            &bands,
            &occupation_rows,
            k_fractional,
            q_fractional,
        )?;
        rebuilds += 1;
        let fresh_band_feedback = scalar_exchange_feedback(&rebuilt)?;
        let valence_global_feedback = lift_scalar_global_feedback(&bands, &fresh_band_feedback)?;
        let fresh_global_feedback =
            add_scalar_core_feedback(valence_global_feedback, &core_feedback)?;
        let feedback_residual = previous_global_feedback
            .as_ref()
            .map(|previous| scalar_feedback_difference(previous, &fresh_global_feedback))
            .transpose()?
            .unwrap_or(f64::INFINITY);
        last_feedback_residual = feedback_residual;
        let global_feedback = match &previous_global_feedback {
            Some(previous) => {
                let up_occupations = occupation_rows
                    .iter()
                    .map(|row| row.up.clone())
                    .collect::<Vec<_>>();
                let down_occupations = occupation_rows
                    .iter()
                    .map(|row| row.down.clone())
                    .collect::<Vec<_>>();
                Collinear::new(
                    up_mixer.mix(
                        &bands,
                        &up_occupations,
                        &previous.up,
                        &fresh_global_feedback.up,
                    )?,
                    down_mixer.mix(
                        &bands,
                        &down_occupations,
                        &previous.down,
                        &fresh_global_feedback.down,
                    )?,
                )
            }
            None => fresh_global_feedback.clone(),
        };
        let solved = bands.solve_scalar_global_feedback(
            &global_feedback
                .up
                .iter()
                .cloned()
                .zip(global_feedback.down.iter().cloned())
                .map(|(up, down)| Collinear::new(up, down))
                .collect::<Vec<_>>(),
        )?;
        let solved_occupation =
            solve_occupations(solved.states(), valence_electrons, spec.config.occupations)?;
        let (residual, solved_density) = fixed_point_density_residual(
            physics,
            &bands,
            &occupation.values,
            current_density.take(),
            &solved,
            &solved_occupation.values,
        )?;
        last_residual = residual;
        current_density = Some(solved_density);
        if last_residual <= spec.fock_density_tolerance
            && feedback_residual <= spec.fock_feedback_tolerance.get()
        {
            return Ok(ScalarFixedPotentialResult {
                bands,
                occupation,
                exchange: rebuilt,
                core_feedback,
                fock_iterations: fock_iteration,
                exchange_rebuilds: rebuilds,
                fixed_point_residual: last_residual,
                feedback_residual,
            });
        }
        bands = solved;
        previous_global_feedback = Some(global_feedback);
    }
    Err(GammaValenceHfError::FockNotConverged {
        outer_iteration,
        iterations: spec.max_fock_iterations,
        density_residual: last_residual,
        feedback_residual: last_feedback_residual,
    })
}

fn rebuild_scalar_exchange(
    physics: &CheckpointPhysics,
    spec: &KhSocValenceHfSpec,
    bands: &CheckpointBandSolution,
    occupations: &[Collinear<Vec<f64>>],
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
) -> Result<Collinear<IsdfExchangeResult>, GammaValenceHfError> {
    let inputs = q_fractional
        .iter()
        .map(|&q| {
            physics.scalar_product_input_from_bands(bands, k_fractional, q, ScfRelativity::Scalar)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = inputs.first().ok_or(GammaValenceHfError::QTopology)?;
    let n_k = first.pair_columns.n_k;
    let n_orb = first.pair_columns.n_orb;
    let selections = [0_u8, 1]
        .into_iter()
        .flat_map(|spin| {
            (0..n_k).flat_map(move |k| {
                (0..n_orb).flat_map(move |left_band| {
                    (0..n_orb).map(move |right_band| ScalarMpbSelection {
                        spin,
                        k,
                        left_band,
                        right_band,
                    })
                })
            })
        })
        .collect::<Vec<_>>();
    let mpb = inputs
        .iter()
        .map(|input| {
            build_scalar_mpb(
                input,
                &ScalarMpbSpec {
                    lattice: input.reciprocal,
                    product_l_max: spec.product_l_max,
                    product_g_max: spec.product_g_max,
                    overlap_tolerance: spec.overlap_tolerance,
                    selections: selections.clone(),
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = scalar_k_weights(bands)?;
    let build = |spin: u8| -> Result<IsdfExchangeResult, GammaValenceHfError> {
        let occupations = occupations
            .iter()
            .map(|row| {
                if spin == 0 {
                    row.up.clone()
                } else {
                    row.down.clone()
                }
            })
            .collect();
        Ok(build_scalar_mpb_exchange(
            &inputs,
            &mpb,
            spin,
            &spec.coulomb,
            &IsdfExchangeSpec {
                k_weights: weights.clone(),
                occupations,
                gamma: spec.gamma,
            },
        )?)
    };
    Ok(Collinear::new(build(0)?, build(1)?))
}

fn rebuild_second_variation_exchange(
    physics: &CheckpointPhysics,
    spec: &KhSocValenceHfSpec,
    bands: &CheckpointBandSolution,
    occupations: &[Vec<f64>],
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
) -> Result<IsdfExchangeResult, GammaValenceHfError> {
    let inputs = q_fractional
        .iter()
        .map(|&q| {
            physics.scalar_product_input_from_bands(bands, k_fractional, q, spec.config.relativity)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = inputs.first().ok_or(GammaValenceHfError::QTopology)?;
    let n_k = first.pair_columns.n_k;
    let n_orb = first.pair_columns.n_orb;
    let selections = (0..n_k)
        .flat_map(|k| {
            (0..n_orb).flat_map(move |left_band| {
                (0..n_orb).map(move |right_band| SecondVariationMpbSelection {
                    k,
                    left_band,
                    right_band,
                })
            })
        })
        .collect::<Vec<_>>();
    let mpb = inputs
        .iter()
        .map(|input| {
            build_second_variation_mpb(
                input,
                &SecondVariationMpbSpec {
                    lattice: input.reciprocal,
                    product_l_max: spec.product_l_max,
                    product_g_max: spec.product_g_max,
                    overlap_tolerance: spec.overlap_tolerance,
                    selections: selections.clone(),
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(build_second_variation_mpb_exchange(
        &inputs,
        &mpb,
        &spec.coulomb,
        &IsdfExchangeSpec {
            k_weights: scalar_k_weights(bands)?,
            occupations: occupations.to_vec(),
            gamma: spec.gamma,
        },
    )?)
}

fn scalar_exchange_feedback(
    exchange: &Collinear<IsdfExchangeResult>,
) -> Result<Vec<Collinear<DenseHermitianMatrix>>, GammaValenceHfError> {
    let up = exchange_feedback(&exchange.up)?;
    let down = exchange_feedback(&exchange.down)?;
    if up.len() != down.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: up.len(),
            actual: down.len(),
        });
    }
    Ok(up
        .into_iter()
        .zip(down)
        .map(|(up, down)| Collinear::new(up, down))
        .collect())
}

fn lift_scalar_global_feedback(
    bands: &CheckpointBandSolution,
    feedback: &[Collinear<DenseHermitianMatrix>],
) -> Result<Collinear<Vec<DenseHermitianMatrix>>, GammaValenceHfError> {
    if bands.points().len() != feedback.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: bands.points().len(),
            actual: feedback.len(),
        });
    }
    let mut up = Vec::with_capacity(feedback.len());
    let mut down = Vec::with_capacity(feedback.len());
    for (point, feedback) in bands.points().iter().zip(feedback) {
        let CheckpointKPointSolution::Collinear {
            eigenproblems,
            solutions,
            up_occupations,
            down_occupations,
            ..
        } = &point.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        if up_occupations == down_occupations {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        }
        up.push(lift_band_hermitian_feedback(
            &eigenproblems.up.overlap,
            &solutions.up.eigenvectors,
            &feedback.up,
        )?);
        down.push(lift_band_hermitian_feedback(
            &eigenproblems.down.overlap,
            &solutions.down.eigenvectors,
            &feedback.down,
        )?);
    }
    Ok(Collinear::new(up, down))
}

fn add_scalar_core_feedback(
    valence: Collinear<Vec<DenseHermitianMatrix>>,
    core: &[Collinear<DenseHermitianMatrix>],
) -> Result<Collinear<Vec<DenseHermitianMatrix>>, GammaValenceHfError> {
    if valence.up.len() != core.len() || valence.down.len() != core.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: valence.up.len(),
            actual: core.len(),
        });
    }
    let add = |valence: Vec<DenseHermitianMatrix>, spin: usize| {
        valence
            .into_iter()
            .zip(core)
            .map(|(valence, core)| {
                let core = if spin == 0 { &core.up } else { &core.down };
                require_feedback_layout(
                    std::slice::from_ref(&valence),
                    std::slice::from_ref(core),
                )?;
                Ok(DenseHermitianMatrix::from_upper_triangle(
                    valence.dimension(),
                    Axis::GlobalBasis,
                    |row, column| valence.at(row, column) + core.at(row, column),
                )?)
            })
            .collect::<Result<Vec<_>, GammaValenceHfError>>()
    };
    Ok(Collinear::new(add(valence.up, 0)?, add(valence.down, 1)?))
}

fn scalar_feedback_difference(
    left: &Collinear<Vec<DenseHermitianMatrix>>,
    right: &Collinear<Vec<DenseHermitianMatrix>>,
) -> Result<f64, GammaValenceHfError> {
    Ok(global_feedback_difference(&left.up, &right.up)?
        .max(global_feedback_difference(&left.down, &right.down)?))
}

fn solve_fixed_potential(
    physics: &CheckpointPhysics,
    spec: &ValenceHfSpec,
    mut bands: CheckpointBandSolution,
    outer_iteration: usize,
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
) -> Result<FixedPotentialResult, GammaValenceHfError> {
    let mut rebuilds = 0;
    let mut first_one_shot_exchange = None;
    let mut first_global_solve_identity_residual = None;
    let mut last_residual = f64::INFINITY;
    let mut last_feedback_residual = f64::INFINITY;
    let mut previous_global_feedback = None;
    let mut feedback_mixer = FeedbackMixer::new(spec.fock_mixing);
    let mut current_density = None;
    for fock_iteration in 1..=spec.max_fock_iterations {
        let occupation = solve_occupations(
            bands.states(),
            spec.config.electron_count,
            spec.config.occupations,
        )?;
        let occupation_rows = occupation_rows(&occupation.values, &bands)?;
        if outer_iteration == 1 && fock_iteration == 1 {
            let driver = rebuild_exchange(
                physics,
                spec,
                &bands,
                &occupation_rows,
                k_fractional,
                q_fractional,
            )?;
            rebuilds += 1;
            first_one_shot_exchange = Some(driver.clone());
            let band_feedback = exchange_feedback(&driver)?;
            let global_feedback = lift_global_feedback(&bands, &band_feedback)?;
            let lifting_identity_residual =
                lifting_identity(&bands, &band_feedback, &global_feedback)?;
            require_gate(
                "band-feedback lifting",
                lifting_identity_residual,
                IDENTITY_TOLERANCE,
            )?;
            let solved = bands.solve_spinor_global_feedback(&global_feedback)?;
            let solve_identity = first_global_solve_identity(&bands, &band_feedback, &solved)?;
            require_gate(
                "first global generalized solve",
                solve_identity,
                IDENTITY_TOLERANCE,
            )?;
            first_global_solve_identity_residual = Some(solve_identity);
            let solved_occupation = solve_occupations(
                solved.states(),
                spec.config.electron_count,
                spec.config.occupations,
            )?;
            let (residual, solved_density) = fixed_point_density_residual(
                physics,
                &bands,
                &occupation.values,
                current_density.take(),
                &solved,
                &solved_occupation.values,
            )?;
            last_residual = residual;
            current_density = Some(solved_density);
            bands = solved;
            previous_global_feedback = Some(global_feedback);
            if fock_iteration == spec.max_fock_iterations {
                return Err(GammaValenceHfError::FockNotConverged {
                    outer_iteration,
                    iterations: fock_iteration,
                    density_residual: last_residual,
                    feedback_residual: last_feedback_residual,
                });
            }
            continue;
        }

        let rebuilt = rebuild_exchange(
            physics,
            spec,
            &bands,
            &occupation_rows,
            k_fractional,
            q_fractional,
        )?;
        rebuilds += 1;
        let fresh_band_feedback = exchange_feedback(&rebuilt)?;
        let fresh_global_feedback = lift_global_feedback(&bands, &fresh_band_feedback)?;
        let feedback_fixed_residual = previous_global_feedback
            .as_ref()
            .map(|previous| global_feedback_difference(previous, &fresh_global_feedback))
            .transpose()?
            .unwrap_or(f64::INFINITY);
        last_feedback_residual = feedback_fixed_residual;
        let global_feedback = match &previous_global_feedback {
            Some(previous) => {
                feedback_mixer.mix(&bands, &occupation_rows, previous, &fresh_global_feedback)?
            }
            None => fresh_global_feedback.clone(),
        };
        let lifting_identity_residual =
            lifting_identity(&bands, &fresh_band_feedback, &fresh_global_feedback)?;
        require_gate(
            "band-feedback lifting",
            lifting_identity_residual,
            IDENTITY_TOLERANCE,
        )?;
        let solved = bands.solve_spinor_global_feedback(&global_feedback)?;
        let solved_occupation = solve_occupations(
            solved.states(),
            spec.config.electron_count,
            spec.config.occupations,
        )?;
        let (residual, solved_density) = fixed_point_density_residual(
            physics,
            &bands,
            &occupation.values,
            current_density.take(),
            &solved,
            &solved_occupation.values,
        )?;
        last_residual = residual;
        current_density = Some(solved_density);
        if last_residual <= spec.fock_density_tolerance
            && feedback_fixed_residual <= spec.fock_feedback_tolerance.get()
        {
            return Ok(FixedPotentialResult {
                bands,
                occupation,
                exchange: rebuilt,
                fock_iterations: fock_iteration,
                exchange_rebuilds: rebuilds,
                fixed_point_residual: last_residual,
                feedback_residual: feedback_fixed_residual,
                lifting_identity_residual,
                first_global_solve_identity_residual,
                first_one_shot_exchange,
            });
        }
        bands = solved;
        previous_global_feedback = Some(global_feedback);
    }
    Err(GammaValenceHfError::FockNotConverged {
        outer_iteration,
        iterations: spec.max_fock_iterations,
        density_residual: last_residual,
        feedback_residual: last_feedback_residual,
    })
}

fn rebuild_exchange(
    physics: &CheckpointPhysics,
    spec: &ValenceHfSpec,
    bands: &CheckpointBandSolution,
    occupations: &[Vec<f64>],
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
) -> Result<IsdfExchangeResult, GammaValenceHfError> {
    let inputs = q_fractional
        .iter()
        .map(|&q| physics.spinor_product_input_from_bands(bands, k_fractional, q))
        .collect::<Result<Vec<_>, _>>()?;
    let first = inputs.first().ok_or(GammaValenceHfError::QTopology)?;
    let n_k = first.pair_columns.n_k;
    let n_orb = first.pair_columns.n_orb;
    let selections: Vec<SpinorMpbSelection> = (0..n_k)
        .flat_map(|k| {
            (0..n_orb).flat_map(move |left_band| {
                (0..n_orb).map(move |right_band| SpinorMpbSelection {
                    k,
                    left_band,
                    right_band,
                })
            })
        })
        .collect();
    let mpb = inputs
        .iter()
        .map(|input| {
            build_spinor_mpb(
                input,
                &SpinorMpbSpec {
                    product_l_max: spec.product_l_max,
                    product_g_max: spec.product_g_max,
                    overlap_tolerance: spec.overlap_tolerance,
                    selections: selections.clone(),
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let k_weights = k_weights(bands)?;
    Ok(build_spinor_mpb_exchange(
        &inputs,
        &mpb,
        &spec.coulomb,
        &IsdfExchangeSpec {
            k_weights,
            occupations: occupations.to_vec(),
            gamma: GammaExchangeTreatment::FiniteBody,
        },
    )?)
}

fn exchange_feedback(
    exchange: &IsdfExchangeResult,
) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
    exchange
        .band_matrices
        .iter()
        .enumerate()
        .map(|(k, matrix)| {
            if matrix.k_index() != k {
                return Err(GammaValenceHfError::ExchangeKIndex {
                    expected: k,
                    actual: matrix.k_index(),
                });
            }
            Ok(DenseHermitianMatrix::from_upper_triangle(
                matrix.n_bands(),
                Axis::Band,
                |row, column| {
                    matrix
                        .element(row, column)
                        .expect("validated band matrix index")
                },
            )?)
        })
        .collect()
}

fn lift_global_feedback(
    bands: &CheckpointBandSolution,
    feedback: &[DenseHermitianMatrix],
) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
    if bands.points().len() != feedback.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: bands.points().len(),
            actual: feedback.len(),
        });
    }
    bands
        .points()
        .iter()
        .zip(feedback)
        .map(|(point, band_feedback)| {
            let CheckpointKPointSolution::Spinor {
                eigenproblem,
                solution,
                ..
            } = &point.solution
            else {
                return Err(GammaValenceHfError::SpinorFirstVariation);
            };
            Ok(lift_band_hermitian_feedback(
                &eigenproblem.overlap,
                &solution.eigenvectors,
                band_feedback,
            )?)
        })
        .collect()
}

#[derive(Clone, Debug)]
struct FeedbackMixRecord {
    candidate: Vec<DenseHermitianMatrix>,
    error: Vec<Complex64>,
}

#[derive(Clone, Debug)]
struct FeedbackMixer {
    mixing: FockMixing,
    history: Vec<FeedbackMixRecord>,
}

impl FeedbackMixer {
    fn new(mixing: FockMixing) -> Self {
        Self {
            mixing,
            history: Vec::new(),
        }
    }

    fn mix(
        &mut self,
        bands: &CheckpointBandSolution,
        occupations: &[Vec<f64>],
        input: &[DenseHermitianMatrix],
        output: &[DenseHermitianMatrix],
    ) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
        let (candidate, error) = match self.mixing {
            FockMixing::Linear { alpha } => {
                return mix_global_feedback(input, output, alpha);
            }
            FockMixing::PulayAnderson { alpha, .. } => (
                mix_global_feedback(input, output, alpha)?,
                flatten_global_feedback(&subtract_global_feedback(input, output)?),
            ),
            FockMixing::CommutatorDiis { .. } => (
                output.to_vec(),
                commutator_diis_error(bands, occupations, output, None)?,
            ),
            FockMixing::QuasiNewtonDiis { level_shift, .. } => (
                output.to_vec(),
                commutator_diis_error(bands, occupations, output, Some(level_shift))?,
            ),
        };
        let max_history = self
            .mixing
            .history()
            .expect("nonlinear Fock mixing has a history");
        let record = FeedbackMixRecord { candidate, error };
        self.history.push(record);
        if self.history.len() > max_history {
            self.history.remove(0);
        }
        if self.history.len() == 1 {
            return Ok(self.history[0].candidate.clone());
        }
        let Some(coefficients) = feedback_pulay_coefficients(&self.history)? else {
            return Ok(self
                .history
                .last()
                .expect("history is nonempty")
                .candidate
                .clone());
        };
        combine_feedback_records(&self.history, &coefficients)
    }
}

fn mix_global_feedback(
    previous: &[DenseHermitianMatrix],
    fresh: &[DenseHermitianMatrix],
    alpha: f64,
) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
    if previous.len() != fresh.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: previous.len(),
            actual: fresh.len(),
        });
    }
    previous
        .iter()
        .zip(fresh)
        .map(|(previous, fresh)| {
            if previous.axis() != Axis::GlobalBasis || fresh.axis() != Axis::GlobalBasis {
                return Err(GammaValenceHfError::Tensor(TensorError::Axis {
                    index: 0,
                    expected: Axis::GlobalBasis,
                    actual: if previous.axis() != Axis::GlobalBasis {
                        previous.axis()
                    } else {
                        fresh.axis()
                    },
                }));
            }
            if previous.dimension() != fresh.dimension() {
                return Err(GammaValenceHfError::ExchangeKIndex {
                    expected: previous.dimension(),
                    actual: fresh.dimension(),
                });
            }
            Ok(DenseHermitianMatrix::from_upper_triangle(
                fresh.dimension(),
                Axis::GlobalBasis,
                |row, column| {
                    (1.0 - alpha) * previous.at(row, column) + alpha * fresh.at(row, column)
                },
            )?)
        })
        .collect()
}

fn subtract_global_feedback(
    input: &[DenseHermitianMatrix],
    output: &[DenseHermitianMatrix],
) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
    require_feedback_layout(input, output)?;
    input
        .iter()
        .zip(output)
        .map(|(input, output)| {
            Ok(DenseHermitianMatrix::from_upper_triangle(
                input.dimension(),
                Axis::GlobalBasis,
                |row, column| input.at(row, column) - output.at(row, column),
            )?)
        })
        .collect()
}

fn flatten_global_feedback(feedback: &[DenseHermitianMatrix]) -> Vec<Complex64> {
    feedback
        .iter()
        .flat_map(DenseHermitianMatrix::to_host_row_major)
        .collect()
}

fn commutator_diis_error(
    bands: &CheckpointBandSolution,
    occupations: &[Vec<f64>],
    feedback: &[DenseHermitianMatrix],
    level_shift: Option<Hartree>,
) -> Result<Vec<Complex64>, GammaValenceHfError> {
    if bands.points().len() != occupations.len() || bands.points().len() != feedback.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: bands.points().len(),
            actual: occupations.len().min(feedback.len()),
        });
    }
    let mut error = Vec::new();
    for ((point, occupations), feedback) in bands.points().iter().zip(occupations).zip(feedback) {
        let CheckpointKPointSolution::Spinor {
            eigenproblem,
            solution,
            ..
        } = &point.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let basis_count = solution.eigenvectors.rows();
        let band_count = solution.eigenvectors.columns();
        if occupations.len() != band_count {
            return Err(GammaValenceHfError::ExchangeKIndex {
                expected: band_count,
                actual: occupations.len(),
            });
        }
        if feedback.dimension() != basis_count {
            return Err(GammaValenceHfError::ExchangeKIndex {
                expected: basis_count,
                actual: feedback.dimension(),
            });
        }
        let conjugate = solution.eigenvectors.as_tensor().conjugate();
        let fock = DenseHermitianMatrix::from_upper_triangle(
            basis_count,
            Axis::GlobalBasis,
            |row, column| eigenproblem.hamiltonian.at(row, column) + feedback.at(row, column),
        )?;
        let weight = point.weight().sqrt();
        if let Some(level_shift) = level_shift {
            let projected_fock = einsum(
                "ia,ij,jb->ab",
                &[
                    &conjugate,
                    fock.as_tensor(),
                    solution.eigenvectors.as_tensor(),
                ],
            )?
            .to_host_row_major();
            let mut preconditioned = Vec::with_capacity(band_count * band_count);
            for row in 0..band_count {
                for column in 0..band_count {
                    let occupation_difference = occupations[row] - occupations[column];
                    if occupation_difference == 0.0 {
                        preconditioned.push(Complex64::default());
                        continue;
                    }
                    let denominator = (solution.eigenvalues[row].get()
                        - solution.eigenvalues[column].get())
                    .abs()
                        + level_shift.get();
                    if denominator == 0.0 {
                        return Err(GammaValenceHfError::FockMixingAlgebra);
                    }
                    preconditioned.push(
                        occupation_difference * projected_fock[row * band_count + column]
                            / denominator,
                    );
                }
            }
            let preconditioned = ComplexTensor::from_host_row_major(
                &[band_count, band_count],
                &[Axis::Band, Axis::Band],
                preconditioned,
            )?;
            let rotated = einsum(
                "ia,ab,jb->ij",
                &[
                    solution.eigenvectors.as_tensor(),
                    &preconditioned,
                    &conjugate,
                ],
            )?;
            error.extend(
                einsum(
                    "ij,jk,kl->il",
                    &[
                        eigenproblem.overlap.as_tensor(),
                        &rotated,
                        eigenproblem.overlap.as_tensor(),
                    ],
                )?
                .to_host_row_major()
                .into_iter()
                .map(|value| weight * value),
            );
        } else {
            let occupation_tensor = ComplexTensor::from_host_row_major(
                &[band_count],
                &[Axis::Band],
                occupations
                    .iter()
                    .map(|&value| Complex64::new(value, 0.0))
                    .collect(),
            )?;
            let density = einsum(
                "ia,a,ja->ij",
                &[
                    solution.eigenvectors.as_tensor(),
                    &occupation_tensor,
                    &conjugate,
                ],
            )?;
            let sdf = einsum(
                "ij,jk,kl->il",
                &[eigenproblem.overlap.as_tensor(), &density, fock.as_tensor()],
            )?
            .to_host_row_major();
            for row in 0..basis_count {
                for column in 0..basis_count {
                    error.push(
                        weight
                            * (sdf[column * basis_count + row].conj()
                                - sdf[row * basis_count + column]),
                    );
                }
            }
        }
    }
    if error
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        Ok(error)
    } else {
        Err(GammaValenceHfError::FockMixingAlgebra)
    }
}

fn require_feedback_layout(
    left: &[DenseHermitianMatrix],
    right: &[DenseHermitianMatrix],
) -> Result<(), GammaValenceHfError> {
    if left.len() != right.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: left.len(),
            actual: right.len(),
        });
    }
    for (left, right) in left.iter().zip(right) {
        if left.axis() != Axis::GlobalBasis || right.axis() != Axis::GlobalBasis {
            return Err(GammaValenceHfError::Tensor(TensorError::Axis {
                index: 0,
                expected: Axis::GlobalBasis,
                actual: if left.axis() != Axis::GlobalBasis {
                    left.axis()
                } else {
                    right.axis()
                },
            }));
        }
        if left.dimension() != right.dimension() {
            return Err(GammaValenceHfError::ExchangeKIndex {
                expected: left.dimension(),
                actual: right.dimension(),
            });
        }
    }
    Ok(())
}

fn feedback_pulay_coefficients(
    history: &[FeedbackMixRecord],
) -> Result<Option<Vec<f64>>, GammaValenceHfError> {
    let number = history.len();
    let dimension = number + 1;
    let mut constrained = vec![vec![0.0; dimension]; dimension];
    for row in 0..number {
        for column in 0..=row {
            let value = feedback_error_inner_product(&history[row].error, &history[column].error)?;
            constrained[row][column] = value;
            constrained[column][row] = value;
        }
        constrained[row][number] = 1.0;
        constrained[number][row] = 1.0;
    }
    let mut right = vec![0.0; dimension];
    right[number] = 1.0;
    Ok(solve_feedback_dense(constrained, right)?.map(|solution| solution[..number].to_vec()))
}

fn feedback_error_inner_product(
    left: &[Complex64],
    right: &[Complex64],
) -> Result<f64, GammaValenceHfError> {
    if left.len() != right.len() {
        return Err(GammaValenceHfError::FockMixingAlgebra);
    }
    let value = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left.conj() * right).re)
        .sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GammaValenceHfError::FockMixingAlgebra)
    }
}

fn combine_feedback_records(
    history: &[FeedbackMixRecord],
    coefficients: &[f64],
) -> Result<Vec<DenseHermitianMatrix>, GammaValenceHfError> {
    let k_count = history[0].candidate.len();
    let mut mixed = Vec::with_capacity(k_count);
    for k in 0..k_count {
        let dimension = history[0].candidate[k].dimension();
        mixed.push(DenseHermitianMatrix::from_upper_triangle(
            dimension,
            Axis::GlobalBasis,
            |row, column| {
                history
                    .iter()
                    .zip(coefficients)
                    .map(|(record, &coefficient)| coefficient * record.candidate[k].at(row, column))
                    .sum()
            },
        )?);
    }
    Ok(mixed)
}

fn solve_feedback_dense(
    mut matrix: Vec<Vec<f64>>,
    mut right: Vec<f64>,
) -> Result<Option<Vec<f64>>, GammaValenceHfError> {
    let dimension = right.len();
    if matrix.iter().flatten().any(|value| !value.is_finite())
        || right.iter().any(|value| !value.is_finite())
    {
        return Err(GammaValenceHfError::FockMixingAlgebra);
    }
    let scale = matrix
        .iter()
        .flatten()
        .chain(&right)
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(None);
    }
    let tolerance = 256.0 * f64::EPSILON * scale * dimension.max(1) as f64;
    for column in 0..dimension {
        let pivot = (column..dimension)
            .max_by(|&left, &right_row| {
                matrix[left][column]
                    .abs()
                    .total_cmp(&matrix[right_row][column].abs())
            })
            .expect("the constrained Pulay matrix is nonempty");
        if matrix[pivot][column].abs() <= tolerance {
            return Ok(None);
        }
        matrix.swap(column, pivot);
        right.swap(column, pivot);
        for row in column + 1..dimension {
            let factor = matrix[row][column] / matrix[column][column];
            matrix[row][column] = 0.0;
            let (upper, lower) = matrix.split_at_mut(row);
            let pivot_tail = &upper[column][column + 1..];
            let target_tail = &mut lower[0][column + 1..];
            for (target, &pivot_entry) in target_tail.iter_mut().zip(pivot_tail) {
                *target -= factor * pivot_entry;
            }
            right[row] -= factor * right[column];
        }
    }
    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let tail = matrix[row][row + 1..]
            .iter()
            .zip(&solution[row + 1..])
            .map(|(coefficient, value)| coefficient * value)
            .sum::<f64>();
        solution[row] = (right[row] - tail) / matrix[row][row];
        if !solution[row].is_finite() {
            return Err(GammaValenceHfError::FockMixingAlgebra);
        }
    }
    Ok(Some(solution))
}

fn global_feedback_difference(
    left: &[DenseHermitianMatrix],
    right: &[DenseHermitianMatrix],
) -> Result<f64, GammaValenceHfError> {
    if left.len() != right.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: left.len(),
            actual: right.len(),
        });
    }
    let mut maximum = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        if left.axis() != Axis::GlobalBasis || right.axis() != Axis::GlobalBasis {
            return Err(GammaValenceHfError::Tensor(TensorError::Axis {
                index: 0,
                expected: Axis::GlobalBasis,
                actual: if left.axis() != Axis::GlobalBasis {
                    left.axis()
                } else {
                    right.axis()
                },
            }));
        }
        if left.dimension() != right.dimension() {
            return Err(GammaValenceHfError::ExchangeKIndex {
                expected: left.dimension(),
                actual: right.dimension(),
            });
        }
        for row in 0..left.dimension() {
            for column in 0..left.dimension() {
                maximum = maximum.max((left.at(row, column) - right.at(row, column)).norm());
            }
        }
    }
    Ok(maximum)
}

/// Fock fixed-point residual, returning the solved density for the next step.
///
/// `carried` is the solved density of the previous step. The next step reaches
/// this function with exactly those bands and occupations, so synthesizing the
/// current density again would repeat the previous step's work.
fn fixed_point_density_residual(
    physics: &CheckpointPhysics,
    current: &CheckpointBandSolution,
    current_occupations: &[f64],
    carried: Option<RegionalDensity>,
    solved: &CheckpointBandSolution,
    solved_occupations: &[f64],
) -> Result<(f64, RegionalDensity), GammaValenceHfError> {
    let current_density = match carried {
        Some(density) => density,
        None => physics
            .kernel
            .synthesize_bands(current, current_occupations)?,
    };
    let solved_density = physics
        .kernel
        .synthesize_bands(solved, solved_occupations)?;
    let residual = current_density.difference_rms(&solved_density)?;
    Ok((residual, solved_density))
}

fn lifting_identity(
    bands: &CheckpointBandSolution,
    band_feedback: &[DenseHermitianMatrix],
    global_feedback: &[DenseHermitianMatrix],
) -> Result<f64, GammaValenceHfError> {
    if bands.points().len() != band_feedback.len() || band_feedback.len() != global_feedback.len() {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: bands.points().len(),
            actual: band_feedback.len().min(global_feedback.len()),
        });
    }
    let mut maximum = 0.0_f64;
    for ((point, expected), lifted) in bands
        .points()
        .iter()
        .zip(band_feedback)
        .zip(global_feedback)
    {
        let CheckpointKPointSolution::Spinor { solution, .. } = &point.solution else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let conjugate = solution.eigenvectors.as_tensor().conjugate();
        let projected = DenseHermitianMatrix::from_tensor(einsum(
            "ia,ij,jb->ab",
            &[
                &conjugate,
                lifted.as_tensor(),
                solution.eigenvectors.as_tensor(),
            ],
        )?)?;
        for row in 0..expected.dimension() {
            for column in 0..expected.dimension() {
                maximum =
                    maximum.max((projected.at(row, column) - expected.at(row, column)).norm());
            }
        }
    }
    Ok(maximum)
}

fn first_global_solve_identity(
    source: &CheckpointBandSolution,
    feedback: &[DenseHermitianMatrix],
    solved: &CheckpointBandSolution,
) -> Result<f64, GammaValenceHfError> {
    let mut maximum = 0.0_f64;
    for ((source, feedback), solved) in source.points().iter().zip(feedback).zip(solved.points()) {
        let CheckpointKPointSolution::Spinor {
            solution: source_solution,
            ..
        } = &source.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let CheckpointKPointSolution::Spinor {
            solution: solved_solution,
            ..
        } = &solved.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let dimension = source_solution.eigenvalues.len();
        let band_fock = DenseHermitianMatrix::from_upper_triangle(
            dimension,
            Axis::GlobalBasis,
            |row, column| {
                feedback.at(row, column)
                    + if row == column {
                        Complex64::new(source_solution.eigenvalues[row].get(), 0.0)
                    } else {
                        Complex64::default()
                    }
            },
        )?;
        let identity = DenseHermitianMatrix::from_upper_triangle(
            dimension,
            Axis::GlobalBasis,
            |row, column| {
                if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::default()
                }
            },
        )?;
        let band_solution = solve_generalized_hermitian(&band_fock, &identity, 0.0)?;
        for (&left, &right) in band_solution
            .eigenvalues
            .iter()
            .zip(&solved_solution.eigenvalues)
        {
            maximum = maximum.max((left.get() - right.get()).abs());
        }
    }
    Ok(maximum)
}

struct EnergyDiagnostic {
    total: Hartree,
    exchange_identity_residual: f64,
    eigenvalue_identity_residual: f64,
    total_identity_residual: f64,
}

fn energy_diagnostic(
    bands: &CheckpointBandSolution,
    occupation: &OccupationSolution,
    exchange: &IsdfExchangeResult,
    electrostatic: &muffintin_dft::RegionalElectrostaticResult,
) -> Result<EnergyDiagnostic, GammaValenceHfError> {
    let rows = occupation_rows(&occupation.values, bands)?;
    let mut h0_expectation = 0.0;
    let mut exchange_trace = 0.0;
    for (k, point) in bands.points().iter().enumerate() {
        let CheckpointKPointSolution::Spinor {
            eigenproblem,
            solution,
            occupations: state_range,
            ..
        } = &point.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        let conjugate = solution.eigenvectors.as_tensor().conjugate();
        let projected_h0 = DenseHermitianMatrix::from_tensor(einsum(
            "ia,ij,jb->ab",
            &[
                &conjugate,
                eigenproblem.hamiltonian.as_tensor(),
                solution.eigenvectors.as_tensor(),
            ],
        )?)?;
        let matrix = &exchange.band_matrices[k];
        for (band, &value) in rows[k].iter().enumerate() {
            let weight = bands.states()[state_range.start + band].k_weight;
            h0_expectation += weight * value * projected_h0.at(band, band).re;
            exchange_trace += weight
                * value
                * matrix
                    .element(band, band)
                    .expect("validated exchange band index")
                    .re;
        }
    }
    let exchange_identity_residual = (exchange.exchange_energy.get() - 0.5 * exchange_trace).abs();
    let eigenvalue_identity_residual =
        (occupation.band_energy.get() - h0_expectation - exchange_trace).abs();
    let direct = h0_expectation - electrostatic.electron_hartree.get()
        + electrostatic.nuclear_nuclear.get()
        + exchange.exchange_energy.get()
        + occupation.correction.get();
    let eigenvalue = occupation.band_energy.get() - electrostatic.electron_hartree.get()
        + electrostatic.nuclear_nuclear.get()
        - exchange.exchange_energy.get()
        + occupation.correction.get();
    Ok(EnergyDiagnostic {
        total: Hartree(direct),
        exchange_identity_residual,
        eigenvalue_identity_residual,
        total_identity_residual: (direct - eigenvalue).abs(),
    })
}

#[derive(Clone, Copy)]
struct FrozenCoreEnergyTerms {
    h0: Hartree,
    cc_trace: Hartree,
}

struct KhSocEnergyDiagnostic {
    total: Hartree,
    exchange_identity_residual: f64,
    eigenvalue_identity_residual: f64,
    total_identity_residual: f64,
    core_valence_exchange: Hartree,
}

fn kh_soc_energy_diagnostic(
    bands: &CheckpointBandSolution,
    occupation: &OccupationSolution,
    exchange: &Collinear<IsdfExchangeResult>,
    core_feedback: &[Collinear<DenseHermitianMatrix>],
    electrostatic: &muffintin_dft::RegionalElectrostaticResult,
    core: FrozenCoreEnergyTerms,
) -> Result<KhSocEnergyDiagnostic, GammaValenceHfError> {
    let rows = scalar_occupation_rows(&occupation.values, bands)?;
    if exchange.up.band_matrices.len() != bands.points().len()
        || exchange.down.band_matrices.len() != bands.points().len()
        || core_feedback.len() != bands.points().len()
    {
        return Err(GammaValenceHfError::ExchangeKIndex {
            expected: bands.points().len(),
            actual: exchange
                .up
                .band_matrices
                .len()
                .min(exchange.down.band_matrices.len()),
        });
    }
    let mut h0_expectation = 0.0;
    let mut exchange_trace = 0.0;
    let mut core_trace = 0.0;
    for (k, point) in bands.points().iter().enumerate() {
        let CheckpointKPointSolution::Collinear {
            eigenproblems,
            solutions,
            up_occupations,
            down_occupations,
            ..
        } = &point.solution
        else {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        };
        if up_occupations == down_occupations {
            return Err(GammaValenceHfError::SpinorFirstVariation);
        }
        for (problem, solution, values, matrix, core_matrix, range) in [
            (
                &eigenproblems.up,
                &solutions.up,
                &rows[k].up,
                &exchange.up.band_matrices[k],
                &core_feedback[k].up,
                up_occupations,
            ),
            (
                &eigenproblems.down,
                &solutions.down,
                &rows[k].down,
                &exchange.down.band_matrices[k],
                &core_feedback[k].down,
                down_occupations,
            ),
        ] {
            let conjugate = solution.eigenvectors.as_tensor().conjugate();
            let projected_h0 = DenseHermitianMatrix::from_tensor(einsum(
                "ia,ij,jb->ab",
                &[
                    &conjugate,
                    problem.hamiltonian.as_tensor(),
                    solution.eigenvectors.as_tensor(),
                ],
            )?)?;
            let projected_core = DenseHermitianMatrix::from_tensor(einsum(
                "ia,ij,jb->ab",
                &[
                    &conjugate,
                    core_matrix.as_tensor(),
                    solution.eigenvectors.as_tensor(),
                ],
            )?)?;
            for (band, &value) in values.iter().enumerate() {
                let weight = bands.states()[range.start + band].k_weight;
                h0_expectation += weight * value * projected_h0.at(band, band).re;
                exchange_trace += weight
                    * value
                    * matrix
                        .element(band, band)
                        .expect("validated scalar exchange band index")
                        .re;
                core_trace += weight * value * projected_core.at(band, band).re;
            }
        }
    }
    let exchange_energy = exchange.up.exchange_energy.get() + exchange.down.exchange_energy.get();
    let exchange_identity_residual = (exchange_energy - 0.5 * exchange_trace).abs();
    let eigenvalue_identity_residual =
        (occupation.band_energy.get() - h0_expectation - exchange_trace - core_trace).abs();
    let core_core_exchange = 0.5 * core.cc_trace.get();
    let direct = h0_expectation + core.h0.get() - electrostatic.electron_hartree.get()
        + electrostatic.nuclear_nuclear.get()
        + exchange_energy
        + core_trace
        + core_core_exchange
        + occupation.correction.get();
    let eigenvalue = occupation.band_energy.get() + core.h0.get()
        - electrostatic.electron_hartree.get()
        + electrostatic.nuclear_nuclear.get()
        - exchange_energy
        + core_core_exchange
        + occupation.correction.get();
    Ok(KhSocEnergyDiagnostic {
        total: Hartree(direct),
        exchange_identity_residual,
        eigenvalue_identity_residual,
        total_identity_residual: (direct - eigenvalue).abs(),
        core_valence_exchange: Hartree(core_trace),
    })
}

struct RelaxedEnergyDiagnostic {
    total: Hartree,
    valence_eigenvalue_identity_residual: f64,
}

fn relaxed_energy_diagnostic(
    bands: &CheckpointBandSolution,
    occupation: &OccupationSolution,
    exchange: &FrozenSpinorSectorExchange,
    electrostatic: &muffintin_dft::RegionalElectrostaticResult,
    core_h0_trace: Hartree,
) -> Result<RelaxedEnergyDiagnostic, RelaxedCoreHfError> {
    let rows = occupation_rows_relaxed(&occupation.values, bands)?;
    let mut valence_h0_trace = 0.0;
    for (k, point) in bands.points().iter().enumerate() {
        let CheckpointKPointSolution::Spinor {
            eigenproblem,
            solution,
            occupations: state_range,
            ..
        } = &point.solution
        else {
            return Err(RelaxedCoreHfError::SpinorFirstVariation);
        };
        let conjugate = solution.eigenvectors.as_tensor().conjugate();
        let projected_h0 = DenseHermitianMatrix::from_tensor(einsum(
            "ia,ij,jb->ab",
            &[
                &conjugate,
                eigenproblem.hamiltonian.as_tensor(),
                solution.eigenvectors.as_tensor(),
            ],
        )?)?;
        for (band, &value) in rows[k].iter().enumerate() {
            let weight = bands.states()[state_range.start + band].k_weight;
            valence_h0_trace += weight * value * projected_h0.at(band, band).re;
        }
    }
    let valence_eigenvalue_identity_residual = (occupation.band_energy.get()
        - valence_h0_trace
        - exchange.vv.trace.get()
        - exchange.cv.trace.get())
    .abs();
    let total = Hartree(
        valence_h0_trace + core_h0_trace.get() - electrostatic.electron_hartree.get()
            + electrostatic.nuclear_nuclear.get()
            + exchange.exchange_total.get()
            + occupation.correction.get(),
    );
    Ok(RelaxedEnergyDiagnostic {
        total,
        valence_eigenvalue_identity_residual,
    })
}

fn frozen_core_energy_terms(
    sidecars: &[CoreShellOrbitals],
) -> Result<FrozenCoreEnergyTerms, GammaValenceHfError> {
    let h0 = Hartree(
        sidecars
            .iter()
            .map(core_local_one_body_trace)
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .map(|trace| trace.total.get())
            .sum(),
    );
    let mut cc_trace = 0.0;
    for sidecar in sidecars {
        let shells = sidecar
            .shells
            .iter()
            .map(|shell| {
                let CoreShellOccupations::MuResolved(occupations) = &shell.occupations else {
                    return Err(GammaValenceHfError::CoreStates);
                };
                let Some(&(_, occupation)) = occupations.first() else {
                    return Err(GammaValenceHfError::CoreStates);
                };
                if occupations
                    .iter()
                    .any(|(_, value)| value.to_bits() != occupation.to_bits())
                {
                    return Err(GammaValenceHfError::CoreStates);
                }
                Ok(CoreCoreFockShell {
                    kappa: shell.state.kappa,
                    p: &shell.p,
                    q: &shell.q,
                    normalization: shell.norm_total,
                    occupation_per_mu: occupation,
                })
            })
            .collect::<Result<Vec<_>, GammaValenceHfError>>()?;
        cc_trace += core_core_fock_actions(&sidecar.extended_mesh, &shells)?
            .trace
            .total
            .get();
    }
    Ok(FrozenCoreEnergyTerms {
        h0,
        cc_trace: Hartree(cc_trace),
    })
}

fn sum_kh_density(
    valence: &RegionalDensity,
    core: &RegionalDensity,
) -> Result<RegionalDensity, GammaValenceHfError> {
    let mut total = valence.clone();
    total.add_scaled(1.0, core)?;
    Ok(total)
}

fn synthesize_core_density(
    template: &RegionalDensity,
    sidecars: &[CoreShellOrbitals],
) -> Result<RegionalDensity, RelaxedCoreHfError> {
    let mut density = template.zero_like();
    for sidecar in sidecars {
        let contribution = build_regional_core_contribution_from_sidecar(sidecar, template)?;
        density.add_scaled(1.0, &contribution.contribution.density)?;
    }
    Ok(density)
}

fn sum_density(
    left: &RegionalDensity,
    right: &RegionalDensity,
) -> Result<RegionalDensity, RelaxedCoreHfError> {
    let mut sum = left.clone();
    sum.add_scaled(1.0, right)?;
    Ok(sum)
}

fn maximum_core_residuals(results: &[CoreFixedPotentialResult]) -> (Hartree, f64) {
    results
        .iter()
        .filter_map(|result| result.diagnostics.last())
        .fold((Hartree(0.0), 0.0_f64), |(energy, radial), item| {
            (
                Hartree(energy.get().max(item.maximum_energy_change.get())),
                radial.max(item.maximum_radial_residual),
            )
        })
}

fn maximum_sector_antihermitian(exchange: &FrozenSpinorSectorExchange) -> f64 {
    [
        exchange.vv.maximum_antihermitian_residual,
        exchange.cv.maximum_antihermitian_residual,
        exchange.vc.maximum_antihermitian_residual,
        exchange.cc.maximum_antihermitian_residual,
    ]
    .into_iter()
    .fold(0.0_f64, f64::max)
}

fn occupation_rows_relaxed(
    flat: &[f64],
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<f64>>, RelaxedCoreHfError> {
    Ok(occupation_rows(flat, bands)?)
}

fn spinor_energies_relaxed(
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<Hartree>>, RelaxedCoreHfError> {
    Ok(spinor_energies(bands)?)
}

fn k_weights_relaxed(bands: &CheckpointBandSolution) -> Result<Vec<f64>, RelaxedCoreHfError> {
    Ok(k_weights(bands)?)
}

fn occupation_rows(
    flat: &[f64],
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<f64>>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Spinor { occupations, .. }
                if occupations.end <= flat.len() =>
            {
                Ok(flat[occupations.clone()].to_vec())
            }
            CheckpointKPointSolution::Spinor { .. }
            | CheckpointKPointSolution::Collinear { .. } => {
                Err(GammaValenceHfError::SpinorFirstVariation)
            }
        })
        .collect()
}

fn scalar_occupation_rows(
    flat: &[f64],
    bands: &CheckpointBandSolution,
) -> Result<Vec<Collinear<Vec<f64>>>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Collinear {
                up_occupations,
                down_occupations,
                ..
            } if up_occupations != down_occupations
                && up_occupations.end <= flat.len()
                && down_occupations.end <= flat.len() =>
            {
                Ok(Collinear::new(
                    flat[up_occupations.clone()].to_vec(),
                    flat[down_occupations.clone()].to_vec(),
                ))
            }
            _ => Err(GammaValenceHfError::SpinorFirstVariation),
        })
        .collect()
}

fn second_variation_occupation_rows(
    flat: &[f64],
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<f64>>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Collinear {
                up_occupations,
                down_occupations,
                ..
            } if up_occupations == down_occupations && up_occupations.end <= flat.len() => {
                Ok(flat[up_occupations.clone()].to_vec())
            }
            _ => Err(GammaValenceHfError::SpinorFirstVariation),
        })
        .collect()
}

fn second_variation_energies(
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<Hartree>>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Collinear {
                solutions,
                up_occupations,
                down_occupations,
                ..
            } if up_occupations == down_occupations => Ok(solutions.up.eigenvalues.clone()),
            _ => Err(GammaValenceHfError::SpinorFirstVariation),
        })
        .collect()
}

fn scalar_k_weights(bands: &CheckpointBandSolution) -> Result<Vec<f64>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match point.solution {
            CheckpointKPointSolution::Collinear { .. } => Ok(point.weight()),
            CheckpointKPointSolution::Spinor { .. } => {
                Err(GammaValenceHfError::SpinorFirstVariation)
            }
        })
        .collect()
}

fn spinor_energies(
    bands: &CheckpointBandSolution,
) -> Result<Vec<Vec<Hartree>>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Spinor { solution, .. } => Ok(solution.eigenvalues.clone()),
            CheckpointKPointSolution::Collinear { .. } => {
                Err(GammaValenceHfError::SpinorFirstVariation)
            }
        })
        .collect()
}

fn k_weights(bands: &CheckpointBandSolution) -> Result<Vec<f64>, GammaValenceHfError> {
    bands
        .points()
        .iter()
        .map(|point| match &point.solution {
            CheckpointKPointSolution::Spinor { .. } => Ok(point.weight()),
            CheckpointKPointSolution::Collinear { .. } => {
                Err(GammaValenceHfError::SpinorFirstVariation)
            }
        })
        .collect()
}

fn density_mixer(spec: ScfMixing) -> Result<DensityMixer, MixingError> {
    match spec {
        ScfMixing::Linear { alpha } => DensityMixer::linear(alpha),
        ScfMixing::Broyden2 { alpha, history } => DensityMixer::broyden2(alpha, history),
        ScfMixing::PulayAnderson { alpha, history } => DensityMixer::pulay_anderson(alpha, history),
    }
}

fn valid_fock_mixing(mixing: FockMixing) -> bool {
    match mixing {
        FockMixing::Linear { alpha } => alpha.is_finite() && alpha > 0.0 && alpha <= 1.0,
        FockMixing::PulayAnderson { alpha, history } => {
            alpha.is_finite() && alpha > 0.0 && alpha <= 1.0 && history >= 2
        }
        FockMixing::CommutatorDiis { history } => history >= 2,
        FockMixing::QuasiNewtonDiis {
            history,
            level_shift,
        } => history >= 2 && level_shift.get().is_finite() && level_shift.get() >= 0.0,
    }
}

fn validate_spec(spec: &GammaValenceHfSpec) -> Result<(), GammaValenceHfError> {
    if spec.config.k_mesh.reduction != ScfKReduction::Full {
        return Err(GammaValenceHfError::SymmetryReduction);
    }
    if spec.config.relativity != ScfRelativity::SpinorFirstVariation {
        return Err(GammaValenceHfError::SpinorFirstVariation);
    }
    if spec
        .config
        .core_sites
        .iter()
        .any(|site| !site.states.is_empty())
    {
        return Err(GammaValenceHfError::CoreStates);
    }
    if spec.max_fock_iterations < 2 {
        return Err(GammaValenceHfError::FockIterations);
    }
    if !spec.fock_density_tolerance.is_finite() || spec.fock_density_tolerance <= 0.0 {
        return Err(GammaValenceHfError::FockTolerance);
    }
    if !spec.fock_feedback_tolerance.get().is_finite() || spec.fock_feedback_tolerance.get() <= 0.0
    {
        return Err(GammaValenceHfError::FockFeedbackTolerance);
    }
    if !valid_fock_mixing(spec.fock_mixing) {
        return Err(GammaValenceHfError::FockMixing);
    }
    Ok(())
}

fn validate_kh_soc_spec(spec: &KhSocValenceHfSpec) -> Result<f64, KhSocValenceHfError> {
    if spec.config.k_mesh.reduction != ScfKReduction::Full {
        return Err(KhSocValenceHfError::Hf(
            GammaValenceHfError::SymmetryReduction,
        ));
    }
    let core_electrons: f64 = spec
        .config
        .core_sites
        .iter()
        .flat_map(|site| &site.states)
        .map(|state| state.occupation)
        .sum();
    let core_configured = core_electrons > 0.0;
    if (core_configured && spec.core_treatment != KhSocCoreTreatment::Frozen)
        || (!core_configured && spec.core_treatment != KhSocCoreTreatment::ValenceOnly)
    {
        return Err(KhSocValenceHfError::CoreTreatment);
    }
    let valence_electrons = spec.config.electron_count - core_electrons;
    if !valence_electrons.is_finite() || valence_electrons <= 0.0 {
        return Err(KhSocValenceHfError::Hf(GammaValenceHfError::CoreStates));
    }
    if spec.max_fock_iterations < 2 {
        return Err(KhSocValenceHfError::Hf(GammaValenceHfError::FockIterations));
    }
    if !spec.fock_density_tolerance.is_finite() || spec.fock_density_tolerance <= 0.0 {
        return Err(KhSocValenceHfError::Hf(GammaValenceHfError::FockTolerance));
    }
    if !spec.fock_feedback_tolerance.get().is_finite() || spec.fock_feedback_tolerance.get() <= 0.0
    {
        return Err(KhSocValenceHfError::Hf(
            GammaValenceHfError::FockFeedbackTolerance,
        ));
    }
    if !valid_fock_mixing(spec.fock_mixing) {
        return Err(KhSocValenceHfError::Hf(GammaValenceHfError::FockMixing));
    }
    Ok(valence_electrons)
}

fn validate_gamma_spec(spec: &GammaValenceHfSpec) -> Result<(), GammaValenceHfError> {
    let mesh = spec.config.k_mesh;
    if mesh.divisions != [1, 1, 1]
        || mesh.shift != [0.0; 3]
        || mesh.reduction != ScfKReduction::Full
    {
        return Err(GammaValenceHfError::GammaMesh);
    }
    Ok(())
}

fn validate_relaxed_core_spec(spec: &RelaxedCoreHfSpec) -> Result<f64, RelaxedCoreHfError> {
    if spec.config.k_mesh.reduction != ScfKReduction::Full {
        return Err(RelaxedCoreHfError::SymmetryReduction);
    }
    if spec.config.relativity != ScfRelativity::SpinorFirstVariation {
        return Err(RelaxedCoreHfError::SpinorFirstVariation);
    }
    let core_electrons: f64 = spec
        .config
        .core_sites
        .iter()
        .flat_map(|site| &site.states)
        .map(|state| state.occupation)
        .sum();
    if !core_electrons.is_finite() || core_electrons <= 0.0 {
        return Err(RelaxedCoreHfError::CoreStates);
    }
    let valence_electrons = spec.config.electron_count - core_electrons;
    if !valence_electrons.is_finite() || valence_electrons <= 0.0 {
        return Err(RelaxedCoreHfError::ValenceElectronCount);
    }
    if spec.max_fock_iterations < 2 {
        return Err(RelaxedCoreHfError::FockIterations);
    }
    if !spec.fock_density_tolerance.is_finite() || spec.fock_density_tolerance <= 0.0 {
        return Err(RelaxedCoreHfError::FockTolerance);
    }
    if !spec.fock_feedback_tolerance.get().is_finite() || spec.fock_feedback_tolerance.get() <= 0.0
    {
        return Err(RelaxedCoreHfError::FockFeedbackTolerance);
    }
    if !valid_fock_mixing(spec.fock_mixing) {
        return Err(RelaxedCoreHfError::FockMixing);
    }
    if !spec.sector_numerical_tolerance.get().is_finite()
        || spec.sector_numerical_tolerance.get() < 0.0
    {
        return Err(RelaxedCoreHfError::SectorTolerance);
    }
    if !spec.maximum_core_shell_spill.is_finite() || spec.maximum_core_shell_spill < 0.0 {
        return Err(RelaxedCoreHfError::CoreSpillTolerance);
    }
    Ok(valence_electrons)
}

fn q_topology_error(_error: CanonicalQMapError) -> GammaValenceHfError {
    GammaValenceHfError::QTopology
}

fn require_gate(
    gate: &'static str,
    residual: f64,
    tolerance: f64,
) -> Result<(), GammaValenceHfError> {
    if residual.is_finite() && residual <= tolerance {
        Ok(())
    } else {
        Err(GammaValenceHfError::Gate {
            gate,
            residual,
            tolerance,
        })
    }
}

fn require_relaxed_gate(
    gate: &'static str,
    residual: f64,
    tolerance: f64,
) -> Result<(), RelaxedCoreHfError> {
    if residual.is_finite() && residual <= tolerance {
        Ok(())
    } else {
        Err(RelaxedCoreHfError::Gate {
            gate,
            residual,
            tolerance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use muffintin_tensor::DenseEigenvectors;

    #[test]
    fn physical_feedback_residual_is_invariant_under_band_gauge_rotation() {
        let inverse_sqrt_two = 1.0 / 2.0_f64.sqrt();
        let overlap =
            DenseHermitianMatrix::from_upper_triangle(2, Axis::GlobalBasis, |row, column| {
                if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::default()
                }
            })
            .unwrap();
        let identity = DenseEigenvectors::from_host_column_major(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::default(),
                Complex64::default(),
                Complex64::new(1.0, 0.0),
            ],
        )
        .unwrap();
        let rotated = DenseEigenvectors::from_host_column_major(
            2,
            2,
            vec![
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(inverse_sqrt_two, 0.0),
                Complex64::new(-inverse_sqrt_two, 0.0),
                Complex64::new(inverse_sqrt_two, 0.0),
            ],
        )
        .unwrap();
        let band_feedback =
            DenseHermitianMatrix::from_upper_triangle(2, Axis::Band, |row, column| {
                match (row, column) {
                    (0, 0) => Complex64::new(1.0, 0.0),
                    (0, 1) => Complex64::new(0.2, 0.3),
                    (1, 1) => Complex64::new(-0.4, 0.0),
                    _ => unreachable!(),
                }
            })
            .unwrap();
        let rotated_band_feedback =
            DenseHermitianMatrix::from_upper_triangle(2, Axis::Band, |row, column| {
                let mut value = Complex64::default();
                for left in 0..2 {
                    for right in 0..2 {
                        value += rotated.at(left, row).conj()
                            * band_feedback.at(left, right)
                            * rotated.at(right, column);
                    }
                }
                value
            })
            .unwrap();
        let original_global =
            lift_band_hermitian_feedback(&overlap, &identity, &band_feedback).unwrap();
        let rotated_global =
            lift_band_hermitian_feedback(&overlap, &rotated, &rotated_band_feedback).unwrap();

        let residual = global_feedback_difference(&[original_global], &[rotated_global]).unwrap();
        assert!(residual <= 1.0e-12, "physical feedback residual {residual}");
    }
}
