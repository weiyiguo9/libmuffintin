//! Full-regular-BZ spinor-first valence Hartree--Fock SCF.

use muffintin_core::{FourierLayout, Hartree, InverseBohr};
use muffintin_coulomb::{CoulombRequest, HartreeError, WeinertHartreeSpec};
use muffintin_dft::{
    BandState, CheckpointBandSolution, CheckpointKPointSolution, CoreDensityError,
    CoreFixedPotentialResult, CoreFixedPotentialSpec, CoreLocalOneBodyError, CoreLocalOneBodyTrace,
    CoreShellOrbitals, DensityError, DensityMixer, ElectrostaticSpec, MaterialKernelError,
    MixingError, OccupationError, RegionalDensity, RegionalElectrostaticError, RegionalError,
    RegionalPotential, ScfConfig, ScfConfigError, ScfKReduction, ScfLoop, ScfMixing,
    ScfOccupations, ScfPhysics, ScfRelativity, build_regional_core_contribution_from_sidecar,
    core_local_one_body_trace, electron_count, evaluate_regional_electrostatics,
};
use muffintin_operators::{
    OperatorError, lift_band_hermitian_feedback, solve_generalized_hermitian,
};
use muffintin_tensor::{Axis, DenseHermitianMatrix, TensorError, einsum};
use num_complex::Complex64;
use thiserror::Error;

use crate::q_mesh::{CanonicalQMapError, canonical_q_points};
use crate::{
    CheckpointPhysics, CheckpointPhysicsError, CoreValenceComparisonSpec,
    CoreValenceDeltaDiagnostic, FrozenCoreValenceComparison, FrozenCoreValenceError,
    FrozenSpinorSectorExchange, FrozenSpinorSectorExchangeError, GammaExchangeTreatment,
    IsdfExchangeError, IsdfExchangeResult, IsdfExchangeSpec, SectorOccupations,
    SpinorCoreInputError, SpinorExchangeMpbError, SpinorExchangeMpbResult, SpinorExchangeMpbSpec,
    SpinorMpbError, SpinorMpbSelection, SpinorMpbSpec, SpinorProductInput,
    build_frozen_core_valence_exchange, build_frozen_site_valence_densities,
    build_frozen_spinor_sector_exchange, build_spinor_exchange_mpb, build_spinor_mpb,
    build_spinor_mpb_exchange, compare_frozen_core_valence, relax_frozen_core_at_fixed_potential,
};

const SPECTRAL_REFINEMENT_PASSES: usize = 16;
const IDENTITY_TOLERANCE: f64 = 1.0e-8;
const RELAXED_VALENCE_EIGENVALUE_TOLERANCE: f64 = 1.0e-6;
const ELECTRON_COUNT_TOLERANCE: f64 = 1.0e-8;

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
    /// Weight of the freshly rebuilt band-space exchange in the next Fock solve.
    pub fock_mixing: f64,
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
    pub fock_mixing: f64,
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
    #[error("relaxed-core HF fock_mixing must be finite and in (0, 1]")]
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
    #[error("valence HF fock_mixing must be finite and in (0, 1]")]
    FockMixing,
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
    Mpb(#[from] SpinorMpbError),
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
/// Only the valence component enters the configured mixer; every core density
/// is synthesized from the freshly relaxed sidecars and replaces the preceding
/// core density without mixing.
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
            RELAXED_VALENCE_EIGENVALUE_TOLERANCE,
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
            let mixed_valence = mixer.mix(&valence_density, &valence_output)?.density;
            let mixed_total = sum_density(&mixed_valence, &fresh_core_density)?;
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
) -> Result<(CheckpointBandSolution, OccupationSolution), GammaValenceHfError> {
    let mut one_particle = physics
        .kernel
        .materialize_checkpoint_one_particle(potential, &config.basis)?;
    let mut bands = physics.kernel.solve_points(
        one_particle.potential(),
        one_particle.basis(),
        k_fractional,
        ScfRelativity::SpinorFirstVariation,
    )?;
    let mut occupation = solve_occupations(bands.states(), electron_count, config.occupations)?;
    for pass in 1..=SPECTRAL_REFINEMENT_PASSES {
        let Some(refined) = physics.kernel.refine_spectral_basis(
            &config.basis,
            &one_particle,
            &bands,
            &occupation.values,
            occupation.chemical_potential,
            ScfRelativity::SpinorFirstVariation,
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
            ScfRelativity::SpinorFirstVariation,
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

struct RelaxedFixedPotentialResult {
    bands: CheckpointBandSolution,
    occupation: OccupationSolution,
    exchange: RelaxedSectorFrame,
    fock_iterations: usize,
    exchange_rebuilds: usize,
    fixed_point_residual: f64,
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
    // The next step's input bands and occupations are this step's solved ones,
    // so its input density is the density already synthesized here.
    let mut current_density = None;

    for fock_iteration in 1..=spec.max_fock_iterations {
        let occupation =
            solve_occupations(bands.states(), valence_electrons, spec.config.occupations)?;
        if outer_iteration == 1 && fock_iteration == 1 {
            let driver = rebuild_relaxed_sector_frame(
                physics,
                spec,
                &bands,
                &occupation,
                k_fractional,
                q_fractional,
                core_sidecars,
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

        let rebuilt = rebuild_relaxed_sector_frame(
            physics,
            spec,
            &bands,
            &occupation,
            k_fractional,
            q_fractional,
            core_sidecars,
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
                mix_global_feedback(previous, &fresh_global_feedback, spec.fock_mixing)?
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
            && feedback_fixed_residual <= IDENTITY_TOLERANCE
        {
            return Ok(RelaxedFixedPotentialResult {
                bands,
                occupation,
                exchange: rebuilt,
                fock_iterations: fock_iteration,
                exchange_rebuilds: rebuilds,
                fixed_point_residual: last_residual,
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

fn rebuild_relaxed_sector_frame(
    physics: &CheckpointPhysics,
    spec: &RelaxedCoreHfSpec,
    bands: &CheckpointBandSolution,
    occupation: &OccupationSolution,
    k_fractional: &[[f64; 3]],
    q_fractional: &[[f64; 3]],
    core_sidecars: &[CoreShellOrbitals],
) -> Result<RelaxedSectorFrame, RelaxedCoreHfError> {
    let inputs =
        build_q_inputs_with_cores(physics, bands, k_fractional, q_fractional, core_sidecars)?;
    let first = inputs.first().ok_or(RelaxedCoreHfError::QTopology)?;
    let n_k = first.pair_columns.n_k;
    let n_orb = first.pair_columns.n_orb;
    let selections = (0..n_k)
        .flat_map(|k| {
            (0..n_orb).flat_map(move |left_band| {
                (0..n_orb).map(move |right_band| SpinorMpbSelection {
                    k,
                    left_band,
                    right_band,
                })
            })
        })
        .collect::<Vec<_>>();
    let vv_mpb = inputs
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
    let core_mpb = inputs
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
    let occupations = sector_occupations(bands, occupation, &inputs, spec.gamma)?;
    let exchange = build_frozen_spinor_sector_exchange(
        &inputs,
        &vv_mpb,
        &core_mpb,
        &spec.coulomb,
        &occupations,
    )?;
    Ok(RelaxedSectorFrame {
        inputs,
        core_mpb,
        occupations,
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
    exchange: &FrozenSpinorSectorExchange,
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
    lifting_identity_residual: f64,
    first_global_solve_identity_residual: Option<f64>,
    first_one_shot_exchange: Option<IsdfExchangeResult>,
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
                mix_global_feedback(previous, &fresh_global_feedback, spec.fock_mixing)?
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
            && feedback_fixed_residual <= IDENTITY_TOLERANCE
        {
            return Ok(FixedPotentialResult {
                bands,
                occupation,
                exchange: rebuilt,
                fock_iterations: fock_iteration,
                exchange_rebuilds: rebuilds,
                fixed_point_residual: last_residual,
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
    if !spec.fock_mixing.is_finite() || spec.fock_mixing <= 0.0 || spec.fock_mixing > 1.0 {
        return Err(GammaValenceHfError::FockMixing);
    }
    Ok(())
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
    if !spec.fock_mixing.is_finite() || spec.fock_mixing <= 0.0 || spec.fock_mixing > 1.0 {
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
