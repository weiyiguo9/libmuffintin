//! Production self-consistent-field state machine and basis-neutral physics seam.

use std::error::Error;

use muffintin_core::{Hartree, Kappa};
use thiserror::Error;

use crate::soc::FirstVariationWindow;
use crate::xc::XcFunctional;
use crate::xc_field::NoncollinearXcRoute;
use crate::{
    BandState, DensityMixer, EnergyError, GeneratedLinearizationEnergy,
    LinearizationEnergyGenerator, MixStatus, MixingError, OccupationEnergy, OccupationError,
    RegionalDensity, RegionalError, RegionalPotential, ScfEnergy, TetrahedronDosBins,
    TetrahedronError, assemble_scf_energy, solve_fermi_dirac, solve_gaussian, tetrahedron_dos_bins,
};

const ELECTRON_TOLERANCE: f64 = 1.0e-12;
const OCCUPATION_MAX_ITERATIONS: usize = 256;
const BASIS_REFINEMENT_MAX_PASSES: usize = 16;

/// A regular full-Brillouin-zone mesh in fractional reciprocal coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfKMesh {
    pub divisions: [usize; 3],
    pub shift: [f64; 3],
}

/// LAPW basis controls that do not prescribe a concrete basis representation.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfBasis {
    pub plane_wave_cutoff: f64,
    pub l_max: u32,
    /// Immutable, normalized channel requests for each outer SCF iteration.
    pub channels: Vec<ScfChannelRecipe>,
    /// Current-potential energies materialized from `channels`.
    pub resolved_channels: Vec<ScfResolvedChannelEnergy>,
}

/// Route-independent radial-channel identity.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ScfChannelIdentity {
    ScalarL { n: u32, l: u32 },
    Kappa { n: u32, kappa: i32 },
}

/// Physical role assigned to one radial channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScfChannelTreatment {
    Core,
    Valence,
    Lo,
    Hdlo,
}

/// Stable origin category retained with a normalized channel request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScfChannelProvenance {
    BuiltIn,
    ExternalRecipe { source: Option<String> },
    TaskDefault,
    Species,
    Site,
}

/// One site-resolved normalized channel request.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfChannelRecipe {
    pub site: String,
    pub identity: ScfChannelIdentity,
    pub treatment: ScfChannelTreatment,
    pub derivative_order: u32,
    pub generator: LinearizationEnergyGenerator,
    pub seed: Option<Hartree>,
    pub provenance: ScfChannelProvenance,
}

/// One materialized channel energy and all generator components used to form it.
///
/// Scalar channels may retain both signed-`kappa` partner diagnostics while
/// exposing their degeneracy-weighted average as `energy`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfResolvedChannelEnergy {
    pub recipe: ScfChannelRecipe,
    pub energy: Hartree,
    pub components: Vec<GeneratedLinearizationEnergy>,
}

/// Finite-temperature occupation functional used during SCF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScfOccupations {
    FermiDirac { temperature: Hartree },
    Gaussian { width: Hartree },
}

/// Functional and noncollinear reduction selected for XC evaluation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScfExchangeCorrelation {
    pub functional: XcFunctional,
    pub noncollinear_route: NoncollinearXcRoute,
}

/// Density mixer and its persistent-history controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScfMixing {
    Linear { alpha: f64 },
    Broyden2 { alpha: f64, history: usize },
    PulayAnderson { alpha: f64, history: usize },
}

impl ScfMixing {
    fn build(self) -> Result<DensityMixer, MixingError> {
        match self {
            Self::Linear { alpha } => DensityMixer::linear(alpha),
            Self::Broyden2 { alpha, history } => DensityMixer::broyden2(alpha, history),
            Self::PulayAnderson { alpha, history } => DensityMixer::pulay_anderson(alpha, history),
        }
    }
}

/// One-particle relativistic route. Its numerical implementation belongs to the kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScfRelativity {
    Scalar,
    SocSecondVariation { window: FirstVariationWindow },
    SpinorFirstVariation,
}

/// One requested occupied bound-core channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfCoreState {
    pub principal_quantum_number: u32,
    pub kappa: i32,
    pub occupation: f64,
}

/// All requested core channels at one physical site.
///
/// Every site appears exactly once in [`ScfConfig::core_sites`], including sites
/// with no explicitly selected channels, so the four-component core solve remains
/// an observable per-iteration, per-site operation.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfCoreSite {
    pub id: String,
    pub states: Vec<ScfCoreState>,
}

/// Energy and density stopping thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfConvergence {
    pub energy_tolerance: Hartree,
    pub density_tolerance: f64,
    pub max_iterations: usize,
}

/// Complete basis-neutral controls for one SCF state machine.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfConfig {
    pub electron_count: f64,
    pub k_mesh: ScfKMesh,
    pub basis: ScfBasis,
    pub occupations: ScfOccupations,
    pub exchange_correlation: ScfExchangeCorrelation,
    pub mixing: ScfMixing,
    pub relativity: ScfRelativity,
    pub convergence: ScfConvergence,
    pub core_sites: Vec<ScfCoreSite>,
}

impl ScfConfig {
    fn validate(&self) -> Result<(), ScfConfigError> {
        if !self.electron_count.is_finite() || self.electron_count <= 0.0 {
            return Err(ScfConfigError::InvalidElectronCount(self.electron_count));
        }
        for (axis, &division) in self.k_mesh.divisions.iter().enumerate() {
            if division == 0 {
                return Err(ScfConfigError::ZeroKMeshDivision { axis });
            }
        }
        for (axis, &shift) in self.k_mesh.shift.iter().enumerate() {
            if !shift.is_finite() {
                return Err(ScfConfigError::NonFiniteKMeshShift { axis, shift });
            }
        }
        if !self.basis.plane_wave_cutoff.is_finite() || self.basis.plane_wave_cutoff <= 0.0 {
            return Err(ScfConfigError::InvalidPlaneWaveCutoff(
                self.basis.plane_wave_cutoff,
            ));
        }
        if self.basis.l_max == 0 {
            return Err(ScfConfigError::ZeroLMax);
        }
        for recipe in &self.basis.channels {
            validate_channel_recipe(recipe)?;
            if matches!(self.relativity, ScfRelativity::SocSecondVariation { .. })
                && matches!(recipe.identity, ScfChannelIdentity::Kappa { .. })
                && matches!(
                    recipe.treatment,
                    ScfChannelTreatment::Lo | ScfChannelTreatment::Hdlo
                )
            {
                return Err(
                    ScfConfigError::SignedKappaLocalOrbitalUnsupportedInSecondVariation {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                    },
                );
            }
        }
        for resolved in &self.basis.resolved_channels {
            validate_channel_recipe(&resolved.recipe)?;
            if !self.basis.channels.contains(&resolved.recipe) {
                return Err(ScfConfigError::ResolvedChannelRecipeNotRequested {
                    site: resolved.recipe.site.clone(),
                    identity: resolved.recipe.identity,
                    derivative_order: resolved.recipe.derivative_order,
                });
            }
            if !resolved.energy.get().is_finite() {
                return Err(ScfConfigError::NonFiniteResolvedChannelEnergy {
                    site: resolved.recipe.site.clone(),
                    energy: resolved.energy.get(),
                });
            }
            if resolved.components.is_empty() {
                return Err(ScfConfigError::EmptyResolvedChannelComponents {
                    site: resolved.recipe.site.clone(),
                    identity: resolved.recipe.identity,
                });
            }
            for (component, generated) in resolved.components.iter().enumerate() {
                if let Some(seed) = generated.seed {
                    if !seed.get().is_finite() {
                        return Err(ScfConfigError::NonFiniteResolvedChannelComponentSeed {
                            site: resolved.recipe.site.clone(),
                            component,
                            seed: seed.get(),
                        });
                    }
                }
                if !generated.energy.get().is_finite() {
                    return Err(ScfConfigError::NonFiniteResolvedChannelComponent {
                        site: resolved.recipe.site.clone(),
                        component,
                        energy: generated.energy.get(),
                    });
                }
            }
        }
        let scale = match self.occupations {
            ScfOccupations::FermiDirac { temperature } => temperature.get(),
            ScfOccupations::Gaussian { width } => width.get(),
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(ScfConfigError::InvalidOccupationScale(scale));
        }
        self.mixing.build()?;
        if !self.convergence.energy_tolerance.get().is_finite()
            || self.convergence.energy_tolerance.get() <= 0.0
        {
            return Err(ScfConfigError::InvalidEnergyTolerance(
                self.convergence.energy_tolerance.get(),
            ));
        }
        if !self.convergence.density_tolerance.is_finite()
            || self.convergence.density_tolerance <= 0.0
        {
            return Err(ScfConfigError::InvalidDensityTolerance(
                self.convergence.density_tolerance,
            ));
        }
        if self.convergence.max_iterations == 0 {
            return Err(ScfConfigError::ZeroMaxIterations);
        }
        if self.core_sites.is_empty() {
            return Err(ScfConfigError::NoCoreSites);
        }
        for site in &self.core_sites {
            if site.id.trim().is_empty() {
                return Err(ScfConfigError::EmptyCoreSiteId);
            }
            for state in &site.states {
                if state.principal_quantum_number == 0 {
                    return Err(ScfConfigError::ZeroPrincipalQuantumNumber {
                        site: site.id.clone(),
                    });
                }
                let kappa = Kappa::new(state.kappa).map_err(|_| ScfConfigError::InvalidKappa {
                    site: site.id.clone(),
                    kappa: state.kappa,
                })?;
                let capacity = f64::from(kappa.degeneracy());
                if !state.occupation.is_finite()
                    || state.occupation <= 0.0
                    || state.occupation > capacity
                {
                    return Err(ScfConfigError::InvalidCoreOccupation {
                        site: site.id.clone(),
                        occupation: state.occupation,
                        capacity,
                    });
                }
            }
        }
        let core_electrons = self.core_electron_count();
        if core_electrons >= self.electron_count {
            return Err(ScfConfigError::NoValenceElectrons {
                total: self.electron_count,
                core: core_electrons,
            });
        }
        Ok(())
    }

    fn core_electron_count(&self) -> f64 {
        self.core_sites
            .iter()
            .flat_map(|site| &site.states)
            .map(|state| state.occupation)
            .sum()
    }
}

fn validate_channel_recipe(recipe: &ScfChannelRecipe) -> Result<(), ScfConfigError> {
    if recipe.site.trim().is_empty() {
        return Err(ScfConfigError::EmptyChannelSite);
    }
    match recipe.identity {
        ScfChannelIdentity::ScalarL { n, l } if n <= l => {
            return Err(ScfConfigError::InvalidScalarChannelIdentity {
                site: recipe.site.clone(),
                n,
                l,
            });
        }
        ScfChannelIdentity::Kappa { n, kappa } => {
            let kappa =
                Kappa::new(kappa).map_err(|_| ScfConfigError::InvalidKappaChannelIdentity {
                    site: recipe.site.clone(),
                    n,
                    kappa,
                })?;
            if n <= kappa.large_l() {
                return Err(ScfConfigError::InvalidKappaChannelIdentity {
                    site: recipe.site.clone(),
                    n,
                    kappa: kappa.get(),
                });
            }
        }
        ScfChannelIdentity::ScalarL { .. } => {}
    }
    if let Some(seed) = recipe.seed {
        if !seed.get().is_finite() {
            return Err(ScfConfigError::NonFiniteChannelSeed {
                site: recipe.site.clone(),
                seed: seed.get(),
            });
        }
    }
    Ok(())
}

/// Terms evaluated by the physics kernel after density synthesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfEnergyTerms {
    pub madelung: Hartree,
    pub coulomb: Hartree,
    pub exchange_correlation: Hartree,
    pub exchange_correlation_potential: Hartree,
}

/// Explicit output of one site's four-component core Dirac solve.
///
/// `density` is the global regional contribution formed from `P^2 + Q^2`.
#[derive(Clone, Debug, PartialEq)]
pub struct CoreContribution {
    pub site_id: String,
    pub density: RegionalDensity,
    pub eigenvalue_sum: Hartree,
}

/// Read-only context supplied to the kernel for full-potential energy terms.
#[derive(Debug)]
pub struct ScfEnergyContext<'a, OneParticle, BandSolution> {
    pub iteration: usize,
    pub input_density: &'a RegionalDensity,
    pub output_density: &'a RegionalDensity,
    pub potential: &'a RegionalPotential,
    pub one_particle: &'a OneParticle,
    pub bands: &'a BandSolution,
    pub occupations: &'a [f64],
    pub chemical_potential: Hartree,
    pub core_eigenvalue_sum: Hartree,
}

/// One completed iteration, retained in stable iteration order.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfIterationDiagnostic {
    pub iteration: usize,
    pub chemical_potential: Hartree,
    pub density_rms: f64,
    pub energy_change: Option<Hartree>,
    pub energy: ScfEnergy,
    /// Exact current-potential channel energies used for this iteration.
    pub resolved_channels: Vec<ScfResolvedChannelEnergy>,
    /// Mixer status that produced the next input, or [`MixStatus::NotMixed`]
    /// when this iteration performed no mix.
    pub mixing: MixStatus,
}

/// Converged state consumed by later SCF, bands, and DOS tasks.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfState {
    /// Accepted fixed-point input density that generated `potential`.
    pub density: RegionalDensity,
    pub potential: RegionalPotential,
    /// Exact materialized basis and channel energies for frozen-potential consumers.
    pub basis: ScfBasis,
    pub chemical_potential: Hartree,
    pub energy: ScfEnergy,
    pub relativity: ScfRelativity,
    pub diagnostics: Vec<ScfIterationDiagnostic>,
}

impl ScfState {
    pub fn iterations(&self) -> usize {
        self.diagnostics.len()
    }
}

/// One labeled reciprocal point requested by a band-path task.
#[derive(Clone, Debug, PartialEq)]
pub struct BandPathPoint {
    pub label: String,
    pub k: [f64; 3],
}

/// Frozen-potential band-path request.
#[derive(Clone, Debug, PartialEq)]
pub struct BandPathRequest {
    pub bands: usize,
    pub points: Vec<BandPathPoint>,
}

/// Energies at one labeled band-path point.
#[derive(Clone, Debug, PartialEq)]
pub struct BandPathPointResult {
    pub label: String,
    pub k: [f64; 3],
    pub energies: Vec<Hartree>,
}

/// Frozen-potential band result in request order.
#[derive(Clone, Debug, PartialEq)]
pub struct BandPathResult {
    pub points: Vec<BandPathPointResult>,
}

/// Regular-mesh tetrahedron DOS request.
#[derive(Clone, Debug, PartialEq)]
pub struct DosRequest {
    pub k_mesh: ScfKMesh,
    pub edges: Vec<Hartree>,
    /// Requested downstream presentation broadening. Tetrahedron integration
    /// itself remains unsmeared and is always the source of the returned bins.
    pub broadening: Hartree,
}

/// DOS obtained from a regular full-BZ spectrum by tetrahedron integration.
#[derive(Clone, Debug, PartialEq)]
pub struct DosResult {
    pub mesh: ScfKMesh,
    pub broadening: Hartree,
    pub tetrahedron: TetrahedronDosBins,
}

/// One stable, basis-neutral seam for all material-specific DFT work.
///
/// The driver owns ordering, occupations, core-density accumulation, energy
/// bookkeeping, convergence, and mixing. The implementation owns Hartree+XC,
/// four-component core physics, radial/basis/H/S assembly, scalar/SV/spinor
/// routing, orbital density synthesis, and frozen-potential band spectra.
pub trait ScfPhysics {
    type Error: Error + Send + Sync + 'static;
    type OneParticle;
    type BandSolution;

    fn initial_density(&mut self, config: &ScfConfig) -> Result<RegionalDensity, Self::Error>;

    fn build_potential(
        &mut self,
        iteration: usize,
        density: &RegionalDensity,
        exchange_correlation: ScfExchangeCorrelation,
    ) -> Result<RegionalPotential, Self::Error>;

    fn solve_core(
        &mut self,
        iteration: usize,
        site: &ScfCoreSite,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        relativity: ScfRelativity,
    ) -> Result<CoreContribution, Self::Error>;

    fn assemble_one_particle(
        &mut self,
        iteration: usize,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        relativity: ScfRelativity,
    ) -> Result<Self::OneParticle, Self::Error>;

    /// Basis controls retained in a converged state.
    ///
    /// Kernels that resolve potential-dependent basis parameters may replace
    /// the requested controls with the exact materialized iteration basis.
    fn retained_basis(&self, requested: &ScfBasis, _one_particle: &Self::OneParticle) -> ScfBasis {
        requested.clone()
    }

    fn solve_regular_bands(
        &mut self,
        iteration: usize,
        one_particle: &Self::OneParticle,
        k_mesh: ScfKMesh,
        relativity: ScfRelativity,
    ) -> Result<Self::BandSolution, Self::Error>;

    fn band_states<'a>(&self, bands: &'a Self::BandSolution) -> &'a [BandState];

    /// Resolve generators that depend on the provisional spectrum in this outer iteration.
    ///
    /// Returning a replacement one-particle problem requests another band and
    /// occupation pass against the same potential and immutable requested basis.
    #[allow(clippy::too_many_arguments)]
    fn refine_one_particle(
        &mut self,
        _iteration: usize,
        _potential: &RegionalPotential,
        _requested_basis: &ScfBasis,
        _one_particle: &Self::OneParticle,
        _bands: &Self::BandSolution,
        _occupations: &[f64],
        _chemical_potential: Hartree,
        _relativity: ScfRelativity,
    ) -> Result<Option<Self::OneParticle>, Self::Error> {
        Ok(None)
    }

    fn synthesize_valence_density(
        &mut self,
        iteration: usize,
        bands: &Self::BandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, Self::Error>;

    fn energy_terms(
        &mut self,
        context: ScfEnergyContext<'_, Self::OneParticle, Self::BandSolution>,
    ) -> Result<ScfEnergyTerms, Self::Error>;

    /// Return one vector of `request.bands` finite energies for each request point.
    fn solve_band_path(
        &mut self,
        state: &ScfState,
        request: &BandPathRequest,
    ) -> Result<Vec<Vec<Hartree>>, Self::Error>;

    /// Return an explicitly regular full-BZ spectrum for tetrahedron integration.
    fn solve_dos_spectrum(
        &mut self,
        state: &ScfState,
        request: &DosRequest,
    ) -> Result<crate::RegularSpectrum, Self::Error>;
}

/// Execute one SCF task, optionally restarting from an earlier converged state.
pub fn run_scf<P: ScfPhysics>(
    physics: &mut P,
    config: &ScfConfig,
    source: Option<&ScfState>,
) -> Result<ScfState, ScfError<P::Error>> {
    config.validate()?;
    let valence_electron_count = config.electron_count - config.core_electron_count();
    let mut mixer = config.mixing.build().map_err(ScfConfigError::from)?;
    let mut input_density = match source {
        Some(state) => state.density.clone(),
        None => physics
            .initial_density(config)
            .map_err(|source| ScfError::Kernel {
                operation: "initial density",
                source,
            })?,
    };
    let mut previous_energy = None;
    let mut diagnostics = Vec::with_capacity(config.convergence.max_iterations);

    for iteration in 1..=config.convergence.max_iterations {
        let potential = physics
            .build_potential(iteration, &input_density, config.exchange_correlation)
            .map_err(|source| ScfError::Kernel {
                operation: "Hartree+XC potential",
                source,
            })?;

        let mut core_density = input_density.zero_like();
        let mut core_eigenvalue_sum = Hartree(0.0);
        for site in &config.core_sites {
            let contribution = physics
                .solve_core(
                    iteration,
                    site,
                    &potential,
                    &config.basis,
                    config.relativity,
                )
                .map_err(|source| ScfError::Kernel {
                    operation: "four-component core solve",
                    source,
                })?;
            if contribution.site_id != site.id {
                return Err(ScfError::WrongCoreSite {
                    expected: site.id.clone(),
                    actual: contribution.site_id,
                });
            }
            core_density.add_scaled(1.0, &contribution.density)?;
            core_eigenvalue_sum += contribution.eigenvalue_sum;
        }

        let mut one_particle = physics
            .assemble_one_particle(iteration, &potential, &config.basis, config.relativity)
            .map_err(|source| ScfError::Kernel {
                operation: "radial/basis/H/S assembly",
                source,
            })?;
        let (bands, occupation) = {
            let mut passes = 0;
            loop {
                passes += 1;
                let bands = physics
                    .solve_regular_bands(iteration, &one_particle, config.k_mesh, config.relativity)
                    .map_err(|source| ScfError::Kernel {
                        operation: "regular full-BZ band solve",
                        source,
                    })?;
                let occupation = solve_occupations(
                    physics.band_states(&bands),
                    valence_electron_count,
                    config.occupations,
                )?;
                let refinement = physics
                    .refine_one_particle(
                        iteration,
                        &potential,
                        &config.basis,
                        &one_particle,
                        &bands,
                        &occupation.occupations,
                        occupation.chemical_potential,
                        config.relativity,
                    )
                    .map_err(|source| ScfError::Kernel {
                        operation: "spectral basis refinement",
                        source,
                    })?;
                match refinement {
                    None => break (bands, occupation),
                    Some(_) if passes == BASIS_REFINEMENT_MAX_PASSES => {
                        return Err(ScfError::BasisRefinementNotConverged { iteration, passes });
                    }
                    Some(refined) => one_particle = refined,
                }
            }
        };
        let materialized_basis = physics.retained_basis(&config.basis, &one_particle);
        let mut output_density = physics
            .synthesize_valence_density(iteration, &bands, &occupation.occupations)
            .map_err(|source| ScfError::Kernel {
                operation: "valence density synthesis",
                source,
            })?;
        output_density.add_scaled(1.0, &core_density)?;

        let energy_terms = physics
            .energy_terms(ScfEnergyContext {
                iteration,
                input_density: &input_density,
                output_density: &output_density,
                potential: &potential,
                one_particle: &one_particle,
                bands: &bands,
                occupations: &occupation.occupations,
                chemical_potential: occupation.chemical_potential,
                core_eigenvalue_sum,
            })
            .map_err(|source| ScfError::Kernel {
                operation: "full-potential energy terms",
                source,
            })?;
        let energy = assemble_scf_energy(
            occupation.band_energy,
            core_eigenvalue_sum,
            energy_terms.madelung,
            energy_terms.coulomb,
            energy_terms.exchange_correlation,
            energy_terms.exchange_correlation_potential,
            occupation.energy,
        )?;

        let density_rms = input_density.difference_rms(&output_density)?;
        let energy_change = previous_energy
            .map(|previous: Hartree| Hartree((energy.total.get() - previous.get()).abs()));
        let mut diagnostic = ScfIterationDiagnostic {
            iteration,
            chemical_potential: occupation.chemical_potential,
            density_rms,
            energy_change,
            energy,
            resolved_channels: materialized_basis.resolved_channels.clone(),
            mixing: MixStatus::NotMixed,
        };

        let converged = density_rms <= config.convergence.density_tolerance
            && energy_change
                .is_some_and(|change| change.get() <= config.convergence.energy_tolerance.get());
        if converged {
            diagnostics.push(diagnostic);
            return Ok(ScfState {
                density: input_density,
                potential,
                basis: materialized_basis,
                chemical_potential: occupation.chemical_potential,
                energy,
                relativity: config.relativity,
                diagnostics,
            });
        }
        if iteration == config.convergence.max_iterations {
            diagnostics.push(diagnostic);
            return Err(ScfError::NotConverged {
                iterations: iteration,
                density_rms,
                energy_change,
                diagnostics,
            });
        }
        let mixed = mixer
            .mix(&input_density, &output_density)
            .map_err(|source| ScfError::MixingFailed { iteration, source })?;
        diagnostic.mixing = mixed.status;
        diagnostics.push(diagnostic);
        previous_energy = Some(energy.total);
        input_density = mixed.density;
    }

    unreachable!("positive SCF iteration limit exits through convergence or failure")
}

/// Execute one frozen-potential band task against a converged SCF state.
pub fn run_band_path<P: ScfPhysics>(
    physics: &mut P,
    state: &ScfState,
    request: &BandPathRequest,
) -> Result<BandPathResult, ScfError<P::Error>> {
    if request.bands == 0 || request.points.len() < 2 {
        return Err(ScfError::InvalidBandRequest);
    }
    if request
        .points
        .iter()
        .flat_map(|point| point.k)
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(ScfError::InvalidBandRequest);
    }
    let energies = physics
        .solve_band_path(state, request)
        .map_err(|source| ScfError::Kernel {
            operation: "frozen-potential band path",
            source,
        })?;
    if energies.len() != request.points.len() {
        return Err(ScfError::InvalidBandResult {
            expected_points: request.points.len(),
            actual_points: energies.len(),
            expected_bands: request.bands,
        });
    }
    let mut points = Vec::with_capacity(request.points.len());
    for (point, energies) in request.points.iter().zip(energies) {
        if energies.len() != request.bands
            || energies.iter().any(|energy| !energy.get().is_finite())
        {
            return Err(ScfError::InvalidBandResult {
                expected_points: request.points.len(),
                actual_points: request.points.len(),
                expected_bands: request.bands,
            });
        }
        points.push(BandPathPointResult {
            label: point.label.clone(),
            k: point.k,
            energies,
        });
    }
    Ok(BandPathResult { points })
}

/// Execute one frozen-potential DOS task using the linear tetrahedron method.
pub fn run_dos<P: ScfPhysics>(
    physics: &mut P,
    state: &ScfState,
    request: &DosRequest,
) -> Result<DosResult, ScfError<P::Error>> {
    if !request.broadening.get().is_finite() || request.broadening.get() <= 0.0 {
        return Err(ScfError::InvalidDosBroadening(request.broadening.get()));
    }
    let spectrum = physics
        .solve_dos_spectrum(state, request)
        .map_err(|source| ScfError::Kernel {
            operation: "regular full-BZ DOS spectrum",
            source,
        })?;
    if spectrum.divisions != request.k_mesh.divisions {
        return Err(ScfError::DosMeshMismatch {
            requested: request.k_mesh.divisions,
            actual: spectrum.divisions,
        });
    }
    let tetrahedron = tetrahedron_dos_bins(&spectrum, &request.edges)?;
    Ok(DosResult {
        mesh: request.k_mesh,
        broadening: request.broadening,
        tetrahedron,
    })
}

#[derive(Debug)]
struct OccupationSolution {
    chemical_potential: Hartree,
    occupations: Vec<f64>,
    band_energy: Hartree,
    energy: OccupationEnergy,
}

fn solve_occupations(
    states: &[BandState],
    electron_count: f64,
    occupations: ScfOccupations,
) -> Result<OccupationSolution, OccupationError> {
    match occupations {
        ScfOccupations::FermiDirac { temperature } => {
            let result = solve_fermi_dirac(
                states,
                electron_count,
                temperature,
                ELECTRON_TOLERANCE,
                OCCUPATION_MAX_ITERATIONS,
            )?;
            Ok(OccupationSolution {
                chemical_potential: result.chemical_potential,
                occupations: result.occupations,
                band_energy: result.band_energy,
                energy: OccupationEnergy::FermiDirac {
                    minus_temperature_entropy: result.minus_temperature_entropy,
                },
            })
        }
        ScfOccupations::Gaussian { width } => {
            let result = solve_gaussian(
                states,
                electron_count,
                width,
                ELECTRON_TOLERANCE,
                OCCUPATION_MAX_ITERATIONS,
            )?;
            Ok(OccupationSolution {
                chemical_potential: result.chemical_potential,
                occupations: result.occupations,
                band_energy: result.band_energy,
                energy: OccupationEnergy::Gaussian {
                    smearing_correction: result.smearing_correction,
                },
            })
        }
    }
}

/// Invalid basis-neutral SCF controls.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ScfConfigError {
    #[error("electron count must be finite and positive, got {0}")]
    InvalidElectronCount(f64),
    #[error("SCF k-mesh division on axis {axis} is zero")]
    ZeroKMeshDivision { axis: usize },
    #[error("SCF k-mesh shift on axis {axis} is not finite: {shift}")]
    NonFiniteKMeshShift { axis: usize, shift: f64 },
    #[error("plane-wave cutoff must be finite and positive, got {0}")]
    InvalidPlaneWaveCutoff(f64),
    #[error("l_max must be positive")]
    ZeroLMax,
    #[error("SCF channel site must not be empty")]
    EmptyChannelSite,
    #[error("SCF channel on site {site:?} has invalid scalar identity n={n}, l={l}")]
    InvalidScalarChannelIdentity { site: String, n: u32, l: u32 },
    #[error("SCF channel on site {site:?} has invalid signed-kappa identity n={n}, kappa={kappa}")]
    InvalidKappaChannelIdentity { site: String, n: u32, kappa: i32 },
    #[error("SCF channel on site {site:?} has non-finite seed {seed} Ha")]
    NonFiniteChannelSeed { site: String, seed: f64 },
    #[error(
        "SOC second variation cannot represent signed-kappa local orbital {identity:?} on site {site:?}; use spinor first variation or remove the channel"
    )]
    SignedKappaLocalOrbitalUnsupportedInSecondVariation {
        site: String,
        identity: ScfChannelIdentity,
    },
    #[error(
        "resolved SCF channel on site {site:?} for {identity:?} at derivative order {derivative_order} has no matching requested recipe"
    )]
    ResolvedChannelRecipeNotRequested {
        site: String,
        identity: ScfChannelIdentity,
        derivative_order: u32,
    },
    #[error("resolved SCF channel on site {site:?} has non-finite energy {energy} Ha")]
    NonFiniteResolvedChannelEnergy { site: String, energy: f64 },
    #[error("resolved SCF channel on site {site:?} for {identity:?} has no generator components")]
    EmptyResolvedChannelComponents {
        site: String,
        identity: ScfChannelIdentity,
    },
    #[error(
        "resolved SCF channel component {component} on site {site:?} has non-finite energy {energy} Ha"
    )]
    NonFiniteResolvedChannelComponent {
        site: String,
        component: usize,
        energy: f64,
    },
    #[error(
        "resolved SCF channel component {component} on site {site:?} has non-finite seed {seed} Ha"
    )]
    NonFiniteResolvedChannelComponentSeed {
        site: String,
        component: usize,
        seed: f64,
    },
    #[error("occupation energy scale must be finite and positive, got {0} Ha")]
    InvalidOccupationScale(f64),
    #[error("SCF energy tolerance must be finite and positive, got {0} Ha")]
    InvalidEnergyTolerance(f64),
    #[error("SCF density tolerance must be finite and positive, got {0}")]
    InvalidDensityTolerance(f64),
    #[error("SCF maximum iteration count must be positive")]
    ZeroMaxIterations,
    #[error("SCF requires one explicit core-solve entry per physical site")]
    NoCoreSites,
    #[error("core site id must not be empty")]
    EmptyCoreSiteId,
    #[error("core site {site:?} has principal quantum number zero")]
    ZeroPrincipalQuantumNumber { site: String },
    #[error("core site {site:?} has invalid kappa {kappa}")]
    InvalidKappa { site: String, kappa: i32 },
    #[error("core site {site:?} has occupation {occupation} outside (0, {capacity}]")]
    InvalidCoreOccupation {
        site: String,
        occupation: f64,
        capacity: f64,
    },
    #[error("total electron count {total} leaves no valence electrons after {core} core electrons")]
    NoValenceElectrons { total: f64, core: f64 },
    #[error(transparent)]
    Mixing(#[from] MixingError),
}

/// SCF state-machine, material-kernel, or frozen-potential task failure.
#[derive(Debug, Error)]
pub enum ScfError<E: Error + Send + Sync + 'static> {
    #[error(transparent)]
    InvalidConfig(#[from] ScfConfigError),
    #[error("physics kernel failed during {operation}: {source}")]
    Kernel {
        operation: &'static str,
        #[source]
        source: E,
    },
    #[error("core kernel returned site {actual:?} while solving {expected:?}")]
    WrongCoreSite { expected: String, actual: String },
    #[error(transparent)]
    Regional(#[from] RegionalError),
    #[error("density mixing failed at SCF iteration {iteration}: {source}")]
    MixingFailed {
        iteration: usize,
        #[source]
        source: MixingError,
    },
    #[error(
        "spectral basis refinement did not converge at SCF iteration {iteration} after {passes} passes"
    )]
    BasisRefinementNotConverged { iteration: usize, passes: usize },
    #[error(transparent)]
    Occupation(#[from] OccupationError),
    #[error(transparent)]
    Energy(#[from] EnergyError),
    #[error(transparent)]
    Tetrahedron(#[from] TetrahedronError),
    #[error(
        "SCF did not converge after {iterations} iterations (density RMS {density_rms}, energy change {energy_change:?})"
    )]
    NotConverged {
        iterations: usize,
        density_rms: f64,
        energy_change: Option<Hartree>,
        diagnostics: Vec<ScfIterationDiagnostic>,
    },
    #[error("band request must contain at least two finite points and at least one band")]
    InvalidBandRequest,
    #[error(
        "band kernel returned an invalid shape or non-finite energy: expected {expected_points} points with {expected_bands} bands, got {actual_points} points"
    )]
    InvalidBandResult {
        expected_points: usize,
        actual_points: usize,
        expected_bands: usize,
    },
    #[error("DOS broadening must be finite and positive, got {0} Ha")]
    InvalidDosBroadening(f64),
    #[error("DOS kernel returned mesh {actual:?}, requested {requested:?}")]
    DosMeshMismatch {
        requested: [usize; 3],
        actual: [usize; 3],
    },
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::convert::Infallible;
    use std::f64::consts::TAU;

    use muffintin_core::{
        FourierLayout, GVector, InterstitialGeometry, InverseBohr, ReciprocalLattice, VolumeBohr3,
    };
    use num_complex::Complex64;

    use super::*;
    use crate::{
        InterstitialField, LinearizationEnergyDiagnostic, RegionalScalarField, RegularSpectrum,
    };

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum MockRefinement {
        None,
        OncePerIteration,
        NeverConverges,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct MockOneParticle {
        generation: usize,
    }

    struct MockBandSolution {
        generation: usize,
        states: Vec<BandState>,
    }

    struct MockPhysics {
        template: RegionalDensity,
        core_per_site: f64,
        events: Vec<String>,
        exchange_correlations: Vec<ScfExchangeCorrelation>,
        valence_occupation_sum: Option<f64>,
        refinement: MockRefinement,
        band_passes: Vec<(usize, usize)>,
        refinement_occupation_sums: Vec<(usize, usize, f64)>,
        density_generations: Vec<(usize, usize)>,
    }

    impl MockPhysics {
        fn new(core_per_site: f64) -> Self {
            Self {
                template: regional_density(1.0, 0.4),
                core_per_site,
                events: Vec::new(),
                exchange_correlations: Vec::new(),
                valence_occupation_sum: None,
                refinement: MockRefinement::None,
                band_passes: Vec::new(),
                refinement_occupation_sums: Vec::new(),
                density_generations: Vec::new(),
            }
        }

        fn with_refinement(mut self, refinement: MockRefinement) -> Self {
            self.refinement = refinement;
            self
        }

        fn core_density(&self) -> RegionalDensity {
            regional_density(self.core_per_site, 0.0)
        }
    }

    impl ScfPhysics for MockPhysics {
        type Error = Infallible;
        type OneParticle = MockOneParticle;
        type BandSolution = MockBandSolution;

        fn initial_density(&mut self, _config: &ScfConfig) -> Result<RegionalDensity, Self::Error> {
            self.events.push("initial".to_owned());
            Ok(self.template.clone())
        }

        fn build_potential(
            &mut self,
            iteration: usize,
            density: &RegionalDensity,
            exchange_correlation: ScfExchangeCorrelation,
        ) -> Result<RegionalPotential, Self::Error> {
            self.events.push(format!("potential:{iteration}"));
            self.exchange_correlations.push(exchange_correlation);
            Ok(RegionalPotential::new(
                density.charge().clone(),
                density
                    .magnetization()
                    .each_ref()
                    .map(|component| component.clone()),
            )
            .unwrap())
        }

        fn solve_core(
            &mut self,
            iteration: usize,
            site: &ScfCoreSite,
            _potential: &RegionalPotential,
            _basis: &ScfBasis,
            _relativity: ScfRelativity,
        ) -> Result<CoreContribution, Self::Error> {
            self.events.push(format!("core:{iteration}:{}", site.id));
            Ok(CoreContribution {
                site_id: site.id.clone(),
                density: self.core_density(),
                eigenvalue_sum: Hartree(-0.5),
            })
        }

        fn assemble_one_particle(
            &mut self,
            iteration: usize,
            _potential: &RegionalPotential,
            _basis: &ScfBasis,
            _relativity: ScfRelativity,
        ) -> Result<Self::OneParticle, Self::Error> {
            self.events.push(format!("assemble:{iteration}"));
            Ok(MockOneParticle { generation: 0 })
        }

        fn retained_basis(
            &self,
            requested: &ScfBasis,
            one_particle: &Self::OneParticle,
        ) -> ScfBasis {
            let mut retained = requested.clone();
            if one_particle.generation > 0 {
                retained.resolved_channels = requested
                    .channels
                    .first()
                    .cloned()
                    .map(resolved_channel)
                    .into_iter()
                    .collect();
            }
            retained
        }

        fn solve_regular_bands(
            &mut self,
            iteration: usize,
            one_particle: &Self::OneParticle,
            _k_mesh: ScfKMesh,
            relativity: ScfRelativity,
        ) -> Result<Self::BandSolution, Self::Error> {
            let route = match relativity {
                ScfRelativity::Scalar => "scalar",
                ScfRelativity::SocSecondVariation { .. } => "sv",
                ScfRelativity::SpinorFirstVariation => "spinor",
            };
            self.events.push(format!("bands:{iteration}:{route}"));
            self.band_passes.push((iteration, one_particle.generation));
            Ok(MockBandSolution {
                generation: one_particle.generation,
                states: vec![
                    BandState::new(Hartree(-1.0), 1.0, 1),
                    BandState::new(Hartree(1.0), 1.0, 1),
                ],
            })
        }

        fn band_states<'a>(&self, bands: &'a Self::BandSolution) -> &'a [BandState] {
            &bands.states
        }

        fn refine_one_particle(
            &mut self,
            iteration: usize,
            _potential: &RegionalPotential,
            _requested_basis: &ScfBasis,
            one_particle: &Self::OneParticle,
            _bands: &Self::BandSolution,
            occupations: &[f64],
            _chemical_potential: Hartree,
            _relativity: ScfRelativity,
        ) -> Result<Option<Self::OneParticle>, Self::Error> {
            self.refinement_occupation_sums.push((
                iteration,
                one_particle.generation,
                occupations.iter().sum(),
            ));
            let refine = match self.refinement {
                MockRefinement::None => false,
                MockRefinement::OncePerIteration => one_particle.generation == 0,
                MockRefinement::NeverConverges => true,
            };
            Ok(refine.then_some(MockOneParticle {
                generation: one_particle.generation + 1,
            }))
        }

        fn synthesize_valence_density(
            &mut self,
            iteration: usize,
            bands: &Self::BandSolution,
            occupations: &[f64],
        ) -> Result<RegionalDensity, Self::Error> {
            self.valence_occupation_sum = Some(occupations.iter().sum());
            self.density_generations.push((iteration, bands.generation));
            self.events.push(format!("valence:{iteration}"));
            Ok(regional_density(0.0, 0.1))
        }

        fn energy_terms(
            &mut self,
            context: ScfEnergyContext<'_, Self::OneParticle, Self::BandSolution>,
        ) -> Result<ScfEnergyTerms, Self::Error> {
            self.events.push(format!("energy:{}", context.iteration));
            Ok(ScfEnergyTerms {
                madelung: Hartree(0.0),
                coulomb: Hartree(0.0),
                exchange_correlation: Hartree(0.0),
                exchange_correlation_potential: Hartree(0.0),
            })
        }

        fn solve_band_path(
            &mut self,
            _state: &ScfState,
            request: &BandPathRequest,
        ) -> Result<Vec<Vec<Hartree>>, Self::Error> {
            self.events.push("path".to_owned());
            Ok(request
                .points
                .iter()
                .map(|_| {
                    (0..request.bands)
                        .map(|band| Hartree(band as f64))
                        .collect()
                })
                .collect())
        }

        fn solve_dos_spectrum(
            &mut self,
            _state: &ScfState,
            request: &DosRequest,
        ) -> Result<RegularSpectrum, Self::Error> {
            self.events.push("dos".to_owned());
            let count = request.k_mesh.divisions.into_iter().product();
            Ok(RegularSpectrum::new(
                request.k_mesh.divisions,
                [
                    [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
                    [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
                    [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
                ],
                vec![Hartree(0.0); count],
                vec![2],
            )
            .unwrap())
        }
    }

    fn regional_density(charge_value: f64, magnetization_x_value: f64) -> RegionalDensity {
        let reciprocal = ReciprocalLattice::new([
            [InverseBohr(1.0), InverseBohr(0.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(1.0), InverseBohr(0.0)],
            [InverseBohr(0.0), InverseBohr(0.0), InverseBohr(1.0)],
        ])
        .unwrap();
        let layout = FourierLayout::new(
            reciprocal,
            vec![GVector {
                index: [0; 3],
                cartesian: [InverseBohr(0.0); 3],
                norm: InverseBohr(0.0),
            }],
        )
        .unwrap();
        let field = |value| {
            InterstitialField::new(
                layout.clone(),
                BTreeMap::from([([0; 3], Complex64::new(value, 0.0))]),
            )
            .unwrap()
        };
        let geometry = InterstitialGeometry::new(VolumeBohr3(TAU.powi(3)), Vec::new()).unwrap();
        let scalar =
            |value| RegionalScalarField::new(geometry.clone(), Vec::new(), field(value)).unwrap();
        RegionalDensity::new(
            scalar(charge_value),
            [scalar(magnetization_x_value), scalar(0.0), scalar(0.0)],
        )
        .unwrap()
    }

    fn channel_recipe() -> ScfChannelRecipe {
        ScfChannelRecipe {
            site: "a".to_owned(),
            identity: ScfChannelIdentity::Kappa { n: 2, kappa: 1 },
            treatment: ScfChannelTreatment::Lo,
            derivative_order: 0,
            generator: LinearizationEnergyGenerator::Explicit,
            seed: Some(Hartree(-0.25)),
            provenance: ScfChannelProvenance::Site,
        }
    }

    fn resolved_channel(recipe: ScfChannelRecipe) -> ScfResolvedChannelEnergy {
        ScfResolvedChannelEnergy {
            recipe,
            energy: Hartree(-0.25),
            components: vec![GeneratedLinearizationEnergy {
                generator: LinearizationEnergyGenerator::Explicit,
                seed: Some(Hartree(-0.25)),
                energy: Hartree(-0.25),
                diagnostic: LinearizationEnergyDiagnostic::Stored,
            }],
        }
    }

    fn config(mixing: ScfMixing, occupations: ScfOccupations) -> ScfConfig {
        ScfConfig {
            electron_count: 1.0,
            k_mesh: ScfKMesh {
                divisions: [2, 2, 2],
                shift: [0.0; 3],
            },
            basis: ScfBasis {
                plane_wave_cutoff: 4.0,
                l_max: 8,
                channels: Vec::new(),
                resolved_channels: Vec::new(),
            },
            occupations,
            exchange_correlation: ScfExchangeCorrelation {
                functional: XcFunctional::LdaPw92,
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            },
            mixing,
            relativity: ScfRelativity::SocSecondVariation {
                window: FirstVariationWindow::new(0, 2).unwrap(),
            },
            convergence: ScfConvergence {
                energy_tolerance: Hartree(1.0e-12),
                density_tolerance: 1.0e-10,
                max_iterations: 50,
            },
            core_sites: vec![
                ScfCoreSite {
                    id: "a".to_owned(),
                    states: vec![ScfCoreState {
                        principal_quantum_number: 1,
                        kappa: -1,
                        occupation: 0.25,
                    }],
                },
                ScfCoreSite {
                    id: "b".to_owned(),
                    states: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn second_variation_rejects_signed_kappa_local_orbitals() {
        let mut config = config(
            ScfMixing::Linear { alpha: 1.0 },
            ScfOccupations::FermiDirac {
                temperature: Hartree(0.1),
            },
        );
        config.basis.channels.push(channel_recipe());
        assert!(matches!(
            config.validate(),
            Err(ScfConfigError::SignedKappaLocalOrbitalUnsupportedInSecondVariation {
                site,
                identity: ScfChannelIdentity::Kappa { n: 2, kappa: 1 },
            }) if site == "a"
        ));
    }

    #[test]
    fn state_machine_preserves_transverse_density_and_frozen_sv_route() {
        let mut physics = MockPhysics::new(0.125);
        let state = run_scf(
            &mut physics,
            &config(
                ScfMixing::Linear { alpha: 1.0 },
                ScfOccupations::FermiDirac {
                    temperature: Hartree(0.1),
                },
            ),
            None,
        )
        .unwrap();
        assert_eq!(state.iterations(), 2);
        assert!((physics.valence_occupation_sum.unwrap() - 0.75).abs() < ELECTRON_TOLERANCE);
        assert!(
            state
                .density
                .difference_rms(&regional_density(0.25, 0.1))
                .unwrap()
                < 1.0e-14
        );
        assert!(
            (state.density.magnetization()[0]
                .interstitial()
                .coefficient([0; 3])
                .unwrap()
                .re
                - 0.1)
                .abs()
                < 1.0e-15
        );
        assert!(physics.exchange_correlations.iter().all(|selection| {
            selection.noncollinear_route == NoncollinearXcRoute::LocalSpinFrame
        }));
        assert_eq!(
            physics.events,
            [
                "initial",
                "potential:1",
                "core:1:a",
                "core:1:b",
                "assemble:1",
                "bands:1:sv",
                "valence:1",
                "energy:1",
                "potential:2",
                "core:2:a",
                "core:2:b",
                "assemble:2",
                "bands:2:sv",
                "valence:2",
                "energy:2",
            ]
        );
        assert!(!physics.events.iter().any(|event| event == "dos"));
    }

    #[test]
    fn spectral_refinement_repeats_bands_before_consuming_the_final_basis() {
        let mut physics = MockPhysics::new(0.125).with_refinement(MockRefinement::OncePerIteration);
        let mut config = config(
            ScfMixing::Linear { alpha: 1.0 },
            ScfOccupations::FermiDirac {
                temperature: Hartree(0.1),
            },
        );
        config.basis.channels.push(channel_recipe());

        let state = run_scf(&mut physics, &config, None).unwrap();
        assert_eq!(physics.band_passes, [(1, 0), (1, 1), (2, 0), (2, 1)]);
        assert_eq!(physics.density_generations, [(1, 1), (2, 1)]);
        assert_eq!(physics.refinement_occupation_sums.len(), 4);
        assert!(
            physics
                .refinement_occupation_sums
                .iter()
                .all(|(_, _, sum)| (*sum - 0.75).abs() < ELECTRON_TOLERANCE)
        );

        let expected = vec![resolved_channel(channel_recipe())];
        assert!(
            state
                .diagnostics
                .iter()
                .all(|diagnostic| diagnostic.resolved_channels == expected)
        );
        assert_eq!(state.basis.channels, vec![channel_recipe()]);
        assert_eq!(state.basis.resolved_channels, expected);
    }

    #[test]
    fn spectral_refinement_stops_after_sixteen_passes() {
        let mut physics = MockPhysics::new(0.125).with_refinement(MockRefinement::NeverConverges);
        let error = run_scf(
            &mut physics,
            &config(
                ScfMixing::Linear { alpha: 1.0 },
                ScfOccupations::FermiDirac {
                    temperature: Hartree(0.1),
                },
            ),
            None,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ScfError::BasisRefinementNotConverged {
                iteration: 1,
                passes: BASIS_REFINEMENT_MAX_PASSES,
            }
        ));
        assert_eq!(physics.band_passes.len(), BASIS_REFINEMENT_MAX_PASSES);
        assert!(physics.density_generations.is_empty());
    }

    #[test]
    fn one_physics_seam_carries_both_xc_routes_through_all_relativistic_routes() {
        for relativity in [
            ScfRelativity::Scalar,
            ScfRelativity::SocSecondVariation {
                window: FirstVariationWindow::new(0, 2).unwrap(),
            },
            ScfRelativity::SpinorFirstVariation,
        ] {
            for noncollinear_route in [
                NoncollinearXcRoute::LocalSpinFrame,
                NoncollinearXcRoute::MagnetizationField,
            ] {
                let mut physics = MockPhysics::new(0.125);
                let mut config = config(
                    ScfMixing::Linear { alpha: 1.0 },
                    ScfOccupations::FermiDirac {
                        temperature: Hartree(0.1),
                    },
                );
                config.relativity = relativity;
                config.exchange_correlation.noncollinear_route = noncollinear_route;
                let state = run_scf(&mut physics, &config, None).unwrap();
                assert!(
                    state
                        .density
                        .difference_rms(&regional_density(0.25, 0.1))
                        .unwrap()
                        < 1.0e-14
                );
                assert!(physics.exchange_correlations.iter().all(|selection| {
                    selection.functional == XcFunctional::LdaPw92
                        && selection.noncollinear_route == noncollinear_route
                }));
            }
        }
    }

    #[test]
    fn all_three_density_mixers_advance_to_convergence() {
        for mixing in [
            ScfMixing::Linear { alpha: 0.5 },
            ScfMixing::Broyden2 {
                alpha: 0.5,
                history: 4,
            },
            ScfMixing::PulayAnderson {
                alpha: 0.5,
                history: 4,
            },
        ] {
            let mut physics = MockPhysics::new(0.125);
            let state = run_scf(
                &mut physics,
                &config(
                    mixing,
                    ScfOccupations::FermiDirac {
                        temperature: Hartree(0.1),
                    },
                ),
                None,
            )
            .unwrap();
            assert!(state.diagnostics.last().unwrap().density_rms <= 1.0e-10);
        }
    }

    #[test]
    fn both_occupation_functionals_reach_the_energy_result() {
        for occupations in [
            ScfOccupations::FermiDirac {
                temperature: Hartree(0.1),
            },
            ScfOccupations::Gaussian {
                width: Hartree(0.1),
            },
        ] {
            let mut physics = MockPhysics::new(0.125);
            let state = run_scf(
                &mut physics,
                &config(ScfMixing::Linear { alpha: 1.0 }, occupations),
                None,
            )
            .unwrap();
            match (occupations, state.energy.occupation) {
                (
                    ScfOccupations::FermiDirac { .. },
                    OccupationEnergy::FermiDirac {
                        minus_temperature_entropy,
                    },
                ) => assert!(minus_temperature_entropy.get() < 0.0),
                (
                    ScfOccupations::Gaussian { .. },
                    OccupationEnergy::Gaussian {
                        smearing_correction,
                    },
                ) => assert!(smearing_correction.get() < 0.0),
                _ => panic!("occupation correction route changed"),
            }
        }
    }

    #[test]
    fn maximum_iteration_exhaustion_is_an_error_after_core_work() {
        let mut physics = MockPhysics::new(0.125);
        let mut config = config(
            ScfMixing::Linear { alpha: 0.5 },
            ScfOccupations::FermiDirac {
                temperature: Hartree(0.1),
            },
        );
        config.convergence.max_iterations = 1;
        let Err(ScfError::NotConverged {
            iterations,
            diagnostics,
            ..
        }) = run_scf(&mut physics, &config, None)
        else {
            panic!("maximum-iteration exhaustion must fail")
        };
        assert_eq!(iterations, 1);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            physics
                .events
                .iter()
                .filter(|event| event.starts_with("core:"))
                .count(),
            2
        );
        assert_eq!(diagnostics[0].mixing, MixStatus::NotMixed);
    }

    #[test]
    fn mixer_configuration_errors_are_not_iteration_failures() {
        let mut physics = MockPhysics::new(0.125);
        let mut config = config(
            ScfMixing::Linear { alpha: 0.0 },
            ScfOccupations::FermiDirac {
                temperature: Hartree(0.1),
            },
        );
        assert!(matches!(
            run_scf(&mut physics, &config, None),
            Err(ScfError::InvalidConfig(ScfConfigError::Mixing(
                MixingError::InvalidAlpha(_)
            )))
        ));
        config.mixing = ScfMixing::Broyden2 {
            alpha: 0.5,
            history: 1,
        };
        assert!(matches!(
            run_scf(&mut physics, &config, None),
            Err(ScfError::InvalidConfig(ScfConfigError::Mixing(
                MixingError::HistoryTooShort(1)
            )))
        ));
    }

    #[test]
    fn iteration_diagnostics_record_mixer_status() {
        let mut linear_physics = MockPhysics::new(0.125);
        let linear = run_scf(
            &mut linear_physics,
            &config(
                ScfMixing::Linear { alpha: 1.0 },
                ScfOccupations::FermiDirac {
                    temperature: Hartree(0.1),
                },
            ),
            None,
        )
        .unwrap();
        assert_eq!(linear.diagnostics.len(), 2);
        assert_eq!(linear.diagnostics[0].mixing, MixStatus::Linear);
        assert_eq!(linear.diagnostics[1].mixing, MixStatus::NotMixed);

        let mut broyden_physics = MockPhysics::new(0.125);
        let broyden = run_scf(
            &mut broyden_physics,
            &config(
                ScfMixing::Broyden2 {
                    alpha: 0.5,
                    history: 4,
                },
                ScfOccupations::FermiDirac {
                    temperature: Hartree(0.1),
                },
            ),
            None,
        )
        .unwrap();
        assert_eq!(broyden.diagnostics[0].mixing, MixStatus::NonlinearWarmup);
        assert_eq!(
            broyden.diagnostics.last().unwrap().mixing,
            MixStatus::NotMixed
        );
        assert!(
            broyden.diagnostics[..broyden.diagnostics.len() - 1]
                .iter()
                .all(|diagnostic| diagnostic.mixing != MixStatus::NotMixed)
        );
    }
}
