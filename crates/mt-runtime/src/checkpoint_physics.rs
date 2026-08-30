//! Concrete production DFT kernel reconstructed from a V1 checkpoint.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::ops::Range;

use muffintin_core::{
    Bohr, ExponentialMesh, FourierFieldError, FourierLayout, GVector, Hartree,
    HermitianFourierField, InterstitialGeometry, InverseBohr, Kappa, LatticeError, MeshError,
    ReciprocalLattice, Sphere, StepFunctionError, VolumeBohr3,
};
use muffintin_dft::{
    ChannelKappaError, ScfPotentialBuildError, build_scf_potential,
    channel_kappas, channel_l, channel_n, scalar_component_energy, spin_resolved_energy,
    spinor_kappas_for_l,
    AtomicEnergyRequest, BandPathRequest, BandState, CollinearKPoint, CoreContribution,
    CoreDensityError, CorePotentialBuildError, CorePotentialBuildSpec, CoreSpinPartition,
    DensityError, FirstVariationRoute, FirstVariationSubspace, FullSpinorKPoint,
    GeneratedLinearizationEnergy, InterstitialField, LinearizationEnergyDiagnostic,
    LinearizationEnergyError, LinearizationEnergyGenerator, LocalPauliPotential, MuffinTinField,
    OccupationError, PdosEnergySample, RegionalCoreShellInput,
    RegionalDensity, RegionalElectrostaticError, RegionalElectrostaticResult, RegionalPotential,
    RegionalScalarField, RegionalXcError, RegionalXcResult, RegularSpectrum, ScalarBuilderError,
    ScalarIterationBasis, ScalarLocalOrbitalRequest, ScalarSiteInput, ScfBasis, ScfChannelIdentity,
    ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfCoreSite, ScfEnergyContext,
    ScfEnergyTerms, ScfExchangeCorrelation, ScfKMesh, ScfOccupations, ScfPhysics, ScfRelativity,
    ScfResolvedChannelEnergy, ScfState, SecondVariationError, SpinorBuilderError,
    SpinorFirstVariationError, SpinorIterationBasis, SpinorLinearizationEnergy,
    SpinorLocalOrbitalRequest, SpinorSiteInput, TetrahedronError,
    build_collinear_scalar_iteration_bases, build_extended_core_potentials,
    build_extended_checkpoint_core_potentials, build_regional_core_contribution,
    build_spinor_iteration_basis,
    generate_atomic_energy, generate_band_center_energy, generate_band_cog_energy,
    generate_explicit_energy, generate_fermi_offset_energy, generate_frozen_checkpoint_energy,
    generate_log_derivative_energy, kappa_degeneracy_average, physical_site_band_projections,
    solve_fermi_dirac, solve_gaussian, solve_soc_second_variation, solve_spinor_k_point,
    synthesize_collinear_valence_density, synthesize_full_spinor_valence_density,
};
use muffintin_envelope::{PlaneWave, PlaneWaveEnvelope};
use muffintin_io::{
    AngularBasis, Complex64V2, DensityV2, FieldRepresentationV2, FieldUnitV2,
    FourierCoefficientV2, GeometryV2, InitialV2, InterstitialFieldV2, IoError, MuffinTinFieldV2,
    PotentialV2, RadialBasisSpinV2, RadialEquationTag, RegionalFieldV2, CheckpointV2,
    SpexMaterialBasisRecipeV1, SpexMaterialChannelKind, SphericalChannelV2,
};
use muffintin_operators::lapw::{Collinear, GeneralizedEigensolution, InterstitialPotential, LapwError};
use muffintin_operators::{
    CompiledSiteProjection, OperatorError, SiteSpinOrbitBlock, SocOperatorError,
    SpinorSiteOperatorBlocks,
};
use muffintin_sphere::{
    CoreBracketSearch, CoreDiracSpec, CorePotentialContinuationSpec, CoreState, DiracError,
    EnergyBracket, ExtendedCorePotential, RadialEquation, SpexSpinOrbitPotential,
    SpinOrbitRadialError, isolate_core_dirac_bracket, solve_core_dirac,
    spex_spin_orbit_radial_shell,
};
use muffintin_sphere::{HarmonicConvention, SphereField, SphereFieldError};
use muffintin_tensor::{DenseEigenvectors, TensorError};
use num_complex::Complex64;
use thiserror::Error;

mod atomic_checkpoint;
mod basis_materialization;
mod convert_v2;

pub use atomic_checkpoint::{
    AtomicCheckpointError, AtomicCheckpointRequest, AtomicCheckpointResult,
    materialize_atomic_checkpoint_v2,
};
pub use convert_v2::checkpoint_v2_from_state;
use convert_v2::{convert_v2_site_bases, regional_density_from_v2, regional_potential_from_v2};

const OVERLAP_THRESHOLD: f64 = 1.0e-10;
const OCCUPATION_TOLERANCE: f64 = 1.0e-12;
const OCCUPATION_ITERATIONS: usize = 256;
const CHECKPOINT_RADIUS_TOLERANCE: f64 = 1.0e-10;
const TRANSVERSE_FIELD_TOLERANCE: f64 = 1.0e-10;
const SPECTRAL_REFINEMENT_TOLERANCE: f64 = 1.0e-10;
const DEFAULT_FERMI_OFFSET_HARTREE: f64 = -0.1;

/// Checkpoint-backed material kernel shared by SCF, bands, and DOS tasks.
///
/// Construction performs only convention conversion and topology validation.
/// The initial density is obtained by a frozen-checkpoint one-particle solve;
/// no atomic-density or artificial `G=0` guess is installed.
#[derive(Debug)]
pub struct CheckpointPhysics {
    checkpoint_template: CheckpointV2,
    reciprocal: ReciprocalLattice,
    geometry: InterstitialGeometry,
    sites: Vec<CheckpointSite>,
    frozen_potential: RegionalPotential,
    restart_density: Option<RegionalDensity>,
    nuclear_charges: Vec<f64>,
    core_potentials: BTreeMap<usize, CorePotentialContext>,
    density_template: Option<RegionalDensity>,
    energy_terms: BTreeMap<usize, ScfEnergyTerms>,
    spex_spinor_binding: Option<SpexSpinorMaterialBinding>,
}

#[derive(Clone, Debug)]
struct SpexSpinorMaterialBinding {
    channels: Vec<SpexBoundSpinorChannel>,
}

#[derive(Clone, Debug)]
struct SpexBoundSpinorChannel {
    l: u32,
    requested: ScfChannelRecipe,
    resolved: ScfResolvedChannelEnergy,
}

#[derive(Clone, Debug)]
struct CorePotentialContext {
    electrostatic: RegionalElectrostaticResult,
    exchange_correlation: RegionalXcResult,
    density: RegionalDensity,
    spec: CorePotentialBuildSpec,
}

#[derive(Clone, Debug)]
struct CheckpointSite {
    id: String,
    position: [Bohr; 3],
    radius: Bohr,
    up: CheckpointSpin,
    down: CheckpointSpin,
    nonmagnetic_scalar: bool,
}

struct ConvertedCheckpointGeometry {
    direct: [[Bohr; 3]; 3],
    reciprocal: ReciprocalLattice,
    geometry: InterstitialGeometry,
    sites: Vec<CheckpointSite>,
    nuclear_charges: Vec<f64>,
}


#[derive(Clone, Debug)]
struct CheckpointSpin {
    equation: RadialEquationTag,
    mesh: ExponentialMesh,
    linearization: BTreeMap<u32, Hartree>,
    local_orbitals: Vec<(u32, Hartree)>,
}

/// One iteration's potential and basis-neutral controls. Concrete k-dependent
/// APW matching is intentionally deferred until the requested k points exist.
#[derive(Clone, Debug)]
pub struct CheckpointOneParticle {
    potential: RegionalPotential,
    basis: ScfBasis,
}

/// Concrete regular-mesh solutions retained for occupations and density synthesis.
#[derive(Clone, Debug)]
pub struct CheckpointBandSolution {
    points: Vec<CheckpointKPoint>,
    states: Vec<BandState>,
}

impl CheckpointBandSolution {
    pub(crate) fn points(&self) -> &[CheckpointKPoint] {
        &self.points
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CheckpointKPoint {
    weight: f64,
    pub(crate) solution: CheckpointKPointSolution,
    energies: Vec<Hartree>,
}

#[derive(Clone, Debug)]
pub(crate) enum CheckpointKPointSolution {
    Collinear {
        bases: Box<Collinear<ScalarIterationBasis>>,
        solutions: Collinear<GeneralizedEigensolution>,
        up_occupations: Range<usize>,
        down_occupations: Range<usize>,
    },
    Spinor {
        basis: SpinorIterationBasis,
        site_blocks: Vec<SpinorSiteOperatorBlocks>,
        solution: GeneralizedEigensolution,
        occupations: Range<usize>,
    },
}

impl CheckpointPhysics {
    /// Convert a validated V2 checkpoint into exact internal units and conventions.
    pub fn new(checkpoint: &CheckpointV2) -> Result<Self, CheckpointPhysicsError> {
        checkpoint.validate()?;
        let converted = convert_checkpoint_geometry(&checkpoint.geometry)?;
        let restart_density = match &checkpoint.initial {
            InitialV2::FrozenPotential { .. } => None,
            InitialV2::Restart { density, .. } => Some(density),
        };
        let potential = match &checkpoint.initial {
            InitialV2::FrozenPotential { potential } | InitialV2::Restart { potential, .. } => {
                potential
            }
        };
        let frozen_potential = regional_potential_from_v2(
            potential,
            &converted.geometry,
            &converted.sites,
            converted.reciprocal,
        )?;
        let restart_density = restart_density
            .map(|density| {
                regional_density_from_v2(
                    density,
                    &converted.geometry,
                    &converted.sites,
                    converted.reciprocal,
                )
            })
            .transpose()?;
        Ok(Self {
            checkpoint_template: checkpoint.clone(),
            reciprocal: converted.reciprocal,
            geometry: converted.geometry,
            sites: converted.sites,
            frozen_potential,
            restart_density,
            nuclear_charges: converted.nuclear_charges,
            core_potentials: BTreeMap::new(),
            density_template: None,
            energy_terms: BTreeMap::new(),
            spex_spinor_binding: None,
        })
    }

    /// Bind a caller-owned signed-kappa material recipe to one runtime basis.
    ///
    /// The SPEX snapshot remains scalar Koelling-Harmon source provenance.
    /// This constructor authorizes a target full-Dirac solve only after every
    /// recipe channel binds exactly to a runtime request and its resolved
    /// energy. The Dirac radial functions are then solved from the checkpoint
    /// `V0` monopole; they are not imported from SPEX.
    pub fn new_spex_material(
        checkpoint: &CheckpointV2,
        recipe: &SpexMaterialBasisRecipeV1,
        basis: &ScfBasis,
    ) -> Result<Self, CheckpointPhysicsError> {
        let mut physics = Self::new(checkpoint)?;
        let recorded_sha256 = checkpoint
            .meta
            .annotations
            .get("material_basis.recipe_sha256");
        let recorded_producer = checkpoint.meta.annotations.get("material_basis.producer");
        if recorded_sha256 != Some(&recipe.recipe_sha256)
            || recorded_producer != Some(&recipe.producer)
        {
            return Err(CheckpointPhysicsError::SpexMaterialProvenanceMismatch);
        }
        for site in &physics.sites {
            for (spin, source) in [&site.up, &site.down].into_iter().enumerate() {
                if source.equation != RadialEquationTag::ScalarKoellingHarmon {
                    return Err(CheckpointPhysicsError::SpexMaterialSourceRadialEquation {
                        site: site.id.clone(),
                        spin,
                        equation: source.equation,
                    });
                }
            }
        }

        let mut keys = BTreeSet::new();
        let mut channels = Vec::with_capacity(recipe.channels.len());
        for channel in &recipe.channels {
            let treatment = match channel.kind {
                SpexMaterialChannelKind::Lo | SpexMaterialChannelKind::Rlo => {
                    ScfChannelTreatment::Lo
                }
                SpexMaterialChannelKind::Hdlo => ScfChannelTreatment::Hdlo,
            };
            let identity = ScfChannelIdentity::Kappa {
                n: channel.n,
                kappa: channel.kappa,
            };
            let key = (
                channel.site_id.clone(),
                channel.n,
                channel.l,
                channel.kappa,
                match treatment {
                    ScfChannelTreatment::Lo => 0_u8,
                    ScfChannelTreatment::Hdlo => 1_u8,
                    ScfChannelTreatment::Core | ScfChannelTreatment::Valence => unreachable!(),
                },
                channel.derivative_order,
            );
            let matches = basis
                .channels
                .iter()
                .filter(|requested| {
                    requested.site == channel.site_id
                        && requested.identity == identity
                        && requested.treatment == treatment
                        && requested.derivative_order == channel.derivative_order
                        && requested.generator == LinearizationEnergyGenerator::FrozenCheckpoint
                })
                .collect::<Vec<_>>();
            if channel_l(identity) != channel.l || !keys.insert(key) || matches.len() != 1 {
                return Err(spex_material_channel_mismatch(channel, treatment));
            }
            let requested = matches[0].clone();
            let generated = generate_frozen_checkpoint_energy(Hartree(channel.energy))
                .map_err(|source| channel_generator_error(&requested, source))?;
            channels.push(SpexBoundSpinorChannel {
                l: channel.l,
                requested: requested.clone(),
                resolved: ScfResolvedChannelEnergy {
                    recipe: requested,
                    energy: generated.energy,
                    components: vec![generated],
                },
            });
        }
        physics.spex_spinor_binding = Some(SpexSpinorMaterialBinding { channels });
        physics.validate_spex_requested_basis(basis)?;
        Ok(physics)
    }

    fn validate_spex_requested_basis(&self, basis: &ScfBasis) -> Result<(), CheckpointPhysicsError> {
        let Some(binding) = &self.spex_spinor_binding else {
            return Ok(());
        };
        for bound in &binding.channels {
            if basis
                .channels
                .iter()
                .filter(|requested| **requested == bound.requested)
                .count()
                != 1
            {
                return Err(spex_bound_channel_mismatch(bound));
            }
        }
        Ok(())
    }

    fn validate_spex_resolved_basis(&self, basis: &ScfBasis) -> Result<(), CheckpointPhysicsError> {
        let Some(binding) = &self.spex_spinor_binding else {
            return Ok(());
        };
        self.validate_spex_requested_basis(basis)?;
        for bound in &binding.channels {
            if basis
                .resolved_channels
                .iter()
                .filter(|resolved| **resolved == bound.resolved)
                .count()
                != 1
            {
                return Err(spex_bound_channel_mismatch(bound));
            }
        }
        Ok(())
    }

    fn spex_bound_channel(&self, requested: &ScfChannelRecipe) -> Option<&SpexBoundSpinorChannel> {
        self.spex_spinor_binding.as_ref().and_then(|binding| {
            binding
                .channels
                .iter()
                .find(|bound| bound.requested == *requested)
        })
    }

    pub const fn reciprocal(&self) -> &ReciprocalLattice {
        &self.reciprocal
    }

    pub const fn geometry(&self) -> &InterstitialGeometry {
        &self.geometry
    }

    pub const fn frozen_potential(&self) -> &RegionalPotential {
        &self.frozen_potential
    }

    pub(crate) fn nuclear_charges(&self) -> &[f64] {
        &self.nuclear_charges
    }

    /// Serialize a converged state as a V2 restart while preserving this
    /// kernel's immutable geometry and radial-basis identity.
    pub fn restart_checkpoint(&self, state: &ScfState) -> Result<CheckpointV2, CheckpointPhysicsError> {
        checkpoint_v2_from_state(&self.checkpoint_template, state)
    }

    pub(crate) fn solve_points(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        points: &[[f64; 3]],
        relativity: ScfRelativity,
    ) -> Result<CheckpointBandSolution, CheckpointPhysicsError> {
        if points.is_empty() {
            return Err(CheckpointPhysicsError::EmptyKPointSet);
        }
        if relativity == ScfRelativity::SpinorFirstVariation {
            return self.solve_spinor_points(potential, basis, points);
        }
        self.require_collinear_route(potential)?;
        let site_inputs = self.scalar_site_inputs(potential, basis)?;
        let interstitial = collinear_interstitial_potential(potential)?;
        let weight = 1.0 / points.len() as f64;
        let mut solved_points = Vec::with_capacity(points.len());
        let mut states = Vec::new();

        for &k in points {
            let envelope = self.plane_wave_envelope(k, basis.plane_wave_cutoff)?;
            let bases = build_collinear_scalar_iteration_bases(
                &envelope,
                &self.geometry,
                Collinear::new(&site_inputs.up, &site_inputs.down),
            )?;
            let scalar = muffintin_dft::solve_collinear_scalar_k_point(
                Collinear::new(&bases.up, &bases.down),
                &self.geometry,
                Collinear::new(&interstitial.up, &interstitial.down),
                OVERLAP_THRESHOLD,
            )?;
            let state_start = states.len();
            let (solutions, up_occupations, down_occupations, energies) = match relativity {
                ScfRelativity::Scalar => {
                    let up_start = states.len();
                    states.extend(
                        scalar
                            .up
                            .solution
                            .eigenvalues
                            .iter()
                            .copied()
                            .map(|energy| BandState::new(energy, weight, 1)),
                    );
                    let up_end = states.len();
                    let down_start = states.len();
                    states.extend(
                        scalar
                            .down
                            .solution
                            .eigenvalues
                            .iter()
                            .copied()
                            .map(|energy| BandState::new(energy, weight, 1)),
                    );
                    let down_end = states.len();
                    let energies = scalar
                        .up
                        .solution
                        .eigenvalues
                        .iter()
                        .chain(&scalar.down.solution.eigenvalues)
                        .copied()
                        .collect();
                    (
                        Collinear::new(scalar.up.solution, scalar.down.solution),
                        up_start..up_end,
                        down_start..down_end,
                        energies,
                    )
                }
                ScfRelativity::SocSecondVariation { window } => {
                    self.require_second_variation_route(potential)?;
                    if window.start() != 0 {
                        return Err(CheckpointPhysicsError::SecondVariationDropsLowerBands {
                            start: window.start(),
                        });
                    }
                    let first = FirstVariationSubspace::select(
                        window,
                        &scalar.up.solution.eigenvalues,
                        &scalar.up.solution.eigenvectors,
                    )?;
                    let blocks = second_variation_blocks(&bases.up, &site_inputs.up)?;
                    let second = solve_soc_second_variation(
                        FirstVariationRoute::NonmagneticScalarKoellingHarmon,
                        &bases.up.compiled,
                        &first,
                        &blocks,
                    )?;
                    let split = split_second_variation(&second)?;
                    let start = states.len();
                    states.extend(
                        second
                            .eigenvalues
                            .iter()
                            .copied()
                            .map(|energy| BandState::new(energy, weight, 1)),
                    );
                    let end = states.len();
                    (split, start..end, start..end, second.eigenvalues)
                }
                ScfRelativity::SpinorFirstVariation => unreachable!(),
            };
            debug_assert!(states.len() > state_start);
            solved_points.push(CheckpointKPoint {
                weight,
                solution: CheckpointKPointSolution::Collinear {
                    bases: Box::new(bases),
                    solutions,
                    up_occupations,
                    down_occupations,
                },
                energies,
            });
        }
        Ok(CheckpointBandSolution {
            points: solved_points,
            states,
        })
    }

    fn solve_spinor_points(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        points: &[[f64; 3]],
    ) -> Result<CheckpointBandSolution, CheckpointPhysicsError> {
        let site_inputs = self.spinor_site_inputs(potential, basis)?;
        let interstitial = potential.to_lapw_interstitial()?;
        let weight = 1.0 / points.len() as f64;
        let mut solved_points = Vec::with_capacity(points.len());
        let mut states = Vec::new();
        for &k in points {
            let envelope = self.plane_wave_envelope(k, basis.plane_wave_cutoff)?;
            let spinor_basis =
                build_spinor_iteration_basis(&envelope, &self.geometry, &site_inputs)?;
            let solved = solve_spinor_k_point(
                &spinor_basis,
                &self.geometry,
                &interstitial,
                OVERLAP_THRESHOLD,
            )?;
            let start = states.len();
            states.extend(
                solved
                    .solution
                    .eigenvalues
                    .iter()
                    .copied()
                    .map(|energy| BandState::new(energy, weight, 1)),
            );
            let end = states.len();
            solved_points.push(CheckpointKPoint {
                weight,
                energies: solved.solution.eigenvalues.clone(),
                solution: CheckpointKPointSolution::Spinor {
                    basis: spinor_basis,
                    site_blocks: solved.site_blocks,
                    solution: solved.solution,
                    occupations: start..end,
                },
            });
        }
        Ok(CheckpointBandSolution {
            points: solved_points,
            states,
        })
    }

    fn scalar_site_inputs(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<Collinear<Vec<ScalarSiteInput>>, CheckpointPhysicsError> {
        self.require_potential_site_count(potential)?;
        let build_spin = |spin: usize| {
            self.sites
                .iter()
                .enumerate()
                .map(|(site_index, site)| {
                    let template = if spin == 0 { &site.up } else { &site.down };
                    if template.equation != RadialEquationTag::ScalarKoellingHarmon {
                        return Err(CheckpointPhysicsError::ScalarRadialEquation {
                            site: site.id.clone(),
                            spin,
                            equation: template.equation,
                        });
                    }
                    let field = combine_muffin_tin_fields(
                        &potential.scalar().muffin_tins()[site_index],
                        1.0,
                        &potential.magnetic()[2].muffin_tins()[site_index],
                        if spin == 0 { 1.0 } else { -1.0 },
                    )?;
                    let monopole = field
                        .field()
                        .channel(0, 0)
                        .ok_or_else(|| CheckpointPhysicsError::MissingMonopole(site.id.clone()))?;
                    let spherical_potential = monopole
                        .iter()
                        .map(|value| value.re / (4.0 * PI).sqrt())
                        .collect();
                    let linearization_energies =
                        self.scalar_linearization_energies(basis, &site.id, spin)?;
                    let local_orbitals = self.scalar_local_orbitals(basis, &site.id, spin)?;
                    Ok(ScalarSiteInput {
                        position: site.position,
                        radius: site.radius,
                        mesh: template.mesh.clone(),
                        spherical_potential,
                        potential: field.field().clone(),
                        linearization_energies,
                        local_orbitals,
                    })
                })
                .collect::<Result<Vec<_>, CheckpointPhysicsError>>()
        };
        Ok(Collinear::new(build_spin(0)?, build_spin(1)?))
    }

    fn spinor_site_inputs(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<Vec<SpinorSiteInput>, CheckpointPhysicsError> {
        self.require_potential_site_count(potential)?;
        let source_is_dirac = self.sites.iter().all(|site| {
            [&site.up, &site.down]
                .into_iter()
                .all(|source| source.equation == RadialEquationTag::FullyRelativisticDirac)
        });
        if !source_is_dirac && self.spex_spinor_binding.is_some() {
            self.validate_spex_resolved_basis(basis)?;
        }
        self.sites
            .iter()
            .enumerate()
            .map(|(site_index, site)| {
                if !source_is_dirac && self.spex_spinor_binding.is_none() {
                    for (spin, template) in [&site.up, &site.down].into_iter().enumerate() {
                        if template.equation != RadialEquationTag::FullyRelativisticDirac {
                            return Err(CheckpointPhysicsError::SpinorRadialEquation {
                                site: site.id.clone(),
                                spin,
                                equation: template.equation,
                            });
                        }
                    }
                }
                let scalar = potential.scalar().muffin_tins()[site_index].field().clone();
                let magnetic = potential
                    .magnetic()
                    .each_ref()
                    .map(|component| component.muffin_tins()[site_index].field().clone());
                let monopole = scalar
                    .channel(0, 0)
                    .ok_or_else(|| CheckpointPhysicsError::MissingMonopole(site.id.clone()))?;
                let spherical_potential = monopole
                    .iter()
                    .map(|value| value.re / (4.0 * PI).sqrt())
                    .collect();
                let linearization_energies = self.spinor_linearization_energies(basis, &site.id)?;
                let local_orbitals = self.spinor_local_orbitals(basis, &site.id)?;
                Ok(SpinorSiteInput {
                    position: site.position,
                    radius: site.radius,
                    mesh: site.up.mesh.clone(),
                    spherical_potential,
                    potential: LocalPauliPotential::new(scalar, magnetic)?,
                    l_max: basis.l_max,
                    linearization_energies,
                    local_orbitals,
                })
            })
            .collect()
    }

    fn site_index(&self, site: &str) -> Result<usize, CheckpointPhysicsError> {
        self.sites
            .iter()
            .position(|candidate| candidate.id == site)
            .ok_or_else(|| CheckpointPhysicsError::UnknownCoreSite(site.to_owned()))
    }

    fn refine_spectral_basis(
        &self,
        requested: &ScfBasis,
        one_particle: &CheckpointOneParticle,
        bands: &CheckpointBandSolution,
        occupations: &[f64],
        chemical_potential: Hartree,
        relativity: ScfRelativity,
    ) -> Result<Option<CheckpointOneParticle>, CheckpointPhysicsError> {
        let spectral = requested
            .channels
            .iter()
            .filter(|recipe| {
                recipe.treatment != ScfChannelTreatment::Core
                    && matches!(
                        recipe.generator,
                        LinearizationEnergyGenerator::BandCog
                            | LinearizationEnergyGenerator::FermiOffset
                    )
            })
            .collect::<Vec<_>>();
        if spectral.is_empty() {
            return Ok(None);
        }
        self.validate_band_cog_projection_keys(&spectral, relativity)?;
        let mut basis = one_particle.basis.clone();
        let mut changed = false;
        for recipe in spectral {
            let resolved = match recipe.generator {
                LinearizationEnergyGenerator::BandCog
                    if relativity == ScfRelativity::SpinorFirstVariation
                        && matches!(recipe.identity, ScfChannelIdentity::ScalarL { .. }) =>
                {
                    let l = channel_l(recipe.identity);
                    let mut components = Vec::new();
                    let mut partner_energies = Vec::new();
                    for kappa in spinor_kappas_for_l(l)? {
                        let mut partner = recipe.clone();
                        partner.identity = ScfChannelIdentity::Kappa {
                            n: channel_n(recipe.identity),
                            kappa: kappa.get(),
                        };
                        let generated = generate_band_cog_energy(&self.band_cog_samples(
                            bands,
                            occupations,
                            &partner,
                            relativity,
                        )?)
                        .map_err(|source| channel_generator_error(recipe, source))?;
                        partner_energies.push((kappa, generated.energy));
                        components.push(generated);
                    }
                    ScfResolvedChannelEnergy {
                        recipe: recipe.clone(),
                        energy: kappa_degeneracy_average(l, &partner_energies)
                            .map_err(|source| channel_generator_error(recipe, source))?,
                        components,
                    }
                }
                LinearizationEnergyGenerator::BandCog => {
                    let generated = generate_band_cog_energy(&self.band_cog_samples(
                        bands,
                        occupations,
                        recipe,
                        relativity,
                    )?)
                    .map_err(|source| channel_generator_error(recipe, source))?;
                    ScfResolvedChannelEnergy {
                        recipe: recipe.clone(),
                        energy: generated.energy,
                        components: vec![generated],
                    }
                }
                LinearizationEnergyGenerator::FermiOffset => {
                    let generated = generate_fermi_offset_energy(
                        chemical_potential,
                        recipe.seed.unwrap_or(Hartree(DEFAULT_FERMI_OFFSET_HARTREE)),
                    )
                    .map_err(|source| channel_generator_error(recipe, source))?;
                    ScfResolvedChannelEnergy {
                        recipe: recipe.clone(),
                        energy: generated.energy,
                        components: vec![generated],
                    }
                }
                _ => unreachable!(),
            };
            let old = basis
                .resolved_channels
                .iter()
                .position(|candidate| candidate.recipe == *recipe)
                .ok_or_else(|| CheckpointPhysicsError::MissingProvisionalChannel {
                    site: recipe.site.clone(),
                    identity: recipe.identity,
                    generator: recipe.generator,
                })?;
            let previous = &basis.resolved_channels[old];
            let provisional = previous
                .components
                .iter()
                .any(|component| component.diagnostic == LinearizationEnergyDiagnostic::Stored);
            let scale = previous
                .energy
                .get()
                .abs()
                .max(resolved.energy.get().abs())
                .max(1.0);
            changed |= provisional
                || (resolved.energy.get() - previous.energy.get()).abs()
                    > SPECTRAL_REFINEMENT_TOLERANCE * scale;
            basis.resolved_channels[old] = resolved;
        }
        Ok(changed.then(|| CheckpointOneParticle {
            potential: one_particle.potential.clone(),
            basis,
        }))
    }

    fn validate_band_cog_projection_keys(
        &self,
        spectral: &[&ScfChannelRecipe],
        relativity: ScfRelativity,
    ) -> Result<(), CheckpointPhysicsError> {
        let band_cog = spectral
            .iter()
            .copied()
            .filter(|recipe| recipe.generator == LinearizationEnergyGenerator::BandCog)
            .collect::<Vec<_>>();
        for (index, recipe) in band_cog.iter().enumerate() {
            for prior in &band_cog[..index] {
                if prior.site != recipe.site {
                    continue;
                }
                let projections_overlap = match relativity {
                    ScfRelativity::SpinorFirstVariation => {
                        let current = channel_kappas(recipe.identity)?
                            .into_iter()
                            .collect::<BTreeSet<_>>();
                        channel_kappas(prior.identity)?
                            .into_iter()
                            .any(|kappa| current.contains(&kappa))
                    }
                    ScfRelativity::Scalar | ScfRelativity::SocSecondVariation { .. } => {
                        channel_l(prior.identity) == channel_l(recipe.identity)
                    }
                };
                if projections_overlap {
                    return Err(CheckpointPhysicsError::AmbiguousBandCogProjection {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                    });
                }
            }
        }
        Ok(())
    }

    fn band_cog_samples(
        &self,
        bands: &CheckpointBandSolution,
        occupations: &[f64],
        recipe: &ScfChannelRecipe,
        relativity: ScfRelativity,
    ) -> Result<Vec<PdosEnergySample>, CheckpointPhysicsError> {
        if occupations.len() != bands.states.len() {
            return Err(CheckpointPhysicsError::OccupationCount {
                expected: bands.states.len(),
                actual: occupations.len(),
            });
        }
        let site = self.site_index(&recipe.site)?;
        let l = channel_l(recipe.identity);
        let mut samples = Vec::new();
        for point in &bands.points {
            match &point.solution {
                CheckpointKPointSolution::Collinear {
                    bases,
                    solutions,
                    up_occupations,
                    down_occupations,
                } => {
                    if matches!(recipe.identity, ScfChannelIdentity::Kappa { .. }) {
                        return Err(CheckpointPhysicsError::KappaBandCogUnavailableInScalar {
                            site: recipe.site.clone(),
                            identity: recipe.identity,
                        });
                    }
                    let up = scalar_channel_band_weights(
                        &bases.up,
                        &solutions.up.eigenvectors,
                        site,
                        l,
                    )?;
                    let down = scalar_channel_band_weights(
                        &bases.down,
                        &solutions.down.eigenvectors,
                        site,
                        l,
                    )?;
                    if matches!(relativity, ScfRelativity::SocSecondVariation { .. }) {
                        if up_occupations != down_occupations || up.len() != down.len() {
                            return Err(CheckpointPhysicsError::InconsistentRelativityRoute);
                        }
                        for band in 0..up.len() {
                            let global = up_occupations.start + band;
                            samples.push(PdosEnergySample::new(
                                bands.states[global],
                                occupations[global],
                                up[band] + down[band],
                            ));
                        }
                    } else {
                        append_pdos_samples(
                            &mut samples,
                            &bands.states,
                            occupations,
                            up_occupations,
                            &up,
                        )?;
                        append_pdos_samples(
                            &mut samples,
                            &bands.states,
                            occupations,
                            down_occupations,
                            &down,
                        )?;
                    }
                }
                CheckpointKPointSolution::Spinor {
                    basis,
                    site_blocks,
                    solution,
                    occupations: state_range,
                } => {
                    let kappas = channel_kappas(recipe.identity)?
                        .into_iter()
                        .map(Kappa::get)
                        .collect::<BTreeSet<_>>();
                    let density_site = &basis.density_sites[site];
                    let coordinates = density_site
                        .orbitals
                        .iter()
                        .enumerate()
                        .filter(|(_, orbital)| kappas.contains(&orbital.channel().kappa().get()))
                        .map(|(coordinate, _)| coordinate)
                        .collect::<Vec<_>>();
                    let projection = CompiledSiteProjection::spinor(
                        &basis.compiled,
                        site,
                        &density_site.channels,
                    )?;
                    let weights = physical_site_band_projections(
                        &projection,
                        &solution.eigenvectors,
                        &site_blocks[site].overlap,
                        &coordinates,
                    )?;
                    append_pdos_samples(
                        &mut samples,
                        &bands.states,
                        occupations,
                        state_range,
                        &weights,
                    )?;
                }
            }
        }
        Ok(samples)
    }

    fn plane_wave_envelope(
        &self,
        fractional_k: [f64; 3],
        cutoff: f64,
    ) -> Result<PlaneWaveEnvelope, CheckpointPhysicsError> {
        production_plane_wave_envelope(self.reciprocal, fractional_k, cutoff)
    }

    fn require_potential_site_count(
        &self,
        potential: &RegionalPotential,
    ) -> Result<(), CheckpointPhysicsError> {
        let expected = self.sites.len();
        for (component, actual) in
            std::iter::once(("scalar", potential.scalar().muffin_tins().len())).chain(
                potential
                    .magnetic()
                    .iter()
                    .zip(["Bx", "By", "Bz"])
                    .map(|(field, name)| (name, field.muffin_tins().len())),
            )
        {
            if actual != expected {
                return Err(CheckpointPhysicsError::PotentialComponentSiteCount {
                    component,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn require_collinear_route(
        &self,
        potential: &RegionalPotential,
    ) -> Result<(), CheckpointPhysicsError> {
        let transverse_rms = [
            potential.magnetic()[0].residual_rms()?,
            potential.magnetic()[1].residual_rms()?,
        ];
        if transverse_rms
            .iter()
            .any(|&rms| rms > TRANSVERSE_FIELD_TOLERANCE)
        {
            return Err(CheckpointPhysicsError::TransversePotentialUnsupported {
                x_rms: transverse_rms[0],
                y_rms: transverse_rms[1],
                tolerance: TRANSVERSE_FIELD_TOLERANCE,
            });
        }
        Ok(())
    }

    fn require_second_variation_route(
        &self,
        potential: &RegionalPotential,
    ) -> Result<(), CheckpointPhysicsError> {
        self.require_collinear_route(potential)?;
        let magnetic = potential
            .magnetic()
            .iter()
            .map(RegionalScalarField::residual_rms)
            .collect::<Result<Vec<_>, _>>()?;
        if self.sites.iter().any(|site| !site.nonmagnetic_scalar)
            || magnetic.iter().any(|&rms| rms > TRANSVERSE_FIELD_TOLERANCE)
        {
            return Err(CheckpointPhysicsError::SecondVariationRequiresNonmagneticScalar);
        }
        Ok(())
    }

    fn synthesize(
        &self,
        bands: &CheckpointBandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, CheckpointPhysicsError> {
        if occupations.len() != bands.states.len() {
            return Err(CheckpointPhysicsError::OccupationCount {
                expected: bands.states.len(),
                actual: occupations.len(),
            });
        }
        let density_layout = self.density_layout(&bands.points)?;
        match &bands.points[0].solution {
            CheckpointKPointSolution::Collinear { bases, .. } => {
                let up_points = bands
                    .points
                    .iter()
                    .map(|point| match &point.solution {
                        CheckpointKPointSolution::Collinear {
                            bases,
                            solutions,
                            up_occupations,
                            ..
                        } => Ok(CollinearKPoint {
                            weight: point.weight,
                            compiled: &bases.up.compiled,
                            solutions: Collinear::new(&solutions.up, &solutions.up),
                            // The duplicate channel only ensures a complete regional
                            // field layout; it is discarded below.
                            occupations: Collinear::new(
                                &occupations[up_occupations.clone()],
                                &occupations[up_occupations.clone()],
                            ),
                        }),
                        CheckpointKPointSolution::Spinor { .. } => {
                            Err(CheckpointPhysicsError::InconsistentRelativityRoute)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let down_points = bands
                    .points
                    .iter()
                    .map(|point| match &point.solution {
                        CheckpointKPointSolution::Collinear {
                            bases,
                            solutions,
                            down_occupations,
                            ..
                        } => Ok(CollinearKPoint {
                            weight: point.weight,
                            compiled: &bases.down.compiled,
                            solutions: Collinear::new(&solutions.down, &solutions.down),
                            occupations: Collinear::new(
                                &occupations[down_occupations.clone()],
                                &occupations[down_occupations.clone()],
                            ),
                        }),
                        CheckpointKPointSolution::Spinor { .. } => {
                            Err(CheckpointPhysicsError::InconsistentRelativityRoute)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let up = synthesize_collinear_valence_density(
                    self.geometry.clone(),
                    density_layout.clone(),
                    &bases.up.density_sites,
                    &up_points,
                )?;
                let down = synthesize_collinear_valence_density(
                    self.geometry.clone(),
                    density_layout,
                    &bases.down.density_sites,
                    &down_points,
                )?;
                let charge = combine_scalar_fields(up.charge(), 0.5, down.charge(), 0.5)?;
                let longitudinal = combine_scalar_fields(up.charge(), 0.5, down.charge(), -0.5)?;
                let zero = charge.zero_like();
                Ok(RegionalDensity::new(
                    charge,
                    [zero.clone(), zero, longitudinal],
                )?)
            }
            CheckpointKPointSolution::Spinor { basis, .. } => {
                let spinor_points = bands
                    .points
                    .iter()
                    .map(|point| match &point.solution {
                        CheckpointKPointSolution::Spinor {
                            basis,
                            solution,
                            occupations: band_occupations,
                            ..
                        } => Ok(FullSpinorKPoint {
                            weight: point.weight,
                            compiled: &basis.compiled,
                            solution,
                            occupations: &occupations[band_occupations.clone()],
                        }),
                        CheckpointKPointSolution::Collinear { .. } => {
                            Err(CheckpointPhysicsError::InconsistentRelativityRoute)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(synthesize_full_spinor_valence_density(
                    self.geometry.clone(),
                    density_layout,
                    &basis.density_sites,
                    &spinor_points,
                )?)
            }
        }
    }

    fn core_contribution(
        &self,
        site: &ScfCoreSite,
        extended: &ExtendedCorePotential,
        template: &RegionalDensity,
    ) -> Result<CoreContribution, CheckpointPhysicsError> {
        let site_index = self
            .sites
            .iter()
            .position(|candidate| candidate.id == site.id)
            .ok_or_else(|| CheckpointPhysicsError::UnknownCoreSite(site.id.clone()))?;
        if site.states.is_empty() {
            return Ok(CoreContribution {
                site_id: site.id.clone(),
                density: template.zero_like(),
                eigenvalue_sum: Hartree(0.0),
            });
        }
        let converted = &self.sites[site_index];

        let mut solved = Vec::with_capacity(site.states.len());
        for requested in &site.states {
            let solution = self.solve_bound_dirac_state(
                site_index,
                requested.principal_quantum_number,
                requested.kappa,
                extended,
            )?;
            solved.push((solution, requested.occupation));
        }
        let shells = solved
            .iter()
            .map(|(solution, occupation)| RegionalCoreShellInput {
                mesh: &extended.mesh,
                solution,
                occupation: *occupation,
                spin: CoreSpinPartition::ClosedShellAverage,
            })
            .collect::<Vec<_>>();
        Ok(build_regional_core_contribution(
            site.id.clone(),
            &self.geometry,
            site_index,
            &converted.up.mesh,
            &shells,
            template,
        )?
        .contribution)
    }

    fn solve_bound_dirac_state(
        &self,
        site_index: usize,
        principal_quantum_number: u32,
        kappa: i32,
        extended: &ExtendedCorePotential,
    ) -> Result<muffintin_sphere::CoreDiracSolution, CheckpointPhysicsError> {
        let state = CoreState::new(principal_quantum_number, Kappa::new(kappa)?)?;
        let charge = self.nuclear_charges[site_index];
        let converted = &self.sites[site_index];
        // Scan the complete negative atomic scale. Node-count selection, not
        // an energy estimate, identifies both core and relLO bound states.
        let continuum = *extended
            .values
            .last()
            .expect("extended core potential follows a nonempty mesh");
        let atomic_scale = (charge * charge / f64::from(state.n).powi(2)).max(1.0);
        let lower = continuum - 2.0 * charge * charge;
        let upper = continuum - 1.0e-8 * atomic_scale;
        let window = EnergyBracket::from_values(lower, upper)?;
        let bracket = isolate_core_dirac_bracket(
            &extended.mesh,
            &extended.values,
            CoreBracketSearch::new(state, converted.radius, window).with_intervals(512),
        )?;
        Ok(solve_core_dirac(
            &extended.mesh,
            &extended.values,
            CoreDiracSpec::new(state, bracket, converted.radius),
        )?)
    }

    fn extended_core_meshes(
        &self,
        requested_site: usize,
        states: &[muffintin_dft::ScfCoreState],
    ) -> Result<Vec<ExponentialMesh>, CheckpointPhysicsError> {
        self.sites
            .iter()
            .enumerate()
            .map(|(site_index, site)| {
                let maximum_n = if site_index == requested_site {
                    states
                        .iter()
                        .map(|state| state.principal_quantum_number)
                        .max()
                        .unwrap_or(1)
                } else {
                    1
                };
                // Forty hydrogenic length scales leave an exponentially small
                // bound tail; four MT radii also cover compact deep cores.
                let orbital_scale =
                    f64::from(maximum_n).powi(2) / self.nuclear_charges[site_index].max(1.0);
                let outer_radius = (4.0 * site.radius.get()).max(40.0 * orbital_scale);
                extend_mesh(&site.up.mesh, outer_radius)
            })
            .collect()
    }

    fn density_layout(&self, points: &[CheckpointKPoint]) -> Result<FourierLayout, CheckpointPhysicsError> {
        let mut indices = BTreeSet::new();
        for point in points {
            let compiled = match &point.solution {
                CheckpointKPointSolution::Collinear { bases, .. } => {
                    vec![
                        &bases.up.compiled.plane_waves,
                        &bases.down.compiled.plane_waves,
                    ]
                }
                CheckpointKPointSolution::Spinor { basis, .. } => {
                    vec![&basis.compiled.plane_waves]
                }
            };
            for plane_waves in compiled {
                insert_plane_wave_differences(&mut indices, plane_waves);
            }
        }
        let vectors = indices
            .into_iter()
            .map(|index| g_vector(self.reciprocal, index))
            .collect();
        Ok(FourierLayout::new(self.reciprocal, vectors)?)
    }
}


impl ScfPhysics for CheckpointPhysics {
    type Error = CheckpointPhysicsError;
    type OneParticle = CheckpointOneParticle;
    type BandSolution = CheckpointBandSolution;

    fn initial_density(&mut self, config: &ScfConfig) -> Result<RegionalDensity, Self::Error> {
        if let Some(density) = &self.restart_density {
            self.density_template = Some(density.clone());
            return Ok(density.clone());
        }
        let meshes = self.channel_meshes(&config.basis)?;
        let initial_extended = build_extended_checkpoint_core_potentials(
            &self.frozen_potential,
            &self.geometry,
            &self.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )?;
        let basis = self.materialize_nonspectral_basis(
            &self.frozen_potential,
            &config.basis,
            &initial_extended,
        )?;
        let mut one_particle = CheckpointOneParticle {
            potential: self.frozen_potential.clone(),
            basis,
        };
        let points = regular_k_points(config.k_mesh)?;
        let (bands, occupations) = {
            let mut passes = 0;
            loop {
                passes += 1;
                let bands = self.solve_points(
                    &one_particle.potential,
                    &one_particle.basis,
                    &points,
                    config.relativity,
                )?;
                let occupation = solve_initial_occupations(&bands.states, config)?;
                match self.refine_spectral_basis(
                    &config.basis,
                    &one_particle,
                    &bands,
                    &occupation.occupations,
                    occupation.chemical_potential,
                    config.relativity,
                )? {
                    None => break (bands, occupation.occupations),
                    Some(_) if passes == 16 => {
                        return Err(CheckpointPhysicsError::InitialBasisRefinementNotConverged {
                            passes,
                        });
                    }
                    Some(refined) => one_particle = refined,
                }
            }
        };
        let mut density = self.synthesize(&bands, &occupations)?;
        if config.core_sites.iter().any(|site| !site.states.is_empty()) {
            for site in &config.core_sites {
                if site.states.is_empty() {
                    continue;
                }
                let site_index = self
                    .sites
                    .iter()
                    .position(|candidate| candidate.id == site.id)
                    .ok_or_else(|| CheckpointPhysicsError::UnknownCoreSite(site.id.clone()))?;
                let contribution = self.core_contribution(
                    site,
                    &initial_extended[site_index].potential,
                    &density,
                )?;
                density.add_scaled(1.0, &contribution.density)?;
            }
        }
        self.density_template = Some(density.clone());
        Ok(density)
    }

    fn build_potential(
        &mut self,
        iteration: usize,
        density: &RegionalDensity,
        exchange_correlation: ScfExchangeCorrelation,
    ) -> Result<RegionalPotential, Self::Error> {
        self.density_template = Some(density.clone());
        let built =
            build_scf_potential(density, &self.nuclear_charges, exchange_correlation)?;
        self.core_potentials.insert(
            iteration,
            CorePotentialContext {
                electrostatic: built.electrostatic.clone(),
                exchange_correlation: built.exchange_correlation.clone(),
                density: density.clone(),
                spec: built.core_spec,
            },
        );
        self.energy_terms.insert(iteration, built.energy_terms);
        Ok(built.potential)
    }

    fn solve_core(
        &mut self,
        iteration: usize,
        site: &ScfCoreSite,
        _potential: &RegionalPotential,
        _basis: &ScfBasis,
        _relativity: ScfRelativity,
    ) -> Result<CoreContribution, Self::Error> {
        let template = self
            .density_template
            .as_ref()
            .ok_or(CheckpointPhysicsError::MissingDensityTemplate)?
            .clone();
        let context = self
            .core_potentials
            .get(&iteration)
            .ok_or(CheckpointPhysicsError::MissingCoreContinuation(iteration))?
            .clone();
        let site_index = self
            .sites
            .iter()
            .position(|candidate| candidate.id == site.id)
            .ok_or_else(|| CheckpointPhysicsError::UnknownCoreSite(site.id.clone()))?;
        if site.states.is_empty() {
            return Ok(CoreContribution {
                site_id: site.id.clone(),
                density: template.zero_like(),
                eigenvalue_sum: Hartree(0.0),
            });
        }
        let meshes = self.extended_core_meshes(site_index, &site.states)?;
        let continued = build_extended_core_potentials(
            &context.electrostatic,
            &context.exchange_correlation,
            &context.density,
            &meshes,
            context.spec,
        )?;
        self.core_contribution(site, &continued[site_index].potential, &template)
    }

    fn assemble_one_particle(
        &mut self,
        iteration: usize,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        _relativity: ScfRelativity,
    ) -> Result<Self::OneParticle, Self::Error> {
        let basis = self.materialize_current_basis(iteration, potential, basis)?;
        Ok(CheckpointOneParticle {
            potential: potential.clone(),
            basis,
        })
    }

    fn retained_basis(&self, _requested: &ScfBasis, one_particle: &Self::OneParticle) -> ScfBasis {
        one_particle.basis.clone()
    }

    fn solve_regular_bands(
        &mut self,
        _iteration: usize,
        one_particle: &Self::OneParticle,
        k_mesh: ScfKMesh,
        relativity: ScfRelativity,
    ) -> Result<Self::BandSolution, Self::Error> {
        self.solve_points(
            &one_particle.potential,
            &one_particle.basis,
            &regular_k_points(k_mesh)?,
            relativity,
        )
    }

    fn band_states<'a>(&self, bands: &'a Self::BandSolution) -> &'a [BandState] {
        &bands.states
    }

    fn refine_one_particle(
        &mut self,
        _iteration: usize,
        _potential: &RegionalPotential,
        requested_basis: &ScfBasis,
        one_particle: &Self::OneParticle,
        bands: &Self::BandSolution,
        occupations: &[f64],
        chemical_potential: Hartree,
        relativity: ScfRelativity,
    ) -> Result<Option<Self::OneParticle>, Self::Error> {
        self.refine_spectral_basis(
            requested_basis,
            one_particle,
            bands,
            occupations,
            chemical_potential,
            relativity,
        )
    }

    fn synthesize_valence_density(
        &mut self,
        _iteration: usize,
        bands: &Self::BandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, Self::Error> {
        self.synthesize(bands, occupations)
    }

    fn energy_terms(
        &mut self,
        context: ScfEnergyContext<'_, Self::OneParticle, Self::BandSolution>,
    ) -> Result<ScfEnergyTerms, Self::Error> {
        self.energy_terms
            .get(&context.iteration)
            .copied()
            .ok_or(CheckpointPhysicsError::MissingEnergyTerms(context.iteration))
    }

    fn solve_band_path(
        &mut self,
        state: &ScfState,
        request: &BandPathRequest,
    ) -> Result<Vec<Vec<Hartree>>, Self::Error> {
        let points = request
            .points
            .iter()
            .map(|point| point.k)
            .collect::<Vec<_>>();
        let solved =
            self.solve_points(&state.potential, &state.basis, &points, state.relativity)?;
        solved
            .points
            .into_iter()
            .map(|point| {
                let mut energies = point.energies;
                energies.sort_by(|left, right| left.get().total_cmp(&right.get()));
                if energies.len() < request.bands {
                    return Err(CheckpointPhysicsError::TooFewBands {
                        requested: request.bands,
                        available: energies.len(),
                    });
                }
                energies.truncate(request.bands);
                Ok(energies)
            })
            .collect()
    }

    fn solve_dos_spectrum(
        &mut self,
        state: &ScfState,
        request: &muffintin_dft::DosRequest,
    ) -> Result<RegularSpectrum, Self::Error> {
        let solved = self.solve_points(
            &state.potential,
            &state.basis,
            &regular_k_points(request.k_mesh)?,
            state.relativity,
        )?;
        let band_count = solved
            .points
            .first()
            .map(|point| point.energies.len())
            .ok_or(CheckpointPhysicsError::EmptyKPointSet)?;
        if solved
            .points
            .iter()
            .any(|point| point.energies.len() != band_count)
        {
            return Err(CheckpointPhysicsError::InconsistentBandCount);
        }
        let mut energies = Vec::with_capacity(band_count * solved.points.len());
        for band in 0..band_count {
            for point in &solved.points {
                energies.push(point.energies[band]);
            }
        }
        Ok(RegularSpectrum::new(
            request.k_mesh.divisions,
            *self.reciprocal.basis(),
            energies,
            vec![1; band_count],
        )?)
    }
}

fn second_variation_blocks(
    basis: &ScalarIterationBasis,
    inputs: &[ScalarSiteInput],
) -> Result<Vec<SiteSpinOrbitBlock>, CheckpointPhysicsError> {
    basis
        .radial_sites
        .iter()
        .zip(&basis.recipe_sites)
        .zip(inputs)
        .map(|((radials, recipe), input)| {
            let potential = SpexSpinOrbitPotential::new(&input.mesh, &input.spherical_potential)?;
            let shells = radials
                .linearized
                .iter()
                .zip(&radials.local_orbitals)
                .map(|(linearized, locals)| {
                    let locals = locals
                        .iter()
                        .map(|local| local.orbital.clone())
                        .collect::<Vec<_>>();
                    spex_spin_orbit_radial_shell(&input.mesh, &potential, linearized, &locals)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(SiteSpinOrbitBlock::from_radial_shells(
                &recipe.local_orbitals,
                &shells,
            )?)
        })
        .collect()
}

fn split_second_variation(
    solution: &muffintin_dft::SecondVariationResult,
) -> Result<Collinear<GeneralizedEigensolution>, CheckpointPhysicsError> {
    let rows = solution.eigenvectors.rows() / 2;
    let columns = solution.eigenvectors.columns();
    let split = |spin: usize| -> Result<GeneralizedEigensolution, CheckpointPhysicsError> {
        let mut values = Vec::with_capacity(rows * columns);
        for band in 0..columns {
            for row in 0..rows {
                values.push(solution.eigenvectors.at(spin * rows + row, band));
            }
        }
        Ok(GeneralizedEigensolution {
            eigenvalues: solution.eigenvalues.clone(),
            eigenvectors: DenseEigenvectors::from_host_column_major(rows, columns, values)?,
            retained_dimension: columns,
            filtered_dimension: 0,
            residuals: Vec::new(),
        })
    };
    Ok(Collinear::new(split(0)?, split(1)?))
}

fn combine_muffin_tin_fields(
    left: &MuffinTinField,
    left_scale: f64,
    right: &MuffinTinField,
    right_scale: f64,
) -> Result<MuffinTinField, CheckpointPhysicsError> {
    let mut result = left.zero_like();
    result.add_scaled(left_scale, left)?;
    result.add_scaled(right_scale, right)?;
    Ok(result)
}

fn combine_interstitial_fields(
    left: &InterstitialField,
    left_scale: f64,
    right: &InterstitialField,
    right_scale: f64,
) -> Result<InterstitialField, CheckpointPhysicsError> {
    let mut result = left.zero_like();
    result.add_scaled(left_scale, left)?;
    result.add_scaled(right_scale, right)?;
    Ok(result)
}

fn combine_scalar_fields(
    left: &RegionalScalarField,
    left_scale: f64,
    right: &RegionalScalarField,
    right_scale: f64,
) -> Result<RegionalScalarField, CheckpointPhysicsError> {
    let mut result = left.zero_like();
    result.add_scaled(left_scale, left)?;
    result.add_scaled(right_scale, right)?;
    Ok(result)
}

fn collinear_interstitial_potential(
    potential: &RegionalPotential,
) -> Result<Collinear<InterstitialPotential>, CheckpointPhysicsError> {
    let up = combine_interstitial_fields(
        potential.scalar().interstitial(),
        1.0,
        potential.magnetic()[2].interstitial(),
        1.0,
    )?;
    let down = combine_interstitial_fields(
        potential.scalar().interstitial(),
        1.0,
        potential.magnetic()[2].interstitial(),
        -1.0,
    )?;
    Ok(Collinear::new(
        InterstitialPotential::try_from(&up)?,
        InterstitialPotential::try_from(&down)?,
    ))
}

fn extend_mesh(
    muffin_tin: &ExponentialMesh,
    target_radius: f64,
) -> Result<ExponentialMesh, CheckpointPhysicsError> {
    let extra = (target_radius / muffin_tin.last().get()).ln().max(0.0) / muffin_tin.increment();
    let count = muffin_tin
        .len()
        .checked_add(extra.ceil() as usize)
        .and_then(|count| count.checked_add(1))
        .ok_or(CheckpointPhysicsError::CoreMeshCountOverflow)?;
    Ok(ExponentialMesh::new(
        muffin_tin.first(),
        muffin_tin.increment(),
        count,
    )?)
}







fn spex_material_channel_mismatch(
    channel: &muffintin_io::SpexMaterialChannelV1,
    treatment: ScfChannelTreatment,
) -> CheckpointPhysicsError {
    CheckpointPhysicsError::SpexMaterialChannelMismatch {
        site: channel.site_id.clone(),
        n: channel.n,
        l: channel.l,
        kappa: channel.kappa,
        treatment,
        derivative_order: channel.derivative_order,
        energy: channel.energy,
    }
}

fn spex_bound_channel_mismatch(bound: &SpexBoundSpinorChannel) -> CheckpointPhysicsError {
    let (n, kappa) = match bound.requested.identity {
        ScfChannelIdentity::Kappa { n, kappa } => (n, kappa),
        ScfChannelIdentity::ScalarL { .. } => unreachable!("SPEX material binding is signed kappa"),
    };
    CheckpointPhysicsError::SpexMaterialChannelMismatch {
        site: bound.requested.site.clone(),
        n,
        l: bound.l,
        kappa,
        treatment: bound.requested.treatment,
        derivative_order: bound.requested.derivative_order,
        energy: bound.resolved.energy.get(),
    }
}

fn channel_generator_error(
    recipe: &ScfChannelRecipe,
    source: LinearizationEnergyError,
) -> CheckpointPhysicsError {
    CheckpointPhysicsError::ChannelGenerator {
        site: recipe.site.clone(),
        identity: recipe.identity,
        treatment: recipe.treatment,
        generator: recipe.generator,
        source,
    }
}

fn spherical_scalar_potential(
    potential: &RegionalPotential,
    site_index: usize,
    site: &str,
) -> Result<Vec<f64>, CheckpointPhysicsError> {
    let monopole = potential.scalar().muffin_tins()[site_index]
        .field()
        .channel(0, 0)
        .ok_or_else(|| CheckpointPhysicsError::MissingMonopole(site.to_owned()))?;
    monopole
        .iter()
        .enumerate()
        .map(|(radial, value)| {
            if value.im.abs() > TRANSVERSE_FIELD_TOLERANCE * (1.0 + value.re.abs()) {
                return Err(CheckpointPhysicsError::NonRealMonopole {
                    site: site.to_owned(),
                    radial,
                    value: *value,
                });
            }
            Ok(value.re / (4.0 * PI).sqrt())
        })
        .collect()
}

fn scalar_channel_band_weights(
    basis: &ScalarIterationBasis,
    eigenvectors: &DenseEigenvectors,
    site: usize,
    l: u32,
) -> Result<Vec<f64>, CheckpointPhysicsError> {
    let coordinates = basis.density_sites[site]
        .orbitals
        .iter()
        .enumerate()
        .filter(|(_, orbital)| orbital.angular().l == l)
        .map(|(coordinate, _)| coordinate)
        .collect::<Vec<_>>();
    let projection = CompiledSiteProjection::scalar(&basis.compiled, site)?;
    Ok(physical_site_band_projections(
        &projection,
        eigenvectors,
        &basis.site_blocks[site].overlap,
        &coordinates,
    )?)
}

fn append_pdos_samples(
    samples: &mut Vec<PdosEnergySample>,
    states: &[BandState],
    occupations: &[f64],
    range: &Range<usize>,
    projections: &[f64],
) -> Result<(), CheckpointPhysicsError> {
    if range.len() != projections.len() || range.end > states.len() || range.end > occupations.len()
    {
        return Err(CheckpointPhysicsError::BandProjectionCount {
            states: range.len(),
            projections: projections.len(),
        });
    }
    for (band, &projection) in projections.iter().enumerate() {
        let global = range.start + band;
        samples.push(PdosEnergySample::new(
            states[global],
            occupations[global],
            projection,
        ));
    }
    Ok(())
}

struct InitialOccupationSolution {
    chemical_potential: Hartree,
    occupations: Vec<f64>,
}

fn solve_initial_occupations(
    states: &[BandState],
    config: &ScfConfig,
) -> Result<InitialOccupationSolution, CheckpointPhysicsError> {
    let core: f64 = config
        .core_sites
        .iter()
        .flat_map(|site| &site.states)
        .map(|state| state.occupation)
        .sum();
    let electrons = config.electron_count - core;
    let (chemical_potential, occupations) = match config.occupations {
        ScfOccupations::FermiDirac { temperature } => {
            let solved = solve_fermi_dirac(
                states,
                electrons,
                temperature,
                OCCUPATION_TOLERANCE,
                OCCUPATION_ITERATIONS,
            )?;
            (solved.chemical_potential, solved.occupations)
        }
        ScfOccupations::Gaussian { width } => {
            let solved = solve_gaussian(
                states,
                electrons,
                width,
                OCCUPATION_TOLERANCE,
                OCCUPATION_ITERATIONS,
            )?;
            (solved.chemical_potential, solved.occupations)
        }
    };
    Ok(InitialOccupationSolution {
        chemical_potential,
        occupations,
    })
}

pub(crate) fn regular_k_points(mesh: ScfKMesh) -> Result<Vec<[f64; 3]>, CheckpointPhysicsError> {
    let count = mesh
        .divisions
        .into_iter()
        .try_fold(1_usize, usize::checked_mul)
        .ok_or(CheckpointPhysicsError::KPointCountOverflow)?;
    let mut points = Vec::with_capacity(count);
    for k3 in 0..mesh.divisions[2] {
        for k2 in 0..mesh.divisions[1] {
            for k1 in 0..mesh.divisions[0] {
                points.push([
                    (k1 as f64 + mesh.shift[0]) / mesh.divisions[0] as f64,
                    (k2 as f64 + mesh.shift[1]) / mesh.divisions[1] as f64,
                    (k3 as f64 + mesh.shift[2]) / mesh.divisions[2] as f64,
                ]);
            }
        }
    }
    Ok(points)
}

fn production_plane_wave_envelope(
    reciprocal: ReciprocalLattice,
    fractional_k: [f64; 3],
    cutoff: f64,
) -> Result<PlaneWaveEnvelope, CheckpointPhysicsError> {
    if fractional_k.iter().any(|value| !value.is_finite()) {
        return Err(CheckpointPhysicsError::NonFiniteKPoint(fractional_k));
    }
    let k = fractional_to_reciprocal(fractional_k, reciprocal.basis());
    let k_norm = squared_norm(k.map(InverseBohr::get)).sqrt();
    let candidates = reciprocal.enumerate(InverseBohr(cutoff + k_norm))?;
    let waves = candidates
        .into_iter()
        .filter(|g| {
            let wave = std::array::from_fn(|axis| k[axis].get() + g.cartesian[axis].get());
            squared_norm(wave) <= cutoff * cutoff * (1.0 + 64.0 * f64::EPSILON)
        })
        .map(|g| PlaneWave::new(k, g))
        .collect::<Vec<_>>();
    if waves.is_empty() {
        return Err(CheckpointPhysicsError::EmptyPlaneWaveBasis {
            k: fractional_k,
            cutoff,
        });
    }
    Ok(PlaneWaveEnvelope::new(waves))
}

fn production_density_layout(
    reciprocal: ReciprocalLattice,
    k_mesh: ScfKMesh,
    cutoff: f64,
) -> Result<FourierLayout, CheckpointPhysicsError> {
    let points = regular_k_points(k_mesh)?;
    if points.is_empty() {
        return Err(CheckpointPhysicsError::EmptyKPointSet);
    }
    let mut indices = BTreeSet::new();
    for point in points {
        let envelope = production_plane_wave_envelope(reciprocal, point, cutoff)?;
        insert_plane_wave_differences(&mut indices, envelope.waves());
    }
    let vectors = indices
        .into_iter()
        .map(|index| g_vector(reciprocal, index))
        .collect();
    Ok(FourierLayout::new(reciprocal, vectors)?)
}

fn insert_plane_wave_differences(indices: &mut BTreeSet<[i32; 3]>, waves: &[PlaneWave]) {
    for left in waves {
        for right in waves {
            indices.insert([
                right.g.index[0] - left.g.index[0],
                right.g.index[1] - left.g.index[1],
                right.g.index[2] - left.g.index[2],
            ]);
        }
    }
}


fn convert_checkpoint_geometry(
    checkpoint: &GeometryV2,
) -> Result<ConvertedCheckpointGeometry, CheckpointPhysicsError> {
    let direct = checkpoint.lattice.vectors.map(|vector| vector.map(Bohr));
    let reciprocal = ReciprocalLattice::from_direct(direct)?;
    let mut sites = Vec::with_capacity(checkpoint.sites.len());
    for site in &checkpoint.sites {
        let position = fractional_to_cartesian(site.fractional_position, direct);
        let (up, down, nonmagnetic_scalar) =
            convert_v2_site_bases(&site.id, &checkpoint.radial_basis)?;
        if up.mesh != down.mesh {
            return Err(CheckpointPhysicsError::SpinMeshMismatch {
                site: site.id.clone(),
            });
        }
        let radius = up.mesh.last();
        let scale = site
            .muffin_tin_radius
            .abs()
            .max(radius.get().abs())
            .max(1.0);
        if (site.muffin_tin_radius - radius.get()).abs() > CHECKPOINT_RADIUS_TOLERANCE * scale {
            return Err(CheckpointPhysicsError::MuffinTinMeshRadius {
                site: site.id.clone(),
                declared: site.muffin_tin_radius,
                mesh: radius.get(),
            });
        }
        sites.push(CheckpointSite {
            id: site.id.clone(),
            position,
            radius,
            up,
            down,
            nonmagnetic_scalar,
        });
    }
    let geometry = InterstitialGeometry::new(
        VolumeBohr3(determinant(checkpoint.lattice.vectors)),
        sites
            .iter()
            .map(|site| Sphere {
                center: site.position,
                radius: site.radius,
            })
            .collect::<Vec<_>>(),
    )?;
    Ok(ConvertedCheckpointGeometry {
        direct,
        reciprocal,
        geometry,
        sites,
        nuclear_charges: checkpoint
            .sites
            .iter()
            .map(|site| f64::from(site.atomic_number))
            .collect(),
    })
}

fn determinant(matrix: [[f64; 3]; 3]) -> f64 {
    let [a, b, c] = matrix;
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn fractional_to_cartesian(fractional: [f64; 3], direct: [[Bohr; 3]; 3]) -> [Bohr; 3] {
    let fractional = fractional.map(|value| value.rem_euclid(1.0));
    std::array::from_fn(|axis| {
        Bohr(
            fractional
                .iter()
                .zip(direct)
                .map(|(&coefficient, vector)| coefficient * vector[axis].get())
                .sum(),
        )
    })
}

fn fractional_to_reciprocal(
    fractional: [f64; 3],
    reciprocal: &[[InverseBohr; 3]; 3],
) -> [InverseBohr; 3] {
    std::array::from_fn(|axis| {
        InverseBohr(
            fractional
                .iter()
                .zip(reciprocal)
                .map(|(&coefficient, vector)| coefficient * vector[axis].get())
                .sum(),
        )
    })
}

pub(crate) fn g_vector(reciprocal: ReciprocalLattice, index: [i32; 3]) -> GVector {
    let cartesian = reciprocal.cartesian(index);
    GVector {
        index,
        cartesian,
        norm: InverseBohr(squared_norm(cartesian.map(InverseBohr::get)).sqrt()),
    }
}

fn squared_norm(vector: [f64; 3]) -> f64 {
    vector.into_iter().map(|value| value * value).sum()
}

/// Checkpoint conversion or concrete DFT-kernel failure.
#[derive(Debug, Error)]
pub enum CheckpointPhysicsError {
    #[error(transparent)]
    ChannelKappa(#[from] ChannelKappaError),
    #[error(transparent)]
    PotentialBuild(#[from] ScfPotentialBuildError),
    #[error(transparent)]
    Checkpoint(#[from] IoError),
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    StepFunction(#[from] StepFunctionError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Sphere(#[from] SphereFieldError),
    #[error(transparent)]
    Regional(#[from] muffintin_dft::RegionalError),
    #[error(transparent)]
    Scalar(#[from] ScalarBuilderError),
    #[error(transparent)]
    SpinorBuilder(#[from] SpinorBuilderError),
    #[error(transparent)]
    SpinorFirstVariation(#[from] SpinorFirstVariationError),
    #[error(transparent)]
    Density(#[from] DensityError),
    #[error(transparent)]
    Occupation(#[from] OccupationError),
    #[error(transparent)]
    Electrostatic(#[from] RegionalElectrostaticError),
    #[error(transparent)]
    Xc(#[from] RegionalXcError),
    #[error(transparent)]
    SecondVariation(#[from] SecondVariationError),
    #[error(transparent)]
    SpinOrbitRadial(#[from] SpinOrbitRadialError),
    #[error(transparent)]
    SocOperator(#[from] SocOperatorError),
    #[error(transparent)]
    Operator(#[from] OperatorError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Lapw(#[from] LapwError),
    #[error(transparent)]
    Kappa(#[from] muffintin_core::KappaError),
    #[error(transparent)]
    Tetrahedron(#[from] TetrahedronError),
    #[error(transparent)]
    Hartree(#[from] muffintin_coulomb::HartreeError),
    #[error(transparent)]
    Dirac(#[from] DiracError),
    #[error(transparent)]
    CorePotential(#[from] CorePotentialBuildError),
    #[error(transparent)]
    CoreDensity(#[from] CoreDensityError),
    #[error("site {site:?} radial basis must be exactly scalar or an up/down pair")]
    InvalidRadialBasisSpins { site: String },
    #[error("V2 regional field is missing site {0:?}")]
    MissingV2FieldSite(String),
    #[error("cannot export {actual} muffin-tin fields against {expected} checkpoint sites")]
    ExportSiteCount { expected: usize, actual: usize },
    #[error("cannot export {from:?} spherical fields as {target:?} fields")]
    UnsupportedAngularConversion {
        from: HarmonicConvention,
        target: HarmonicConvention,
    },
    #[error("real-tesseral channel l={l}, m={m} has no signed-m partner")]
    UnpairedRealTesseralChannel { l: u32, m: i32 },
    #[error("site {site:?} has different up/down radial meshes")]
    SpinMeshMismatch { site: String },
    #[error("site {site:?} muffin-tin radius is {declared}, radial mesh ends at {mesh}")]
    MuffinTinMeshRadius {
        site: String,
        declared: f64,
        mesh: f64,
    },
    #[error("checkpoint interstitial potential must contain G=0")]
    MissingInterstitialZero,
    #[error("potential component {component} has {actual} sites, expected {expected}")]
    PotentialComponentSiteCount {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "scalar route needs Koelling-Harmon input at site {site:?}, spin {spin}; got {equation:?}"
    )]
    ScalarRadialEquation {
        site: String,
        spin: usize,
        equation: RadialEquationTag,
    },
    #[error(
        "full-spinor route needs fully relativistic input at site {site:?}, spin {spin}; got {equation:?}"
    )]
    SpinorRadialEquation {
        site: String,
        spin: usize,
        equation: RadialEquationTag,
    },
    #[error("SPEX material checkpoint annotations do not match the caller-owned recipe")]
    SpexMaterialProvenanceMismatch,
    #[error(
        "SPEX material source at site {site:?}, spin {spin} must remain scalar Koelling-Harmon; got {equation:?}"
    )]
    SpexMaterialSourceRadialEquation {
        site: String,
        spin: usize,
        equation: RadialEquationTag,
    },
    #[error(
        "SPEX material channel site={site:?}, n={n}, l={l}, kappa={kappa}, treatment={treatment:?}, derivative_order={derivative_order}, energy={energy} is not bound exactly to the runtime basis"
    )]
    SpexMaterialChannelMismatch {
        site: String,
        n: u32,
        l: u32,
        kappa: i32,
        treatment: ScfChannelTreatment,
        derivative_order: u32,
        energy: f64,
    },
    #[error("site {0:?} potential has no spherical monopole")]
    MissingMonopole(String),
    #[error("site {site:?} scalar monopole at radial index {radial} is not real: {value}")]
    NonRealMonopole {
        site: String,
        radial: usize,
        value: Complex64,
    },
    #[error(
        "site {site:?} channel {identity:?} ({treatment:?}, {generator:?}) energy generation failed: {source}"
    )]
    ChannelGenerator {
        site: String,
        identity: ScfChannelIdentity,
        treatment: ScfChannelTreatment,
        generator: LinearizationEnergyGenerator,
        #[source]
        source: LinearizationEnergyError,
    },
    #[error("site {site:?} signed-kappa valence partners cannot form scalar l={l}: {source}")]
    ScalarKappaAverage {
        site: String,
        l: u32,
        #[source]
        source: LinearizationEnergyError,
    },
    #[error("site {site:?} channel {identity:?} generator {generator:?} requires a recipe seed")]
    MissingChannelSeed {
        site: String,
        identity: ScfChannelIdentity,
        generator: LinearizationEnergyGenerator,
    },
    #[error(
        "site {site:?} channel {identity:?} generator {generator:?} requires an l-resolved identity"
    )]
    ScalarGeneratorRequiresLIdentity {
        site: String,
        identity: ScfChannelIdentity,
        generator: LinearizationEnergyGenerator,
    },
    #[error("site {site:?} has no materialized valence base for l={l}")]
    MissingMaterializedBaseChannel { site: String, l: u32 },
    #[error("site {site:?} has ambiguous materialized valence bases for l={l}")]
    AmbiguousBaseChannel { site: String, l: u32 },
    #[error("site {site:?} is missing the spinor base l={l}, kappa={kappa}")]
    MissingSpinorBaseChannel { site: String, l: u32, kappa: i32 },
    #[error(
        "site {site:?} channel {identity:?} ({treatment:?}) has no matching frozen-checkpoint anchor"
    )]
    MissingFrozenCheckpointAnchor {
        site: String,
        identity: ScfChannelIdentity,
        treatment: ScfChannelTreatment,
    },
    #[error("site {site:?}, spin {spin} has no frozen-checkpoint base anchor for l={l}")]
    MissingFrozenCheckpointBase { site: String, l: u32, spin: usize },
    #[error(
        "site {site:?}, spin {spin} has no frozen-checkpoint LO anchor for l={l}, ordinal={ordinal}"
    )]
    MissingFrozenCheckpointLo {
        site: String,
        l: u32,
        ordinal: usize,
        spin: usize,
    },
    #[error(
        "site {site:?} channel {identity:?} generator {generator:?} has no provisional materialized value"
    )]
    MissingProvisionalChannel {
        site: String,
        identity: ScfChannelIdentity,
        generator: LinearizationEnergyGenerator,
    },
    #[error("site {site:?} channel {identity:?} has an ambiguous n-resolved band-cog projection")]
    AmbiguousBandCogProjection {
        site: String,
        identity: ScfChannelIdentity,
    },
    #[error("site {site:?} signed channel {identity:?} cannot be projected in a scalar band solve")]
    KappaBandCogUnavailableInScalar {
        site: String,
        identity: ScfChannelIdentity,
    },
    #[error("band projection returned {projections} values for {states} states")]
    BandProjectionCount { states: usize, projections: usize },
    #[error("initial frozen-potential basis refinement did not converge after {passes} passes")]
    InitialBasisRefinementNotConverged { passes: usize },
    #[error("k point {0:?} contains a non-finite coordinate")]
    NonFiniteKPoint([f64; 3]),
    #[error("plane-wave cutoff {cutoff} produces no basis at k={k:?}")]
    EmptyPlaneWaveBasis { k: [f64; 3], cutoff: f64 },
    #[error("regular k-point set is empty")]
    EmptyKPointSet,
    #[error("regular k-point count overflows usize")]
    KPointCountOverflow,
    #[error("scalar product input requires scalar Koelling-Harmon relativity")]
    ScalarProductRequiresScalarRelativity,
    #[error(
        "spinor product input requires ScfRelativity::SpinorFirstVariation, not scalar Koelling-Harmon"
    )]
    SpinorProductRejectsScalarRelativity,
    #[error(
        "spinor product input requires ScfRelativity::SpinorFirstVariation, not SOC second variation; signed-kappa is not routed through second variation"
    )]
    SpinorProductRejectsSocSecondVariation,
    #[error(
        "spinor product k-mesh, compiled bases, eigenvectors, energies, available-band counts, and k-q map must share one ordered k slice"
    )]
    SpinorProductKSliceMismatch,
    #[error("spinor product source transfer q does not match the frozen q-slice")]
    SpinorProductTransferQMismatch,
    #[error(transparent)]
    DiracProduct(#[from] muffintin_prodbasis::DiracProductError),
    #[error(
        "folded k-q {folded:?} from k={k:?} q_in={q_in:?} q_canonical={q_canonical:?} is not on the regular mesh"
    )]
    OffMeshTransfer {
        k: [f64; 3],
        q_in: [f64; 3],
        q_canonical: [f64; 3],
        folded: [f64; 3],
    },
    #[error("collinear product input needs equal up/down band counts, got {up} and {down}")]
    CollinearBandCount { up: usize, down: usize },
    #[error(transparent)]
    Product(#[from] muffintin_prodbasis::AuxiliaryIrError),
    #[error("second variation requires a nonmagnetic scalar checkpoint and potential")]
    SecondVariationRequiresNonmagneticScalar,
    #[error(
        "second-variation window starts at {start}; runtime requires start=0 so occupied lower scalar bands are not dropped"
    )]
    SecondVariationDropsLowerBands { start: usize },
    #[error("angular momentum does not fit the signed-kappa representation")]
    AngularMomentumOverflow,
    #[error("one band solution mixed scalar and spinor k-point routes")]
    InconsistentRelativityRoute,
    #[error(
        "scalar/second-variation route cannot consume transverse potential RMS ({x_rms}, {y_rms}) above {tolerance}"
    )]
    TransversePotentialUnsupported {
        x_rms: f64,
        y_rms: f64,
        tolerance: f64,
    },
    #[error("core site {0:?} is not present in the checkpoint")]
    UnknownCoreSite(String),
    #[error("extended core mesh point count overflows usize")]
    CoreMeshCountOverflow,
    #[error("core solve has no regional density template")]
    MissingDensityTemplate,
    #[error("iteration {0} has no raw periodic continuation for the core solve")]
    MissingCoreContinuation(usize),
    #[error("received {actual} occupations for {expected} states")]
    OccupationCount { expected: usize, actual: usize },
    #[error("requested {requested} bands, but only {available} are available")]
    TooFewBands { requested: usize, available: usize },
    #[error("k points retained inconsistent band counts")]
    InconsistentBandCount,
    #[error("iteration {0} has no cached electrostatic/XC energy terms")]
    MissingEnergyTerms(usize),
}

#[cfg(test)]
mod atomic_checkpoint_test;
#[cfg(test)]
mod tests;
