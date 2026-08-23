//! Concrete production DFT kernel reconstructed from a V1 snapshot.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::ops::Range;

use muffintin_core::{
    Bohr, ExponentialMesh, FourierFieldError, FourierLayout, GVector, Hartree,
    HermitianFourierField, InterstitialGeometry, InverseBohr, Kappa, LatticeError, MeshError,
    ReciprocalLattice, Sphere, StepFunctionError, VolumeBohr3,
};
use muffintin_dft::{
    AtomicEnergyRequest, BandPathRequest, BandState, CollinearKPoint, CoreContribution,
    CoreDensityError, CorePotentialBuildError, CorePotentialBuildSpec, CoreSpinPartition,
    DensityError, ElectrostaticSpec, FirstVariationRoute, FirstVariationSubspace, FullSpinorKPoint,
    GeneratedLinearizationEnergy, InterstitialField, LinearizationEnergyDiagnostic,
    LinearizationEnergyError, LinearizationEnergyGenerator, LocalPauliPotential, MuffinTinField,
    NoncollinearXcRoute, OccupationError, PdosEnergySample, RegionalCoreShellInput,
    RegionalDensity, RegionalElectrostaticError, RegionalElectrostaticResult, RegionalPotential,
    RegionalScalarField, RegionalXcError, RegionalXcResult, RegularSpectrum, ScalarBuilderError,
    ScalarIterationBasis, ScalarLocalOrbitalRequest, ScalarSiteInput, ScfBasis, ScfChannelIdentity,
    ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfCoreSite, ScfEnergyContext,
    ScfEnergyTerms, ScfExchangeCorrelation, ScfKMesh, ScfOccupations, ScfPhysics, ScfRelativity,
    ScfResolvedChannelEnergy, ScfState, SecondVariationError, SpinorBuilderError,
    SpinorFirstVariationError, SpinorIterationBasis, SpinorLinearizationEnergy,
    SpinorLocalOrbitalRequest, SpinorSiteInput, TetrahedronError, XcFieldSpec,
    build_collinear_scalar_iteration_bases, build_extended_core_potentials,
    build_extended_snapshot_core_potentials, build_regional_core_contribution,
    build_spinor_iteration_basis, evaluate_regional_electrostatics, evaluate_regional_xc,
    generate_atomic_energy, generate_band_center_energy, generate_band_cog_energy,
    generate_explicit_energy, generate_fermi_offset_energy, generate_frozen_snapshot_energy,
    generate_log_derivative_energy, kappa_degeneracy_average, physical_site_band_projections,
    solve_fermi_dirac, solve_gaussian, solve_soc_second_variation, solve_spinor_k_point,
    synthesize_collinear_valence_density, synthesize_full_spinor_valence_density,
};
use muffintin_envelope::{PlaneWave, PlaneWaveEnvelope};
use muffintin_io::{
    AngularBasisV1, Complex64V2, DensityV2, FieldRepresentationV2, FieldUnitV2,
    FourierCoefficientV2, InitialV2, InterstitialFieldV2, IoError, MuffinTinFieldV2, PotentialV2,
    RadialBasisSpinV2, RadialEquationTagV1, RegionalFieldV2, SnapshotV2, SphericalChannelV2,
};
use muffintin_lapw::{Collinear, GeneralizedEigensolution, InterstitialPotential, LapwError};
use muffintin_operators::{
    CompiledSiteProjection, OperatorError, SiteSpinOrbitBlock, SocOperatorError,
    SpinorSiteOperatorBlocks,
};
use muffintin_radial::{
    CoreBracketSearch, CoreDiracSpec, CorePotentialContinuationSpec, CoreState, DiracError,
    EnergyBracket, ExtendedCorePotential, RadialEquation, SpexSpinOrbitPotential,
    SpinOrbitRadialError, isolate_core_dirac_bracket, solve_core_dirac,
    spex_spin_orbit_radial_shell,
};
use muffintin_sphere::{HarmonicConvention, SphereField, SphereFieldError};
use muffintin_tensor::{DenseEigenvectors, TensorError};
use num_complex::Complex64;
use thiserror::Error;

const OVERLAP_THRESHOLD: f64 = 1.0e-10;
const OCCUPATION_TOLERANCE: f64 = 1.0e-12;
const OCCUPATION_ITERATIONS: usize = 256;
const SNAPSHOT_RADIUS_TOLERANCE: f64 = 1.0e-10;
const TRANSVERSE_FIELD_TOLERANCE: f64 = 1.0e-10;
const SPECTRAL_REFINEMENT_TOLERANCE: f64 = 1.0e-10;
const DEFAULT_FERMI_OFFSET_HARTREE: f64 = -0.1;

/// Snapshot-backed material kernel shared by SCF, bands, and DOS tasks.
///
/// Construction performs only convention conversion and topology validation.
/// The initial density is obtained by a frozen-snapshot one-particle solve;
/// no atomic-density or artificial `G=0` guess is installed.
#[derive(Debug)]
pub struct SnapshotDftPhysics {
    snapshot_template: SnapshotV2,
    reciprocal: ReciprocalLattice,
    geometry: InterstitialGeometry,
    sites: Vec<SnapshotSite>,
    frozen_potential: RegionalPotential,
    restart_density: Option<RegionalDensity>,
    nuclear_charges: Vec<f64>,
    core_potentials: BTreeMap<usize, CorePotentialContext>,
    density_template: Option<RegionalDensity>,
    energy_terms: BTreeMap<usize, ScfEnergyTerms>,
}

#[derive(Clone, Debug)]
struct CorePotentialContext {
    electrostatic: RegionalElectrostaticResult,
    exchange_correlation: RegionalXcResult,
    density: RegionalDensity,
    spec: CorePotentialBuildSpec,
}

#[derive(Clone, Debug)]
struct SnapshotSite {
    id: String,
    position: [Bohr; 3],
    radius: Bohr,
    up: SnapshotSpin,
    down: SnapshotSpin,
    nonmagnetic_scalar: bool,
}

#[derive(Clone, Debug)]
struct SnapshotSpin {
    equation: RadialEquationTagV1,
    mesh: ExponentialMesh,
    linearization: BTreeMap<u32, Hartree>,
    local_orbitals: Vec<(u32, Hartree)>,
}

/// One iteration's potential and basis-neutral controls. Concrete k-dependent
/// APW matching is intentionally deferred until the requested k points exist.
#[derive(Clone, Debug)]
pub struct SnapshotOneParticle {
    potential: RegionalPotential,
    basis: ScfBasis,
}

/// Concrete regular-mesh solutions retained for occupations and density synthesis.
#[derive(Clone, Debug)]
pub struct SnapshotBandSolution {
    points: Vec<SnapshotKPoint>,
    states: Vec<BandState>,
}

#[derive(Clone, Debug)]
struct SnapshotKPoint {
    weight: f64,
    solution: SnapshotKPointSolution,
    energies: Vec<Hartree>,
}

#[derive(Clone, Debug)]
enum SnapshotKPointSolution {
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

impl SnapshotDftPhysics {
    /// Convert a validated V2 snapshot into exact internal units and conventions.
    pub fn new(snapshot: &SnapshotV2) -> Result<Self, SnapshotDftError> {
        snapshot.validate()?;
        let direct = snapshot
            .geometry
            .lattice
            .vectors
            .map(|vector| vector.map(Bohr));
        let reciprocal = ReciprocalLattice::from_direct(direct)?;
        let volume = determinant(snapshot.geometry.lattice.vectors);

        let mut converted_sites = Vec::with_capacity(snapshot.geometry.sites.len());
        for site in &snapshot.geometry.sites {
            let position = fractional_to_cartesian(site.fractional_position, direct);
            let (up, down, nonmagnetic_scalar) =
                convert_v2_site_bases(&site.id, &snapshot.geometry.radial_basis)?;
            if up.mesh != down.mesh {
                return Err(SnapshotDftError::SpinMeshMismatch {
                    site: site.id.clone(),
                });
            }
            let radius = up.mesh.last();
            let scale = site
                .muffin_tin_radius
                .abs()
                .max(radius.get().abs())
                .max(1.0);
            if (site.muffin_tin_radius - radius.get()).abs() > SNAPSHOT_RADIUS_TOLERANCE * scale {
                return Err(SnapshotDftError::MuffinTinMeshRadius {
                    site: site.id.clone(),
                    declared: site.muffin_tin_radius,
                    mesh: radius.get(),
                });
            }
            converted_sites.push(SnapshotSite {
                id: site.id.clone(),
                position,
                radius,
                up,
                down,
                nonmagnetic_scalar,
            });
        }
        let geometry = InterstitialGeometry::new(
            VolumeBohr3(volume),
            converted_sites
                .iter()
                .map(|site| Sphere {
                    center: site.position,
                    radius: site.radius,
                })
                .collect::<Vec<_>>(),
        )?;
        let restart_density = match &snapshot.initial {
            InitialV2::FrozenPotential { .. } => None,
            InitialV2::Restart { density, .. } => Some(density),
        };
        let potential = match &snapshot.initial {
            InitialV2::FrozenPotential { potential } | InitialV2::Restart { potential, .. } => {
                potential
            }
        };
        let frozen_potential =
            regional_potential_from_v2(potential, &geometry, &converted_sites, reciprocal)?;
        let restart_density = restart_density
            .map(|density| {
                regional_density_from_v2(density, &geometry, &converted_sites, reciprocal)
            })
            .transpose()?;
        Ok(Self {
            snapshot_template: snapshot.clone(),
            reciprocal,
            geometry,
            sites: converted_sites,
            frozen_potential,
            restart_density,
            nuclear_charges: snapshot
                .geometry
                .sites
                .iter()
                .map(|site| f64::from(site.atomic_number))
                .collect(),
            core_potentials: BTreeMap::new(),
            density_template: None,
            energy_terms: BTreeMap::new(),
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

    /// Serialize a converged state as a V2 restart while preserving this
    /// kernel's immutable geometry and radial-basis identity.
    pub fn restart_snapshot(&self, state: &ScfState) -> Result<SnapshotV2, SnapshotDftError> {
        snapshot_v2_from_state(&self.snapshot_template, state)
    }

    fn solve_points(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        points: &[[f64; 3]],
        relativity: ScfRelativity,
    ) -> Result<SnapshotBandSolution, SnapshotDftError> {
        if points.is_empty() {
            return Err(SnapshotDftError::EmptyKPointSet);
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
                        return Err(SnapshotDftError::SecondVariationDropsLowerBands {
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
            solved_points.push(SnapshotKPoint {
                weight,
                solution: SnapshotKPointSolution::Collinear {
                    bases: Box::new(bases),
                    solutions,
                    up_occupations,
                    down_occupations,
                },
                energies,
            });
        }
        Ok(SnapshotBandSolution {
            points: solved_points,
            states,
        })
    }

    fn solve_spinor_points(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        points: &[[f64; 3]],
    ) -> Result<SnapshotBandSolution, SnapshotDftError> {
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
            solved_points.push(SnapshotKPoint {
                weight,
                energies: solved.solution.eigenvalues.clone(),
                solution: SnapshotKPointSolution::Spinor {
                    basis: spinor_basis,
                    site_blocks: solved.site_blocks,
                    solution: solved.solution,
                    occupations: start..end,
                },
            });
        }
        Ok(SnapshotBandSolution {
            points: solved_points,
            states,
        })
    }

    fn scalar_site_inputs(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<Collinear<Vec<ScalarSiteInput>>, SnapshotDftError> {
        self.require_potential_site_count(potential)?;
        let build_spin = |spin: usize| {
            self.sites
                .iter()
                .enumerate()
                .map(|(site_index, site)| {
                    let template = if spin == 0 { &site.up } else { &site.down };
                    if template.equation != RadialEquationTagV1::ScalarKoellingHarmon {
                        return Err(SnapshotDftError::ScalarRadialEquation {
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
                        .ok_or_else(|| SnapshotDftError::MissingMonopole(site.id.clone()))?;
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
                .collect::<Result<Vec<_>, SnapshotDftError>>()
        };
        Ok(Collinear::new(build_spin(0)?, build_spin(1)?))
    }

    fn spinor_site_inputs(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<Vec<SpinorSiteInput>, SnapshotDftError> {
        self.require_potential_site_count(potential)?;
        self.sites
            .iter()
            .enumerate()
            .map(|(site_index, site)| {
                for (spin, template) in [&site.up, &site.down].into_iter().enumerate() {
                    if template.equation != RadialEquationTagV1::FullyRelativisticDirac {
                        return Err(SnapshotDftError::SpinorRadialEquation {
                            site: site.id.clone(),
                            spin,
                            equation: template.equation,
                        });
                    }
                }
                let scalar = potential.scalar().muffin_tins()[site_index].field().clone();
                let magnetic = potential
                    .magnetic()
                    .each_ref()
                    .map(|component| component.muffin_tins()[site_index].field().clone());
                let monopole = scalar
                    .channel(0, 0)
                    .ok_or_else(|| SnapshotDftError::MissingMonopole(site.id.clone()))?;
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

    fn scalar_linearization_energies(
        &self,
        basis: &ScfBasis,
        site: &str,
        spin: usize,
    ) -> Result<Vec<Hartree>, SnapshotDftError> {
        (0..=basis.l_max)
            .map(|l| {
                let channels = basis
                    .resolved_channels
                    .iter()
                    .filter(|resolved| {
                        resolved.recipe.site == site
                            && resolved.recipe.treatment == ScfChannelTreatment::Valence
                            && channel_l(resolved.recipe.identity) == l
                    })
                    .collect::<Vec<_>>();
                let scalar = channels
                    .iter()
                    .filter(|resolved| {
                        matches!(resolved.recipe.identity, ScfChannelIdentity::ScalarL { .. })
                    })
                    .copied()
                    .collect::<Vec<_>>();
                if scalar.len() == 1 && channels.len() == 1 {
                    return Ok(spin_resolved_energy(scalar[0], spin));
                }
                if scalar.is_empty() && !channels.is_empty() {
                    let n = channel_n(channels[0].recipe.identity);
                    if channels
                        .iter()
                        .any(|resolved| channel_n(resolved.recipe.identity) != n)
                    {
                        return Err(SnapshotDftError::AmbiguousBaseChannel {
                            site: site.to_owned(),
                            l,
                        });
                    }
                    let partners = channels
                        .iter()
                        .map(|resolved| match resolved.recipe.identity {
                            ScfChannelIdentity::Kappa { kappa, .. } => {
                                Ok((Kappa::new(kappa)?, resolved.energy))
                            }
                            ScfChannelIdentity::ScalarL { .. } => unreachable!(),
                        })
                        .collect::<Result<Vec<_>, SnapshotDftError>>()?;
                    return kappa_degeneracy_average(l, &partners).map_err(|source| {
                        SnapshotDftError::ScalarKappaAverage {
                            site: site.to_owned(),
                            l,
                            source,
                        }
                    });
                }
                if channels.is_empty() {
                    Err(SnapshotDftError::MissingMaterializedBaseChannel {
                        site: site.to_owned(),
                        l,
                    })
                } else {
                    Err(SnapshotDftError::AmbiguousBaseChannel {
                        site: site.to_owned(),
                        l,
                    })
                }
            })
            .collect()
    }

    fn scalar_local_orbitals(
        &self,
        basis: &ScfBasis,
        site: &str,
        spin: usize,
    ) -> Result<Vec<ScalarLocalOrbitalRequest>, SnapshotDftError> {
        basis
            .resolved_channels
            .iter()
            .filter(|resolved| {
                resolved.recipe.site == site
                    && matches!(resolved.recipe.identity, ScfChannelIdentity::ScalarL { .. })
                    && matches!(
                        resolved.recipe.treatment,
                        ScfChannelTreatment::Lo | ScfChannelTreatment::Hdlo
                    )
            })
            .map(|resolved| {
                let l = channel_l(resolved.recipe.identity);
                Ok(match resolved.recipe.treatment {
                    ScfChannelTreatment::Lo => ScalarLocalOrbitalRequest::Lo {
                        l,
                        energy: spin_resolved_energy(resolved, spin),
                    },
                    ScfChannelTreatment::Hdlo => ScalarLocalOrbitalRequest::Hdlo { l },
                    ScfChannelTreatment::Core | ScfChannelTreatment::Valence => unreachable!(),
                })
            })
            .collect()
    }

    fn spinor_linearization_energies(
        &self,
        basis: &ScfBasis,
        site: &str,
    ) -> Result<Vec<SpinorLinearizationEnergy>, SnapshotDftError> {
        let mut energies = Vec::new();
        for l in 0..=basis.l_max {
            let channels = basis
                .resolved_channels
                .iter()
                .filter(|resolved| {
                    resolved.recipe.site == site
                        && resolved.recipe.treatment == ScfChannelTreatment::Valence
                        && channel_l(resolved.recipe.identity) == l
                })
                .collect::<Vec<_>>();
            let scalar = channels
                .iter()
                .filter(|resolved| {
                    matches!(resolved.recipe.identity, ScfChannelIdentity::ScalarL { .. })
                })
                .copied()
                .collect::<Vec<_>>();
            if scalar.len() == 1 && channels.len() == 1 {
                for kappa in spinor_kappas_for_l(l)? {
                    energies.push(SpinorLinearizationEnergy {
                        kappa,
                        energy: scalar_component_energy(scalar[0], kappa),
                    });
                }
                continue;
            }
            if scalar.is_empty() && !channels.is_empty() {
                for kappa in spinor_kappas_for_l(l)? {
                    let matches = channels
                        .iter()
                        .filter(|resolved| {
                            matches!(
                                resolved.recipe.identity,
                                ScfChannelIdentity::Kappa {
                                    kappa: candidate,
                                    ..
                                } if candidate == kappa.get()
                            )
                        })
                        .copied()
                        .collect::<Vec<_>>();
                    if matches.len() != 1 {
                        return Err(SnapshotDftError::MissingSpinorBaseChannel {
                            site: site.to_owned(),
                            l,
                            kappa: kappa.get(),
                        });
                    }
                    energies.push(SpinorLinearizationEnergy {
                        kappa,
                        energy: matches[0].energy,
                    });
                }
                continue;
            }
            return Err(if channels.is_empty() {
                SnapshotDftError::MissingMaterializedBaseChannel {
                    site: site.to_owned(),
                    l,
                }
            } else {
                SnapshotDftError::AmbiguousBaseChannel {
                    site: site.to_owned(),
                    l,
                }
            });
        }
        Ok(energies)
    }

    fn spinor_local_orbitals(
        &self,
        basis: &ScfBasis,
        site: &str,
    ) -> Result<Vec<SpinorLocalOrbitalRequest>, SnapshotDftError> {
        let mut orbitals = Vec::new();
        for resolved in basis.resolved_channels.iter().filter(|resolved| {
            resolved.recipe.site == site
                && matches!(
                    resolved.recipe.treatment,
                    ScfChannelTreatment::Lo | ScfChannelTreatment::Hdlo
                )
        }) {
            let kappas = match resolved.recipe.identity {
                ScfChannelIdentity::ScalarL { l, .. } => spinor_kappas_for_l(l)?,
                ScfChannelIdentity::Kappa { kappa, .. } => vec![Kappa::new(kappa)?],
            };
            for kappa in kappas {
                orbitals.push(match resolved.recipe.treatment {
                    ScfChannelTreatment::Lo => SpinorLocalOrbitalRequest::Lo {
                        kappa,
                        energy: scalar_component_energy(resolved, kappa),
                    },
                    ScfChannelTreatment::Hdlo => SpinorLocalOrbitalRequest::Hdlo { kappa },
                    ScfChannelTreatment::Core | ScfChannelTreatment::Valence => unreachable!(),
                });
            }
        }
        Ok(orbitals)
    }

    fn materialize_current_basis(
        &self,
        iteration: usize,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<ScfBasis, SnapshotDftError> {
        let context = self
            .core_potentials
            .get(&iteration)
            .ok_or(SnapshotDftError::MissingCoreContinuation(iteration))?;
        let meshes = self.channel_meshes(basis)?;
        let extended = build_extended_core_potentials(
            &context.electrostatic,
            &context.exchange_correlation,
            &context.density,
            &meshes,
            context.spec,
        )?;
        self.materialize_nonspectral_basis(potential, basis, &extended)
    }

    fn materialize_nonspectral_basis(
        &self,
        potential: &RegionalPotential,
        requested: &ScfBasis,
        extended: &[muffintin_dft::BuiltExtendedCorePotential],
    ) -> Result<ScfBasis, SnapshotDftError> {
        self.require_potential_site_count(potential)?;
        let mut basis = requested.clone();
        basis.resolved_channels.clear();
        let mut lo_ordinals = BTreeMap::<(String, u32), usize>::new();
        for recipe in requested
            .channels
            .iter()
            .filter(|recipe| recipe.treatment != ScfChannelTreatment::Core)
        {
            let site_index = self.site_index(&recipe.site)?;
            let lo_ordinal = if recipe.treatment == ScfChannelTreatment::Lo {
                let key = (recipe.site.clone(), channel_l(recipe.identity));
                let ordinal = lo_ordinals.entry(key).or_default();
                let current = *ordinal;
                *ordinal += 1;
                Some(current)
            } else {
                None
            };
            let generated = if matches!(
                recipe.generator,
                LinearizationEnergyGenerator::BandCog | LinearizationEnergyGenerator::FermiOffset
            ) {
                self.provisional_spectral_channel(recipe, lo_ordinal)?
            } else {
                self.materialize_potential_channel(
                    recipe,
                    site_index,
                    potential,
                    &extended[site_index].potential,
                    lo_ordinal,
                )?
            };
            basis.resolved_channels.push(generated);
        }
        Ok(basis)
    }

    fn materialize_potential_channel(
        &self,
        recipe: &ScfChannelRecipe,
        site_index: usize,
        potential: &RegionalPotential,
        extended: &ExtendedCorePotential,
        lo_ordinal: Option<usize>,
    ) -> Result<ScfResolvedChannelEnergy, SnapshotDftError> {
        let site = &self.sites[site_index];
        let l = channel_l(recipe.identity);
        let one = |generated: GeneratedLinearizationEnergy| ScfResolvedChannelEnergy {
            recipe: recipe.clone(),
            energy: generated.energy,
            components: vec![generated],
        };
        let generated = match recipe.generator {
            LinearizationEnergyGenerator::Explicit => {
                let seed = recipe
                    .seed
                    .ok_or_else(|| SnapshotDftError::MissingChannelSeed {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                        generator: recipe.generator,
                    })?;
                return generate_explicit_energy(seed)
                    .map(one)
                    .map_err(|source| channel_generator_error(recipe, source));
            }
            LinearizationEnergyGenerator::FrozenSnapshot => {
                let up = self.snapshot_anchor_spin(recipe, lo_ordinal, 0)?;
                let down = self.snapshot_anchor_spin(recipe, lo_ordinal, 1)?;
                let mut components = vec![
                    generate_frozen_snapshot_energy(up)
                        .map_err(|source| channel_generator_error(recipe, source))?,
                ];
                let energy = if site.nonmagnetic_scalar {
                    up
                } else {
                    components.push(
                        generate_frozen_snapshot_energy(down)
                            .map_err(|source| channel_generator_error(recipe, source))?,
                    );
                    Hartree(0.5 * (up.get() + down.get()))
                };
                return Ok(ScfResolvedChannelEnergy {
                    recipe: recipe.clone(),
                    energy,
                    components,
                });
            }
            LinearizationEnergyGenerator::Atomic => {
                let kappas = channel_kappas(recipe.identity)?;
                let mut components = Vec::with_capacity(kappas.len());
                let mut partner_energies = Vec::with_capacity(kappas.len());
                for kappa in kappas {
                    let state = CoreState::new(channel_n(recipe.identity), kappa)?;
                    let generated = generate_atomic_energy(
                        &extended.mesh,
                        &extended.values,
                        AtomicEnergyRequest::new(
                            state,
                            self.nuclear_charges[site_index],
                            site.radius,
                        ),
                    )
                    .map_err(|source| channel_generator_error(recipe, source))?;
                    partner_energies.push((kappa, generated.energy));
                    components.push(generated);
                }
                let energy = match recipe.identity {
                    ScfChannelIdentity::ScalarL { .. } => {
                        kappa_degeneracy_average(l, &partner_energies)
                            .map_err(|source| channel_generator_error(recipe, source))?
                    }
                    ScfChannelIdentity::Kappa { .. } => components[0].energy,
                };
                return Ok(ScfResolvedChannelEnergy {
                    recipe: recipe.clone(),
                    energy,
                    components,
                });
            }
            LinearizationEnergyGenerator::BandCenter
            | LinearizationEnergyGenerator::LogDerivative => {
                if matches!(recipe.identity, ScfChannelIdentity::Kappa { .. }) {
                    return Err(SnapshotDftError::ScalarGeneratorRequiresLIdentity {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                        generator: recipe.generator,
                    });
                }
                let seed = recipe
                    .seed
                    .unwrap_or(self.snapshot_anchor(recipe, lo_ordinal)?);
                let spherical = spherical_scalar_potential(potential, site_index, &site.id)?;
                if recipe.generator == LinearizationEnergyGenerator::BandCenter {
                    generate_band_center_energy(
                        &site.up.mesh,
                        &spherical,
                        RadialEquation::ScalarKoellingHarmon,
                        l,
                        seed,
                    )
                } else {
                    generate_log_derivative_energy(
                        &site.up.mesh,
                        &spherical,
                        RadialEquation::ScalarKoellingHarmon,
                        channel_n(recipe.identity),
                        l,
                        seed,
                        InverseBohr(-(f64::from(l) + 1.0) / site.radius.get()),
                    )
                }
            }
            LinearizationEnergyGenerator::BandCog | LinearizationEnergyGenerator::FermiOffset => {
                unreachable!("spectral generators are materialized after occupations")
            }
        };
        generated
            .map(one)
            .map_err(|source| channel_generator_error(recipe, source))
    }

    fn provisional_spectral_channel(
        &self,
        recipe: &ScfChannelRecipe,
        lo_ordinal: Option<usize>,
    ) -> Result<ScfResolvedChannelEnergy, SnapshotDftError> {
        let energy = self.snapshot_anchor(recipe, lo_ordinal)?;
        Ok(ScfResolvedChannelEnergy {
            recipe: recipe.clone(),
            energy,
            components: vec![GeneratedLinearizationEnergy {
                generator: recipe.generator,
                seed: Some(energy),
                energy,
                diagnostic: LinearizationEnergyDiagnostic::Stored,
            }],
        })
    }

    fn snapshot_anchor(
        &self,
        recipe: &ScfChannelRecipe,
        lo_ordinal: Option<usize>,
    ) -> Result<Hartree, SnapshotDftError> {
        let site_index = self.site_index(&recipe.site)?;
        let up = self.snapshot_anchor_spin(recipe, lo_ordinal, 0)?;
        let down = self.snapshot_anchor_spin(recipe, lo_ordinal, 1)?;
        Ok(if self.sites[site_index].nonmagnetic_scalar {
            up
        } else {
            Hartree(0.5 * (up.get() + down.get()))
        })
    }

    fn snapshot_anchor_spin(
        &self,
        recipe: &ScfChannelRecipe,
        lo_ordinal: Option<usize>,
        spin: usize,
    ) -> Result<Hartree, SnapshotDftError> {
        let site_index = self.site_index(&recipe.site)?;
        let site = &self.sites[site_index];
        let l = channel_l(recipe.identity);
        let radial = if spin == 0 { &site.up } else { &site.down };
        match recipe.treatment {
            ScfChannelTreatment::Lo => {
                let ordinal =
                    lo_ordinal.ok_or_else(|| SnapshotDftError::MissingFrozenSnapshotAnchor {
                        site: recipe.site.clone(),
                        identity: recipe.identity,
                        treatment: recipe.treatment,
                    })?;
                radial
                    .local_orbitals
                    .iter()
                    .filter(|(candidate_l, _)| *candidate_l == l)
                    .nth(ordinal)
                    .map(|(_, energy)| *energy)
                    .ok_or_else(|| SnapshotDftError::MissingFrozenSnapshotLo {
                        site: site.id.clone(),
                        l,
                        ordinal,
                        spin,
                    })
            }
            ScfChannelTreatment::Core
            | ScfChannelTreatment::Valence
            | ScfChannelTreatment::Hdlo => radial.linearization.get(&l).copied().ok_or_else(|| {
                SnapshotDftError::MissingFrozenSnapshotBase {
                    site: site.id.clone(),
                    l,
                    spin,
                }
            }),
        }
    }

    fn channel_meshes(&self, basis: &ScfBasis) -> Result<Vec<ExponentialMesh>, SnapshotDftError> {
        self.sites
            .iter()
            .enumerate()
            .map(|(site_index, site)| {
                let maximum_n = basis
                    .channels
                    .iter()
                    .filter(|recipe| recipe.site == site.id)
                    .map(|recipe| channel_n(recipe.identity))
                    .max()
                    .unwrap_or(1);
                let orbital_scale =
                    f64::from(maximum_n).powi(2) / self.nuclear_charges[site_index].max(1.0);
                let outer_radius = (4.0 * site.radius.get()).max(40.0 * orbital_scale);
                extend_mesh(&site.up.mesh, outer_radius)
            })
            .collect()
    }

    fn site_index(&self, site: &str) -> Result<usize, SnapshotDftError> {
        self.sites
            .iter()
            .position(|candidate| candidate.id == site)
            .ok_or_else(|| SnapshotDftError::UnknownCoreSite(site.to_owned()))
    }

    fn refine_spectral_basis(
        &self,
        requested: &ScfBasis,
        one_particle: &SnapshotOneParticle,
        bands: &SnapshotBandSolution,
        occupations: &[f64],
        chemical_potential: Hartree,
        relativity: ScfRelativity,
    ) -> Result<Option<SnapshotOneParticle>, SnapshotDftError> {
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
                .ok_or_else(|| SnapshotDftError::MissingProvisionalChannel {
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
        Ok(changed.then(|| SnapshotOneParticle {
            potential: one_particle.potential.clone(),
            basis,
        }))
    }

    fn validate_band_cog_projection_keys(
        &self,
        spectral: &[&ScfChannelRecipe],
        relativity: ScfRelativity,
    ) -> Result<(), SnapshotDftError> {
        let band_cog = spectral
            .iter()
            .copied()
            .filter(|recipe| recipe.generator == LinearizationEnergyGenerator::BandCog)
            .collect::<Vec<_>>();
        for (index, recipe) in band_cog.iter().enumerate() {
            let duplicate = band_cog[..index].iter().any(|prior| {
                prior.site == recipe.site
                    && match relativity {
                        ScfRelativity::SpinorFirstVariation => prior.identity == recipe.identity,
                        ScfRelativity::Scalar | ScfRelativity::SocSecondVariation { .. } => {
                            channel_l(prior.identity) == channel_l(recipe.identity)
                        }
                    }
            });
            if duplicate {
                return Err(SnapshotDftError::AmbiguousBandCogProjection {
                    site: recipe.site.clone(),
                    identity: recipe.identity,
                });
            }
        }
        Ok(())
    }

    fn band_cog_samples(
        &self,
        bands: &SnapshotBandSolution,
        occupations: &[f64],
        recipe: &ScfChannelRecipe,
        relativity: ScfRelativity,
    ) -> Result<Vec<PdosEnergySample>, SnapshotDftError> {
        if occupations.len() != bands.states.len() {
            return Err(SnapshotDftError::OccupationCount {
                expected: bands.states.len(),
                actual: occupations.len(),
            });
        }
        let site = self.site_index(&recipe.site)?;
        let l = channel_l(recipe.identity);
        let mut samples = Vec::new();
        for point in &bands.points {
            match &point.solution {
                SnapshotKPointSolution::Collinear {
                    bases,
                    solutions,
                    up_occupations,
                    down_occupations,
                } => {
                    if matches!(recipe.identity, ScfChannelIdentity::Kappa { .. }) {
                        return Err(SnapshotDftError::KappaBandCogUnavailableInScalar {
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
                            return Err(SnapshotDftError::InconsistentRelativityRoute);
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
                SnapshotKPointSolution::Spinor {
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
    ) -> Result<PlaneWaveEnvelope, SnapshotDftError> {
        if fractional_k.iter().any(|value| !value.is_finite()) {
            return Err(SnapshotDftError::NonFiniteKPoint(fractional_k));
        }
        let k = fractional_to_reciprocal(fractional_k, self.reciprocal.basis());
        let k_norm = squared_norm(k.map(InverseBohr::get)).sqrt();
        let candidates = self.reciprocal.enumerate(InverseBohr(cutoff + k_norm))?;
        let waves = candidates
            .into_iter()
            .filter(|g| {
                let wave = std::array::from_fn(|axis| k[axis].get() + g.cartesian[axis].get());
                squared_norm(wave) <= cutoff * cutoff * (1.0 + 64.0 * f64::EPSILON)
            })
            .map(|g| PlaneWave::new(k, g))
            .collect::<Vec<_>>();
        if waves.is_empty() {
            return Err(SnapshotDftError::EmptyPlaneWaveBasis {
                k: fractional_k,
                cutoff,
            });
        }
        Ok(PlaneWaveEnvelope::new(waves))
    }

    fn require_potential_site_count(
        &self,
        potential: &RegionalPotential,
    ) -> Result<(), SnapshotDftError> {
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
                return Err(SnapshotDftError::PotentialComponentSiteCount {
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
    ) -> Result<(), SnapshotDftError> {
        let transverse_rms = [
            potential.magnetic()[0].residual_rms()?,
            potential.magnetic()[1].residual_rms()?,
        ];
        if transverse_rms
            .iter()
            .any(|&rms| rms > TRANSVERSE_FIELD_TOLERANCE)
        {
            return Err(SnapshotDftError::TransversePotentialUnsupported {
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
    ) -> Result<(), SnapshotDftError> {
        self.require_collinear_route(potential)?;
        let magnetic = potential
            .magnetic()
            .iter()
            .map(RegionalScalarField::residual_rms)
            .collect::<Result<Vec<_>, _>>()?;
        if self.sites.iter().any(|site| !site.nonmagnetic_scalar)
            || magnetic.iter().any(|&rms| rms > TRANSVERSE_FIELD_TOLERANCE)
        {
            return Err(SnapshotDftError::SecondVariationRequiresNonmagneticScalar);
        }
        Ok(())
    }

    fn synthesize(
        &self,
        bands: &SnapshotBandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, SnapshotDftError> {
        if occupations.len() != bands.states.len() {
            return Err(SnapshotDftError::OccupationCount {
                expected: bands.states.len(),
                actual: occupations.len(),
            });
        }
        let density_layout = self.density_layout(&bands.points)?;
        match &bands.points[0].solution {
            SnapshotKPointSolution::Collinear { bases, .. } => {
                let up_points = bands
                    .points
                    .iter()
                    .map(|point| match &point.solution {
                        SnapshotKPointSolution::Collinear {
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
                        SnapshotKPointSolution::Spinor { .. } => {
                            Err(SnapshotDftError::InconsistentRelativityRoute)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let down_points = bands
                    .points
                    .iter()
                    .map(|point| match &point.solution {
                        SnapshotKPointSolution::Collinear {
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
                        SnapshotKPointSolution::Spinor { .. } => {
                            Err(SnapshotDftError::InconsistentRelativityRoute)
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
            SnapshotKPointSolution::Spinor { basis, .. } => {
                let spinor_points = bands
                    .points
                    .iter()
                    .map(|point| match &point.solution {
                        SnapshotKPointSolution::Spinor {
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
                        SnapshotKPointSolution::Collinear { .. } => {
                            Err(SnapshotDftError::InconsistentRelativityRoute)
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
    ) -> Result<CoreContribution, SnapshotDftError> {
        let site_index = self
            .sites
            .iter()
            .position(|candidate| candidate.id == site.id)
            .ok_or_else(|| SnapshotDftError::UnknownCoreSite(site.id.clone()))?;
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
    ) -> Result<muffintin_radial::CoreDiracSolution, SnapshotDftError> {
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
    ) -> Result<Vec<ExponentialMesh>, SnapshotDftError> {
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

    fn density_layout(&self, points: &[SnapshotKPoint]) -> Result<FourierLayout, SnapshotDftError> {
        let mut indices = BTreeSet::new();
        for point in points {
            let compiled = match &point.solution {
                SnapshotKPointSolution::Collinear { bases, .. } => {
                    vec![
                        &bases.up.compiled.plane_waves,
                        &bases.down.compiled.plane_waves,
                    ]
                }
                SnapshotKPointSolution::Spinor { basis, .. } => {
                    vec![&basis.compiled.plane_waves]
                }
            };
            for plane_waves in compiled {
                for left in plane_waves {
                    for right in plane_waves {
                        indices.insert([
                            right.g.index[0] - left.g.index[0],
                            right.g.index[1] - left.g.index[1],
                            right.g.index[2] - left.g.index[2],
                        ]);
                    }
                }
            }
        }
        let vectors = indices
            .into_iter()
            .map(|index| g_vector(self.reciprocal, index))
            .collect();
        Ok(FourierLayout::new(self.reciprocal, vectors)?)
    }
}

impl ScfPhysics for SnapshotDftPhysics {
    type Error = SnapshotDftError;
    type OneParticle = SnapshotOneParticle;
    type BandSolution = SnapshotBandSolution;

    fn initial_density(&mut self, config: &ScfConfig) -> Result<RegionalDensity, Self::Error> {
        if let Some(density) = &self.restart_density {
            self.density_template = Some(density.clone());
            return Ok(density.clone());
        }
        let meshes = self.channel_meshes(&config.basis)?;
        let initial_extended = build_extended_snapshot_core_potentials(
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
        let mut one_particle = SnapshotOneParticle {
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
                        return Err(SnapshotDftError::InitialBasisRefinementNotConverged {
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
                    .ok_or_else(|| SnapshotDftError::UnknownCoreSite(site.id.clone()))?;
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
        let electrostatic = evaluate_regional_electrostatics(
            density.charge(),
            &ElectrostaticSpec::new(
                muffintin_coulomb::WeinertHartreeSpec::electronic(4)?,
                self.nuclear_charges.clone(),
            )?,
        )?;
        let output_l_max = std::iter::once(density.charge())
            .chain(density.magnetization())
            .flat_map(RegionalScalarField::muffin_tins)
            .flat_map(|field| field.field().channels().map(|(channel, _)| channel.l))
            .max()
            .unwrap_or(0);
        let xc_functional = exchange_correlation.functional;
        let xc_field_spec = xc_spec(
            density,
            output_l_max,
            exchange_correlation.noncollinear_route,
        );
        let xc = evaluate_regional_xc(xc_functional, density, xc_field_spec)?;
        let mut scalar = electrostatic.potential.clone();
        scalar.add_scaled(1.0, xc.potential.scalar())?;
        let potential = RegionalPotential::new(scalar, xc.potential.magnetic().clone())?;
        self.core_potentials.insert(
            iteration,
            CorePotentialContext {
                electrostatic: electrostatic.clone(),
                exchange_correlation: xc.clone(),
                density: density.clone(),
                spec: CorePotentialBuildSpec {
                    continuation: CorePotentialContinuationSpec::default(),
                    xc_functional,
                    xc_noncollinear_route: exchange_correlation.noncollinear_route,
                    xc_angular_point_count: xc_field_spec.angular_point_count,
                },
            },
        );
        self.energy_terms.insert(
            iteration,
            ScfEnergyTerms {
                madelung: electrostatic.madelung,
                coulomb: electrostatic.coulomb,
                exchange_correlation: xc.exchange_correlation_energy,
                exchange_correlation_potential: xc.density_potential_integral,
            },
        );
        Ok(potential)
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
            .ok_or(SnapshotDftError::MissingDensityTemplate)?
            .clone();
        let context = self
            .core_potentials
            .get(&iteration)
            .ok_or(SnapshotDftError::MissingCoreContinuation(iteration))?
            .clone();
        let site_index = self
            .sites
            .iter()
            .position(|candidate| candidate.id == site.id)
            .ok_or_else(|| SnapshotDftError::UnknownCoreSite(site.id.clone()))?;
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
        Ok(SnapshotOneParticle {
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
            .ok_or(SnapshotDftError::MissingEnergyTerms(context.iteration))
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
                    return Err(SnapshotDftError::TooFewBands {
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
            .ok_or(SnapshotDftError::EmptyKPointSet)?;
        if solved
            .points
            .iter()
            .any(|point| point.energies.len() != band_count)
        {
            return Err(SnapshotDftError::InconsistentBandCount);
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

/// Build a validated V2 restart snapshot from a converged SCF state.
///
/// `template` supplies the immutable cell, sites, radial equations, and
/// linearization metadata. The state supplies the complete noncollinear
/// density and potential without reducing their Cartesian Pauli components.
pub fn snapshot_v2_from_state(
    template: &SnapshotV2,
    state: &ScfState,
) -> Result<SnapshotV2, SnapshotDftError> {
    template.validate()?;
    let template_potential = match &template.initial {
        InitialV2::FrozenPotential { potential } | InitialV2::Restart { potential, .. } => {
            potential
        }
    };
    let mut potential_hints = template_potential.basis_hints;
    potential_hints.plane_wave_cutoff = Some(state.basis.plane_wave_cutoff);
    let mut density_hints = match &template.initial {
        InitialV2::Restart { density, .. } => density.basis_hints,
        InitialV2::FrozenPotential { .. } => template_potential.basis_hints,
    };
    density_hints.plane_wave_cutoff = Some(state.basis.plane_wave_cutoff);
    let angular_basis = template.meta.potential_convention.angular_basis;
    let density = DensityV2 {
        unit: FieldUnitV2::BohrMinus3,
        representation: FieldRepresentationV2::PeriodicExtension,
        angular_basis,
        basis_hints: density_hints,
        n: regional_scalar_to_v2(state.density.charge(), &template.geometry.sites)?,
        mx: regional_scalar_to_v2(&state.density.magnetization()[0], &template.geometry.sites)?,
        my: regional_scalar_to_v2(&state.density.magnetization()[1], &template.geometry.sites)?,
        mz: regional_scalar_to_v2(&state.density.magnetization()[2], &template.geometry.sites)?,
    };
    let potential = PotentialV2 {
        unit: FieldUnitV2::Hartree,
        representation: FieldRepresentationV2::MaskedOperator,
        angular_basis,
        basis_hints: potential_hints,
        v0: regional_scalar_to_v2(state.potential.scalar(), &template.geometry.sites)?,
        bx: regional_scalar_to_v2(&state.potential.magnetic()[0], &template.geometry.sites)?,
        by: regional_scalar_to_v2(&state.potential.magnetic()[1], &template.geometry.sites)?,
        bz: regional_scalar_to_v2(&state.potential.magnetic()[2], &template.geometry.sites)?,
    };
    let snapshot = SnapshotV2::new(
        template.meta.clone(),
        template.geometry.clone(),
        InitialV2::Restart { density, potential },
    );
    snapshot.validate()?;
    Ok(snapshot)
}

fn convert_v2_site_bases(
    site_id: &str,
    bases: &[muffintin_io::SiteRadialBasisV2],
) -> Result<(SnapshotSpin, SnapshotSpin, bool), SnapshotDftError> {
    let scalar = bases
        .iter()
        .find(|basis| basis.site_id == site_id && basis.spin == RadialBasisSpinV2::Scalar);
    let up = bases
        .iter()
        .find(|basis| basis.site_id == site_id && basis.spin == RadialBasisSpinV2::Up);
    let down = bases
        .iter()
        .find(|basis| basis.site_id == site_id && basis.spin == RadialBasisSpinV2::Down);
    match (scalar, up, down) {
        (Some(scalar), None, None) => {
            let converted = convert_v2_radial_basis(scalar)?;
            Ok((converted.clone(), converted, true))
        }
        (None, Some(up), Some(down)) => Ok((
            convert_v2_radial_basis(up)?,
            convert_v2_radial_basis(down)?,
            false,
        )),
        _ => Err(SnapshotDftError::InvalidRadialBasisSpins {
            site: site_id.to_owned(),
        }),
    }
}

fn convert_v2_radial_basis(
    basis: &muffintin_io::SiteRadialBasisV2,
) -> Result<SnapshotSpin, SnapshotDftError> {
    let mesh = ExponentialMesh::new(
        Bohr(basis.mesh.first),
        basis.mesh.log_increment,
        basis.mesh.point_count,
    )?;
    Ok(SnapshotSpin {
        equation: basis.radial_equation,
        mesh,
        linearization: basis
            .linearization
            .linearization_energies
            .iter()
            .map(|parameter| (parameter.l, Hartree(parameter.energy)))
            .collect(),
        local_orbitals: basis
            .linearization
            .local_orbital_energies
            .iter()
            .map(|parameter| (parameter.l, Hartree(parameter.energy)))
            .collect(),
    })
}

fn regional_potential_from_v2(
    potential: &PotentialV2,
    geometry: &InterstitialGeometry,
    sites: &[SnapshotSite],
    reciprocal: ReciprocalLattice,
) -> Result<RegionalPotential, SnapshotDftError> {
    let scalar = regional_scalar_from_v2(
        &potential.v0,
        potential.angular_basis,
        geometry,
        sites,
        reciprocal,
    )?;
    let magnetic = [&potential.bx, &potential.by, &potential.bz]
        .map(|field| {
            regional_scalar_from_v2(field, potential.angular_basis, geometry, sites, reciprocal)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("three V2 magnetic components remain three components");
    Ok(RegionalPotential::new(scalar, magnetic)?)
}

fn regional_density_from_v2(
    density: &DensityV2,
    geometry: &InterstitialGeometry,
    sites: &[SnapshotSite],
    reciprocal: ReciprocalLattice,
) -> Result<RegionalDensity, SnapshotDftError> {
    let charge = regional_scalar_from_v2(
        &density.n,
        density.angular_basis,
        geometry,
        sites,
        reciprocal,
    )?;
    let magnetization = [&density.mx, &density.my, &density.mz]
        .map(|field| {
            regional_scalar_from_v2(field, density.angular_basis, geometry, sites, reciprocal)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("three V2 magnetization components remain three components");
    Ok(RegionalDensity::new(charge, magnetization)?)
}

fn regional_scalar_from_v2(
    field: &RegionalFieldV2,
    angular_basis: AngularBasisV1,
    geometry: &InterstitialGeometry,
    sites: &[SnapshotSite],
    reciprocal: ReciprocalLattice,
) -> Result<RegionalScalarField, SnapshotDftError> {
    let convention = match angular_basis {
        AngularBasisV1::ComplexCondonShortley => HarmonicConvention::Complex,
        AngularBasisV1::RealTesseralCondonShortley => HarmonicConvention::Real,
    };
    let by_site = field
        .muffin_tins
        .iter()
        .map(|field| (field.site_id.as_str(), field))
        .collect::<BTreeMap<_, _>>();
    let muffin_tins = sites
        .iter()
        .map(|site| {
            let source = by_site
                .get(site.id.as_str())
                .ok_or_else(|| SnapshotDftError::MissingV2FieldSite(site.id.clone()))?;
            let channels = source.channels.iter().map(|channel| {
                let scale = if (channel.l, channel.m) == (0, 0) {
                    (4.0 * PI).sqrt()
                } else {
                    1.0
                };
                let values = channel
                    .real
                    .iter()
                    .enumerate()
                    .map(|(index, &real)| {
                        Complex64::new(
                            scale * real,
                            scale * channel.imaginary.get(index).copied().unwrap_or(0.0),
                        )
                    })
                    .collect();
                ((channel.l, channel.m), values)
            });
            Ok(MuffinTinField::new(
                site.up.mesh.clone(),
                SphereField::new(convention, channels)?,
            )?)
        })
        .collect::<Result<Vec<_>, SnapshotDftError>>()?;
    let mut coefficients = field.interstitial.coefficients.clone();
    coefficients.sort_by_key(|coefficient| coefficient.g);
    let vectors = coefficients
        .iter()
        .map(|coefficient| g_vector(reciprocal, coefficient.g))
        .collect();
    let layout = FourierLayout::new(reciprocal, vectors)?;
    if layout.index([0; 3]).is_none() {
        return Err(SnapshotDftError::MissingInterstitialZero);
    }
    let values = coefficients
        .iter()
        .map(|coefficient| Complex64::new(coefficient.value.real, coefficient.value.imaginary))
        .collect();
    let interstitial =
        InterstitialField::from_fourier_field(HermitianFourierField::new(layout, values)?);
    Ok(RegionalScalarField::new(
        geometry.clone(),
        muffin_tins,
        interstitial,
    )?)
}

fn regional_scalar_to_v2(
    field: &RegionalScalarField,
    sites: &[muffintin_io::SiteV2],
) -> Result<RegionalFieldV2, SnapshotDftError> {
    if field.muffin_tins().len() != sites.len() {
        return Err(SnapshotDftError::ExportSiteCount {
            expected: sites.len(),
            actual: field.muffin_tins().len(),
        });
    }
    let muffin_tins = sites
        .iter()
        .zip(field.muffin_tins())
        .map(|(site, field)| MuffinTinFieldV2 {
            site_id: site.id.clone(),
            channels: field
                .field()
                .channels()
                .map(|(channel, values)| {
                    let scale = if (channel.l, channel.m) == (0, 0) {
                        1.0 / (4.0 * PI).sqrt()
                    } else {
                        1.0
                    };
                    SphericalChannelV2 {
                        l: channel.l,
                        m: channel.m,
                        real: values.iter().map(|value| scale * value.re).collect(),
                        imaginary: values.iter().map(|value| scale * value.im).collect(),
                    }
                })
                .collect(),
        })
        .collect();
    let coefficients = field
        .interstitial()
        .field()
        .iter()
        .map(|(vector, &value)| FourierCoefficientV2 {
            g: vector.index,
            value: Complex64V2 {
                real: value.re,
                imaginary: value.im,
            },
        })
        .collect();
    Ok(RegionalFieldV2 {
        muffin_tins,
        interstitial: InterstitialFieldV2 { coefficients },
    })
}

fn second_variation_blocks(
    basis: &ScalarIterationBasis,
    inputs: &[ScalarSiteInput],
) -> Result<Vec<SiteSpinOrbitBlock>, SnapshotDftError> {
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
) -> Result<Collinear<GeneralizedEigensolution>, SnapshotDftError> {
    let rows = solution.eigenvectors.rows() / 2;
    let columns = solution.eigenvectors.columns();
    let split = |spin: usize| -> Result<GeneralizedEigensolution, SnapshotDftError> {
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
) -> Result<MuffinTinField, SnapshotDftError> {
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
) -> Result<InterstitialField, SnapshotDftError> {
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
) -> Result<RegionalScalarField, SnapshotDftError> {
    let mut result = left.zero_like();
    result.add_scaled(left_scale, left)?;
    result.add_scaled(right_scale, right)?;
    Ok(result)
}

fn collinear_interstitial_potential(
    potential: &RegionalPotential,
) -> Result<Collinear<InterstitialPotential>, SnapshotDftError> {
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
) -> Result<ExponentialMesh, SnapshotDftError> {
    let extra = (target_radius / muffin_tin.last().get()).ln().max(0.0) / muffin_tin.increment();
    let count = muffin_tin
        .len()
        .checked_add(extra.ceil() as usize)
        .and_then(|count| count.checked_add(1))
        .ok_or(SnapshotDftError::CoreMeshCountOverflow)?;
    Ok(ExponentialMesh::new(
        muffin_tin.first(),
        muffin_tin.increment(),
        count,
    )?)
}

fn spinor_kappas_for_l(l: u32) -> Result<Vec<Kappa>, SnapshotDftError> {
    let l = i32::try_from(l).map_err(|_| SnapshotDftError::AngularMomentumOverflow)?;
    let negative = l
        .checked_add(1)
        .and_then(i32::checked_neg)
        .ok_or(SnapshotDftError::AngularMomentumOverflow)?;
    let mut kappas = vec![Kappa::new(negative)?];
    if l != 0 {
        kappas.push(Kappa::new(l)?);
    }
    Ok(kappas)
}

fn channel_l(identity: ScfChannelIdentity) -> u32 {
    match identity {
        ScfChannelIdentity::ScalarL { l, .. } => l,
        ScfChannelIdentity::Kappa { kappa, .. } if kappa > 0 => kappa as u32,
        ScfChannelIdentity::Kappa { kappa, .. } => (-kappa - 1) as u32,
    }
}

fn channel_n(identity: ScfChannelIdentity) -> u32 {
    match identity {
        ScfChannelIdentity::ScalarL { n, .. } | ScfChannelIdentity::Kappa { n, .. } => n,
    }
}

fn channel_kappas(identity: ScfChannelIdentity) -> Result<Vec<Kappa>, SnapshotDftError> {
    match identity {
        ScfChannelIdentity::ScalarL { l, .. } => spinor_kappas_for_l(l),
        ScfChannelIdentity::Kappa { kappa, .. } => Ok(vec![Kappa::new(kappa)?]),
    }
}

fn scalar_component_energy(resolved: &ScfResolvedChannelEnergy, kappa: Kappa) -> Hartree {
    if let Some(energy) =
        resolved
            .components
            .iter()
            .find_map(|component| match component.diagnostic {
                LinearizationEnergyDiagnostic::Atomic { state, .. } if state.kappa == kappa => {
                    Some(component.energy)
                }
                _ => None,
            })
    {
        return energy;
    }
    if resolved.recipe.generator == LinearizationEnergyGenerator::BandCog
        && let ScfChannelIdentity::ScalarL { l, .. } = resolved.recipe.identity
        && let Ok(kappas) = spinor_kappas_for_l(l)
        && kappas.len() == resolved.components.len()
        && let Some(index) = kappas.iter().position(|candidate| *candidate == kappa)
    {
        return resolved.components[index].energy;
    }
    resolved.energy
}

fn spin_resolved_energy(resolved: &ScfResolvedChannelEnergy, spin: usize) -> Hartree {
    if resolved.recipe.generator == LinearizationEnergyGenerator::FrozenSnapshot
        && resolved.components.len() == 2
    {
        resolved.components[spin].energy
    } else {
        resolved.energy
    }
}

fn channel_generator_error(
    recipe: &ScfChannelRecipe,
    source: LinearizationEnergyError,
) -> SnapshotDftError {
    SnapshotDftError::ChannelGenerator {
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
) -> Result<Vec<f64>, SnapshotDftError> {
    let monopole = potential.scalar().muffin_tins()[site_index]
        .field()
        .channel(0, 0)
        .ok_or_else(|| SnapshotDftError::MissingMonopole(site.to_owned()))?;
    monopole
        .iter()
        .enumerate()
        .map(|(radial, value)| {
            if value.im.abs() > TRANSVERSE_FIELD_TOLERANCE * (1.0 + value.re.abs()) {
                return Err(SnapshotDftError::NonRealMonopole {
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
) -> Result<Vec<f64>, SnapshotDftError> {
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
) -> Result<(), SnapshotDftError> {
    if range.len() != projections.len() || range.end > states.len() || range.end > occupations.len()
    {
        return Err(SnapshotDftError::BandProjectionCount {
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
) -> Result<InitialOccupationSolution, SnapshotDftError> {
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

fn regular_k_points(mesh: ScfKMesh) -> Result<Vec<[f64; 3]>, SnapshotDftError> {
    let count = mesh
        .divisions
        .into_iter()
        .try_fold(1_usize, usize::checked_mul)
        .ok_or(SnapshotDftError::KPointCountOverflow)?;
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

fn xc_spec(
    density: &RegionalDensity,
    output_l_max: u32,
    noncollinear_route: NoncollinearXcRoute,
) -> XcFieldSpec {
    let layout = density.charge().interstitial().layout();
    let divisions = std::array::from_fn(|axis| {
        let maximum = layout
            .vectors()
            .iter()
            .map(|vector| vector.index[axis].unsigned_abs() as usize)
            .max()
            .unwrap_or(0);
        (2 * maximum + 1).max(4)
    });
    let angular_point_count = ((output_l_max as usize + 1).pow(2) * 2).max(50);
    XcFieldSpec {
        interstitial_divisions: divisions,
        angular_point_count,
        output_l_max,
        noncollinear_route,
    }
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

fn g_vector(reciprocal: ReciprocalLattice, index: [i32; 3]) -> GVector {
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

/// Snapshot conversion or concrete DFT-kernel failure.
#[derive(Debug, Error)]
pub enum SnapshotDftError {
    #[error(transparent)]
    Snapshot(#[from] IoError),
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
    #[error("cannot export {actual} muffin-tin fields against {expected} snapshot sites")]
    ExportSiteCount { expected: usize, actual: usize },
    #[error("site {site:?} has different up/down radial meshes")]
    SpinMeshMismatch { site: String },
    #[error("site {site:?} muffin-tin radius is {declared}, radial mesh ends at {mesh}")]
    MuffinTinMeshRadius {
        site: String,
        declared: f64,
        mesh: f64,
    },
    #[error("snapshot interstitial potential must contain G=0")]
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
        equation: RadialEquationTagV1,
    },
    #[error(
        "full-spinor route needs fully relativistic input at site {site:?}, spin {spin}; got {equation:?}"
    )]
    SpinorRadialEquation {
        site: String,
        spin: usize,
        equation: RadialEquationTagV1,
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
        "site {site:?} channel {identity:?} ({treatment:?}) has no matching frozen-snapshot anchor"
    )]
    MissingFrozenSnapshotAnchor {
        site: String,
        identity: ScfChannelIdentity,
        treatment: ScfChannelTreatment,
    },
    #[error("site {site:?}, spin {spin} has no frozen-snapshot base anchor for l={l}")]
    MissingFrozenSnapshotBase { site: String, l: u32, spin: usize },
    #[error(
        "site {site:?}, spin {spin} has no frozen-snapshot LO anchor for l={l}, ordinal={ordinal}"
    )]
    MissingFrozenSnapshotLo {
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
    #[error("site {site:?}, spin {spin} has no linearization energy for l={l}")]
    MissingLinearizationEnergy { site: String, spin: usize, l: u32 },
    #[error(
        "bound relativistic local-orbital energy is unavailable for site {site:?}, n={principal_quantum_number}, kappa={kappa}"
    )]
    MissingRelativisticLocalOrbitalEnergy {
        site: String,
        principal_quantum_number: u32,
        kappa: i32,
    },
    #[error("k point {0:?} contains a non-finite coordinate")]
    NonFiniteKPoint([f64; 3]),
    #[error("plane-wave cutoff {cutoff} produces no basis at k={k:?}")]
    EmptyPlaneWaveBasis { k: [f64; 3], cutoff: f64 },
    #[error("regular k-point set is empty")]
    EmptyKPointSet,
    #[error("regular k-point count overflows usize")]
    KPointCountOverflow,
    #[error("second variation requires a nonmagnetic scalar snapshot and potential")]
    SecondVariationRequiresNonmagneticScalar,
    #[error(
        "second-variation window starts at {start}; runtime requires start=0 so occupied lower scalar bands are not dropped"
    )]
    SecondVariationDropsLowerBands { start: usize },
    #[error("site {site:?} has incompatible up/down l-resolved spinor local-orbital channels")]
    SpinorLocalOrbitalChannels { site: String },
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
    #[error("core site {0:?} is not present in the snapshot")]
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
mod tests {
    use super::*;
    use muffintin_dft::{
        BandPathPoint, FirstVariationWindow, ScfConvergence, ScfCoreSite, ScfCoreState, ScfMixing,
        XcFunctional, run_scf,
    };
    use muffintin_io::{
        BasisHintsV1, Complex64V1, EnergyParameterV1, EnergyUnitV1, ExponentialMeshSpecV1,
        FourierCoefficientV1, FourierNormalizationV1, FourierPhaseV1, GeometryV1, InterstitialV1,
        InverseLengthUnitV1, LatticeV1, LengthUnitV1, LinearizationV1, MetaV1, PotentialChannelV1,
        PotentialConventionV1, PotentialRadialQuantityV1, SiteSpinV1, SiteV1, SnapshotFile,
        SnapshotV1, SphericalChannelConventionV1, SpinTagV1, snapshot_file_from_toml,
        snapshot_file_to_toml,
    };

    fn snapshot_v1() -> muffintin_io::SnapshotV1 {
        let point_count = 61;
        let first: f64 = 1.0e-4;
        let radius: f64 = 1.0;
        let increment = (radius / first).ln() / (point_count - 1) as f64;
        let radii = (0..point_count)
            .map(|index| first * (index as f64 * increment).exp())
            .collect::<Vec<_>>();
        SnapshotV1::new(
            MetaV1 {
                title: "snapshot kernel hydrogen smoke".to_owned(),
                producer: "mt-runtime test".to_owned(),
                producer_version: None,
                energy_zero: "zero interstitial Fourier mean".to_owned(),
                potential_convention: PotentialConventionV1 {
                    angular_basis: AngularBasisV1::ComplexCondonShortley,
                    radial_quantity: PotentialRadialQuantityV1::Potential,
                    spherical_channel: SphericalChannelConventionV1::PhysicalValue,
                },
                annotations: BTreeMap::new(),
            },
            GeometryV1 {
                lattice: LatticeV1 {
                    unit: LengthUnitV1::Bohr,
                    vectors: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
                },
                sites: vec![SiteV1 {
                    id: "H-1".to_owned(),
                    atomic_number: 1,
                    fractional_position: [1.25, -0.5, 0.5],
                    muffin_tin_radius_unit: LengthUnitV1::Bohr,
                    muffin_tin_radius: radius,
                    spins: vec![SiteSpinV1 {
                        spin: SpinTagV1::Scalar,
                        mesh: ExponentialMeshSpecV1 {
                            radius_unit: LengthUnitV1::Bohr,
                            first,
                            log_increment: increment,
                            point_count,
                            last: first * ((point_count - 1) as f64 * increment).exp(),
                            consistency_tolerance: 1.0e-12,
                        },
                        radial_equation: RadialEquationTagV1::ScalarKoellingHarmon,
                        potential_unit: EnergyUnitV1::Hartree,
                        potential_channels: vec![PotentialChannelV1 {
                            l: 0,
                            m: 0,
                            real: radii.iter().map(|radius| -1.0 / radius).collect(),
                            imaginary: Vec::new(),
                        }],
                        linearization: LinearizationV1 {
                            energy_unit: EnergyUnitV1::Hartree,
                            linearization_energies: vec![
                                EnergyParameterV1 { l: 0, energy: -0.3 },
                                EnergyParameterV1 {
                                    l: 1,
                                    energy: -0.15,
                                },
                            ],
                            local_orbital_energies: Vec::new(),
                        },
                    }],
                }],
            },
            InterstitialV1 {
                coefficient_unit: EnergyUnitV1::Hartree,
                coefficients: vec![FourierCoefficientV1 {
                    g: [0; 3],
                    value: Complex64V1 {
                        real: 0.0,
                        imaginary: 0.0,
                    },
                }],
                basis_hints: BasisHintsV1 {
                    reciprocal_length_unit: InverseLengthUnitV1::BohrInverse,
                    plane_wave_cutoff: Some(0.5),
                    coefficient_cutoff: Some(1.0),
                    normalization: FourierNormalizationV1::CellNormalized,
                    phase: FourierPhaseV1::NegativeExponent,
                },
            },
        )
    }

    fn snapshot() -> SnapshotV2 {
        snapshot_v1().normalize_v2().unwrap()
    }

    fn config(relativity: ScfRelativity) -> ScfConfig {
        ScfConfig {
            electron_count: 1.0,
            k_mesh: ScfKMesh {
                divisions: [1, 1, 1],
                shift: [0.0; 3],
            },
            basis: ScfBasis {
                plane_wave_cutoff: 0.5,
                l_max: 1,
                channels: vec![
                    ScfChannelRecipe {
                        site: "H-1".to_owned(),
                        identity: ScfChannelIdentity::ScalarL { n: 1, l: 0 },
                        treatment: ScfChannelTreatment::Valence,
                        derivative_order: 0,
                        generator: LinearizationEnergyGenerator::FrozenSnapshot,
                        seed: None,
                        provenance: muffintin_dft::ScfChannelProvenance::BuiltIn,
                    },
                    ScfChannelRecipe {
                        site: "H-1".to_owned(),
                        identity: ScfChannelIdentity::ScalarL { n: 2, l: 1 },
                        treatment: ScfChannelTreatment::Valence,
                        derivative_order: 0,
                        generator: LinearizationEnergyGenerator::FrozenSnapshot,
                        seed: None,
                        provenance: muffintin_dft::ScfChannelProvenance::BuiltIn,
                    },
                ],
                resolved_channels: Vec::new(),
            },
            occupations: ScfOccupations::FermiDirac {
                temperature: Hartree(0.02),
            },
            exchange_correlation: ScfExchangeCorrelation {
                functional: XcFunctional::LdaPw92,
                noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
            },
            mixing: ScfMixing::Linear { alpha: 1.0 },
            relativity,
            convergence: ScfConvergence {
                energy_tolerance: Hartree(1.0e100),
                density_tolerance: 1.0e100,
                max_iterations: 2,
            },
            core_sites: vec![ScfCoreSite {
                id: "H-1".to_owned(),
                states: Vec::new(),
            }],
        }
    }

    fn core_snapshot_and_config() -> (SnapshotV2, ScfConfig) {
        let mut snapshot = snapshot_v1();
        let first: f64 = 1.0e-5;
        let radius: f64 = 3.0;
        let point_count = 121;
        let increment = (radius / first).ln() / (point_count - 1) as f64;
        snapshot.geometry.lattice.vectors = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]];
        snapshot.geometry.sites[0].atomic_number = 2;
        snapshot.geometry.sites[0].muffin_tin_radius = radius;
        let spin = &mut snapshot.geometry.sites[0].spins[0];
        spin.mesh = ExponentialMeshSpecV1 {
            radius_unit: LengthUnitV1::Bohr,
            first,
            log_increment: increment,
            point_count,
            last: radius,
            consistency_tolerance: 1.0e-12,
        };
        let mesh = ExponentialMesh::new(Bohr(first), increment, point_count).unwrap();
        spin.potential_channels[0].real = mesh
            .radii()
            .iter()
            .map(|radius| -2.0 / radius.get())
            .collect();
        spin.linearization.linearization_energies[0].energy = -0.8;
        spin.linearization.linearization_energies[1].energy = -0.3;
        let mut config = config(ScfRelativity::Scalar);
        config.electron_count = 2.0;
        config.core_sites[0].states.push(ScfCoreState {
            principal_quantum_number: 1,
            kappa: -1,
            occupation: 1.0,
        });
        (snapshot.normalize_v2().unwrap(), config)
    }

    #[test]
    fn snapshot_conversion_normalizes_monopole_and_wraps_cartesian_site() {
        let physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        assert_eq!(
            physics.geometry.spheres()[0].center,
            [Bohr(2.0), Bohr(4.0), Bohr(4.0)]
        );
        let physical = snapshot_v1().geometry.sites[0].spins[0].potential_channels[0].real[17];
        let normalized = physics.frozen_potential.scalar().muffin_tins()[0]
            .field()
            .channel(0, 0)
            .unwrap()[17]
            .re;
        assert!((normalized - (4.0 * PI).sqrt() * physical).abs() < 1.0e-12);
    }

    #[test]
    fn v2_interstitial_components_are_keyed_independently_of_input_order() {
        fn coefficient(g: [i32; 3], real: f64, imaginary: f64) -> FourierCoefficientV2 {
            FourierCoefficientV2 {
                g,
                value: Complex64V2 { real, imaginary },
            }
        }

        let mut snapshot = snapshot();
        let InitialV2::FrozenPotential { potential } = &mut snapshot.initial else {
            unreachable!()
        };
        potential.v0.interstitial.coefficients = vec![
            coefficient([0, 0, 0], 0.0, 0.0),
            coefficient([1, 0, 0], 1.0, 2.0),
            coefficient([-1, 0, 0], 1.0, -2.0),
        ];
        potential.bx.interstitial.coefficients = vec![
            coefficient([1, 0, 0], 3.0, 4.0),
            coefficient([0, 0, 0], 0.5, 0.0),
            coefficient([-1, 0, 0], 3.0, -4.0),
        ];
        potential.by.interstitial.coefficients = vec![
            coefficient([-1, 0, 0], 5.0, -6.0),
            coefficient([1, 0, 0], 5.0, 6.0),
            coefficient([0, 0, 0], 0.25, 0.0),
        ];
        potential.bz.interstitial.coefficients = vec![
            coefficient([0, 0, 0], -0.5, 0.0),
            coefficient([-1, 0, 0], 7.0, -8.0),
            coefficient([1, 0, 0], 7.0, 8.0),
        ];

        let physics = SnapshotDftPhysics::new(&snapshot).unwrap();
        let potential = physics.frozen_potential();
        for field in potential.magnetic() {
            assert_eq!(
                field.interstitial().layout(),
                potential.scalar().interstitial().layout()
            );
        }
        assert_eq!(
            potential.magnetic()[0]
                .interstitial()
                .field()
                .coefficient([1, 0, 0]),
            Some(Complex64::new(3.0, 4.0))
        );
        assert_eq!(
            potential.magnetic()[1]
                .interstitial()
                .field()
                .coefficient([-1, 0, 0]),
            Some(Complex64::new(5.0, -6.0))
        );
    }

    #[test]
    fn frozen_snapshot_produces_initial_density_without_fake_atomic_g_zero() {
        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let config = config(ScfRelativity::Scalar);
        let meshes = physics.channel_meshes(&config.basis).unwrap();
        let extended = build_extended_snapshot_core_potentials(
            &physics.frozen_potential,
            &physics.geometry,
            &physics.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )
        .unwrap();
        let materialized = physics
            .materialize_nonspectral_basis(&physics.frozen_potential, &config.basis, &extended)
            .unwrap();
        assert_eq!(
            materialized.resolved_channels[0].energy.get().to_bits(),
            (-0.3_f64).to_bits()
        );
        assert_eq!(
            materialized.resolved_channels[1].energy.get().to_bits(),
            (-0.15_f64).to_bits()
        );
        let density = physics.initial_density(&config).unwrap();
        assert!((muffintin_dft::electron_count(&density).unwrap() - 1.0).abs() < 1.0e-10);
        assert!(
            density
                .charge()
                .interstitial()
                .layout()
                .index([0; 3])
                .is_some()
        );
    }

    #[test]
    fn magnetic_frozen_snapshot_does_not_turn_spin_splitting_into_kappa_splitting() {
        let mut snapshot = snapshot_v1();
        let mut up = snapshot.geometry.sites[0].spins[0].clone();
        up.spin = SpinTagV1::Up;
        up.linearization.linearization_energies[1].energy = -0.14;
        let mut down = up.clone();
        down.spin = SpinTagV1::Down;
        down.linearization.linearization_energies[1].energy = -0.16;
        snapshot.geometry.sites[0].spins = vec![up, down];
        let physics = SnapshotDftPhysics::new(&snapshot.normalize_v2().unwrap()).unwrap();
        let basis = config(ScfRelativity::SpinorFirstVariation).basis;
        let meshes = physics.channel_meshes(&basis).unwrap();
        let extended = build_extended_snapshot_core_potentials(
            &physics.frozen_potential,
            &physics.geometry,
            &physics.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )
        .unwrap();
        let materialized = physics
            .materialize_nonspectral_basis(&physics.frozen_potential, &basis, &extended)
            .unwrap();
        assert_eq!(
            physics
                .scalar_linearization_energies(&materialized, "H-1", 0)
                .unwrap()[1]
                .get()
                .to_bits(),
            (-0.14_f64).to_bits()
        );
        assert_eq!(
            physics
                .scalar_linearization_energies(&materialized, "H-1", 1)
                .unwrap()[1]
                .get()
                .to_bits(),
            (-0.16_f64).to_bits()
        );
        let spinor = physics
            .spinor_linearization_energies(&materialized, "H-1")
            .unwrap();
        let p = spinor
            .iter()
            .filter(|parameter| parameter.kappa.large_l() == 1)
            .map(|parameter| parameter.energy)
            .collect::<Vec<_>>();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].get().to_bits(), p[1].get().to_bits());
        assert_eq!(p[0], Hartree(0.5 * (-0.14 - 0.16)));
    }

    #[test]
    fn atomic_recipe_materializes_from_the_current_extended_potential() {
        let physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let mut config = config(ScfRelativity::Scalar);
        for channel in &mut config.basis.channels {
            if channel_l(channel.identity) == 0 {
                channel.generator = LinearizationEnergyGenerator::Atomic;
            }
        }
        let meshes = physics.channel_meshes(&config.basis).unwrap();
        let extended = build_extended_snapshot_core_potentials(
            &physics.frozen_potential,
            &physics.geometry,
            &physics.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )
        .unwrap();
        let first = physics
            .materialize_nonspectral_basis(&physics.frozen_potential, &config.basis, &extended)
            .unwrap();
        let second = physics
            .materialize_nonspectral_basis(&physics.frozen_potential, &config.basis, &extended)
            .unwrap();
        let atomic = first
            .resolved_channels
            .iter()
            .find(|resolved| resolved.recipe.generator == LinearizationEnergyGenerator::Atomic)
            .unwrap();
        assert_eq!(atomic.components.len(), 1);
        assert_eq!(atomic.energy, second.resolved_channels[0].energy);
    }

    #[test]
    fn scalar_single_site_snapshot_runs_two_iteration_scf_smoke() {
        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let state = run_scf(&mut physics, &config(ScfRelativity::Scalar), None).unwrap();
        assert_eq!(state.iterations(), 2);
        assert_eq!(state.relativity, ScfRelativity::Scalar);
    }

    #[test]
    fn fermi_offset_refines_inside_each_scf_iteration() {
        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let mut config = config(ScfRelativity::Scalar);
        let channel = &mut config.basis.channels[0];
        channel.generator = LinearizationEnergyGenerator::FermiOffset;
        channel.seed = Some(Hartree(-0.1));
        let state = run_scf(&mut physics, &config, None).unwrap();
        assert!(state.diagnostics.iter().all(|diagnostic| {
            diagnostic.resolved_channels.iter().any(|resolved| {
                resolved.recipe.generator == LinearizationEnergyGenerator::FermiOffset
                    && matches!(
                        resolved.components[0].diagnostic,
                        LinearizationEnergyDiagnostic::FermiOffset { .. }
                    )
            })
        }));
    }

    #[test]
    fn band_cog_uses_physical_projection_inside_the_scf_iteration() {
        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let mut config = config(ScfRelativity::Scalar);
        config.basis.channels[0].generator = LinearizationEnergyGenerator::BandCog;
        let state = run_scf(&mut physics, &config, None).unwrap();
        assert!(state.diagnostics.iter().all(|diagnostic| {
            diagnostic.resolved_channels.iter().any(|resolved| {
                resolved.recipe.generator == LinearizationEnergyGenerator::BandCog
                    && matches!(
                        resolved.components[0].diagnostic,
                        LinearizationEnergyDiagnostic::BandCog { .. }
                    )
            })
        }));
    }

    #[test]
    fn second_variation_is_routed_and_full_spinor_never_falls_back_to_scalar() {
        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let sv = config(ScfRelativity::SocSecondVariation {
            window: FirstVariationWindow::new(0, 1).unwrap(),
        });
        assert!(physics.initial_density(&sv).is_ok());

        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        assert!(matches!(
            physics.initial_density(&config(ScfRelativity::SpinorFirstVariation)),
            Err(SnapshotDftError::SpinorRadialEquation { .. })
        ));
    }

    #[test]
    fn fully_relativistic_snapshot_uses_full_spinor_solve_and_density() {
        let mut snapshot = snapshot_v1();
        snapshot.geometry.sites[0].spins[0].radial_equation =
            RadialEquationTagV1::FullyRelativisticDirac;
        let mut physics = SnapshotDftPhysics::new(&snapshot.normalize_v2().unwrap()).unwrap();
        let density = physics
            .initial_density(&config(ScfRelativity::SpinorFirstVariation))
            .unwrap();
        assert!((muffintin_dft::electron_count(&density).unwrap() - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn full_spinor_scf_retains_transverse_magnetization_for_two_iterations() {
        let mut snapshot = snapshot_v1();
        snapshot.geometry.sites[0].spins[0].radial_equation =
            RadialEquationTagV1::FullyRelativisticDirac;
        let config = config(ScfRelativity::SpinorFirstVariation);
        let mut physics = SnapshotDftPhysics::new(&snapshot.normalize_v2().unwrap()).unwrap();
        let mut source = run_scf(&mut physics, &config, None).unwrap();
        let charge = source.density.charge().clone();
        let mut transverse = charge.zero_like();
        transverse.add_scaled(0.1, &charge).unwrap();
        let zero = charge.zero_like();
        source.density = RegionalDensity::new(charge, [transverse, zero.clone(), zero]).unwrap();

        let state = run_scf(&mut physics, &config, Some(&source)).unwrap();
        assert_eq!(state.iterations(), 2);
        assert!(state.density.magnetization()[0].residual_rms().unwrap() > 1.0e-8);

        let restart = physics.restart_snapshot(&state).unwrap();
        let encoded = snapshot_file_to_toml(&SnapshotFile::V2(restart)).unwrap();
        let SnapshotFile::V2(reloaded) = snapshot_file_from_toml(&encoded).unwrap() else {
            unreachable!()
        };
        let mut restarted_physics = SnapshotDftPhysics::new(&reloaded).unwrap();
        assert!(
            restarted_physics
                .frozen_potential()
                .scalar()
                .difference_rms(state.potential.scalar())
                .unwrap()
                < 1.0e-10
        );
        for (restarted, expected) in restarted_physics
            .frozen_potential()
            .magnetic()
            .iter()
            .zip(state.potential.magnetic())
        {
            assert!(restarted.difference_rms(expected).unwrap() < 1.0e-10);
        }
        let restarted_density = restarted_physics.initial_density(&config).unwrap();
        assert!(state.density.difference_rms(&restarted_density).unwrap() < 1.0e-12);
    }

    #[test]
    fn scalar_route_rejects_a_transverse_potential() {
        let physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let scalar = physics.frozen_potential.scalar().clone();
        let mut transverse = scalar.zero_like();
        transverse.add_scaled(0.01, &scalar).unwrap();
        let zero = scalar.zero_like();
        let potential = RegionalPotential::new(scalar, [transverse, zero.clone(), zero]).unwrap();
        assert!(matches!(
            physics.solve_points(
                &potential,
                &config(ScfRelativity::Scalar).basis,
                &[[0.0; 3]],
                ScfRelativity::Scalar,
            ),
            Err(SnapshotDftError::TransversePotentialUnsupported { .. })
        ));
    }

    #[test]
    fn signed_kappa_recipe_keeps_multiple_spinor_local_orbitals() {
        let mut snapshot = snapshot_v1();
        let spin = &mut snapshot.geometry.sites[0].spins[0];
        spin.radial_equation = RadialEquationTagV1::FullyRelativisticDirac;
        let physics = SnapshotDftPhysics::new(&snapshot.normalize_v2().unwrap()).unwrap();
        let mut basis = config(ScfRelativity::SpinorFirstVariation).basis;
        for (n, energy) in [(2, -0.1), (3, -0.05)] {
            basis.channels.push(ScfChannelRecipe {
                site: "H-1".to_owned(),
                identity: ScfChannelIdentity::Kappa { n, kappa: 1 },
                treatment: ScfChannelTreatment::Lo,
                derivative_order: 0,
                generator: LinearizationEnergyGenerator::Explicit,
                seed: Some(Hartree(energy)),
                provenance: muffintin_dft::ScfChannelProvenance::Site,
            });
        }
        let meshes = physics.channel_meshes(&basis).unwrap();
        let extended = build_extended_snapshot_core_potentials(
            &physics.frozen_potential,
            &physics.geometry,
            &physics.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )
        .unwrap();
        let basis = physics
            .materialize_nonspectral_basis(&physics.frozen_potential, &basis, &extended)
            .unwrap();
        let inputs = physics
            .spinor_site_inputs(&physics.frozen_potential, &basis)
            .unwrap();
        assert_eq!(
            inputs[0].local_orbitals,
            vec![
                SpinorLocalOrbitalRequest::Lo {
                    kappa: Kappa::new(1).unwrap(),
                    energy: Hartree(-0.1),
                },
                SpinorLocalOrbitalRequest::Lo {
                    kappa: Kappa::new(1).unwrap(),
                    energy: Hartree(-0.05),
                },
            ]
        );
    }

    #[test]
    fn scalar_route_omits_signed_kappa_local_orbitals() {
        let physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let mut basis = config(ScfRelativity::Scalar).basis;
        basis.channels.push(ScfChannelRecipe {
            site: "H-1".to_owned(),
            identity: ScfChannelIdentity::Kappa { n: 2, kappa: 1 },
            treatment: ScfChannelTreatment::Lo,
            derivative_order: 0,
            generator: LinearizationEnergyGenerator::Explicit,
            seed: Some(Hartree(-0.1)),
            provenance: muffintin_dft::ScfChannelProvenance::Site,
        });
        let meshes = physics.channel_meshes(&basis).unwrap();
        let extended = build_extended_snapshot_core_potentials(
            &physics.frozen_potential,
            &physics.geometry,
            &physics.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )
        .unwrap();
        let basis = physics
            .materialize_nonspectral_basis(&physics.frozen_potential, &basis, &extended)
            .unwrap();
        let inputs = physics
            .scalar_site_inputs(&physics.frozen_potential, &basis)
            .unwrap();
        assert!(inputs.up[0].local_orbitals.is_empty());
        assert!(inputs.down[0].local_orbitals.is_empty());
    }

    #[test]
    fn nonempty_core_is_present_initially_and_in_the_scf_iteration() {
        let (snapshot, config) = core_snapshot_and_config();
        let mut physics = SnapshotDftPhysics::new(&snapshot).unwrap();
        let initial = physics.initial_density(&config).unwrap();
        let initial_count = muffintin_dft::electron_count(&initial).unwrap();
        assert!(
            (initial_count - 2.0).abs() < 1.0e-8,
            "initial core+valence electron count was {initial_count}"
        );

        let mut physics = SnapshotDftPhysics::new(&snapshot).unwrap();
        let state = run_scf(&mut physics, &config, None).unwrap();
        assert_eq!(state.iterations(), 2);
    }

    #[test]
    fn frozen_consumers_use_their_source_states_basis_after_a_later_scf() {
        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let first_config = config(ScfRelativity::Scalar);
        let first = run_scf(&mut physics, &first_config, None).unwrap();
        let mut later_config = first_config.clone();
        later_config.basis.plane_wave_cutoff = 0.55;
        let later = run_scf(&mut physics, &later_config, Some(&first)).unwrap();
        assert_eq!(first.basis.plane_wave_cutoff, 0.5);
        assert_eq!(later.basis.plane_wave_cutoff, 0.55);

        let request = BandPathRequest {
            bands: 1,
            points: vec![
                BandPathPoint {
                    label: "G".to_owned(),
                    k: [0.0; 3],
                },
                BandPathPoint {
                    label: "X".to_owned(),
                    k: [0.5, 0.0, 0.0],
                },
            ],
        };
        assert_eq!(physics.solve_band_path(&first, &request).unwrap().len(), 2);
    }

    #[test]
    fn second_variation_rejects_a_window_that_would_drop_lower_scalar_bands() {
        let mut physics = SnapshotDftPhysics::new(&snapshot()).unwrap();
        let config = config(ScfRelativity::SocSecondVariation {
            window: FirstVariationWindow::new(1, 2).unwrap(),
        });
        assert!(matches!(
            physics.initial_density(&config),
            Err(SnapshotDftError::SecondVariationDropsLowerBands { start: 1 })
        ));
    }
}
