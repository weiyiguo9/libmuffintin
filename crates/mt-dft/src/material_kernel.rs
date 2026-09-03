//! Basis-neutral material kernel reconstructed from converted checkpoint state.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::ops::Range;

use crate::core_station::{extend_core_mesh, solve_regional_core_site};
use crate::{
    AtomicEnergyRequest, BandPathRequest, BandState, ChannelKappaError, CollinearKPoint,
    CoreContribution, CorePotentialBuildError, CoreShellOrbitals, CoreSiteRequest,
    CoreSpinPartition, CoreStateRequest, CoreStationError, DensityError, FirstVariationRoute,
    FirstVariationSubspace, FirstVariationWindow, FullSpinorKPoint, GeneratedLinearizationEnergy,
    InterstitialField, LinearizationEnergyDiagnostic, LinearizationEnergyError,
    LinearizationEnergyGenerator, LocalPauliPotential, MuffinTinField, OccupationError,
    PdosEnergySample, RegionalCoreResult, RegionalDensity, RegionalElectrostaticResult,
    RegionalPotential, RegionalScalarField, RegularSpectrum, ScalarBuilderError,
    ScalarIterationBasis, ScalarLocalOrbitalRequest, ScalarSiteInput, ScfBasis, ScfChannelIdentity,
    ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfCoreSite, ScfEnergyContext,
    ScfEnergyTerms, ScfExchangeCorrelation, ScfKMesh, ScfKReduction, ScfKSamplingProvenance,
    ScfOccupations, ScfPhysics, ScfPotentialBuild, ScfPotentialBuildError, ScfRelativity,
    ScfResolvedChannelEnergy, ScfState, SecondVariationBandDiagnostic, SecondVariationError,
    SecondVariationKPoint, SpinorBuilderError, SpinorFirstVariationError, SpinorIterationBasis,
    SpinorLinearizationEnergy, SpinorLocalOrbitalRequest, SpinorSiteInput,
    StaticCoreSiteExchangeError, TetrahedronError, build_collinear_scalar_iteration_bases,
    build_extended_checkpoint_core_potentials, build_extended_core_potentials,
    build_extended_electrostatic_core_potentials, build_scf_potential,
    build_spinor_iteration_basis, build_static_core_exchange_site_blocks, channel_kappas,
    channel_l, channel_n, generate_atomic_energy, generate_band_center_energy,
    generate_band_cog_energy, generate_explicit_energy, generate_fermi_offset_energy,
    generate_frozen_checkpoint_energy, generate_log_derivative_energy, kappa_degeneracy_average,
    physical_site_band_projections, scalar_component_energy, solve_fermi_dirac, solve_gaussian,
    solve_soc_second_variation, solve_spinor_k_point, spin_resolved_energy, spinor_kappas_for_l,
    synthesize_collinear_valence_density, synthesize_full_spinor_valence_density,
    synthesize_second_variation_valence_density,
};
use muffintin_core::{
    Bohr, ExponentialMesh, FourierFieldError, FourierLayout, GVector, Hartree,
    InterstitialGeometry, InverseBohr, Kappa, LatticeError, MeshError, ReciprocalLattice,
};
use muffintin_coulomb::StaticCoreExchangeMode;
use muffintin_envelope::{PlaneWave, PlaneWaveEnvelope};
use muffintin_operators::lapw::{
    Collinear, GeneralizedEigensolution, InterstitialPotential, LapwEigenproblem, LapwError,
};
use muffintin_operators::{
    CompiledSiteProjection, OperatorError, SecondVariationMixing, SiteSpinOrbitBlock,
    SocOperatorError, SpinorSiteOperatorBlocks, assemble_scalar_site_operator,
    lift_band_hermitian_feedback, project_site_soc_to_subspace,
    project_site_spinor_operator_to_subspace, solve_generalized_hermitian,
};
use muffintin_sphere::{
    CorePotentialContinuationSpec, CoreState, DiracError, ExtendedCorePotential, RadialEquation,
    SpexSpinOrbitPotential, SphereField, SpinOrbitRadialError, spex_spin_orbit_radial_shell,
};
use muffintin_symmetry::kmesh::{
    KMeshReduction, KMeshReductionError, RegularKMesh, reduce_regular_mesh,
};
use muffintin_symmetry::moyo_backend::{MoyoDetectionError, detect as detect_symmetry};
use muffintin_symmetry::{CrystalCell, CrystalSymmetryTransform, SymmetryTransformError};
use muffintin_tensor::{Axis, DenseEigenvectors, DenseHermitianMatrix, TensorError};
use num_complex::Complex64;
use thiserror::Error;

mod basis_materialization;

const OVERLAP_THRESHOLD: f64 = 1.0e-10;
const OCCUPATION_TOLERANCE: f64 = 1.0e-12;
const OCCUPATION_ITERATIONS: usize = 256;
const TRANSVERSE_FIELD_TOLERANCE: f64 = 1.0e-10;
const SPECTRAL_REFINEMENT_TOLERANCE: f64 = 1.0e-10;
const DEFAULT_FERMI_OFFSET_HARTREE: f64 = -0.1;

/// Checkpoint-backed material kernel shared by SCF, bands, and DOS tasks.
///
/// Construction performs only convention conversion and topology validation.
/// The initial density is obtained by a frozen-checkpoint one-particle solve;
/// no atomic-density or artificial `G=0` guess is installed.
#[derive(Debug)]
pub struct MaterialKernel {
    pub(super) reciprocal: ReciprocalLattice,
    pub(super) geometry: InterstitialGeometry,
    pub(super) sites: Vec<CheckpointSite>,
    pub(super) frozen_potential: RegionalPotential,
    pub(super) restart_density: Option<RegionalDensity>,
    pub(super) nuclear_charges: Vec<f64>,
    crystal_cell: CrystalCell,
    prepared_symmetry: Option<PreparedSymmetrySampling>,
    core_potentials: BTreeMap<usize, ScfPotentialBuild>,
    pub(super) spex_spinor_binding: Option<SpexSpinorMaterialBinding>,
}

#[derive(Clone, Debug)]
struct PreparedSymmetrySampling {
    mesh: ScfKMesh,
    reduction: KMeshReduction,
    transforms: Vec<CrystalSymmetryTransform>,
    spacegroup_number: Option<i32>,
}

#[derive(Clone, Debug)]
pub struct SpexSpinorMaterialBinding {
    channels: Vec<SpexBoundSpinorChannel>,
}

#[derive(Clone, Debug)]
pub struct SpexBoundSpinorChannel {
    l: u32,
    requested: ScfChannelRecipe,
    resolved: ScfResolvedChannelEnergy,
}

#[derive(Clone, Debug)]
pub struct CheckpointSite {
    id: String,
    position: [Bohr; 3],
    radius: Bohr,
    up: CheckpointSpin,
    down: CheckpointSpin,
    nonmagnetic_scalar: bool,
}

#[derive(Clone, Debug)]
pub struct CheckpointSpin {
    route: RadialRoute,
    mesh: ExponentialMesh,
    linearization: BTreeMap<u32, Hartree>,
    local_orbitals: Vec<(u32, Hartree)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadialRoute {
    Schroedinger,
    ScalarKoellingHarmon,
    Dirac,
}

impl CheckpointSpin {
    pub fn new(
        route: RadialRoute,
        mesh: ExponentialMesh,
        linearization: BTreeMap<u32, Hartree>,
        local_orbitals: Vec<(u32, Hartree)>,
    ) -> Self {
        Self {
            route,
            mesh,
            linearization,
            local_orbitals,
        }
    }

    pub const fn route(&self) -> RadialRoute {
        self.route
    }

    pub const fn mesh(&self) -> &ExponentialMesh {
        &self.mesh
    }
}

impl CheckpointSite {
    pub fn new(
        id: String,
        position: [Bohr; 3],
        radius: Bohr,
        up: CheckpointSpin,
        down: CheckpointSpin,
        nonmagnetic_scalar: bool,
    ) -> Self {
        Self {
            id,
            position,
            radius,
            up,
            down,
            nonmagnetic_scalar,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub const fn position(&self) -> [Bohr; 3] {
        self.position
    }

    pub const fn radius(&self) -> Bohr {
        self.radius
    }

    pub const fn up(&self) -> &CheckpointSpin {
        &self.up
    }

    pub const fn down(&self) -> &CheckpointSpin {
        &self.down
    }

    pub const fn nonmagnetic_scalar(&self) -> bool {
        self.nonmagnetic_scalar
    }
}

impl SpexBoundSpinorChannel {
    pub fn new(l: u32, requested: ScfChannelRecipe, resolved: ScfResolvedChannelEnergy) -> Self {
        Self {
            l,
            requested,
            resolved,
        }
    }
}

impl SpexSpinorMaterialBinding {
    pub fn new(channels: Vec<SpexBoundSpinorChannel>) -> Self {
        Self { channels }
    }
}

/// One iteration's potential and basis-neutral controls. Concrete k-dependent
/// APW matching is intentionally deferred until the requested k points exist.
#[derive(Clone, Debug)]
pub struct CheckpointOneParticle {
    potential: RegionalPotential,
    basis: ScfBasis,
}

/// Valence, core, and unchanged total forms of the initial regional density.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialDensityComponents {
    pub valence: RegionalDensity,
    pub core: RegionalDensity,
    pub total: RegionalDensity,
}

impl CheckpointOneParticle {
    pub const fn potential(&self) -> &RegionalPotential {
        &self.potential
    }

    pub const fn basis(&self) -> &ScfBasis {
        &self.basis
    }
}

/// Concrete regular-mesh solutions retained for occupations and density synthesis.
#[derive(Clone, Debug)]
pub struct CheckpointBandSolution {
    points: Vec<CheckpointKPoint>,
    states: Vec<BandState>,
    reduction: Option<KMeshReduction>,
    density_layout: Option<FourierLayout>,
    symmetry_transforms: Vec<CrystalSymmetryTransform>,
    spacegroup_number: Option<i32>,
}

impl CheckpointBandSolution {
    pub fn points(&self) -> &[CheckpointKPoint] {
        &self.points
    }

    pub fn states(&self) -> &[BandState] {
        &self.states
    }

    /// Select the exact reciprocal layout used when synthesizing this band
    /// solution's regional density.
    ///
    /// The layout must contain every plane-wave difference required by the
    /// retained basis; density synthesis rejects an incomplete layout.
    pub fn set_density_layout(&mut self, layout: FourierLayout) {
        self.density_layout = Some(layout);
    }

    /// Seed scalar orbitals from another Hamiltonian in exactly the same radial
    /// and constrained LAPW space, retaining this solution's physical H0/S.
    ///
    /// The copied eigenpairs are an initial guess, not eigenpairs of the current
    /// Hamiltonian. The next Fock iteration must rebuild exchange and solve it.
    pub fn with_scalar_orbital_guess(mut self, guess: &Self) -> Result<Self, MaterialKernelError> {
        if self.points.len() != guess.points.len() {
            return Err(MaterialKernelError::ScalarOrbitalGuessPointCount {
                actual: guess.points.len(),
                expected: self.points.len(),
            });
        }
        for (point_index, (point, source)) in self.points.iter_mut().zip(&guess.points).enumerate()
        {
            let (
                CheckpointKPointSolution::Collinear {
                    bases,
                    solutions,
                    up_occupations,
                    down_occupations,
                    ..
                },
                CheckpointKPointSolution::Collinear {
                    bases: source_bases,
                    solutions: source_solutions,
                    up_occupations: source_up,
                    down_occupations: source_down,
                    ..
                },
            ) = (&mut point.solution, &source.solution)
            else {
                return Err(MaterialKernelError::FeedbackRequiresScalar { point: point_index });
            };
            let same_space = |target: &ScalarIterationBasis, source: &ScalarIterationBasis| {
                target.compiled == source.compiled
                    && target.radial_sites == source.radial_sites
                    && target.core_orthogonalization == source.core_orthogonalization
            };
            if up_occupations == down_occupations
                || up_occupations != source_up
                || down_occupations != source_down
                || point.weight != source.weight
                || !same_space(&bases.up, &source_bases.up)
                || !same_space(&bases.down, &source_bases.down)
            {
                return Err(MaterialKernelError::ScalarOrbitalGuessSpace { point: point_index });
            }
            for (range, solved) in [
                (up_occupations.clone(), &source_solutions.up),
                (down_occupations.clone(), &source_solutions.down),
            ] {
                for (state, &energy) in self.states[range].iter_mut().zip(&solved.eigenvalues) {
                    state.energy = energy;
                }
            }
            *solutions = source_solutions.clone();
            point.energies = source.energies.clone();
        }
        Ok(self)
    }

    /// Re-solve the retained spinor subspace with fresh Hermitian band-space
    /// feedback at every k point.
    ///
    /// Each feedback matrix is lifted through the current orbitals, but added
    /// to the retained original local-potential Hamiltonian. Repeated calls
    /// therefore never accumulate feedback on an earlier Fock Hamiltonian.
    pub fn solve_spinor_feedback(
        &self,
        feedback: &[DenseHermitianMatrix],
    ) -> Result<Self, MaterialKernelError> {
        if feedback.len() != self.points.len() {
            return Err(MaterialKernelError::FeedbackPointCount {
                actual: feedback.len(),
                expected: self.points.len(),
            });
        }
        let lifted = self
            .points
            .iter()
            .zip(feedback)
            .enumerate()
            .map(|(point_index, (point, band_feedback))| {
                let CheckpointKPointSolution::Spinor {
                    eigenproblem,
                    solution,
                    ..
                } = &point.solution
                else {
                    return Err(MaterialKernelError::FeedbackRequiresSpinor { point: point_index });
                };
                Ok(lift_band_hermitian_feedback(
                    &eigenproblem.overlap,
                    &solution.eigenvectors,
                    band_feedback,
                )?)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.solve_spinor_global_feedback(&lifted)
    }

    /// Re-solve the retained spinor subspace with physical-basis Hermitian
    /// feedback at every k point.
    ///
    /// The matrices are expressed in the unchanged raw basis of the retained
    /// original H0/S problems. They may therefore be compared and mixed
    /// across orbital updates within this one fixed radial-basis frame.
    pub fn solve_spinor_global_feedback(
        &self,
        feedback: &[DenseHermitianMatrix],
    ) -> Result<Self, MaterialKernelError> {
        if feedback.len() != self.points.len() {
            return Err(MaterialKernelError::FeedbackPointCount {
                actual: feedback.len(),
                expected: self.points.len(),
            });
        }
        let mut updated = self.clone();
        let (points, states) = (&mut updated.points, &mut updated.states);
        for (point_index, (point, global_feedback)) in points.iter_mut().zip(feedback).enumerate() {
            let CheckpointKPointSolution::Spinor {
                eigenproblem,
                solution,
                occupations,
                ..
            } = &mut point.solution
            else {
                return Err(MaterialKernelError::FeedbackRequiresSpinor { point: point_index });
            };
            if global_feedback.axis() != Axis::GlobalBasis {
                return Err(TensorError::Axis {
                    index: 0,
                    expected: Axis::GlobalBasis,
                    actual: global_feedback.axis(),
                }
                .into());
            }
            if global_feedback.dimension() != eigenproblem.hamiltonian.dimension() {
                let expected = eigenproblem.hamiltonian.dimension();
                return Err(TensorError::Shape {
                    expected: vec![expected, expected],
                    actual: vec![global_feedback.dimension(), global_feedback.dimension()],
                }
                .into());
            }
            let fock = DenseHermitianMatrix::from_upper_triangle(
                eigenproblem.hamiltonian.dimension(),
                Axis::GlobalBasis,
                |row, column| {
                    eigenproblem.hamiltonian.at(row, column) + global_feedback.at(row, column)
                },
            )?;
            let solved =
                solve_generalized_hermitian(&fock, &eigenproblem.overlap, OVERLAP_THRESHOLD)?;
            if solved.eigenvalues.len() != occupations.len() {
                return Err(MaterialKernelError::FeedbackBandCount {
                    point: point_index,
                    actual: solved.eigenvalues.len(),
                    expected: occupations.len(),
                });
            }
            point.energies = solved.eigenvalues.clone();
            for (state, &energy) in states[occupations.clone()]
                .iter_mut()
                .zip(&solved.eigenvalues)
            {
                state.energy = energy;
            }
            *solution = solved;
        }
        Ok(updated)
    }

    /// Re-solve both scalar KH spin channels with physical-basis exchange
    /// feedback on the retained fixed H0/S frame.
    pub fn solve_scalar_global_feedback(
        &self,
        feedback: &[Collinear<DenseHermitianMatrix>],
    ) -> Result<Self, MaterialKernelError> {
        if feedback.len() != self.points.len() {
            return Err(MaterialKernelError::FeedbackPointCount {
                actual: feedback.len(),
                expected: self.points.len(),
            });
        }
        let mut updated = self.clone();
        let (points, states) = (&mut updated.points, &mut updated.states);
        for (point_index, (point, feedback)) in points.iter_mut().zip(feedback).enumerate() {
            let CheckpointKPointSolution::Collinear {
                bases,
                eigenproblems,
                solutions,
                up_occupations,
                down_occupations,
                ..
            } = &mut point.solution
            else {
                return Err(MaterialKernelError::FeedbackRequiresScalar { point: point_index });
            };
            if up_occupations == down_occupations {
                return Err(MaterialKernelError::FeedbackRequiresScalar { point: point_index });
            }
            let solve = |basis: &ScalarIterationBasis,
                         problem: &LapwEigenproblem,
                         feedback: &DenseHermitianMatrix|
             -> Result<GeneralizedEigensolution, MaterialKernelError> {
                if feedback.axis() != Axis::GlobalBasis {
                    return Err(TensorError::Axis {
                        index: 0,
                        expected: Axis::GlobalBasis,
                        actual: feedback.axis(),
                    }
                    .into());
                }
                if feedback.dimension() != problem.hamiltonian.dimension() {
                    return Err(TensorError::Shape {
                        expected: vec![problem.hamiltonian.dimension(); 2],
                        actual: vec![feedback.dimension(); 2],
                    }
                    .into());
                }
                let fock = DenseHermitianMatrix::from_upper_triangle(
                    problem.hamiltonian.dimension(),
                    Axis::GlobalBasis,
                    |row, column| problem.hamiltonian.at(row, column) + feedback.at(row, column),
                )?;
                Ok(match &basis.core_orthogonalization {
                    Some(core) => muffintin_operators::solve_generalized_hermitian_embedded(
                        &fock,
                        &problem.overlap,
                        &core.embedding,
                        OVERLAP_THRESHOLD,
                    )?,
                    None => {
                        solve_generalized_hermitian(&fock, &problem.overlap, OVERLAP_THRESHOLD)?
                    }
                })
            };
            let up = solve(&bases.up, &eigenproblems.up, &feedback.up)?;
            let down = solve(&bases.down, &eigenproblems.down, &feedback.down)?;
            for (range, solved) in [
                (up_occupations.clone(), &up),
                (down_occupations.clone(), &down),
            ] {
                if solved.eigenvalues.len() != range.len() {
                    return Err(MaterialKernelError::FeedbackBandCount {
                        point: point_index,
                        actual: solved.eigenvalues.len(),
                        expected: range.len(),
                    });
                }
                for (state, &energy) in states[range].iter_mut().zip(&solved.eigenvalues) {
                    state.energy = energy;
                }
            }
            point.energies = up
                .eigenvalues
                .iter()
                .chain(&down.eigenvalues)
                .copied()
                .collect();
            *solutions = Collinear::new(up, down);
        }
        Ok(updated)
    }
}

#[derive(Clone, Debug)]
pub struct CheckpointKPoint {
    weight: f64,
    pub solution: CheckpointKPointSolution,
    energies: Vec<Hartree>,
}

impl CheckpointKPoint {
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

#[derive(Clone, Debug)]
pub enum CheckpointKPointSolution {
    Collinear {
        bases: Box<Collinear<ScalarIterationBasis>>,
        /// Original scalar local-potential H0/S problems for this radial basis.
        eigenproblems: Collinear<LapwEigenproblem>,
        solutions: Collinear<GeneralizedEigensolution>,
        up_occupations: Range<usize>,
        down_occupations: Range<usize>,
    },
    Spinor {
        basis: SpinorIterationBasis,
        site_blocks: Vec<SpinorSiteOperatorBlocks>,
        /// Original local-potential H0/S problem for this fixed radial basis.
        eigenproblem: LapwEigenproblem,
        solution: GeneralizedEigensolution,
        occupations: Range<usize>,
    },
}

/// SOC second-variation bands plus source-band mixing diagnostics per k point.
#[derive(Clone, Debug)]
pub struct CheckpointSecondVariationResult {
    pub bands: CheckpointBandSolution,
    /// Current SOC eigenvectors in the fixed doubled scalar-band frame.
    pub mixings: Vec<SecondVariationMixing>,
    pub diagnostics: Vec<Vec<SecondVariationBandDiagnostic>>,
}

/// Fixed scalar-band frame for repeated SOC second-variation Fock solves.
#[derive(Clone, Debug)]
pub struct CheckpointSecondVariationFrame {
    scalar: CheckpointBandSolution,
    first_variations: Vec<FirstVariationSubspace>,
    site_blocks: Vec<Vec<SiteSpinOrbitBlock>>,
    site_feedback: Vec<Vec<DenseHermitianMatrix>>,
    fixed_hamiltonians: Vec<DenseHermitianMatrix>,
    resolved_core_feedback: Vec<DenseHermitianMatrix>,
}

impl CheckpointSecondVariationFrame {
    /// Scalar-Fock plus SOC plus resolved frozen-core exchange in the fixed
    /// doubled source-band frame, before valence-exchange replacement.
    pub fn fixed_hamiltonians(&self) -> &[DenseHermitianMatrix] {
        &self.fixed_hamiltonians
    }

    /// Resolved frozen-core exchange in the same fixed doubled frame.
    pub fn resolved_core_feedback(&self) -> &[DenseHermitianMatrix] {
        &self.resolved_core_feedback
    }

    /// Re-solve the fixed SOC frame with one additional Hermitian matrix per
    /// k point. An empty slice means zero additional feedback.
    pub fn solve(
        &self,
        feedback: &[DenseHermitianMatrix],
    ) -> Result<CheckpointSecondVariationResult, MaterialKernelError> {
        if !feedback.is_empty() && feedback.len() != self.scalar.points.len() {
            return Err(MaterialKernelError::FeedbackPointCount {
                actual: feedback.len(),
                expected: self.scalar.points.len(),
            });
        }
        let mut points = Vec::with_capacity(self.scalar.points.len());
        let mut states = Vec::new();
        let mut mixings = Vec::with_capacity(self.scalar.points.len());
        let mut diagnostics = Vec::with_capacity(self.scalar.points.len());
        for (point_index, point) in self.scalar.points.iter().enumerate() {
            let CheckpointKPointSolution::Collinear {
                bases,
                eigenproblems,
                ..
            } = &point.solution
            else {
                return Err(MaterialKernelError::SecondVariationRequiresScalarBands {
                    point: point_index,
                });
            };
            let second = solve_soc_second_variation(
                FirstVariationRoute::NonmagneticScalarKoellingHarmon,
                &bases.up.compiled,
                &self.first_variations[point_index],
                &self.site_blocks[point_index],
                &self.site_feedback[point_index],
                feedback.get(point_index),
            )?;
            let split = split_second_variation(&second)?;
            mixings.push(second.mixing.clone());
            diagnostics.push(second.diagnostics.clone());
            let start = states.len();
            states.extend(
                second
                    .eigenvalues
                    .iter()
                    .copied()
                    .map(|energy| BandState::new(energy, point.weight, 1)),
            );
            let end = states.len();
            points.push(CheckpointKPoint {
                weight: point.weight,
                energies: second.eigenvalues,
                solution: CheckpointKPointSolution::Collinear {
                    bases: bases.clone(),
                    eigenproblems: eigenproblems.clone(),
                    solutions: split,
                    up_occupations: start..end,
                    down_occupations: start..end,
                },
            });
        }
        Ok(CheckpointSecondVariationResult {
            bands: CheckpointBandSolution {
                points,
                states,
                reduction: self.scalar.reduction.clone(),
                density_layout: self.scalar.density_layout.clone(),
                symmetry_transforms: self.scalar.symmetry_transforms.clone(),
                spacegroup_number: self.scalar.spacegroup_number,
            },
            mixings,
            diagnostics,
        })
    }
}

impl MaterialKernel {
    pub fn new(
        reciprocal: ReciprocalLattice,
        geometry: InterstitialGeometry,
        sites: Vec<CheckpointSite>,
        frozen_potential: RegionalPotential,
        restart_density: Option<RegionalDensity>,
        nuclear_charges: Vec<f64>,
        crystal_cell: CrystalCell,
    ) -> Result<Self, MaterialKernelError> {
        let kernel = Self {
            reciprocal,
            geometry,
            sites,
            frozen_potential,
            restart_density,
            nuclear_charges,
            crystal_cell,
            prepared_symmetry: None,
            core_potentials: BTreeMap::new(),
            spex_spinor_binding: None,
        };
        kernel.require_topology_site_count("geometry", kernel.geometry.spheres().len())?;
        kernel.require_topology_site_count("nuclear charges", kernel.nuclear_charges.len())?;
        kernel.require_potential_site_count(&kernel.frozen_potential)?;
        if let Some(density) = &kernel.restart_density {
            kernel.require_density_site_count(density)?;
        }
        Ok(kernel)
    }

    pub fn bind_spex_spinor(
        &mut self,
        binding: SpexSpinorMaterialBinding,
        basis: &ScfBasis,
    ) -> Result<(), MaterialKernelError> {
        Self::validate_spex_binding(&binding, basis)?;
        self.spex_spinor_binding = Some(binding);
        Ok(())
    }

    fn validate_spex_requested_basis(&self, basis: &ScfBasis) -> Result<(), MaterialKernelError> {
        let Some(binding) = &self.spex_spinor_binding else {
            return Ok(());
        };
        Self::validate_spex_binding(binding, basis)
    }

    fn validate_spex_binding(
        binding: &SpexSpinorMaterialBinding,
        basis: &ScfBasis,
    ) -> Result<(), MaterialKernelError> {
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

    fn validate_spex_resolved_basis(&self, basis: &ScfBasis) -> Result<(), MaterialKernelError> {
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

    pub fn sites(&self) -> &[CheckpointSite] {
        &self.sites
    }

    pub fn restart_density(&self) -> Option<&RegionalDensity> {
        self.restart_density.as_ref()
    }

    pub fn nuclear_charges(&self) -> &[f64] {
        &self.nuclear_charges
    }

    pub const fn crystal_cell(&self) -> &CrystalCell {
        &self.crystal_cell
    }

    /// Construct the current initial density while retaining its valence/core split.
    ///
    /// A restart checkpoint stores only the total regional density. Its core
    /// component is therefore re-solved once in the frozen checkpoint local
    /// potential and subtracted from that unchanged total. Frozen-potential
    /// inputs retain the valence and core pieces already produced by the
    /// one-particle and core solves.
    pub fn initial_density_components(
        &mut self,
        config: &ScfConfig,
    ) -> Result<InitialDensityComponents, MaterialKernelError> {
        if let Some(mut total) = self.restart_density.clone() {
            let transforms = self
                .reduced_sampling(config.k_mesh)?
                .map(|(_, transforms, _)| transforms);
            if let Some(transforms) = &transforms {
                if config.relativity != ScfRelativity::Scalar {
                    return Err(MaterialKernelError::SymmetryReductionRequiresScalarNonmagnetic);
                }
                self.require_second_variation_route(&self.frozen_potential)?;
                self.require_symmetry_equivalent_config(config, transforms)?;
                total = self.project_scalar_density(&total, transforms)?;
            }
            let meshes = self.channel_meshes(&config.basis)?;
            let extended = build_extended_checkpoint_core_potentials(
                &self.frozen_potential,
                &self.geometry,
                &self.nuclear_charges,
                &meshes,
                CorePotentialContinuationSpec::default(),
            )?;
            let mut core = self.solve_initial_core_density(&total, config, &extended)?;
            if let Some(transforms) = &transforms {
                core = self.project_scalar_density(&core, transforms)?;
            }
            let mut valence = total.clone();
            valence.add_scaled(-1.0, &core)?;
            return Ok(InitialDensityComponents {
                valence,
                core,
                total,
            });
        }

        let meshes = self.channel_meshes(&config.basis)?;
        if let Some((_, transforms, _)) = self.reduced_sampling(config.k_mesh)? {
            self.require_symmetry_equivalent_config(config, &transforms)?;
        }
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
        let (bands, occupations) = {
            let mut passes = 0;
            loop {
                passes += 1;
                let bands =
                    self.solve_regular_bands(0, &one_particle, config.k_mesh, config.relativity)?;
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
                        return Err(MaterialKernelError::InitialBasisRefinementNotConverged {
                            passes,
                        });
                    }
                    Some(refined) => one_particle = refined,
                }
            }
        };
        let mut valence = self.synthesize(&bands, &occupations)?;
        let mut core = self.solve_initial_core_density(&valence, config, &initial_extended)?;
        if !bands.symmetry_transforms.is_empty() {
            valence = self.project_scalar_density(&valence, &bands.symmetry_transforms)?;
            core = self.project_scalar_density(&core, &bands.symmetry_transforms)?;
        }
        let mut total = valence.clone();
        total.add_scaled(1.0, &core)?;
        Ok(InitialDensityComponents {
            valence,
            core,
            total,
        })
    }

    /// Solve the configured core once in the immutable checkpoint potential.
    pub fn frozen_checkpoint_core(
        &self,
        template: &RegionalDensity,
        config: &ScfConfig,
    ) -> Result<RegionalCoreResult, MaterialKernelError> {
        let meshes = self.channel_meshes(&config.basis)?;
        let extended = build_extended_checkpoint_core_potentials(
            &self.frozen_potential,
            &self.geometry,
            &self.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )?;
        self.solve_initial_core(template, config, &extended)
    }

    /// Bootstrap requested core sidecars in a no-XC nuclear-plus-Hartree field.
    pub fn bootstrap_hf_core(
        &self,
        source_density: &RegionalDensity,
        electrostatics: &RegionalElectrostaticResult,
        core_sites: &[ScfCoreSite],
    ) -> Result<RegionalCoreResult, MaterialKernelError> {
        self.require_density_site_count(source_density)?;
        if source_density.geometry() != &self.geometry {
            return Err(MaterialKernelError::DensityGeometryMismatch);
        }
        let mut requested_ids = BTreeSet::new();
        for site in core_sites {
            if !requested_ids.insert(site.id.clone()) {
                return Err(MaterialKernelError::DuplicateCoreSite(site.id.clone()));
            }
            if self.sites.iter().all(|candidate| candidate.id != site.id) {
                return Err(MaterialKernelError::UnknownCoreSite(site.id.clone()));
            }
        }
        let maximum_n = self
            .sites
            .iter()
            .map(|checkpoint_site| {
                core_sites
                    .iter()
                    .find(|site| site.id == checkpoint_site.id)
                    .into_iter()
                    .flat_map(|site| &site.states)
                    .map(|state| state.principal_quantum_number)
                    .max()
                    .unwrap_or(1)
            })
            .collect::<Vec<_>>();
        let extended_meshes = self
            .sites
            .iter()
            .enumerate()
            .map(|(site_index, site)| {
                let orbital_scale = f64::from(maximum_n[site_index]).powi(2)
                    / self.nuclear_charges[site_index].max(1.0);
                let outer_radius = (4.0 * site.radius.get()).max(40.0 * orbital_scale);
                extend_core_mesh(&site.up.mesh, outer_radius).map_err(Into::into)
            })
            .collect::<Result<Vec<_>, MaterialKernelError>>()?;
        let extended = build_extended_electrostatic_core_potentials(
            electrostatics,
            &self.geometry,
            &extended_meshes,
            CorePotentialContinuationSpec::default(),
        )?;
        let mut density = source_density.zero_like();
        let mut eigenvalue_sum = Hartree(0.0);
        let mut sites = Vec::new();
        let mut orbitals = Vec::new();
        for site in core_sites.iter().filter(|site| !site.states.is_empty()) {
            let site_index = self
                .sites
                .iter()
                .position(|candidate| candidate.id == site.id)
                .ok_or_else(|| MaterialKernelError::UnknownCoreSite(site.id.clone()))?;
            let request = self.core_site_request(site_index, site)?;
            let solved = solve_regional_core_site(
                source_density,
                &self.nuclear_charges,
                &request,
                &extended[site_index].potential,
            )?;
            density.add_scaled(1.0, &solved.contribution.contribution.density)?;
            eigenvalue_sum += solved.contribution.contribution.eigenvalue_sum;
            sites.push(solved.contribution);
            orbitals.push(solved.orbitals);
        }
        Ok(RegionalCoreResult {
            density,
            eigenvalue_sum,
            sites,
            orbitals,
        })
    }

    fn reduced_sampling(
        &mut self,
        mesh: ScfKMesh,
    ) -> Result<
        Option<(KMeshReduction, Vec<CrystalSymmetryTransform>, Option<i32>)>,
        MaterialKernelError,
    > {
        let ScfKReduction::Symmetry {
            symprec,
            include_time_reversal,
        } = mesh.reduction
        else {
            return Ok(None);
        };
        if let Some(prepared) = &self.prepared_symmetry
            && prepared.mesh == mesh
        {
            return Ok(Some((
                prepared.reduction.clone(),
                prepared.transforms.clone(),
                prepared.spacegroup_number,
            )));
        }
        let dataset = detect_symmetry(&self.crystal_cell, symprec)?;
        let reduction = reduce_regular_mesh(
            &dataset,
            RegularKMesh {
                divisions: mesh.divisions,
                shift: mesh.shift,
            },
            include_time_reversal,
        )?;
        let active_operation_indices = reduction
            .active_operations
            .iter()
            .map(|operation| operation.operation_index)
            .collect::<BTreeSet<_>>();
        let transforms = active_operation_indices
            .into_iter()
            .map(|index| dataset.operations[index].clone())
            .map(|operation| {
                CrystalSymmetryTransform::from_cell(operation, &self.crystal_cell, symprec)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.prepared_symmetry = Some(PreparedSymmetrySampling {
            mesh,
            reduction: reduction.clone(),
            transforms: transforms.clone(),
            spacegroup_number: dataset.spacegroup_number,
        });
        Ok(Some((reduction, transforms, dataset.spacegroup_number)))
    }

    fn symmetry_projected_potential(
        &self,
        potential: &RegionalPotential,
        transforms: &[CrystalSymmetryTransform],
    ) -> Result<RegionalPotential, MaterialKernelError> {
        let scalar = potential.scalar().symmetry_average(transforms)?;
        let magnetic = std::array::from_fn(|_| scalar.zero_like());
        Ok(RegionalPotential::new(scalar, magnetic)?)
    }

    fn project_scalar_density(
        &self,
        density: &RegionalDensity,
        transforms: &[CrystalSymmetryTransform],
    ) -> Result<RegionalDensity, MaterialKernelError> {
        let charge = density.charge().symmetry_average(transforms)?;
        let zero = charge.zero_like();
        Ok(RegionalDensity::new(
            charge,
            [zero.clone(), zero.clone(), zero],
        )?)
    }

    fn require_symmetry_equivalent_config(
        &self,
        config: &ScfConfig,
        transforms: &[CrystalSymmetryTransform],
    ) -> Result<(), MaterialKernelError> {
        for transform in transforms {
            for (source, &target) in transform.site_map().iter().enumerate() {
                let source_site = &self.sites[source];
                let target_site = &self.sites[target];
                if source_site.radius != target_site.radius
                    || source_site.up.mesh != target_site.up.mesh
                    || source_site.down.mesh != target_site.down.mesh
                    || source_site.up.route != target_site.up.route
                    || source_site.down.route != target_site.down.route
                    || source_site.nonmagnetic_scalar != target_site.nonmagnetic_scalar
                {
                    return Err(MaterialKernelError::SymmetryEquivalentSiteMismatch {
                        source_site: source_site.id.clone(),
                        target_site: target_site.id.clone(),
                    });
                }
                let source_core = config
                    .core_sites
                    .iter()
                    .find(|site| site.id == source_site.id)
                    .map(|site| &site.states);
                let target_core = config
                    .core_sites
                    .iter()
                    .find(|site| site.id == target_site.id)
                    .map(|site| &site.states);
                let source_channels = config
                    .basis
                    .channels
                    .iter()
                    .filter(|recipe| recipe.site == source_site.id)
                    .collect::<Vec<_>>();
                let target_channels = config
                    .basis
                    .channels
                    .iter()
                    .filter(|recipe| recipe.site == target_site.id)
                    .collect::<Vec<_>>();
                if source_core != target_core
                    || source_channels.len() != target_channels.len()
                    || source_channels
                        .iter()
                        .zip(target_channels)
                        .any(|(source, target)| !equivalent_channel_recipe(source, target))
                {
                    return Err(MaterialKernelError::SymmetryEquivalentRecipeMismatch {
                        source_site: source_site.id.clone(),
                        target_site: target_site.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn solve_points(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        points: &[[f64; 3]],
        relativity: ScfRelativity,
        core_orthogonal: &[CoreShellOrbitals],
    ) -> Result<CheckpointBandSolution, MaterialKernelError> {
        if points.is_empty() {
            return Err(MaterialKernelError::EmptyKPointSet);
        }
        let weights = vec![1.0 / points.len() as f64; points.len()];
        self.solve_points_with_weights(
            potential,
            basis,
            points,
            &weights,
            relativity,
            core_orthogonal,
        )
    }

    /// Prepare a fixed doubled scalar-band frame for repeated SOC Fock solves.
    pub fn prepare_soc_second_variation(
        &self,
        potential: &RegionalPotential,
        scalar: &CheckpointBandSolution,
        window: FirstVariationWindow,
        core_sidecars: &[CoreShellOrbitals],
    ) -> Result<CheckpointSecondVariationFrame, MaterialKernelError> {
        self.require_second_variation_route(potential)?;
        if window.start() != 0 {
            return Err(MaterialKernelError::SecondVariationDropsLowerBands {
                start: window.start(),
            });
        }
        let mut first_variations = Vec::with_capacity(scalar.points.len());
        let mut site_blocks = Vec::with_capacity(scalar.points.len());
        let mut site_feedback = Vec::with_capacity(scalar.points.len());
        let mut fixed_hamiltonians = Vec::with_capacity(scalar.points.len());
        let mut resolved_core_feedback = Vec::with_capacity(scalar.points.len());
        for (point_index, point) in scalar.points.iter().enumerate() {
            let CheckpointKPointSolution::Collinear {
                bases,
                solutions,
                up_occupations,
                down_occupations,
                ..
            } = &point.solution
            else {
                return Err(MaterialKernelError::SecondVariationRequiresScalarBands {
                    point: point_index,
                });
            };
            if up_occupations == down_occupations
                || bases.up != bases.down
                || solutions.up != solutions.down
            {
                return Err(MaterialKernelError::SecondVariationRequiresSpinDegenerate {
                    point: point_index,
                });
            }
            let first = FirstVariationSubspace::select(
                window,
                &solutions.up.eigenvalues,
                &solutions.up.eigenvectors,
            )?;
            let blocks = second_variation_blocks_from_potential(&bases.up, potential)?;
            let core_blocks = build_static_core_exchange_site_blocks(
                &bases.up,
                core_sidecars,
                StaticCoreExchangeMode::SpinOrbitResolved,
            )?;
            let core_feedback = core_blocks
                .iter()
                .map(|block| block.matrix().clone())
                .collect::<Vec<_>>();
            let (fixed_hamiltonian, projected_core) = second_variation_frame_matrices(
                &bases.up.compiled,
                &first,
                &blocks,
                &core_feedback,
            )?;
            first_variations.push(first);
            site_blocks.push(blocks);
            site_feedback.push(core_feedback);
            fixed_hamiltonians.push(fixed_hamiltonian);
            resolved_core_feedback.push(projected_core);
        }
        Ok(CheckpointSecondVariationFrame {
            scalar: scalar.clone(),
            first_variations,
            site_blocks,
            site_feedback,
            fixed_hamiltonians,
            resolved_core_feedback,
        })
    }

    /// Assemble the spherical static core-exchange operator on each retained
    /// scalar KH global basis.
    pub fn scalar_static_core_exchange_feedback(
        &self,
        scalar: &CheckpointBandSolution,
        core_sidecars: &[CoreShellOrbitals],
    ) -> Result<Vec<Collinear<DenseHermitianMatrix>>, MaterialKernelError> {
        scalar
            .points
            .iter()
            .enumerate()
            .map(|(point, point_solution)| {
                let CheckpointKPointSolution::Collinear {
                    bases,
                    up_occupations,
                    down_occupations,
                    ..
                } = &point_solution.solution
                else {
                    return Err(MaterialKernelError::FeedbackRequiresScalar { point });
                };
                if up_occupations == down_occupations || bases.up != bases.down {
                    return Err(MaterialKernelError::SecondVariationRequiresSpinDegenerate {
                        point,
                    });
                }
                let sites = build_static_core_exchange_site_blocks(
                    &bases.up,
                    core_sidecars,
                    StaticCoreExchangeMode::ScalarAverage,
                )?;
                let scalar_sites = sites
                    .iter()
                    .map(|site| site.scalar_block())
                    .collect::<Result<Vec<_>, _>>()?;
                let global = assemble_scalar_site_operator(&bases.up.compiled, &scalar_sites)?;
                Ok(Collinear::new(global.clone(), global))
            })
            .collect()
    }

    /// Materialize one local-potential radial/basis problem without entering
    /// the DFT XC/core cache used by [`ScfPhysics`].
    pub fn materialize_checkpoint_one_particle(
        &self,
        potential: &RegionalPotential,
        requested: &ScfBasis,
    ) -> Result<CheckpointOneParticle, MaterialKernelError> {
        let meshes = self.channel_meshes(requested)?;
        let extended = build_extended_checkpoint_core_potentials(
            potential,
            &self.geometry,
            &self.nuclear_charges,
            &meshes,
            CorePotentialContinuationSpec::default(),
        )?;
        let basis = self.materialize_nonspectral_basis(potential, requested, &extended)?;
        Ok(CheckpointOneParticle {
            potential: potential.clone(),
            basis,
        })
    }

    /// Synthesize the valence-only regional density of a retained band solve.
    pub fn synthesize_bands(
        &self,
        bands: &CheckpointBandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, MaterialKernelError> {
        self.synthesize(bands, occupations)
    }

    /// Synthesize the complete Pauli density of bands produced by a prepared
    /// SOC second-variation frame.
    pub fn synthesize_second_variation_bands(
        &self,
        bands: &CheckpointBandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, MaterialKernelError> {
        if occupations.len() != bands.states.len() {
            return Err(MaterialKernelError::OccupationCount {
                expected: bands.states.len(),
                actual: occupations.len(),
            });
        }
        let density_layout = match &bands.density_layout {
            Some(layout) => layout.clone(),
            None => self.density_layout(&bands.points)?,
        };
        let first = bands
            .points
            .first()
            .ok_or(MaterialKernelError::EmptyKPointSet)?;
        let CheckpointKPointSolution::Collinear { bases, .. } = &first.solution else {
            return Err(MaterialKernelError::InconsistentRelativityRoute);
        };
        let points = bands
            .points
            .iter()
            .map(|point| match &point.solution {
                CheckpointKPointSolution::Collinear {
                    bases: point_bases,
                    solutions,
                    up_occupations,
                    down_occupations,
                    ..
                } if up_occupations == down_occupations && point_bases.up == point_bases.down => {
                    Ok(SecondVariationKPoint {
                        weight: point.weight,
                        compiled: &point_bases.up.compiled,
                        solutions: Collinear::new(&solutions.up, &solutions.down),
                        occupations: &occupations[up_occupations.clone()],
                    })
                }
                _ => Err(MaterialKernelError::InconsistentRelativityRoute),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let density = synthesize_second_variation_valence_density(
            self.geometry.clone(),
            density_layout,
            &bases.up.density_sites,
            &points,
        )?;
        self.project_density_muffin_tin_layout(&density)
    }

    fn solve_points_with_weights(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        points: &[[f64; 3]],
        weights: &[f64],
        relativity: ScfRelativity,
        core_orthogonal: &[CoreShellOrbitals],
    ) -> Result<CheckpointBandSolution, MaterialKernelError> {
        if points.is_empty() {
            return Err(MaterialKernelError::EmptyKPointSet);
        }
        if points.len() != weights.len()
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight <= 0.0)
            || (weights.iter().sum::<f64>() - 1.0).abs() > 1.0e-12
        {
            return Err(MaterialKernelError::InvalidKPointWeights);
        }
        if relativity == ScfRelativity::SpinorFirstVariation {
            if !core_orthogonal.is_empty() {
                return Err(MaterialKernelError::InconsistentRelativityRoute);
            }
            return self.solve_spinor_points(potential, basis, points, weights);
        }
        self.require_collinear_route(potential)?;
        let site_inputs = self.scalar_site_inputs(potential, basis)?;
        let radial_reference = if core_orthogonal.is_empty() {
            None
        } else {
            Some(self.scalar_site_inputs(&self.frozen_potential, basis)?)
        };
        let interstitial = collinear_interstitial_potential(potential)?;
        let mut solved_points = Vec::with_capacity(points.len());
        let mut states = Vec::new();

        for (&k, &weight) in points.iter().zip(weights) {
            let envelope = self.plane_wave_envelope(k, basis.plane_wave_cutoff)?;
            let bases = if core_orthogonal.is_empty() {
                build_collinear_scalar_iteration_bases(
                    &envelope,
                    &self.geometry,
                    Collinear::new(&site_inputs.up, &site_inputs.down),
                )?
            } else {
                Collinear::new(
                    crate::build_core_orthogonal_scalar_iteration_basis(
                        &envelope,
                        &self.geometry,
                        &radial_reference.as_ref().unwrap().up,
                        &site_inputs.up,
                        core_orthogonal,
                    )?,
                    crate::build_core_orthogonal_scalar_iteration_basis(
                        &envelope,
                        &self.geometry,
                        &radial_reference.as_ref().unwrap().down,
                        &site_inputs.down,
                        core_orthogonal,
                    )?,
                )
            };
            let scalar = crate::solve_collinear_scalar_k_point(
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
                        return Err(MaterialKernelError::SecondVariationDropsLowerBands {
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
                        &[],
                        None,
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
                    eigenproblems: Collinear::new(scalar.up.eigenproblem, scalar.down.eigenproblem),
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
            reduction: None,
            density_layout: Some(potential.scalar().interstitial().layout().clone()),
            symmetry_transforms: Vec::new(),
            spacegroup_number: None,
        })
    }

    fn solve_spinor_points(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
        points: &[[f64; 3]],
        weights: &[f64],
    ) -> Result<CheckpointBandSolution, MaterialKernelError> {
        let site_inputs = self.spinor_site_inputs(potential, basis)?;
        let interstitial = potential.to_lapw_interstitial()?;
        let mut solved_points = Vec::with_capacity(points.len());
        let mut states = Vec::new();
        for (&k, &weight) in points.iter().zip(weights) {
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
                    eigenproblem: solved.eigenproblem,
                    solution: solved.solution,
                    occupations: start..end,
                },
            });
        }
        Ok(CheckpointBandSolution {
            points: solved_points,
            states,
            reduction: None,
            density_layout: Some(potential.scalar().interstitial().layout().clone()),
            symmetry_transforms: Vec::new(),
            spacegroup_number: None,
        })
    }

    pub fn scalar_site_inputs(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<Collinear<Vec<ScalarSiteInput>>, MaterialKernelError> {
        self.require_potential_site_count(potential)?;
        let build_spin = |spin: usize| {
            self.sites
                .iter()
                .enumerate()
                .map(|(site_index, site)| {
                    let template = if spin == 0 { &site.up } else { &site.down };
                    if template.route != RadialRoute::ScalarKoellingHarmon {
                        return Err(MaterialKernelError::ScalarRadialEquation {
                            site: site.id.clone(),
                            spin,
                            route: template.route,
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
                        .ok_or_else(|| MaterialKernelError::MissingMonopole(site.id.clone()))?;
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
                .collect::<Result<Vec<_>, MaterialKernelError>>()
        };
        Ok(Collinear::new(build_spin(0)?, build_spin(1)?))
    }

    pub fn spinor_site_inputs(
        &self,
        potential: &RegionalPotential,
        basis: &ScfBasis,
    ) -> Result<Vec<SpinorSiteInput>, MaterialKernelError> {
        self.require_potential_site_count(potential)?;
        let source_is_dirac = self.sites.iter().all(|site| {
            [&site.up, &site.down]
                .into_iter()
                .all(|source| source.route == RadialRoute::Dirac)
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
                        if template.route != RadialRoute::Dirac {
                            return Err(MaterialKernelError::SpinorRadialEquation {
                                site: site.id.clone(),
                                spin,
                                route: template.route,
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
                    .ok_or_else(|| MaterialKernelError::MissingMonopole(site.id.clone()))?;
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

    fn site_index(&self, site: &str) -> Result<usize, MaterialKernelError> {
        self.sites
            .iter()
            .position(|candidate| candidate.id == site)
            .ok_or_else(|| MaterialKernelError::UnknownCoreSite(site.to_owned()))
    }

    pub fn refine_spectral_basis(
        &self,
        requested: &ScfBasis,
        one_particle: &CheckpointOneParticle,
        bands: &CheckpointBandSolution,
        occupations: &[f64],
        chemical_potential: Hartree,
        relativity: ScfRelativity,
    ) -> Result<Option<CheckpointOneParticle>, MaterialKernelError> {
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
                .ok_or_else(|| MaterialKernelError::MissingProvisionalChannel {
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

    pub fn validate_band_cog_projection_keys(
        &self,
        spectral: &[&ScfChannelRecipe],
        relativity: ScfRelativity,
    ) -> Result<(), MaterialKernelError> {
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
                    return Err(MaterialKernelError::AmbiguousBandCogProjection {
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
    ) -> Result<Vec<PdosEnergySample>, MaterialKernelError> {
        if occupations.len() != bands.states.len() {
            return Err(MaterialKernelError::OccupationCount {
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
                    ..
                } => {
                    if matches!(recipe.identity, ScfChannelIdentity::Kappa { .. }) {
                        return Err(MaterialKernelError::KappaBandCogUnavailableInScalar {
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
                            return Err(MaterialKernelError::InconsistentRelativityRoute);
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
                    eigenproblem: _,
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
        cutoff: InverseBohr,
    ) -> Result<PlaneWaveEnvelope, MaterialKernelError> {
        production_plane_wave_envelope(self.reciprocal, fractional_k, cutoff)
    }

    fn require_potential_site_count(
        &self,
        potential: &RegionalPotential,
    ) -> Result<(), MaterialKernelError> {
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
                return Err(MaterialKernelError::PotentialComponentSiteCount {
                    component,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }

    fn require_density_site_count(
        &self,
        density: &RegionalDensity,
    ) -> Result<(), MaterialKernelError> {
        self.require_topology_site_count(
            "restart density charge",
            density.charge().muffin_tins().len(),
        )?;
        for (component, field) in ["mx", "my", "mz"].into_iter().zip(density.magnetization()) {
            self.require_topology_site_count(component, field.muffin_tins().len())?;
        }
        Ok(())
    }

    fn require_topology_site_count(
        &self,
        component: &'static str,
        actual: usize,
    ) -> Result<(), MaterialKernelError> {
        let expected = self.sites.len();
        if actual != expected {
            return Err(MaterialKernelError::TopologySiteCount {
                component,
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn require_collinear_route(
        &self,
        potential: &RegionalPotential,
    ) -> Result<(), MaterialKernelError> {
        let transverse_rms = [
            potential.magnetic()[0].residual_rms()?,
            potential.magnetic()[1].residual_rms()?,
        ];
        if transverse_rms
            .iter()
            .any(|&rms| rms > TRANSVERSE_FIELD_TOLERANCE)
        {
            return Err(MaterialKernelError::TransversePotentialUnsupported {
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
    ) -> Result<(), MaterialKernelError> {
        self.require_collinear_route(potential)?;
        let magnetic = potential
            .magnetic()
            .iter()
            .map(RegionalScalarField::residual_rms)
            .collect::<Result<Vec<_>, _>>()?;
        if self.sites.iter().any(|site| !site.nonmagnetic_scalar)
            || magnetic.iter().any(|&rms| rms > TRANSVERSE_FIELD_TOLERANCE)
        {
            return Err(MaterialKernelError::SecondVariationRequiresNonmagneticScalar);
        }
        Ok(())
    }

    fn synthesize(
        &self,
        bands: &CheckpointBandSolution,
        occupations: &[f64],
    ) -> Result<RegionalDensity, MaterialKernelError> {
        if occupations.len() != bands.states.len() {
            return Err(MaterialKernelError::OccupationCount {
                expected: bands.states.len(),
                actual: occupations.len(),
            });
        }
        let density_layout = match &bands.density_layout {
            Some(layout) => layout.clone(),
            None => self.density_layout(&bands.points)?,
        };
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
                            Err(MaterialKernelError::InconsistentRelativityRoute)
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
                            Err(MaterialKernelError::InconsistentRelativityRoute)
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
                let density = RegionalDensity::new(charge, [zero.clone(), zero, longitudinal])?;
                self.project_density_muffin_tin_layout(&density)
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
                            Err(MaterialKernelError::InconsistentRelativityRoute)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let density = synthesize_full_spinor_valence_density(
                    self.geometry.clone(),
                    density_layout,
                    &basis.density_sites,
                    &spinor_points,
                )?;
                self.project_density_muffin_tin_layout(&density)
            }
        }
    }

    fn project_density_muffin_tin_layout(
        &self,
        density: &RegionalDensity,
    ) -> Result<RegionalDensity, MaterialKernelError> {
        let charge =
            project_scalar_muffin_tin_layout(density.charge(), self.frozen_potential.scalar())?;
        let magnetization = density
            .magnetization()
            .iter()
            .zip(self.frozen_potential.magnetic())
            .map(|(source, template)| project_scalar_muffin_tin_layout(source, template))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("density and potential both have three magnetic components");
        Ok(RegionalDensity::new(charge, magnetization)?)
    }

    fn extended_core_meshes(
        &self,
        requested_site: usize,
        states: &[crate::ScfCoreState],
    ) -> Result<Vec<ExponentialMesh>, MaterialKernelError> {
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
                extend_core_mesh(&site.up.mesh, outer_radius).map_err(Into::into)
            })
            .collect()
    }

    fn solve_initial_core_density(
        &self,
        template: &RegionalDensity,
        config: &ScfConfig,
        extended: &[crate::BuiltExtendedCorePotential],
    ) -> Result<RegionalDensity, MaterialKernelError> {
        Ok(self.solve_initial_core(template, config, extended)?.density)
    }

    fn solve_initial_core(
        &self,
        template: &RegionalDensity,
        config: &ScfConfig,
        extended: &[crate::BuiltExtendedCorePotential],
    ) -> Result<RegionalCoreResult, MaterialKernelError> {
        let mut density = template.zero_like();
        let mut eigenvalue_sum = Hartree(0.0);
        let mut sites = Vec::new();
        let mut orbitals = Vec::new();
        for site in config
            .core_sites
            .iter()
            .filter(|site| !site.states.is_empty())
        {
            let site_index = self
                .sites
                .iter()
                .position(|candidate| candidate.id == site.id)
                .ok_or_else(|| MaterialKernelError::UnknownCoreSite(site.id.clone()))?;
            let request = self.core_site_request(site_index, site)?;
            let contribution = solve_regional_core_site(
                template,
                &self.nuclear_charges,
                &request,
                &extended[site_index].potential,
            )?;
            density.add_scaled(1.0, &contribution.contribution.contribution.density)?;
            eigenvalue_sum += contribution.contribution.contribution.eigenvalue_sum;
            sites.push(contribution.contribution);
            orbitals.push(contribution.orbitals);
        }
        Ok(RegionalCoreResult {
            density,
            eigenvalue_sum,
            sites,
            orbitals,
        })
    }

    fn core_site_request(
        &self,
        site_index: usize,
        site: &ScfCoreSite,
    ) -> Result<CoreSiteRequest, MaterialKernelError> {
        let states = site
            .states
            .iter()
            .map(|requested| {
                Ok(CoreStateRequest {
                    state: CoreState::new(
                        requested.principal_quantum_number,
                        Kappa::new(requested.kappa)?,
                    )?,
                    occupation: requested.occupation,
                    spin: CoreSpinPartition::ClosedShellAverage,
                })
            })
            .collect::<Result<Vec<_>, MaterialKernelError>>()?;
        Ok(CoreSiteRequest {
            site_index,
            site_id: site.id.clone(),
            states,
        })
    }

    fn density_layout(
        &self,
        points: &[CheckpointKPoint],
    ) -> Result<FourierLayout, MaterialKernelError> {
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

impl ScfPhysics for MaterialKernel {
    type Error = MaterialKernelError;
    type OneParticle = CheckpointOneParticle;
    type BandSolution = CheckpointBandSolution;

    fn initial_density(&mut self, config: &ScfConfig) -> Result<RegionalDensity, Self::Error> {
        if let Some(density) = self.restart_density.clone() {
            if let Some((_, transforms, _)) = self.reduced_sampling(config.k_mesh)? {
                if config.relativity != ScfRelativity::Scalar {
                    return Err(MaterialKernelError::SymmetryReductionRequiresScalarNonmagnetic);
                }
                self.require_second_variation_route(&self.frozen_potential)?;
                self.require_symmetry_equivalent_config(config, &transforms)?;
                return self.project_scalar_density(&density, &transforms);
            }
            return Ok(density);
        }
        Ok(self.initial_density_components(config)?.total)
    }

    fn build_potential(
        &mut self,
        iteration: usize,
        density: &RegionalDensity,
        exchange_correlation: ScfExchangeCorrelation,
    ) -> Result<RegionalPotential, Self::Error> {
        let built = build_scf_potential(density, &self.nuclear_charges, exchange_correlation)?;
        let potential = built.potential.clone();
        self.core_potentials.insert(iteration, built);
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
        let built = self
            .core_potentials
            .get(&iteration)
            .ok_or(MaterialKernelError::MissingCoreContinuation(iteration))?;
        let site_index = self
            .sites
            .iter()
            .position(|candidate| candidate.id == site.id)
            .ok_or_else(|| MaterialKernelError::UnknownCoreSite(site.id.clone()))?;
        if site.states.is_empty() {
            return Ok(CoreContribution {
                site_id: site.id.clone(),
                density: built.source_density().zero_like(),
                eigenvalue_sum: Hartree(0.0),
            });
        }
        let meshes = self.extended_core_meshes(site_index, &site.states)?;
        let continued = build_extended_core_potentials(
            &built.electrostatic,
            &built.exchange_correlation,
            built.source_density(),
            &meshes,
            built.core_spec,
        )?;
        let request = self.core_site_request(site_index, site)?;
        Ok(solve_regional_core_site(
            built.source_density(),
            &self.nuclear_charges,
            &request,
            &continued[site_index].potential,
        )?
        .contribution
        .contribution)
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
        let Some((reduction, transforms, spacegroup_number)) = self.reduced_sampling(k_mesh)?
        else {
            return self.solve_points(
                &one_particle.potential,
                &one_particle.basis,
                &regular_k_points(k_mesh)?,
                relativity,
                &[],
            );
        };
        if relativity != ScfRelativity::Scalar {
            return Err(MaterialKernelError::SymmetryReductionRequiresScalarNonmagnetic);
        }
        self.require_second_variation_route(&one_particle.potential)?;
        let potential = self.symmetry_projected_potential(&one_particle.potential, &transforms)?;
        let points = reduction
            .irreducible_points
            .iter()
            .map(|point| point.fractional)
            .collect::<Vec<_>>();
        let weights = reduction
            .irreducible_points
            .iter()
            .map(|point| point.weight)
            .collect::<Vec<_>>();
        let mut bands = self.solve_points_with_weights(
            &potential,
            &one_particle.basis,
            &points,
            &weights,
            relativity,
            &[],
        )?;
        bands.reduction = Some(reduction);
        bands.symmetry_transforms = transforms;
        bands.spacegroup_number = spacegroup_number;
        Ok(bands)
    }

    fn band_states<'a>(&self, bands: &'a Self::BandSolution) -> &'a [BandState] {
        &bands.states
    }

    fn k_sampling_provenance(
        &self,
        bands: &Self::BandSolution,
        mesh: ScfKMesh,
    ) -> ScfKSamplingProvenance {
        let Some(reduction) = &bands.reduction else {
            return ScfKSamplingProvenance::Full {
                divisions: mesh.divisions,
                shift: mesh.shift,
                point_count: mesh.divisions.into_iter().product(),
            };
        };
        let ScfKReduction::Symmetry {
            symprec,
            include_time_reversal,
        } = mesh.reduction
        else {
            unreachable!("a reduced band solution requires symmetry sampling")
        };
        ScfKSamplingProvenance::SymmetryReduced {
            divisions: mesh.divisions,
            shift: mesh.shift,
            symprec,
            include_time_reversal,
            spacegroup_number: bands.spacegroup_number,
            full_point_count: reduction.full_points.len(),
            irreducible_point_count: reduction.irreducible_points.len(),
            multiplicities: reduction
                .irreducible_points
                .iter()
                .map(|point| point.multiplicity)
                .collect(),
            operation_count: bands.symmetry_transforms.len(),
            symmetry_provenance: "moyo".to_owned(),
        }
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

    fn project_output_density(
        &mut self,
        _iteration: usize,
        bands: &Self::BandSolution,
        density: RegionalDensity,
    ) -> Result<RegionalDensity, Self::Error> {
        if bands.symmetry_transforms.is_empty() {
            Ok(density)
        } else {
            self.project_scalar_density(&density, &bands.symmetry_transforms)
        }
    }

    fn energy_terms(
        &mut self,
        context: ScfEnergyContext<'_, Self::OneParticle, Self::BandSolution>,
    ) -> Result<ScfEnergyTerms, Self::Error> {
        self.core_potentials
            .get(&context.iteration)
            .map(|build| build.energy_terms)
            .ok_or(MaterialKernelError::MissingEnergyTerms(context.iteration))
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
        let solved = self.solve_points(
            &state.potential,
            &state.basis,
            &points,
            state.relativity,
            &[],
        )?;
        solved
            .points
            .into_iter()
            .map(|point| {
                let mut energies = point.energies;
                energies.sort_by(|left, right| left.get().total_cmp(&right.get()));
                if energies.len() < request.bands {
                    return Err(MaterialKernelError::TooFewBands {
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
        request: &crate::DosRequest,
    ) -> Result<RegularSpectrum, Self::Error> {
        let solved = self.solve_points(
            &state.potential,
            &state.basis,
            &regular_k_points(request.k_mesh)?,
            state.relativity,
            &[],
        )?;
        let band_count = solved
            .points
            .first()
            .map(|point| point.energies.len())
            .ok_or(MaterialKernelError::EmptyKPointSet)?;
        if solved
            .points
            .iter()
            .any(|point| point.energies.len() != band_count)
        {
            return Err(MaterialKernelError::InconsistentBandCount);
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
) -> Result<Vec<SiteSpinOrbitBlock>, MaterialKernelError> {
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

fn second_variation_blocks_from_potential(
    basis: &ScalarIterationBasis,
    potential: &RegionalPotential,
) -> Result<Vec<SiteSpinOrbitBlock>, MaterialKernelError> {
    if basis.radial_sites.len() != potential.scalar().muffin_tins().len() {
        return Err(MaterialKernelError::TopologySiteCount {
            component: "second-variation scalar potential",
            expected: basis.radial_sites.len(),
            actual: potential.scalar().muffin_tins().len(),
        });
    }
    basis
        .radial_sites
        .iter()
        .zip(&basis.recipe_sites)
        .zip(&basis.density_sites)
        .zip(potential.scalar().muffin_tins())
        .map(|(((radials, recipe), density), field)| {
            let monopole = field
                .field()
                .channel(0, 0)
                .ok_or_else(|| MaterialKernelError::MissingMonopole("second variation".into()))?;
            let spherical = monopole
                .iter()
                .map(|value| value.re / (4.0 * PI).sqrt())
                .collect::<Vec<_>>();
            let spin_orbit = SpexSpinOrbitPotential::new(&density.mesh, &spherical)?;
            let shells = radials
                .linearized
                .iter()
                .zip(&radials.local_orbitals)
                .map(|(linearized, locals)| {
                    let locals = locals
                        .iter()
                        .map(|local| local.orbital.clone())
                        .collect::<Vec<_>>();
                    spex_spin_orbit_radial_shell(&density.mesh, &spin_orbit, linearized, &locals)
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
    solution: &crate::SecondVariationResult,
) -> Result<Collinear<GeneralizedEigensolution>, MaterialKernelError> {
    let rows = solution.eigenvectors.rows() / 2;
    let columns = solution.eigenvectors.columns();
    let split = |spin: usize| -> Result<GeneralizedEigensolution, MaterialKernelError> {
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

fn second_variation_frame_matrices(
    compiled: &muffintin_envelope::CompiledBasis,
    first: &FirstVariationSubspace,
    site_blocks: &[SiteSpinOrbitBlock],
    site_feedback: &[DenseHermitianMatrix],
) -> Result<(DenseHermitianMatrix, DenseHermitianMatrix), MaterialKernelError> {
    if site_blocks.len() != compiled.site_count() || site_feedback.len() != compiled.site_count() {
        return Err(MaterialKernelError::SecondVariation(
            SecondVariationError::SiteCount {
                expected: compiled.site_count(),
                actual: site_blocks.len().min(site_feedback.len()),
            },
        ));
    }
    let dimension = 2 * first.eigenvalues.len();
    let mut fixed = vec![Complex64::default(); dimension * dimension];
    let mut core = vec![Complex64::default(); dimension * dimension];
    let bands = first.eigenvalues.len();
    for spin in 0..2 {
        for (band, energy) in first.eigenvalues.iter().enumerate() {
            let index = spin * bands + band;
            fixed[index * dimension + index] = Complex64::new(energy.get(), 0.0);
        }
    }
    for site in 0..compiled.site_count() {
        let projection = CompiledSiteProjection::scalar(compiled, site)?;
        let coefficients = projection.project_eigenvectors(&first.eigenvectors)?;
        let soc = project_site_soc_to_subspace(&site_blocks[site], &coefficients)?;
        let resolved = project_site_spinor_operator_to_subspace(
            projection.coordinate_count(),
            &site_feedback[site],
            &coefficients,
        )?;
        for ((fixed_value, core_value), (soc_value, resolved_value)) in
            fixed.iter_mut().zip(&mut core).zip(
                soc.to_host_row_major()
                    .into_iter()
                    .zip(resolved.to_host_row_major()),
            )
        {
            *fixed_value += soc_value + resolved_value;
            *core_value += resolved_value;
        }
    }
    Ok((
        DenseHermitianMatrix::from_host_row_major(dimension, Axis::Band, fixed)?,
        DenseHermitianMatrix::from_host_row_major(dimension, Axis::Band, core)?,
    ))
}

fn combine_muffin_tin_fields(
    left: &MuffinTinField,
    left_scale: f64,
    right: &MuffinTinField,
    right_scale: f64,
) -> Result<MuffinTinField, MaterialKernelError> {
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
) -> Result<InterstitialField, MaterialKernelError> {
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
) -> Result<RegionalScalarField, MaterialKernelError> {
    let mut result = left.zero_like();
    result.add_scaled(left_scale, left)?;
    result.add_scaled(right_scale, right)?;
    Ok(result)
}

fn project_scalar_muffin_tin_layout(
    source: &RegionalScalarField,
    template: &RegionalScalarField,
) -> Result<RegionalScalarField, MaterialKernelError> {
    let muffin_tins = source
        .muffin_tins()
        .iter()
        .zip(template.muffin_tins())
        .map(|(source, template)| {
            let channels = template.field().channels().map(|(channel, _)| {
                let values = source.field().channel(channel.l, channel.m).map_or_else(
                    || vec![Complex64::new(0.0, 0.0); source.mesh().len()],
                    <[Complex64]>::to_vec,
                );
                ((channel.l, channel.m), values)
            });
            let field = SphereField::new(template.field().convention(), channels)
                .map_err(crate::RegionalError::from)?;
            MuffinTinField::new(source.mesh().clone(), field).map_err(Into::into)
        })
        .collect::<Result<Vec<_>, MaterialKernelError>>()?;
    Ok(RegionalScalarField::new(
        source.geometry().clone(),
        muffin_tins,
        source.interstitial().clone(),
    )?)
}

fn collinear_interstitial_potential(
    potential: &RegionalPotential,
) -> Result<Collinear<InterstitialPotential>, MaterialKernelError> {
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

fn spex_bound_channel_mismatch(bound: &SpexBoundSpinorChannel) -> MaterialKernelError {
    MaterialKernelError::SpexMaterialChannelMismatch {
        site: bound.requested.site.clone(),
        identity: bound.requested.identity,
        l: bound.l,
        treatment: bound.requested.treatment,
        derivative_order: bound.requested.derivative_order,
        energy: bound.resolved.energy.get(),
    }
}

fn channel_generator_error(
    recipe: &ScfChannelRecipe,
    source: LinearizationEnergyError,
) -> MaterialKernelError {
    MaterialKernelError::ChannelGenerator {
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
) -> Result<Vec<f64>, MaterialKernelError> {
    let monopole = potential.scalar().muffin_tins()[site_index]
        .field()
        .channel(0, 0)
        .ok_or_else(|| MaterialKernelError::MissingMonopole(site.to_owned()))?;
    monopole
        .iter()
        .enumerate()
        .map(|(radial, value)| {
            if value.im.abs() > TRANSVERSE_FIELD_TOLERANCE * (1.0 + value.re.abs()) {
                return Err(MaterialKernelError::NonRealMonopole {
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
) -> Result<Vec<f64>, MaterialKernelError> {
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
) -> Result<(), MaterialKernelError> {
    if range.len() != projections.len() || range.end > states.len() || range.end > occupations.len()
    {
        return Err(MaterialKernelError::BandProjectionCount {
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
) -> Result<InitialOccupationSolution, MaterialKernelError> {
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

pub fn regular_k_points(mesh: ScfKMesh) -> Result<Vec<[f64; 3]>, MaterialKernelError> {
    let count = mesh
        .divisions
        .into_iter()
        .try_fold(1_usize, usize::checked_mul)
        .ok_or(MaterialKernelError::KPointCountOverflow)?;
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
    cutoff: InverseBohr,
) -> Result<PlaneWaveEnvelope, MaterialKernelError> {
    if fractional_k.iter().any(|value| !value.is_finite()) {
        return Err(MaterialKernelError::NonFiniteKPoint(fractional_k));
    }
    let k = fractional_to_reciprocal(fractional_k, reciprocal.basis());
    let k_norm = squared_norm(k.map(InverseBohr::get)).sqrt();
    let cutoff_value = cutoff.get();
    let candidates = reciprocal.enumerate(InverseBohr(cutoff_value + k_norm))?;
    let waves = candidates
        .into_iter()
        .filter(|g| {
            let wave = std::array::from_fn(|axis| k[axis].get() + g.cartesian[axis].get());
            squared_norm(wave) <= cutoff_value * cutoff_value * (1.0 + 64.0 * f64::EPSILON)
        })
        .map(|g| PlaneWave::new(k, g))
        .collect::<Vec<_>>();
    if waves.is_empty() {
        return Err(MaterialKernelError::EmptyPlaneWaveBasis {
            k: fractional_k,
            cutoff,
        });
    }
    Ok(PlaneWaveEnvelope::new(waves))
}

pub fn production_density_layout(
    reciprocal: ReciprocalLattice,
    k_mesh: ScfKMesh,
    cutoff: InverseBohr,
) -> Result<FourierLayout, MaterialKernelError> {
    let points = regular_k_points(k_mesh)?;
    if points.is_empty() {
        return Err(MaterialKernelError::EmptyKPointSet);
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

pub fn g_vector(reciprocal: ReciprocalLattice, index: [i32; 3]) -> GVector {
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

fn equivalent_channel_recipe(source: &ScfChannelRecipe, target: &ScfChannelRecipe) -> bool {
    source.identity == target.identity
        && source.treatment == target.treatment
        && source.derivative_order == target.derivative_order
        && source.generator == target.generator
        && source.seed == target.seed
        && source.provenance == target.provenance
}

/// Checkpoint conversion or concrete DFT-kernel failure.
#[derive(Debug, Error)]
pub enum MaterialKernelError {
    #[error(transparent)]
    ChannelKappa(#[from] ChannelKappaError),
    #[error(transparent)]
    PotentialBuild(#[from] ScfPotentialBuildError),
    #[error(transparent)]
    Lattice(#[from] LatticeError),
    #[error(transparent)]
    Mesh(#[from] MeshError),
    #[error(transparent)]
    Fourier(#[from] FourierFieldError),
    #[error(transparent)]
    Regional(#[from] crate::RegionalError),
    #[error(transparent)]
    Scalar(#[from] ScalarBuilderError),
    #[error(transparent)]
    CoreOrthogonalization(#[from] crate::ScalarCoreOrthogonalizationError),
    #[error(transparent)]
    SpinorBuilder(#[from] SpinorBuilderError),
    #[error(transparent)]
    SpinorFirstVariation(#[from] SpinorFirstVariationError),
    #[error(transparent)]
    Density(#[from] DensityError),
    #[error(transparent)]
    Occupation(#[from] OccupationError),
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
    Dirac(#[from] DiracError),
    #[error(transparent)]
    CorePotential(#[from] CorePotentialBuildError),
    #[error(transparent)]
    CoreStation(#[from] CoreStationError),
    #[error(transparent)]
    StaticCoreExchange(#[from] StaticCoreSiteExchangeError),
    #[error(transparent)]
    SymmetryDetection(#[from] MoyoDetectionError),
    #[error(transparent)]
    KMeshReduction(#[from] KMeshReductionError),
    #[error(transparent)]
    SymmetryTransform(#[from] SymmetryTransformError),
    #[error("material-kernel {component} has {actual} sites, expected {expected}")]
    TopologySiteCount {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("potential component {component} has {actual} sites, expected {expected}")]
    PotentialComponentSiteCount {
        component: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error(
        "scalar route needs Koelling-Harmon input at site {site:?}, spin {spin}; got {route:?}"
    )]
    ScalarRadialEquation {
        site: String,
        spin: usize,
        route: RadialRoute,
    },
    #[error(
        "full-spinor route needs fully relativistic input at site {site:?}, spin {spin}; got {route:?}"
    )]
    SpinorRadialEquation {
        site: String,
        spin: usize,
        route: RadialRoute,
    },
    #[error(
        "SPEX material channel site={site:?}, identity={identity:?}, l={l}, treatment={treatment:?}, derivative_order={derivative_order}, energy={energy} is not bound exactly to the runtime basis"
    )]
    SpexMaterialChannelMismatch {
        site: String,
        identity: ScfChannelIdentity,
        l: u32,
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
    EmptyPlaneWaveBasis { k: [f64; 3], cutoff: InverseBohr },
    #[error("regular k-point set is empty")]
    EmptyKPointSet,
    #[error("k-point weights must be finite, positive, match the points, and sum to one")]
    InvalidKPointWeights,
    #[error("regular k-point count overflows usize")]
    KPointCountOverflow,
    #[error("second variation requires a nonmagnetic scalar checkpoint and potential")]
    SecondVariationRequiresNonmagneticScalar,
    #[error("symmetry-reduced SCF currently requires the scalar nonmagnetic route")]
    SymmetryReductionRequiresScalarNonmagnetic,
    #[error("symmetry maps site {source_site:?} to {target_site:?} with incompatible radial data")]
    SymmetryEquivalentSiteMismatch {
        source_site: String,
        target_site: String,
    },
    #[error(
        "symmetry maps site {source_site:?} to {target_site:?} with incompatible channel/core recipes"
    )]
    SymmetryEquivalentRecipeMismatch {
        source_site: String,
        target_site: String,
    },
    #[error(
        "second-variation window starts at {start}; runtime requires start=0 so occupied lower scalar bands are not dropped"
    )]
    SecondVariationDropsLowerBands { start: usize },
    #[error("one band solution mixed scalar and spinor k-point routes")]
    InconsistentRelativityRoute,
    #[error("received {actual} band-feedback matrices for {expected} k points")]
    FeedbackPointCount { actual: usize, expected: usize },
    #[error("band feedback at k-point {point} requires a spinor first-variation solution")]
    FeedbackRequiresSpinor { point: usize },
    #[error("band feedback at k-point {point} requires independent scalar KH spin channels")]
    FeedbackRequiresScalar { point: usize },
    #[error("scalar orbital guess contains {actual} k points, expected {expected}")]
    ScalarOrbitalGuessPointCount { actual: usize, expected: usize },
    #[error(
        "scalar orbital guess at k-point {point} does not share the current radial and constrained LAPW space"
    )]
    ScalarOrbitalGuessSpace { point: usize },
    #[error(
        "feedback solve at k-point {point} returned {actual} bands, expected retained count {expected}"
    )]
    FeedbackBandCount {
        point: usize,
        actual: usize,
        expected: usize,
    },
    #[error("SOC second variation at k-point {point} requires a scalar KH band solution")]
    SecondVariationRequiresScalarBands { point: usize },
    #[error("SOC second variation at k-point {point} requires identical scalar KH spin channels")]
    SecondVariationRequiresSpinDegenerate { point: usize },
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
    #[error("core site {0:?} is requested more than once")]
    DuplicateCoreSite(String),
    #[error("HF core bootstrap density geometry differs from the material kernel")]
    DensityGeometryMismatch,
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
