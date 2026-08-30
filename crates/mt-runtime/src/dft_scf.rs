//! Canonical single-task DFT SCF plan and concrete staged runtime session.

use std::path::Path;

use muffintin_dft::{
    CheckpointBandSolution, CheckpointOneParticle, ContinueStep, CoreStep, EnergyRecord,
    LapwDensityAssembly, LapwSolution, MaterialKernelError, OccupationStep, RegionalDensity,
    RegionalDensityStep, RegionalPotential, RegionalPotentialStep, ScfConfig, ScfError,
    ScfIterationDiagnostic, ScfLoop, ScfState,
};
use muffintin_io::CheckpointV2;
use num_complex::Complex64;
use thiserror::Error;

use crate::checkpoint_physics::CheckpointPhysicsError;
use crate::runner::{PreparedWorkflow, load_input_path};
use crate::{CheckpointPhysics, InputError, SingleDftScfConfigError, single_dft_scf_config};

type ConcreteLapw = LapwSolution<CheckpointOneParticle, CheckpointBandSolution>;
type ConcreteOccupations = OccupationStep<CheckpointOneParticle, CheckpointBandSolution>;
type ConcreteDensity = LapwDensityAssembly<CheckpointOneParticle, CheckpointBandSolution>;

/// Representation-neutral interstitial Pauli components for language bindings.
#[derive(Clone, Debug, PartialEq)]
pub struct DftRegionalFourier {
    pub g_vectors: Vec<[i32; 3]>,
    /// Component-major values in charge/scalar, x, y, z order.
    pub components: [Vec<Complex64>; 4],
}

fn density_fourier(density: &RegionalDensity) -> DftRegionalFourier {
    regional_fourier([
        density.charge(),
        &density.magnetization()[0],
        &density.magnetization()[1],
        &density.magnetization()[2],
    ])
}

fn potential_fourier(potential: &RegionalPotential) -> DftRegionalFourier {
    regional_fourier([
        potential.scalar(),
        &potential.magnetic()[0],
        &potential.magnetic()[1],
        &potential.magnetic()[2],
    ])
}

fn regional_fourier(fields: [&muffintin_dft::RegionalScalarField; 4]) -> DftRegionalFourier {
    DftRegionalFourier {
        g_vectors: fields[0]
            .interstitial()
            .layout()
            .vectors()
            .iter()
            .map(|vector| vector.index)
            .collect(),
        components: fields.map(|field| field.interstitial().field().coefficients().to_vec()),
    }
}

/// A validated workflow containing exactly one `dft-scf` task.
#[derive(Clone, Debug, PartialEq)]
pub struct DftScfPlan {
    workflow: PreparedWorkflow,
    checkpoint: CheckpointV2,
    config: ScfConfig,
}

impl DftScfPlan {
    pub const fn workflow(&self) -> &PreparedWorkflow {
        &self.workflow
    }
    pub const fn checkpoint(&self) -> &CheckpointV2 {
        &self.checkpoint
    }
    pub const fn config(&self) -> &ScfConfig {
        &self.config
    }

    /// Construct an independent concrete material-kernel session.
    pub fn session(&self) -> Result<DftScfSession, DftScfError> {
        let physics = CheckpointPhysics::new(&self.checkpoint)?;
        let scf = ScfLoop::new(self.config.clone(), None)?;
        Ok(DftScfSession { physics, scf })
    }

    /// Bind the plan to an already selected checkpoint context.
    pub fn session_for_checkpoint(
        &self,
        checkpoint: &CheckpointV2,
    ) -> Result<DftScfSession, DftScfError> {
        if checkpoint != &self.checkpoint {
            return Err(DftScfError::CheckpointContextMismatch);
        }
        self.session()
    }
}

/// Concrete checkpoint-backed staged SCF session.
#[derive(Debug)]
pub struct DftScfSession {
    physics: CheckpointPhysics,
    scf: ScfLoop,
}

#[derive(Debug)]
pub struct DftRegionalDensity(RegionalDensityStep);

impl DftRegionalDensity {
    pub const fn iteration(&self) -> usize {
        self.0.iteration()
    }
    pub const fn density(&self) -> &RegionalDensity {
        self.0.density()
    }
    pub fn export_interstitial(&self) -> DftRegionalFourier {
        density_fourier(self.0.density())
    }
}

#[derive(Debug)]
pub struct DftRegionalPotentialStep(RegionalPotentialStep);

impl DftRegionalPotentialStep {
    pub const fn iteration(&self) -> usize {
        self.0.iteration()
    }
    pub const fn potential(&self) -> &RegionalPotential {
        self.0.potential()
    }
    pub fn export_interstitial(&self) -> DftRegionalFourier {
        potential_fourier(self.0.potential())
    }
}

#[derive(Debug)]
pub struct DftCoreStep(CoreStep);

impl DftCoreStep {
    pub const fn iteration(&self) -> usize {
        self.0.iteration()
    }
    pub const fn core_density(&self) -> &RegionalDensity {
        self.0.core_density()
    }
    pub const fn core_eigenvalue_sum(&self) -> muffintin_core::Hartree {
        self.0.core_eigenvalue_sum()
    }
}

#[derive(Debug)]
pub struct DftLapwSolution(ConcreteLapw);

impl DftLapwSolution {
    pub const fn iteration(&self) -> usize {
        self.0.iteration()
    }
}

#[derive(Debug)]
pub struct DftOccupations(ConcreteOccupations);

impl DftOccupations {
    pub const fn iteration(&self) -> usize {
        self.0.iteration()
    }
    pub const fn chemical_potential(&self) -> muffintin_core::Hartree {
        self.0.chemical_potential()
    }
    pub fn values(&self) -> &[f64] {
        self.0.occupations()
    }
}

#[derive(Debug)]
pub struct DftLapwDensityAssembly(ConcreteDensity);

impl DftLapwDensityAssembly {
    pub const fn iteration(&self) -> usize {
        self.0.iteration()
    }
    pub const fn density(&self) -> &RegionalDensity {
        self.0.density()
    }
    pub fn export_interstitial(&self) -> DftRegionalFourier {
        density_fourier(self.0.density())
    }
}

#[derive(Debug)]
pub struct DftEnergyRecord(EnergyRecord);

impl DftEnergyRecord {
    pub const fn iteration(&self) -> usize {
        self.0.iteration()
    }
    pub const fn chemical_potential(&self) -> muffintin_core::Hartree {
        self.0.chemical_potential()
    }
    pub const fn total_energy(&self) -> muffintin_core::Hartree {
        self.0.energy().total
    }
    pub const fn density_rms(&self) -> f64 {
        self.0.density_rms()
    }
    pub const fn energy_change(&self) -> Option<muffintin_core::Hartree> {
        self.0.energy_change()
    }
}

#[derive(Debug)]
pub enum DftConvergenceDecision {
    Continue(Box<ContinueStep>),
    Converged(Box<DftScfResult>),
}

impl DftConvergenceDecision {
    pub const fn is_converged(&self) -> bool {
        matches!(self, Self::Converged(_))
    }

    pub fn iteration(&self) -> usize {
        match self {
            Self::Continue(step) => step.iteration(),
            Self::Converged(result) => result.state.iterations(),
        }
    }

    pub fn result(&self) -> Option<&DftScfResult> {
        match self {
            Self::Continue(_) => None,
            Self::Converged(result) => Some(result.as_ref()),
        }
    }
}

/// Converged restart checkpoint and the exact state/diagnostic history that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct DftScfResult {
    pub checkpoint: CheckpointV2,
    pub state: ScfState,
}

impl DftScfResult {
    pub fn diagnostics(&self) -> &[ScfIterationDiagnostic] {
        &self.state.diagnostics
    }
}

impl DftScfSession {
    pub fn diagnostics(&self) -> &[ScfIterationDiagnostic] {
        self.scf.diagnostics()
    }

    pub fn initial_density(&mut self) -> Result<DftRegionalDensity, DftScfError> {
        Ok(DftRegionalDensity(
            self.scf.initial_density(&mut self.physics.kernel)?,
        ))
    }

    pub fn potential(
        &mut self,
        density: DftRegionalDensity,
    ) -> Result<DftRegionalPotentialStep, DftScfError> {
        Ok(DftRegionalPotentialStep(
            self.scf.potential(&mut self.physics.kernel, density.0)?,
        ))
    }

    pub fn core(
        &mut self,
        potential: DftRegionalPotentialStep,
    ) -> Result<DftCoreStep, DftScfError> {
        Ok(DftCoreStep(
            self.scf.core(&mut self.physics.kernel, potential.0)?,
        ))
    }

    pub fn lapw(&mut self, core: DftCoreStep) -> Result<DftLapwSolution, DftScfError> {
        Ok(DftLapwSolution(
            self.scf.lapw(&mut self.physics.kernel, core.0)?,
        ))
    }

    pub fn occupations(
        &mut self,
        solution: DftLapwSolution,
    ) -> Result<DftOccupations, DftScfError> {
        Ok(DftOccupations(
            self.scf.occupations(&mut self.physics.kernel, solution.0)?,
        ))
    }

    pub fn density(
        &mut self,
        occupations: DftOccupations,
    ) -> Result<DftLapwDensityAssembly, DftScfError> {
        Ok(DftLapwDensityAssembly(
            self.scf.density(&mut self.physics.kernel, occupations.0)?,
        ))
    }

    pub fn energy(
        &mut self,
        density: DftLapwDensityAssembly,
    ) -> Result<DftEnergyRecord, DftScfError> {
        Ok(DftEnergyRecord(
            self.scf.energy(&mut self.physics.kernel, density.0)?,
        ))
    }

    pub fn convergence(
        &mut self,
        energy: DftEnergyRecord,
    ) -> Result<DftConvergenceDecision, DftScfError> {
        match self.scf.convergence(energy.0)? {
            muffintin_dft::ConvergenceDecision::Continue(step) => {
                Ok(DftConvergenceDecision::Continue(Box::new(step)))
            }
            muffintin_dft::ConvergenceDecision::Converged(state) => {
                let checkpoint = self.physics.restart_checkpoint(&state)?;
                Ok(DftConvergenceDecision::Converged(Box::new(DftScfResult {
                    checkpoint,
                    state,
                })))
            }
        }
    }

    /// Mixing accepts only the explicit `Continue` proof from convergence.
    pub fn mix(
        &mut self,
        decision: DftConvergenceDecision,
    ) -> Result<DftRegionalDensity, DftScfError> {
        let DftConvergenceDecision::Continue(step) = decision else {
            return Err(DftScfError::CannotMixConverged);
        };
        Ok(DftRegionalDensity(self.scf.mix(*step)?))
    }

    /// Execute the same staged transitions exposed above until convergence.
    pub fn run(mut self) -> Result<DftScfResult, DftScfError> {
        let mut density = self.initial_density()?;
        loop {
            let potential = self.potential(density)?;
            let core = self.core(potential)?;
            let lapw = self.lapw(core)?;
            let occupations = self.occupations(lapw)?;
            let assembled = self.density(occupations)?;
            let energy = self.energy(assembled)?;
            match self.convergence(energy)? {
                DftConvergenceDecision::Converged(result) => return Ok(*result),
                decision @ DftConvergenceDecision::Continue(_) => {
                    density = self.mix(decision)?;
                }
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum DftScfError {
    #[error(transparent)]
    Input(Box<InputError>),
    #[error(transparent)]
    SingleTask(Box<SingleDftScfConfigError>),
    #[error("DFT SCF plan checkpoint does not match the selected physics context")]
    CheckpointContextMismatch,
    #[error("a converged SCF decision cannot be mixed")]
    CannotMixConverged,
    #[error(transparent)]
    CheckpointPhysics(Box<CheckpointPhysicsError>),
    #[error(transparent)]
    InvalidConfig(Box<muffintin_dft::ScfConfigError>),
    #[error(transparent)]
    Scf(Box<ScfError<MaterialKernelError>>),
}

impl From<InputError> for DftScfError {
    fn from(error: InputError) -> Self {
        Self::Input(Box::new(error))
    }
}

impl From<SingleDftScfConfigError> for DftScfError {
    fn from(error: SingleDftScfConfigError) -> Self {
        Self::SingleTask(Box::new(error))
    }
}

impl From<CheckpointPhysicsError> for DftScfError {
    fn from(error: CheckpointPhysicsError) -> Self {
        Self::CheckpointPhysics(Box::new(error))
    }
}

impl From<muffintin_dft::ScfConfigError> for DftScfError {
    fn from(error: muffintin_dft::ScfConfigError) -> Self {
        Self::InvalidConfig(Box::new(error))
    }
}

impl From<ScfError<MaterialKernelError>> for DftScfError {
    fn from(error: ScfError<MaterialKernelError>) -> Self {
        Self::Scf(Box::new(error))
    }
}

/// Load and validate the canonical single-task DFT SCF plan.
pub fn prepare_dft_scf(path: impl AsRef<Path>) -> Result<DftScfPlan, DftScfError> {
    let workflow = load_input_path(path)?;
    let config = single_dft_scf_config(&workflow)?;
    let checkpoint = workflow.checkpoint.clone();
    Ok(DftScfPlan {
        workflow,
        checkpoint,
        config,
    })
}

/// Canonical global entry point: exactly `prepare_dft_scf(path)?.session()?.run()`.
pub fn run_dft_scf(path: impl AsRef<Path>) -> Result<DftScfResult, DftScfError> {
    prepare_dft_scf(path)?.session()?.run()
}
