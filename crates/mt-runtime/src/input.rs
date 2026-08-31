use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

use muffintin_core::{Hartree, InverseBohr};
use serde::{Deserialize, Serialize};

use crate::error::{finite, fraction, nonempty, positive};
use crate::{ChannelEnergyGenerator, ChannelTreatment, InputError, InputValidationError};

/// Stable discriminator written at the start of every runtime input.
pub const INPUT_FORMAT: &str = "libmuffintin-input";
/// Only runtime input schema version currently supported.
pub const INPUT_VERSION: u32 = 3;

/// A complete workflow input in the currently supported schema.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Input {
    pub format: String,
    pub version: u32,
    /// Checkpoint path relative to the input file that names this workflow.
    /// The pre-rename `snapshot` key is still accepted on read.
    #[serde(alias = "snapshot")]
    pub checkpoint: PathBuf,
    pub workflow: Workflow,
    pub task: BTreeMap<String, Task>,
}

impl Input {
    pub fn new(checkpoint: PathBuf, workflow: Workflow, task: BTreeMap<String, Task>) -> Self {
        Self {
            format: INPUT_FORMAT.to_owned(),
            version: INPUT_VERSION,
            checkpoint,
            workflow,
            task,
        }
    }

    /// Check the header, every DTO, and all workflow graph invariants.
    pub fn validate(&self) -> Result<(), InputError> {
        if self.format != INPUT_FORMAT {
            return Err(InputError::InvalidFormat {
                expected: INPUT_FORMAT,
                found: self.format.clone(),
            });
        }
        if self.version == 1 {
            return Err(InputError::V1MigrationRequired);
        }
        if self.version == 2 {
            return Err(InputError::V2MigrationRequired);
        }
        if self.version != INPUT_VERSION {
            return Err(InputError::UnsupportedVersion {
                format: INPUT_FORMAT,
                supported: INPUT_VERSION,
                found: self.version,
            });
        }
        validate_checkpoint_path(&self.checkpoint)?;
        validate_task_sets(self)?;

        let positions: BTreeMap<&str, usize> = self
            .workflow
            .tasks
            .iter()
            .enumerate()
            .map(|(index, id)| (id.as_str(), index))
            .collect();

        for (index, task_id) in self.workflow.tasks.iter().enumerate() {
            let task = &self.task[task_id];
            task.validate(task_id)?;
            if let Some(source) = task.source() {
                validate_source(self, &positions, index, task_id, task, source)?;
            }
        }
        Ok(())
    }
}

/// Ordered workflow declaration. The map under `[task]` does not define execution order.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Workflow {
    pub tasks: Vec<String>,
}

/// Runtime task declaration, discriminated by the TOML `kind` field.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Task {
    #[serde(rename = "dft-scf", rename_all = "kebab-case")]
    DftScf {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        electron_count: f64,
        k_mesh: KMesh,
        #[serde(default)]
        symmetry: Symmetry,
        basis: Basis,
        occupations: Occupations,
        xc: ExchangeCorrelation,
        mixing: Mixing,
        relativity: Relativity,
        convergence: Convergence,
    },
    #[serde(rename = "dft-bands", rename_all = "kebab-case")]
    DftBands {
        source: String,
        bands: u32,
        path: Vec<BandPathPoint>,
    },
    #[serde(rename = "dft-dos", rename_all = "kebab-case")]
    DftDos {
        source: String,
        k_mesh: KMesh,
        energy_window: EnergyWindow,
        points: usize,
        broadening: f64,
    },
}

impl Task {
    pub const fn kind(&self) -> TaskKind {
        match self {
            Self::DftScf { .. } => TaskKind::DftScf,
            Self::DftBands { .. } => TaskKind::DftBands,
            Self::DftDos { .. } => TaskKind::DftDos,
        }
    }

    pub fn source(&self) -> Option<&str> {
        match self {
            Self::DftScf { source, .. } => source.as_deref(),
            Self::DftBands { source, .. } | Self::DftDos { source, .. } => Some(source),
        }
    }

    fn validate(&self, task_id: &str) -> Result<(), InputValidationError> {
        let base = format!("task.{task_id}");
        match self {
            Self::DftScf {
                electron_count,
                k_mesh,
                symmetry,
                basis,
                occupations,
                mixing,
                relativity,
                convergence,
                ..
            } => {
                positive(format!("{base}.electron-count"), *electron_count)?;
                k_mesh.validate(&format!("{base}.k-mesh"))?;
                symmetry.validate(&format!("{base}.symmetry"))?;
                basis.validate(&format!("{base}.basis"))?;
                occupations.validate(&format!("{base}.occupations"))?;
                mixing.validate(&format!("{base}.mixing"))?;
                relativity.validate(&format!("{base}.relativity"))?;
                convergence.validate(&format!("{base}.convergence"))?;
            }
            Self::DftBands { bands, path, .. } => {
                if *bands == 0 {
                    return Err(InputValidationError::Zero {
                        path: format!("{base}.bands"),
                    });
                }
                if path.len() < 2 {
                    return Err(InputValidationError::TooShort {
                        path: format!("{base}.path"),
                        minimum: 2,
                        actual: path.len(),
                    });
                }
                for (index, point) in path.iter().enumerate() {
                    point.validate(&format!("{base}.path[{index}]"))?;
                }
            }
            Self::DftDos {
                k_mesh,
                energy_window,
                points,
                broadening,
                ..
            } => {
                k_mesh.validate(&format!("{base}.k-mesh"))?;
                energy_window.validate(&format!("{base}.energy-window"))?;
                if *points < 2 {
                    return Err(InputValidationError::TooShort {
                        path: format!("{base}.points"),
                        minimum: 2,
                        actual: *points,
                    });
                }
                positive(format!("{base}.broadening"), *broadening)?;
            }
        }
        Ok(())
    }
}

/// Closed set of executable task kinds in the workflow input.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    DftScf,
    DftBands,
    DftDos,
}

impl fmt::Display for TaskKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::DftScf => "dft-scf",
            Self::DftBands => "dft-bands",
            Self::DftDos => "dft-dos",
        };
        formatter.write_str(name)
    }
}

/// Regular full-Brillouin-zone mesh in reciprocal fractional coordinates.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct KMesh {
    pub mesh: [u32; 3],
    pub shift: [f64; 3],
}

/// Optional crystal-symmetry reduction of an SCF regular k mesh.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Symmetry {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_symprec")]
    pub symprec: f64,
    #[serde(default = "default_true")]
    pub include_time_reversal: bool,
}

impl Default for Symmetry {
    fn default() -> Self {
        Self {
            enabled: false,
            symprec: default_symprec(),
            include_time_reversal: true,
        }
    }
}

impl Symmetry {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        if self.enabled {
            positive(format!("{path}.symprec"), self.symprec)?;
        }
        Ok(())
    }
}

const fn default_symprec() -> f64 {
    1.0e-5
}

const fn default_true() -> bool {
    true
}

impl KMesh {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        for (axis, &count) in self.mesh.iter().enumerate() {
            if count == 0 {
                return Err(InputValidationError::Zero {
                    path: format!("{path}.mesh[{axis}]"),
                });
            }
        }
        for (axis, &shift) in self.shift.iter().enumerate() {
            finite(format!("{path}.shift[{axis}]"), shift)?;
        }
        Ok(())
    }
}

/// Minimal LAPW basis controls shared by the DFT workflow.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Basis {
    pub l_max: u32,
    /// Explicit task-level override. `None` preserves recipe/default precedence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_generator: Option<ChannelEnergyGenerator>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<PathBuf>,
    pub envelope: BasisEnvelope,
    /// Per-scope treatment rows. An empty vector remains distinct from an absent key.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub channels: BTreeMap<String, BTreeMap<ChannelTreatment, Vec<String>>>,
}

impl Basis {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        if self.l_max == 0 {
            return Err(InputValidationError::Zero {
                path: format!("{path}.l-max"),
            });
        }
        if let Some(recipe) = &self.recipe {
            validate_recipe_path(recipe)?;
        }
        self.envelope.validate(&format!("{path}.envelope"))?;
        Ok(())
    }
}

/// Basis envelope family. V2 currently admits only the plane-wave route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BasisEnvelopeKind {
    PlaneWave,
}

/// Envelope controls orthogonal to the radial channel table.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct BasisEnvelope {
    pub kind: BasisEnvelopeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub g_cutoff: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_cutoff: Option<f64>,
}

impl BasisEnvelope {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        match (self.g_cutoff, self.energy_cutoff) {
            (Some(value), None) => positive(format!("{path}.g-cutoff"), value),
            (None, Some(value)) => positive(format!("{path}.energy-cutoff"), value),
            (None, None) => Err(InputValidationError::MissingPlaneWaveCutoff {
                path: path.to_owned(),
            }),
            (Some(_), Some(_)) => Err(InputValidationError::ConflictingPlaneWaveCutoffs {
                path: path.to_owned(),
            }),
        }
    }

    pub(crate) fn normalized_cutoff(&self) -> InverseBohr {
        match (self.g_cutoff, self.energy_cutoff) {
            (Some(value), None) => InverseBohr(value),
            (None, Some(value)) => InverseBohr::from_kinetic_cutoff(Hartree(value)),
            _ => unreachable!("validated envelope has exactly one cutoff"),
        }
    }
}

/// Typed occupation model and its energy-scale parameter, in Hartree.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Occupations {
    #[serde(rename = "fermi-dirac", rename_all = "kebab-case")]
    FermiDirac { temperature: f64 },
    #[serde(rename = "gaussian", rename_all = "kebab-case")]
    Gaussian { width: f64 },
}

impl Occupations {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        match self {
            Self::FermiDirac { temperature } => {
                positive(format!("{path}.temperature"), *temperature)
            }
            Self::Gaussian { width } => positive(format!("{path}.width"), *width),
        }
    }
}

/// Exchange-correlation functional used to construct the local potential.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum ExchangeCorrelation {
    #[serde(rename = "lda-pw92")]
    LdaPw92 {
        #[serde(default, rename = "noncollinear-route")]
        noncollinear_route: NoncollinearXcRoute,
    },
    #[serde(rename = "pbe")]
    Pbe {
        #[serde(default, rename = "noncollinear-route")]
        noncollinear_route: NoncollinearXcRoute,
    },
}

/// Noncollinear reduction used by the pointwise XC kernel.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NoncollinearXcRoute {
    /// Rotate derivatives into the instantaneous magnetization direction.
    #[default]
    LocalSpinFrame,
    /// Differentiate the scalar fields `(n +/- |m|)/2` directly.
    MagnetizationField,
}

/// Density-mixing algorithm and history controls.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Mixing {
    #[serde(rename = "linear", rename_all = "kebab-case")]
    Linear { beta: f64 },
    #[serde(rename = "broyden2", rename_all = "kebab-case")]
    Broyden2 { beta: f64, history: usize },
    #[serde(rename = "pulay-anderson", rename_all = "kebab-case")]
    PulayAnderson { beta: f64, history: usize },
}

impl Mixing {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        let (beta, history) = match self {
            Self::Linear { beta } => (*beta, None),
            Self::Broyden2 { beta, history } | Self::PulayAnderson { beta, history } => {
                (*beta, Some(*history))
            }
        };
        fraction(format!("{path}.beta"), beta)?;
        if let Some(history) = history
            && history < 2
        {
            return Err(InputValidationError::TooShort {
                path: format!("{path}.history"),
                minimum: 2,
                actual: history,
            });
        }
        Ok(())
    }
}

/// Relativistic one-particle formulation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum Relativity {
    #[serde(rename = "scalar")]
    Scalar {},
    #[serde(rename = "soc-second-variation", rename_all = "kebab-case")]
    SocSecondVariation { band_window: [usize; 2] },
    #[serde(rename = "spinor-first-variation")]
    SpinorFirstVariation {},
}

impl Relativity {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        if let Self::SocSecondVariation { band_window } = self
            && band_window[0] >= band_window[1]
        {
            return Err(InputValidationError::InvalidRange {
                path: format!("{path}.band-window"),
                minimum: band_window[0] as f64,
                maximum: band_window[1] as f64,
            });
        }
        Ok(())
    }
}

/// Stopping criteria for an SCF task.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Convergence {
    pub energy_tolerance: f64,
    pub density_tolerance: f64,
    pub max_iterations: usize,
}

impl Convergence {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        positive(format!("{path}.energy-tolerance"), self.energy_tolerance)?;
        positive(format!("{path}.density-tolerance"), self.density_tolerance)?;
        if self.max_iterations == 0 {
            return Err(InputValidationError::Zero {
                path: format!("{path}.max-iterations"),
            });
        }
        Ok(())
    }
}

/// One labeled reciprocal-space point in an ordered band path.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct BandPathPoint {
    pub label: String,
    pub k: [f64; 3],
}

impl BandPathPoint {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        nonempty(format!("{path}.label"), &self.label)?;
        for (axis, &coordinate) in self.k.iter().enumerate() {
            finite(format!("{path}.k[{axis}]"), coordinate)?;
        }
        Ok(())
    }
}

/// DOS sampling window in Hartree relative to the checkpoint energy zero.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct EnergyWindow {
    pub minimum: f64,
    pub maximum: f64,
}

impl EnergyWindow {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        finite(format!("{path}.minimum"), self.minimum)?;
        finite(format!("{path}.maximum"), self.maximum)?;
        if self.minimum >= self.maximum {
            return Err(InputValidationError::InvalidRange {
                path: path.to_owned(),
                minimum: self.minimum,
                maximum: self.maximum,
            });
        }
        Ok(())
    }
}

/// Parse and validate deterministic input TOML without touching the filesystem.
pub fn parse_input_toml(text: &str) -> Result<Input, InputError> {
    #[derive(Deserialize)]
    struct Header {
        format: String,
        version: u32,
    }

    let header: Header = toml::from_str(text)?;
    if header.format != INPUT_FORMAT {
        return Err(InputError::InvalidFormat {
            expected: INPUT_FORMAT,
            found: header.format,
        });
    }
    if header.version == 1 {
        return Err(InputError::V1MigrationRequired);
    }
    if header.version == 2 {
        return Err(InputError::V2MigrationRequired);
    }
    if header.version != INPUT_VERSION {
        return Err(InputError::UnsupportedVersion {
            format: INPUT_FORMAT,
            supported: INPUT_VERSION,
            found: header.version,
        });
    }
    let input: Input = toml::from_str(text)?;
    input.validate()?;
    Ok(input)
}

/// Serialize a validated input as deterministic pretty TOML.
pub fn input_to_toml(input: &Input) -> Result<String, InputError> {
    input.validate()?;
    let mut text = toml::to_string_pretty(input)?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

fn validate_checkpoint_path(path: &std::path::Path) -> Result<(), InputValidationError> {
    if path.as_os_str().is_empty() {
        return Err(InputValidationError::EmptyCheckpointPath);
    }
    if path.is_absolute() {
        return Err(InputValidationError::AbsoluteCheckpointPath {
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn validate_recipe_path(path: &std::path::Path) -> Result<(), InputValidationError> {
    if path.as_os_str().is_empty() {
        return Err(InputValidationError::EmptyRecipePath);
    }
    if path.is_absolute() {
        return Err(InputValidationError::AbsoluteRecipePath {
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn validate_task_sets(input: &Input) -> Result<(), InputValidationError> {
    if input.workflow.tasks.is_empty() {
        return Err(InputValidationError::EmptyWorkflow);
    }
    let mut workflow_ids = BTreeSet::new();
    for id in &input.workflow.tasks {
        validate_task_id(id)?;
        if !workflow_ids.insert(id.as_str()) {
            return Err(InputValidationError::DuplicateTaskId { id: id.clone() });
        }
        if !input.task.contains_key(id) {
            return Err(InputValidationError::MissingTaskBlock { id: id.clone() });
        }
    }
    for id in input.task.keys() {
        validate_task_id(id)?;
        if !workflow_ids.contains(id.as_str()) {
            return Err(InputValidationError::OrphanTaskBlock { id: id.clone() });
        }
    }
    Ok(())
}

fn validate_task_id(id: &str) -> Result<(), InputValidationError> {
    let mut characters = id.chars();
    let valid_first = characters.next().is_some_and(|c| c.is_ascii_alphabetic());
    let valid_rest = characters.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if valid_first && valid_rest {
        Ok(())
    } else {
        Err(InputValidationError::InvalidTaskId { id: id.to_owned() })
    }
}

fn validate_source(
    input: &Input,
    positions: &BTreeMap<&str, usize>,
    consumer_index: usize,
    task_id: &str,
    task: &Task,
    source: &str,
) -> Result<(), InputValidationError> {
    let Some((source_task, output)) = parse_source(source) else {
        return Err(InputValidationError::InvalidSource {
            task_id: task_id.to_owned(),
            source_ref: source.to_owned(),
        });
    };
    let Some(&source_index) = positions.get(source_task) else {
        return Err(InputValidationError::MissingSourceTask {
            task_id: task_id.to_owned(),
            source_task: source_task.to_owned(),
        });
    };
    if source_index >= consumer_index {
        return Err(InputValidationError::ForwardSource {
            task_id: task_id.to_owned(),
            source_task: source_task.to_owned(),
        });
    }
    let producer = &input.task[source_task];
    if producer.kind() != TaskKind::DftScf || output != "state" {
        return Err(InputValidationError::IncompatibleSource {
            task_id: task_id.to_owned(),
            task_kind: task.kind(),
            source_ref: source.to_owned(),
        });
    }
    Ok(())
}

pub(crate) fn parse_source(source: &str) -> Option<(&str, &str)> {
    let (task, output) = source.split_once('.')?;
    if task.is_empty()
        || output.is_empty()
        || output.contains('.')
        || validate_task_id(task).is_err()
        || validate_task_id(output).is_err()
    {
        None
    } else {
        Some((task, output))
    }
}
