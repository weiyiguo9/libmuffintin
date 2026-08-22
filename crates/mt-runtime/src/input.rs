use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{finite, fraction, nonempty, positive};
use crate::{InputError, InputValidationError};

/// Stable discriminator written at the start of every runtime input.
pub const INPUT_FORMAT: &str = "libmuffintin-input";
/// Only runtime input schema version currently supported.
pub const INPUT_VERSION: u32 = 1;

/// A complete V1 workflow input.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct InputV1 {
    pub format: String,
    pub version: u32,
    /// Snapshot path relative to the input file that names this workflow.
    pub snapshot: PathBuf,
    pub workflow: WorkflowV1,
    pub task: BTreeMap<String, TaskV1>,
}

impl InputV1 {
    pub fn new(snapshot: PathBuf, workflow: WorkflowV1, task: BTreeMap<String, TaskV1>) -> Self {
        Self {
            format: INPUT_FORMAT.to_owned(),
            version: INPUT_VERSION,
            snapshot,
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
        if self.version != INPUT_VERSION {
            return Err(InputError::UnsupportedVersion {
                format: INPUT_FORMAT,
                supported: INPUT_VERSION,
                found: self.version,
            });
        }
        validate_snapshot_path(&self.snapshot)?;
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
pub struct WorkflowV1 {
    pub tasks: Vec<String>,
}

/// Runtime task declaration, discriminated by the TOML `kind` field.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum TaskV1 {
    #[serde(rename = "dft-scf", rename_all = "kebab-case")]
    DftScf {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source: Option<String>,
        electron_count: f64,
        k_mesh: KMeshV1,
        basis: BasisV1,
        occupations: OccupationsV1,
        xc: ExchangeCorrelationV1,
        mixing: MixingV1,
        relativity: RelativityV1,
        convergence: ConvergenceV1,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        core_states: Vec<CoreStateV1>,
    },
    #[serde(rename = "dft-bands", rename_all = "kebab-case")]
    DftBands {
        source: String,
        bands: u32,
        path: Vec<BandPathPointV1>,
    },
    #[serde(rename = "dft-dos", rename_all = "kebab-case")]
    DftDos {
        source: String,
        k_mesh: KMeshV1,
        energy_window: EnergyWindowV1,
        points: usize,
        broadening: f64,
    },
}

impl TaskV1 {
    pub const fn kind(&self) -> TaskKindV1 {
        match self {
            Self::DftScf { .. } => TaskKindV1::DftScf,
            Self::DftBands { .. } => TaskKindV1::DftBands,
            Self::DftDos { .. } => TaskKindV1::DftDos,
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
                basis,
                occupations,
                xc,
                mixing,
                relativity,
                convergence,
                core_states,
                ..
            } => {
                positive(format!("{base}.electron-count"), *electron_count)?;
                k_mesh.validate(&format!("{base}.k-mesh"))?;
                basis.validate(&format!("{base}.basis"))?;
                occupations.validate(&format!("{base}.occupations"))?;
                xc.validate();
                mixing.validate(&format!("{base}.mixing"))?;
                relativity.validate(&format!("{base}.relativity"))?;
                convergence.validate(&format!("{base}.convergence"))?;
                for (index, state) in core_states.iter().enumerate() {
                    state.validate(&format!("{base}.core-states[{index}]"))?;
                }
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

/// Closed set of executable task kinds in input V1.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKindV1 {
    DftScf,
    DftBands,
    DftDos,
}

impl fmt::Display for TaskKindV1 {
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
pub struct KMeshV1 {
    pub mesh: [u32; 3],
    pub shift: [f64; 3],
}

impl KMeshV1 {
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
pub struct BasisV1 {
    pub plane_wave_cutoff: f64,
    pub l_max: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub local_orbitals: Vec<LocalOrbitalV1>,
}

impl BasisV1 {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        positive(format!("{path}.plane-wave-cutoff"), self.plane_wave_cutoff)?;
        if self.l_max == 0 {
            return Err(InputValidationError::Zero {
                path: format!("{path}.l-max"),
            });
        }
        for (index, orbital) in self.local_orbitals.iter().enumerate() {
            orbital.validate(&format!("{path}.local-orbitals[{index}]"))?;
        }
        Ok(())
    }
}

/// Construction route for a signed-kappa spinor local orbital.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LocalOrbitalKindV1 {
    Lo,
    Hdlo,
}

/// One site-resolved local-orbital request in Hartree units.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct LocalOrbitalV1 {
    pub site: String,
    pub kappa: i32,
    pub energy: f64,
    pub kind: LocalOrbitalKindV1,
}

impl LocalOrbitalV1 {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        nonempty(format!("{path}.site"), &self.site)?;
        if self.kappa == 0 {
            return Err(InputValidationError::Zero {
                path: format!("{path}.kappa"),
            });
        }
        finite(format!("{path}.energy"), self.energy)
    }
}

/// Typed occupation model and its energy-scale parameter, in Hartree.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum OccupationsV1 {
    #[serde(rename = "fermi-dirac", rename_all = "kebab-case")]
    FermiDirac { temperature: f64 },
    #[serde(rename = "gaussian", rename_all = "kebab-case")]
    Gaussian { width: f64 },
}

impl OccupationsV1 {
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
pub enum ExchangeCorrelationV1 {
    #[serde(rename = "lda-pw92")]
    LdaPw92 {},
    #[serde(rename = "pbe")]
    Pbe {},
}

impl ExchangeCorrelationV1 {
    const fn validate(self) {}
}

/// Density-mixing algorithm and history controls.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum MixingV1 {
    #[serde(rename = "linear", rename_all = "kebab-case")]
    Linear { beta: f64 },
    #[serde(rename = "broyden2", rename_all = "kebab-case")]
    Broyden2 { beta: f64, history: usize },
    #[serde(rename = "pulay-anderson", rename_all = "kebab-case")]
    PulayAnderson { beta: f64, history: usize },
}

impl MixingV1 {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        let (beta, history) = match self {
            Self::Linear { beta } => (*beta, None),
            Self::Broyden2 { beta, history } | Self::PulayAnderson { beta, history } => {
                (*beta, Some(*history))
            }
        };
        fraction(format!("{path}.beta"), beta)?;
        if let Some(history) = history {
            if history < 2 {
                return Err(InputValidationError::TooShort {
                    path: format!("{path}.history"),
                    minimum: 2,
                    actual: history,
                });
            }
        }
        Ok(())
    }
}

/// Relativistic one-particle formulation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum RelativityV1 {
    #[serde(rename = "scalar")]
    Scalar {},
    #[serde(rename = "spex-second-variation", rename_all = "kebab-case")]
    SpexSecondVariation { band_window: [usize; 2] },
    #[serde(rename = "spinor-first-variation")]
    SpinorFirstVariation {},
}

impl RelativityV1 {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        if let Self::SpexSecondVariation { band_window } = self {
            if band_window[0] >= band_window[1] {
                return Err(InputValidationError::InvalidRange {
                    path: format!("{path}.band-window"),
                    minimum: band_window[0] as f64,
                    maximum: band_window[1] as f64,
                });
            }
        }
        Ok(())
    }
}

/// Stopping criteria for an SCF task.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct ConvergenceV1 {
    pub energy_tolerance: f64,
    pub density_tolerance: f64,
    pub max_iterations: usize,
}

impl ConvergenceV1 {
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

/// Requested occupied bound-core channel.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct CoreStateV1 {
    pub site: String,
    pub principal_quantum_number: u32,
    pub kappa: i32,
    pub occupation: f64,
}

impl CoreStateV1 {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        nonempty(format!("{path}.site"), &self.site)?;
        if self.principal_quantum_number == 0 {
            return Err(InputValidationError::Zero {
                path: format!("{path}.principal-quantum-number"),
            });
        }
        if self.kappa == 0 {
            return Err(InputValidationError::Zero {
                path: format!("{path}.kappa"),
            });
        }
        positive(format!("{path}.occupation"), self.occupation)
    }
}

/// One labeled reciprocal-space point in an ordered band path.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct BandPathPointV1 {
    pub label: String,
    pub k: [f64; 3],
}

impl BandPathPointV1 {
    fn validate(&self, path: &str) -> Result<(), InputValidationError> {
        nonempty(format!("{path}.label"), &self.label)?;
        for (axis, &coordinate) in self.k.iter().enumerate() {
            finite(format!("{path}.k[{axis}]"), coordinate)?;
        }
        Ok(())
    }
}

/// DOS sampling window in Hartree relative to the snapshot energy zero.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct EnergyWindowV1 {
    pub minimum: f64,
    pub maximum: f64,
}

impl EnergyWindowV1 {
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

/// Parse and validate deterministic V1 input TOML without touching the filesystem.
pub fn parse_input_toml(text: &str) -> Result<InputV1, InputError> {
    let input: InputV1 = toml::from_str(text)?;
    input.validate()?;
    Ok(input)
}

/// Serialize a validated V1 input as deterministic pretty TOML.
pub fn input_to_toml(input: &InputV1) -> Result<String, InputError> {
    input.validate()?;
    let mut text = toml::to_string_pretty(input)?;
    if !text.ends_with('\n') {
        text.push('\n');
    }
    Ok(text)
}

fn validate_snapshot_path(path: &std::path::Path) -> Result<(), InputValidationError> {
    if path.as_os_str().is_empty() {
        return Err(InputValidationError::EmptySnapshotPath);
    }
    if path.is_absolute() {
        return Err(InputValidationError::AbsoluteSnapshotPath {
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn validate_task_sets(input: &InputV1) -> Result<(), InputValidationError> {
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
    input: &InputV1,
    positions: &BTreeMap<&str, usize>,
    consumer_index: usize,
    task_id: &str,
    task: &TaskV1,
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
    if producer.kind() != TaskKindV1::DftScf || output != "state" {
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
