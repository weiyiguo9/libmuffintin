use std::fs;
use std::path::{Path, PathBuf};

use muffintin_core::Hartree;
use muffintin_dft::{
    BandPathPoint, BandPathRequest, BandPathResult, DosRequest, DosResult, FirstVariationWindow,
    ScfBasis, ScfConfig, ScfConvergence, ScfCoreSite, ScfCoreState, ScfExchangeCorrelation,
    ScfKMesh, ScfLocalOrbital, ScfLocalOrbitalKind, ScfMixing, ScfOccupations, ScfPhysics,
    ScfRelativity, ScfState, run_band_path, run_dos, run_scf,
};
use muffintin_io::{SnapshotV1, snapshot_from_toml};

use crate::input::parse_source;
use crate::{
    ExchangeCorrelationV1, InputError, InputV1, KMeshV1, LocalOrbitalKindV1, MixingV1,
    OccupationsV1, RelativityV1, TaskV1, parse_input_toml,
};

/// A source reference resolved against the workflow's stable task order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedSource {
    pub task_index: usize,
    pub task_id: String,
    pub output: String,
}

/// One validated task plus its resolved optional source edge.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTask {
    pub id: String,
    pub task: TaskV1,
    pub source: Option<PreparedSource>,
}

/// A fully validated workflow and loaded immutable input snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedWorkflow {
    pub snapshot: SnapshotV1,
    pub tasks: Vec<PreparedTask>,
}

/// One completed production task output.
#[derive(Clone, Debug, PartialEq)]
pub enum TaskResult {
    Scf(Box<ScfState>),
    Bands(BandPathResult),
    Dos(DosResult),
}

impl TaskResult {
    pub fn scf_state(&self) -> Option<&ScfState> {
        match self {
            Self::Scf(state) => Some(state.as_ref()),
            Self::Bands(_) | Self::Dos(_) => None,
        }
    }
}

/// Results in the same stable order as [`PreparedWorkflow::tasks`].
#[derive(Clone, Debug, PartialEq)]
pub struct WorkflowResult {
    pub tasks: Vec<TaskResult>,
}

/// Validate and resolve an already decoded input and snapshot without filesystem access.
pub fn prepare_input(
    input: &InputV1,
    snapshot: SnapshotV1,
) -> Result<PreparedWorkflow, InputError> {
    input.validate()?;
    snapshot.validate().map_err(InputError::InvalidSnapshot)?;

    let mut tasks = Vec::with_capacity(input.workflow.tasks.len());
    for id in &input.workflow.tasks {
        let task = input.task[id].clone();
        let source = task.source().map(|source| {
            let (task_id, output) = parse_source(source)
                .expect("InputV1::validate accepted only syntactically valid sources");
            let task_index = input
                .workflow
                .tasks
                .iter()
                .position(|candidate| candidate == task_id)
                .expect("InputV1::validate accepted only existing source tasks");
            PreparedSource {
                task_index,
                task_id: task_id.to_owned(),
                output: output.to_owned(),
            }
        });
        tasks.push(PreparedTask {
            id: id.clone(),
            task,
            source,
        });
    }
    Ok(PreparedWorkflow { snapshot, tasks })
}

/// Read one input and its relative snapshot, then prepare the workflow.
pub fn load_input_path(path: impl AsRef<Path>) -> Result<PreparedWorkflow, InputError> {
    let input_path = path.as_ref();
    let input_text = fs::read_to_string(input_path).map_err(|source| InputError::ReadInput {
        path: input_path.to_owned(),
        source,
    })?;
    let input = parse_input_toml(&input_text)?;
    let snapshot_path = resolve_snapshot_path(input_path, &input.snapshot);
    let snapshot_text =
        fs::read_to_string(&snapshot_path).map_err(|source| InputError::ReadSnapshot {
            path: snapshot_path.clone(),
            source,
        })?;
    let snapshot = snapshot_from_toml(&snapshot_text).map_err(|source| InputError::Snapshot {
        path: snapshot_path,
        source,
    })?;
    prepare_input(&input, snapshot)
}

/// Execute a prepared workflow with one material kernel shared by every task.
///
/// Tasks run strictly in [`PreparedWorkflow::tasks`] order. A source edge can
/// consume only the `ScfState` produced by its resolved earlier SCF task.
pub fn execute_prepared_with<P: ScfPhysics>(
    workflow: &PreparedWorkflow,
    physics: &mut P,
) -> Result<WorkflowResult, InputError> {
    let mut results = Vec::with_capacity(workflow.tasks.len());
    for task in &workflow.tasks {
        let source = source_state(task, &results)?;
        let result = match &task.task {
            TaskV1::DftScf { .. } => {
                let config = scf_config(&task.id, &task.task, &workflow.snapshot)?;
                let state = run_scf(physics, &config, source).map_err(|source| {
                    InputError::TaskExecution {
                        task_id: task.id.clone(),
                        kind: task.task.kind(),
                        source: Box::new(source),
                    }
                })?;
                TaskResult::Scf(Box::new(state))
            }
            TaskV1::DftBands { bands, path, .. } => {
                let state = source.ok_or_else(|| InputError::UnavailableScfSource {
                    task_id: task.id.clone(),
                })?;
                let request = BandPathRequest {
                    bands: usize::try_from(*bands).expect("u32 band count fits usize"),
                    points: path
                        .iter()
                        .map(|point| BandPathPoint {
                            label: point.label.clone(),
                            k: point.k,
                        })
                        .collect(),
                };
                let bands = run_band_path(physics, state, &request).map_err(|source| {
                    InputError::TaskExecution {
                        task_id: task.id.clone(),
                        kind: task.task.kind(),
                        source: Box::new(source),
                    }
                })?;
                TaskResult::Bands(bands)
            }
            TaskV1::DftDos {
                k_mesh,
                energy_window,
                points,
                broadening,
                ..
            } => {
                let state = source.ok_or_else(|| InputError::UnavailableScfSource {
                    task_id: task.id.clone(),
                })?;
                let request = DosRequest {
                    k_mesh: map_k_mesh(*k_mesh),
                    edges: uniform_edges(energy_window.minimum, energy_window.maximum, *points),
                    broadening: Hartree(*broadening),
                };
                let dos = run_dos(physics, state, &request).map_err(|source| {
                    InputError::TaskExecution {
                        task_id: task.id.clone(),
                        kind: task.task.kind(),
                        source: Box::new(source),
                    }
                })?;
                TaskResult::Dos(dos)
            }
        };
        results.push(result);
    }
    Ok(WorkflowResult { tasks: results })
}

fn source_state<'a>(
    task: &PreparedTask,
    results: &'a [TaskResult],
) -> Result<Option<&'a ScfState>, InputError> {
    let Some(source) = &task.source else {
        return Ok(None);
    };
    results
        .get(source.task_index)
        .and_then(TaskResult::scf_state)
        .map(Some)
        .ok_or_else(|| InputError::UnavailableScfSource {
            task_id: task.id.clone(),
        })
}

fn scf_config(
    task_id: &str,
    task: &TaskV1,
    snapshot: &SnapshotV1,
) -> Result<ScfConfig, InputError> {
    let TaskV1::DftScf {
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
    } = task
    else {
        unreachable!("scf_config is called only for DFT SCF tasks")
    };

    for state in core_states {
        if !snapshot
            .geometry
            .sites
            .iter()
            .any(|site| site.id == state.site)
        {
            return Err(InputError::UnknownCoreSite {
                task_id: task_id.to_owned(),
                site: state.site.clone(),
            });
        }
    }
    for orbital in &basis.local_orbitals {
        if !snapshot
            .geometry
            .sites
            .iter()
            .any(|site| site.id == orbital.site)
        {
            return Err(InputError::UnknownLocalOrbitalSite {
                task_id: task_id.to_owned(),
                site: orbital.site.clone(),
            });
        }
    }
    let core_sites = snapshot
        .geometry
        .sites
        .iter()
        .map(|site| ScfCoreSite {
            id: site.id.clone(),
            states: core_states
                .iter()
                .filter(|state| state.site == site.id)
                .map(|state| ScfCoreState {
                    principal_quantum_number: state.principal_quantum_number,
                    kappa: state.kappa,
                    occupation: state.occupation,
                })
                .collect(),
        })
        .collect();

    Ok(ScfConfig {
        electron_count: *electron_count,
        k_mesh: map_k_mesh(*k_mesh),
        basis: ScfBasis {
            plane_wave_cutoff: basis.plane_wave_cutoff,
            l_max: basis.l_max,
            local_orbitals: basis
                .local_orbitals
                .iter()
                .map(|orbital| ScfLocalOrbital {
                    site: orbital.site.clone(),
                    kappa: orbital.kappa,
                    energy: Hartree(orbital.energy),
                    kind: match orbital.kind {
                        LocalOrbitalKindV1::Lo => ScfLocalOrbitalKind::Lo,
                        LocalOrbitalKindV1::Hdlo => ScfLocalOrbitalKind::Hdlo,
                    },
                })
                .collect(),
        },
        occupations: match occupations {
            OccupationsV1::FermiDirac { temperature } => ScfOccupations::FermiDirac {
                temperature: Hartree(*temperature),
            },
            OccupationsV1::Gaussian { width } => ScfOccupations::Gaussian {
                width: Hartree(*width),
            },
        },
        exchange_correlation: match xc {
            ExchangeCorrelationV1::LdaPw92 {} => ScfExchangeCorrelation::LdaPw92,
            ExchangeCorrelationV1::Pbe {} => ScfExchangeCorrelation::Pbe,
        },
        mixing: match mixing {
            MixingV1::Linear { beta } => ScfMixing::Linear { alpha: *beta },
            MixingV1::Broyden2 { beta, history } => ScfMixing::Broyden2 {
                alpha: *beta,
                history: *history,
            },
            MixingV1::PulayAnderson { beta, history } => ScfMixing::PulayAnderson {
                alpha: *beta,
                history: *history,
            },
        },
        relativity: match relativity {
            RelativityV1::Scalar {} => ScfRelativity::Scalar,
            RelativityV1::SpexSecondVariation { band_window } => {
                ScfRelativity::SpexSecondVariation {
                    window: FirstVariationWindow::new(band_window[0], band_window[1])
                        .expect("validated runtime second-variation window is nonempty"),
                }
            }
            RelativityV1::SpinorFirstVariation {} => ScfRelativity::SpinorFirstVariation,
        },
        convergence: ScfConvergence {
            energy_tolerance: Hartree(convergence.energy_tolerance),
            density_tolerance: convergence.density_tolerance,
            max_iterations: convergence.max_iterations,
        },
        core_sites,
    })
}

fn map_k_mesh(mesh: KMeshV1) -> ScfKMesh {
    ScfKMesh {
        divisions: mesh
            .mesh
            .map(|division| usize::try_from(division).expect("u32 k-mesh division fits usize")),
        shift: mesh.shift,
    }
}

fn uniform_edges(minimum: f64, maximum: f64, points: usize) -> Vec<Hartree> {
    let step = (maximum - minimum) / (points - 1) as f64;
    (0..points)
        .map(|index| {
            if index + 1 == points {
                Hartree(maximum)
            } else {
                Hartree(minimum + index as f64 * step)
            }
        })
        .collect()
}

fn resolve_snapshot_path(input_path: &Path, snapshot: &Path) -> PathBuf {
    input_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(snapshot)
}
