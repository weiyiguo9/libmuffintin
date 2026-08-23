use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use muffintin_core::Hartree;
use muffintin_dft::{
    AtomicChannelTreatment, AtomicNumber, BandPathPoint, BandPathRequest, BandPathResult,
    DosRequest, DosResult, FirstVariationWindow, NoncollinearXcRoute as ScfNoncollinearXcRoute,
    ScfBasis, ScfConfig, ScfConvergence, ScfCoreSite, ScfCoreState, ScfExchangeCorrelation,
    ScfKMesh, ScfMixing, ScfOccupations, ScfPhysics, ScfRelativisticLocalOrbital, ScfRelativity,
    ScfState, XcFunctional, fleur_default_atomic_configuration, run_band_path, run_dos, run_scf,
};
use muffintin_io::{SnapshotFile, SnapshotV2, snapshot_file_from_toml};

use crate::input::parse_source;
use crate::{
    ChannelEnergyGenerator, ChannelRecipeArtifact, CompiledChannelRecipe, ExchangeCorrelation,
    ExternalChannelRecipe, Input, InputError, KMesh, Mixing, NoncollinearXcRoute, Occupations,
    RecipeSite, Relativity, Task, compile_channel_recipe, parse_channel_recipe_toml,
    parse_input_toml,
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
    pub task: Task,
    pub source: Option<PreparedSource>,
    pub channel_recipe: Option<CompiledChannelRecipe>,
}

/// A fully validated workflow and loaded immutable input snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedWorkflow {
    pub snapshot: SnapshotV2,
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
    input: &Input,
    snapshot: SnapshotFile,
) -> Result<PreparedWorkflow, InputError> {
    prepare_input_with_recipes(input, snapshot, &BTreeMap::new())
}

/// Validate and resolve decoded input, snapshot, and preloaded recipe artifacts.
///
/// Recipe paths remain workflow-relative keys. This function performs no
/// filesystem access; callers must supply every artifact named by an SCF task.
pub fn prepare_input_with_recipes(
    input: &Input,
    snapshot: SnapshotFile,
    recipe_artifacts: &BTreeMap<PathBuf, ChannelRecipeArtifact>,
) -> Result<PreparedWorkflow, InputError> {
    input.validate()?;
    let snapshot = match snapshot {
        SnapshotFile::V1(snapshot) => snapshot.normalize_v2(),
        SnapshotFile::V2(snapshot) => snapshot.validate().map(|()| snapshot),
    }
    .map_err(InputError::InvalidSnapshot)?;

    let mut tasks = Vec::with_capacity(input.workflow.tasks.len());
    for id in &input.workflow.tasks {
        let task = input.task[id].clone();
        let channel_recipe = match &task {
            Task::DftScf { basis, .. } => {
                let sites = recipe_sites(id, &snapshot)?;
                let external = match &basis.recipe {
                    Some(path) => {
                        let artifact = recipe_artifacts.get(path).ok_or_else(|| {
                            InputError::MissingRecipeArtifact {
                                task_id: id.clone(),
                                path: path.clone(),
                            }
                        })?;
                        let source = path.display().to_string();
                        Some((artifact, source))
                    }
                    None => None,
                };
                let external = external
                    .as_ref()
                    .map(|(artifact, source)| ExternalChannelRecipe {
                        artifact,
                        source: source.as_str(),
                    });
                let compiled = compile_channel_recipe(
                    &sites,
                    external,
                    basis.energy_generator,
                    &basis.channels,
                )
                .map_err(|source| InputError::ChannelRecipe {
                    task_id: id.clone(),
                    path: basis.recipe.clone(),
                    source: Box::new(source),
                })?;
                Some(compiled)
            }
            Task::DftBands { .. } | Task::DftDos { .. } => None,
        };
        let source = task.source().map(|source| {
            let (task_id, output) = parse_source(source)
                .expect("Input::validate accepted only syntactically valid sources");
            let task_index = input
                .workflow
                .tasks
                .iter()
                .position(|candidate| candidate == task_id)
                .expect("Input::validate accepted only existing source tasks");
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
            channel_recipe,
        });
    }
    Ok(PreparedWorkflow { snapshot, tasks })
}

fn recipe_sites(task_id: &str, snapshot: &SnapshotV2) -> Result<Vec<RecipeSite>, InputError> {
    snapshot
        .geometry
        .sites
        .iter()
        .map(|site| {
            let atomic_number = u8::try_from(site.atomic_number)
                .ok()
                .and_then(AtomicNumber::new)
                .ok_or_else(|| InputError::UnsupportedAtomicNumber {
                    task_id: task_id.to_owned(),
                    site: site.id.clone(),
                    atomic_number: site.atomic_number,
                })?;
            Ok(RecipeSite {
                id: site.id.clone(),
                atomic_number,
            })
        })
        .collect()
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
    let snapshot =
        snapshot_file_from_toml(&snapshot_text).map_err(|source| InputError::Snapshot {
            path: snapshot_path,
            source,
        })?;
    let mut recipe_artifacts = BTreeMap::new();
    for task_id in &input.workflow.tasks {
        let Task::DftScf { basis, .. } = &input.task[task_id] else {
            continue;
        };
        let Some(recipe) = &basis.recipe else {
            continue;
        };
        if recipe_artifacts.contains_key(recipe) {
            continue;
        }
        let recipe_path = resolve_input_relative_path(input_path, recipe);
        let recipe_text =
            fs::read_to_string(&recipe_path).map_err(|source| InputError::ReadRecipe {
                task_id: task_id.clone(),
                path: recipe_path,
                source,
            })?;
        let artifact = parse_channel_recipe_toml(&recipe_text).map_err(|source| {
            InputError::ChannelRecipe {
                task_id: task_id.clone(),
                path: Some(recipe.clone()),
                source: Box::new(source),
            }
        })?;
        recipe_artifacts.insert(recipe.clone(), artifact);
    }
    prepare_input_with_recipes(&input, snapshot, &recipe_artifacts)
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
            Task::DftScf { .. } => {
                let config = scf_config(task, &workflow.snapshot)?;
                let state = run_scf(physics, &config, source).map_err(|source| {
                    InputError::TaskExecution {
                        task_id: task.id.clone(),
                        kind: task.task.kind(),
                        source: Box::new(source),
                    }
                })?;
                TaskResult::Scf(Box::new(state))
            }
            Task::DftBands { bands, path, .. } => {
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
            Task::DftDos {
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

fn scf_config(task: &PreparedTask, snapshot: &SnapshotV2) -> Result<ScfConfig, InputError> {
    let Task::DftScf {
        electron_count,
        k_mesh,
        basis,
        occupations,
        xc,
        mixing,
        relativity,
        convergence,
        ..
    } = &task.task
    else {
        unreachable!("scf_config is called only for DFT SCF tasks")
    };

    if let Some((site, channel)) = task.channel_recipe.as_ref().and_then(|recipe| {
        recipe.sites.iter().find_map(|site| {
            site.channels
                .iter()
                .find(|channel| channel.derivative_order >= 3)
                .map(|channel| (site, channel))
        })
    }) {
        return Err(InputError::DerivativeOrderNotImplemented {
            task_id: task.id.clone(),
            site: site.site.clone(),
            identity: channel.identity,
            derivative_order: channel.derivative_order,
        });
    }

    if basis.energy_generator != Some(ChannelEnergyGenerator::FrozenSnapshot)
        || basis.recipe.is_some()
        || !basis.channels.is_empty()
    {
        return Err(InputError::UnsupportedV2OrbitalConfiguration {
            task_id: task.id.clone(),
        });
    }
    let mut atomic_configurations = Vec::with_capacity(snapshot.geometry.sites.len());
    for site in &snapshot.geometry.sites {
        let atomic_number = u8::try_from(site.atomic_number)
            .ok()
            .and_then(AtomicNumber::new)
            .ok_or_else(|| InputError::UnsupportedAtomicNumber {
                task_id: task.id.clone(),
                site: site.id.clone(),
                atomic_number: site.atomic_number,
            })?;
        let configuration = fleur_default_atomic_configuration(atomic_number);
        atomic_configurations.push((site, configuration));
    }
    let core_sites = atomic_configurations
        .iter()
        .map(|(site, configuration)| ScfCoreSite {
            id: site.id.clone(),
            states: configuration
                .occupations()
                .iter()
                .filter(|state| state.treatment == AtomicChannelTreatment::Core)
                .map(|state| ScfCoreState {
                    principal_quantum_number: u32::from(state.orbital.principal_quantum_number()),
                    kappa: i32::from(state.orbital.kappa()),
                    occupation: state.occupation,
                })
                .collect(),
        })
        .collect();
    let relativistic_local_orbitals = if matches!(relativity, Relativity::Scalar {}) {
        Vec::new()
    } else {
        atomic_configurations
            .iter()
            .flat_map(|(site, configuration)| {
                configuration
                    .occupations()
                    .iter()
                    .filter(|state| {
                        state.treatment == AtomicChannelTreatment::RelativisticLocalOrbital
                    })
                    .map(|state| ScfRelativisticLocalOrbital {
                        site: site.id.clone(),
                        principal_quantum_number: u32::from(
                            state.orbital.principal_quantum_number(),
                        ),
                        kappa: i32::from(state.orbital.kappa()),
                    })
            })
            .collect()
    };

    Ok(ScfConfig {
        electron_count: *electron_count,
        k_mesh: map_k_mesh(*k_mesh),
        basis: ScfBasis {
            plane_wave_cutoff: basis.envelope.cutoff,
            l_max: basis.l_max,
            local_orbitals: Vec::new(),
            relativistic_local_orbitals,
        },
        occupations: match occupations {
            Occupations::FermiDirac { temperature } => ScfOccupations::FermiDirac {
                temperature: Hartree(*temperature),
            },
            Occupations::Gaussian { width } => ScfOccupations::Gaussian {
                width: Hartree(*width),
            },
        },
        exchange_correlation: match xc {
            ExchangeCorrelation::LdaPw92 { noncollinear_route } => ScfExchangeCorrelation {
                functional: XcFunctional::LdaPw92,
                noncollinear_route: map_noncollinear_xc_route(*noncollinear_route),
            },
            ExchangeCorrelation::Pbe { noncollinear_route } => ScfExchangeCorrelation {
                functional: XcFunctional::Pbe,
                noncollinear_route: map_noncollinear_xc_route(*noncollinear_route),
            },
        },
        mixing: match mixing {
            Mixing::Linear { beta } => ScfMixing::Linear { alpha: *beta },
            Mixing::Broyden2 { beta, history } => ScfMixing::Broyden2 {
                alpha: *beta,
                history: *history,
            },
            Mixing::PulayAnderson { beta, history } => ScfMixing::PulayAnderson {
                alpha: *beta,
                history: *history,
            },
        },
        relativity: match relativity {
            Relativity::Scalar {} => ScfRelativity::Scalar,
            Relativity::SocSecondVariation { band_window } => ScfRelativity::SocSecondVariation {
                window: FirstVariationWindow::new(band_window[0], band_window[1])
                    .expect("validated runtime second-variation window is nonempty"),
            },
            Relativity::SpinorFirstVariation {} => ScfRelativity::SpinorFirstVariation,
        },
        convergence: ScfConvergence {
            energy_tolerance: Hartree(convergence.energy_tolerance),
            density_tolerance: convergence.density_tolerance,
            max_iterations: convergence.max_iterations,
        },
        core_sites,
    })
}

const fn map_noncollinear_xc_route(route: NoncollinearXcRoute) -> ScfNoncollinearXcRoute {
    match route {
        NoncollinearXcRoute::LocalSpinFrame => ScfNoncollinearXcRoute::LocalSpinFrame,
        NoncollinearXcRoute::MagnetizationField => ScfNoncollinearXcRoute::MagnetizationField,
    }
}

fn map_k_mesh(mesh: KMesh) -> ScfKMesh {
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
    resolve_input_relative_path(input_path, snapshot)
}

fn resolve_input_relative_path(input_path: &Path, relative: &Path) -> PathBuf {
    input_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(relative)
}
