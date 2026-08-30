use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use muffintin_core::Hartree;
use muffintin_dft::{
    AtomicNumber, BandPathPoint, BandPathRequest, BandPathResult, DosRequest, DosResult,
    FirstVariationWindow, LinearizationEnergyGenerator,
    NoncollinearXcRoute as ScfNoncollinearXcRoute, ScfBasis, ScfChannelIdentity,
    ScfChannelProvenance, ScfChannelRecipe, ScfChannelTreatment, ScfConfig, ScfConvergence,
    ScfCoreSite, ScfCoreState, ScfExchangeCorrelation, ScfKMesh, ScfMixing, ScfOccupations,
    ScfPhysics, ScfRelativity, ScfState, XcFunctional, fleur_default_atomic_configuration,
    run_band_path, run_dos, run_scf,
};
use muffintin_io::{CheckpointFile, CheckpointV2, checkpoint_file_from_toml};

use crate::input::parse_source;
use crate::{
    ChannelEnergyGenerator, ChannelIdentity, ChannelProvenance, ChannelRecipeArtifact,
    ChannelTreatment, CompiledChannelRecipe, CompiledSiteRecipe, ExchangeCorrelation,
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

/// A fully validated workflow and loaded immutable input checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedWorkflow {
    pub checkpoint: CheckpointV2,
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

/// Validate and resolve an already decoded input and checkpoint without filesystem access.
pub fn prepare_input(
    input: &Input,
    checkpoint: CheckpointFile,
) -> Result<PreparedWorkflow, InputError> {
    prepare_input_with_recipes(input, checkpoint, &BTreeMap::new())
}

/// Validate and resolve decoded input, checkpoint, and preloaded recipe artifacts.
///
/// Recipe paths remain workflow-relative keys. This function performs no
/// filesystem access; callers must supply every artifact named by an SCF task.
pub fn prepare_input_with_recipes(
    input: &Input,
    checkpoint: CheckpointFile,
    recipe_artifacts: &BTreeMap<PathBuf, ChannelRecipeArtifact>,
) -> Result<PreparedWorkflow, InputError> {
    input.validate()?;
    let checkpoint = checkpoint
        .into_v2_prevalidated()
        .map_err(InputError::InvalidCheckpoint)?;

    let mut tasks = Vec::with_capacity(input.workflow.tasks.len());
    for id in &input.workflow.tasks {
        let task = input.task[id].clone();
        let channel_recipe = match &task {
            Task::DftScf { basis, .. } => {
                let sites = recipe_sites(id, &checkpoint)?;
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
    Ok(PreparedWorkflow { checkpoint, tasks })
}

fn recipe_sites(task_id: &str, checkpoint: &CheckpointV2) -> Result<Vec<RecipeSite>, InputError> {
    checkpoint
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

/// Read one input and its relative checkpoint, then prepare the workflow.
pub fn load_input_path(path: impl AsRef<Path>) -> Result<PreparedWorkflow, InputError> {
    let input_path = path.as_ref();
    let input_text = fs::read_to_string(input_path).map_err(|source| InputError::ReadInput {
        path: input_path.to_owned(),
        source,
    })?;
    let input = parse_input_toml(&input_text)?;
    let checkpoint_path = resolve_checkpoint_path(input_path, &input.checkpoint);
    let checkpoint_text =
        fs::read_to_string(&checkpoint_path).map_err(|source| InputError::ReadCheckpoint {
            path: checkpoint_path.clone(),
            source,
        })?;
    let checkpoint =
        checkpoint_file_from_toml(&checkpoint_text).map_err(|source| InputError::Checkpoint {
            path: checkpoint_path,
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
    prepare_input_with_recipes(&input, checkpoint, &recipe_artifacts)
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
                let config = scf_config(task, &workflow.checkpoint)?;
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

pub(crate) fn scf_config(
    task: &PreparedTask,
    _checkpoint: &CheckpointV2,
) -> Result<ScfConfig, InputError> {
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

    let channel_recipe = task
        .channel_recipe
        .as_ref()
        .expect("prepared DFT SCF tasks always contain a compiled channel recipe");
    let channels = map_basis_channels(
        &task.id,
        basis.l_max,
        basis.energy_generator,
        channel_recipe,
    )?;
    let core_sites = channel_recipe
        .sites
        .iter()
        .map(|site| map_core_site(&task.id, site))
        .collect::<Result<_, _>>()?;

    Ok(ScfConfig {
        electron_count: *electron_count,
        k_mesh: map_k_mesh(*k_mesh),
        basis: ScfBasis {
            plane_wave_cutoff: basis.envelope.normalized_cutoff(),
            l_max: basis.l_max,
            channels,
            resolved_channels: Vec::new(),
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

fn map_basis_channels(
    task_id: &str,
    l_max: u32,
    task_generator: Option<ChannelEnergyGenerator>,
    recipe: &CompiledChannelRecipe,
) -> Result<Vec<ScfChannelRecipe>, InputError> {
    let mut channels = Vec::new();
    for site in &recipe.sites {
        let mut site_channels = map_site_channels(task_id, site)?;
        for l in 0..=l_max {
            if site_channels.iter().any(|channel| {
                channel.treatment == ScfChannelTreatment::Valence
                    && scf_channel_angular_momentum(channel.identity) == l
            }) {
                continue;
            }
            let mut n = l + 1;
            while site_channels.iter().any(|channel| {
                scf_channel_angular_momentum(channel.identity) == l
                    && scf_channel_principal_quantum_number(channel.identity) == n
            }) {
                n += 1;
            }
            let identity = ChannelIdentity::ScalarL { n, l };
            let generator = task_generator.unwrap_or(ChannelEnergyGenerator::Atomic);
            if generator == ChannelEnergyGenerator::Explicit {
                return Err(InputError::MissingExplicitBaseValenceSeed {
                    task_id: task_id.to_owned(),
                    site: site.site.clone(),
                    identity,
                });
            }
            site_channels.push(ScfChannelRecipe {
                site: site.site.clone(),
                identity: ScfChannelIdentity::ScalarL { n, l },
                treatment: ScfChannelTreatment::Valence,
                derivative_order: 0,
                generator: map_channel_energy_generator(generator),
                seed: None,
                provenance: ScfChannelProvenance::BuiltIn,
            });
        }
        channels.extend(site_channels);
    }
    Ok(channels)
}

fn map_site_channels(
    task_id: &str,
    site: &CompiledSiteRecipe,
) -> Result<Vec<ScfChannelRecipe>, InputError> {
    let mut channels = Vec::with_capacity(site.channels.len());
    let mut collapsed = BTreeMap::new();
    for record in &site.channels {
        let collapse_key = match (&record.provenance, record.treatment, record.identity) {
            (
                ChannelProvenance::BuiltIn,
                ChannelTreatment::Valence,
                ChannelIdentity::Kappa { n, kappa },
            ) => Some((n, angular_momentum_from_kappa(kappa))),
            _ => None,
        };
        let Some((n, l)) = collapse_key else {
            channels.push(map_channel_recipe(&site.site, record));
            continue;
        };
        if let Some(&(first_generator, first_seed)) = collapsed.get(&(n, l)) {
            if first_generator != record.generator || first_seed != record.seed {
                return Err(InputError::InconsistentBuiltInValencePartners {
                    task_id: task_id.to_owned(),
                    site: site.site.clone(),
                    n,
                    l,
                    first_generator,
                    first_seed,
                    conflicting_generator: record.generator,
                    conflicting_seed: record.seed,
                });
            }
            continue;
        }
        collapsed.insert((n, l), (record.generator, record.seed));
        let mut channel = map_channel_recipe(&site.site, record);
        channel.identity = ScfChannelIdentity::ScalarL { n, l };
        channels.push(channel);
    }
    Ok(channels)
}

fn map_channel_recipe(site: &str, record: &crate::ChannelRecipeRecord) -> ScfChannelRecipe {
    ScfChannelRecipe {
        site: site.to_owned(),
        identity: match record.identity {
            ChannelIdentity::ScalarL { n, l } => ScfChannelIdentity::ScalarL { n, l },
            ChannelIdentity::Kappa { n, kappa } => ScfChannelIdentity::Kappa { n, kappa },
        },
        treatment: match record.treatment {
            ChannelTreatment::Core => ScfChannelTreatment::Core,
            ChannelTreatment::Valence => ScfChannelTreatment::Valence,
            ChannelTreatment::Lo => ScfChannelTreatment::Lo,
            ChannelTreatment::Hdlo => ScfChannelTreatment::Hdlo,
        },
        derivative_order: record.derivative_order,
        generator: map_channel_energy_generator(record.generator),
        seed: record.seed,
        provenance: match &record.provenance {
            ChannelProvenance::BuiltIn => ScfChannelProvenance::BuiltIn,
            ChannelProvenance::ExternalRecipe { source } => ScfChannelProvenance::ExternalRecipe {
                source: source.clone(),
            },
            ChannelProvenance::TaskDefault => ScfChannelProvenance::TaskDefault,
            ChannelProvenance::Species => ScfChannelProvenance::Species,
            ChannelProvenance::Site => ScfChannelProvenance::Site,
        },
    }
}

const fn map_channel_energy_generator(
    generator: ChannelEnergyGenerator,
) -> LinearizationEnergyGenerator {
    match generator {
        ChannelEnergyGenerator::Explicit => LinearizationEnergyGenerator::Explicit,
        ChannelEnergyGenerator::Atomic => LinearizationEnergyGenerator::Atomic,
        ChannelEnergyGenerator::BandCenter => LinearizationEnergyGenerator::BandCenter,
        ChannelEnergyGenerator::LogDerivative => LinearizationEnergyGenerator::LogDerivative,
        ChannelEnergyGenerator::BandCog => LinearizationEnergyGenerator::BandCog,
        ChannelEnergyGenerator::FermiOffset => LinearizationEnergyGenerator::FermiOffset,
        ChannelEnergyGenerator::FrozenCheckpoint => LinearizationEnergyGenerator::FrozenCheckpoint,
    }
}

fn map_core_site(task_id: &str, site: &CompiledSiteRecipe) -> Result<ScfCoreSite, InputError> {
    let configuration = fleur_default_atomic_configuration(site.atomic_number);
    let mut states = Vec::new();
    for record in site
        .channels
        .iter()
        .filter(|record| record.treatment == ChannelTreatment::Core)
    {
        let mut matched = false;
        for occupation in configuration.occupations().iter().filter(|occupation| {
            let n = u32::from(occupation.orbital.principal_quantum_number());
            let kappa = i32::from(occupation.orbital.kappa());
            match record.identity {
                ChannelIdentity::Kappa {
                    n: requested_n,
                    kappa: requested_kappa,
                } => n == requested_n && kappa == requested_kappa,
                ChannelIdentity::ScalarL { n: requested_n, l } => {
                    n == requested_n && angular_momentum_from_kappa(kappa) == l
                }
            }
        }) {
            matched = true;
            let state = ScfCoreState {
                principal_quantum_number: u32::from(occupation.orbital.principal_quantum_number()),
                kappa: i32::from(occupation.orbital.kappa()),
                occupation: occupation.occupation,
            };
            if !states.iter().any(|present: &ScfCoreState| {
                present.principal_quantum_number == state.principal_quantum_number
                    && present.kappa == state.kappa
            }) {
                states.push(state);
            }
        }
        if !matched {
            return Err(InputError::MissingCoreOccupation {
                task_id: task_id.to_owned(),
                site: site.site.clone(),
                atomic_number: site.atomic_number.get(),
                identity: record.identity,
            });
        }
    }
    Ok(ScfCoreSite {
        id: site.site.clone(),
        states,
    })
}

fn angular_momentum_from_kappa(kappa: i32) -> u32 {
    if kappa > 0 {
        kappa as u32
    } else {
        (-kappa - 1) as u32
    }
}

fn scf_channel_angular_momentum(identity: ScfChannelIdentity) -> u32 {
    match identity {
        ScfChannelIdentity::ScalarL { l, .. } => l,
        ScfChannelIdentity::Kappa { kappa, .. } => angular_momentum_from_kappa(kappa),
    }
}

fn scf_channel_principal_quantum_number(identity: ScfChannelIdentity) -> u32 {
    match identity {
        ScfChannelIdentity::ScalarL { n, .. } | ScfChannelIdentity::Kappa { n, .. } => n,
    }
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

fn resolve_checkpoint_path(input_path: &Path, checkpoint: &Path) -> PathBuf {
    resolve_input_relative_path(input_path, checkpoint)
}

fn resolve_input_relative_path(input_path: &Path, relative: &Path) -> PathBuf {
    input_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(relative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChannelRecipeRecord, ChannelScope};

    fn record(identity: ChannelIdentity, treatment: ChannelTreatment) -> ChannelRecipeRecord {
        ChannelRecipeRecord {
            scope: ChannelScope::Site {
                name: "H-1".to_owned(),
            },
            identity,
            treatment,
            derivative_order: 0,
            generator: ChannelEnergyGenerator::Atomic,
            seed: None,
            provenance: ChannelProvenance::BuiltIn,
        }
    }

    #[test]
    fn base_valence_coverage_uses_first_unrepresented_n_and_task_generator() {
        let mut compiled_channels = vec![
            record(
                ChannelIdentity::Kappa { n: 1, kappa: -1 },
                ChannelTreatment::Lo,
            ),
            record(
                ChannelIdentity::Kappa { n: 2, kappa: 1 },
                ChannelTreatment::Valence,
            ),
            record(
                ChannelIdentity::Kappa { n: 2, kappa: -2 },
                ChannelTreatment::Valence,
            ),
        ];
        for record in &mut compiled_channels {
            record.generator = ChannelEnergyGenerator::BandCog;
        }
        let recipe = CompiledChannelRecipe {
            sites: vec![CompiledSiteRecipe {
                site: "H-1".to_owned(),
                atomic_number: AtomicNumber::new(1).unwrap(),
                channels: compiled_channels,
            }],
        };

        let channels =
            map_basis_channels("scf", 2, Some(ChannelEnergyGenerator::BandCog), &recipe).unwrap();
        let mut scalar_valence: Vec<_> = channels
            .iter()
            .filter(|channel| {
                channel.provenance == ScfChannelProvenance::BuiltIn
                    && channel.treatment == ScfChannelTreatment::Valence
                    && matches!(channel.identity, ScfChannelIdentity::ScalarL { .. })
            })
            .map(|channel| channel.identity)
            .collect();
        scalar_valence.sort();
        assert_eq!(
            scalar_valence,
            vec![
                ScfChannelIdentity::ScalarL { n: 2, l: 0 },
                ScfChannelIdentity::ScalarL { n: 2, l: 1 },
                ScfChannelIdentity::ScalarL { n: 3, l: 2 },
            ]
        );
        assert!(
            channels
                .iter()
                .filter(|channel| channel.treatment == ScfChannelTreatment::Valence)
                .all(|channel| channel.generator == LinearizationEnergyGenerator::BandCog)
        );
        assert!(!channels.iter().any(|channel| {
            channel.treatment == ScfChannelTreatment::Valence
                && matches!(channel.identity, ScfChannelIdentity::Kappa { .. })
        }));
    }

    #[test]
    fn built_in_valence_partners_must_have_matching_generator_and_seed() {
        let first = record(
            ChannelIdentity::Kappa { n: 2, kappa: 1 },
            ChannelTreatment::Valence,
        );
        let mut conflicting = record(
            ChannelIdentity::Kappa { n: 2, kappa: -2 },
            ChannelTreatment::Valence,
        );
        conflicting.generator = ChannelEnergyGenerator::BandCog;
        let recipe = CompiledChannelRecipe {
            sites: vec![CompiledSiteRecipe {
                site: "H-1".to_owned(),
                atomic_number: AtomicNumber::new(1).unwrap(),
                channels: vec![first, conflicting],
            }],
        };

        assert!(matches!(
            map_basis_channels("scf", 1, None, &recipe),
            Err(InputError::InconsistentBuiltInValencePartners {
                task_id,
                site,
                n: 2,
                l: 1,
                first_generator: ChannelEnergyGenerator::Atomic,
                first_seed: None,
                conflicting_generator: ChannelEnergyGenerator::BandCog,
                conflicting_seed: None,
            }) if task_id == "scf" && site == "H-1"
        ));
    }

    #[test]
    fn explicit_task_generator_rejects_seedless_base_valence_injection() {
        let recipe = CompiledChannelRecipe {
            sites: vec![CompiledSiteRecipe {
                site: "H-1".to_owned(),
                atomic_number: AtomicNumber::new(1).unwrap(),
                channels: Vec::new(),
            }],
        };

        assert!(matches!(
            map_basis_channels(
                "scf",
                1,
                Some(ChannelEnergyGenerator::Explicit),
                &recipe,
            ),
            Err(InputError::MissingExplicitBaseValenceSeed {
                task_id,
                site,
                identity: ChannelIdentity::ScalarL { n: 1, l: 0 },
            }) if task_id == "scf" && site == "H-1"
        ));
    }
}
