mod common;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use muffintin::{
    ChannelEnergyGenerator, ChannelIdentity, ChannelProvenance, ChannelRecipeArtifact,
    ChannelRecipeError, ChannelRecipeRecord, ChannelScope, ChannelTreatment, ExchangeCorrelation,
    InputError, InputValidationError, Mixing, NoncollinearXcRoute, Relativity, Task, input_to_toml,
    load_input_path, parse_input_toml, prepare_input, prepare_input_with_recipes,
};
use muffintin_core::Hartree;
use muffintin_io::CheckpointFile;

use common::{FixtureDirectory, sample_checkpoint, sample_input};

#[test]
fn input_round_trips_deterministically_with_header_first() {
    let input = sample_input();
    let encoded = input_to_toml(&input).unwrap();
    assert!(encoded.starts_with(
        "format = \"libmuffintin-input\"\nversion = 3\ncheckpoint = \"data/checkpoint.toml\"\n"
    ));
    let decoded = parse_input_toml(&encoded).unwrap();
    assert_eq!(decoded, input);
    assert_eq!(input_to_toml(&decoded).unwrap(), encoded);
}

#[test]
fn xc_noncollinear_route_defaults_to_local_spin_frame() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let without_route = encoded.replace("noncollinear-route = \"local-spin-frame\"\n", "");
    assert_ne!(without_route, encoded);
    let decoded = parse_input_toml(&without_route).unwrap();
    let Task::DftScf { xc, .. } = &decoded.task["scf"] else {
        unreachable!()
    };
    assert_eq!(
        *xc,
        ExchangeCorrelation::LdaPw92 {
            noncollinear_route: NoncollinearXcRoute::LocalSpinFrame,
        }
    );
}

#[test]
fn envelope_channels_quoted_sites_and_explicit_empty_rows_are_preserved() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    assert!(encoded.contains("[task.scf.k-mesh]"));
    assert!(encoded.contains("[task.scf.basis]"));
    assert!(encoded.contains("[task.scf.basis.envelope]"));
    assert!(encoded.contains("[task.scf.xc]"));
    assert!(encoded.contains("kind = \"soc-second-variation\""));
    assert!(encoded.contains("[[task.bands.path]]"));

    let document: toml::Value = toml::from_str(&encoded).unwrap();
    let scf = &document["task"]["scf"];
    assert_eq!(scf["relativity"]["band-window"][0].as_integer(), Some(0));
    assert_eq!(scf["relativity"]["band-window"][1].as_integer(), Some(12));
    assert_eq!(
        scf["basis"]["envelope"]["kind"].as_str(),
        Some("plane-wave")
    );
    assert_eq!(scf["basis"]["envelope"]["g-cutoff"].as_float(), Some(4.0));
    assert!(scf["basis"]["envelope"].get("energy-cutoff").is_none());
    assert!(
        scf["basis"]["channels"]["Si"]["valence"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        scf["basis"]["channels"]["Si-1"]["lo"][0].as_str(),
        Some("+3d@-0.15")
    );
    assert!(
        scf["basis"]["channels"]["Si-1"]["hdlo"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    let quoted_site = encoded.replace(
        "[task.scf.basis.channels.Si-1]",
        "[task.scf.basis.channels.\"Si-1\"]",
    );
    let quoted = parse_input_toml(&quoted_site).unwrap();
    let Task::DftScf { basis, .. } = &quoted.task["scf"] else {
        panic!("scf task changed kind");
    };
    assert!(basis.channels.contains_key("Si-1"));

    let decoded = parse_input_toml(&encoded).unwrap();
    let Task::DftScf {
        basis, relativity, ..
    } = &decoded.task["scf"]
    else {
        unreachable!()
    };
    assert_eq!(basis.energy_generator, None);
    assert_eq!(basis.recipe, Some(PathBuf::from("recipes/si.toml")));
    assert!(basis.channels["Si"].contains_key(&ChannelTreatment::Valence));
    assert!(matches!(
        relativity,
        Relativity::SocSecondVariation {
            band_window: [0, 12]
        }
    ));
}

#[test]
fn omitted_task_generator_remains_none() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let decoded = parse_input_toml(&encoded).unwrap();
    let Task::DftScf { basis, .. } = &decoded.task["scf"] else {
        panic!("scf task changed kind");
    };
    assert_eq!(basis.energy_generator, None);
    assert!(!encoded.contains("energy-generator"));
}

#[test]
fn omitted_channels_normalize_to_an_empty_table() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let mut document: toml::Value = toml::from_str(&encoded).unwrap();
    document["task"]["scf"]["basis"]
        .as_table_mut()
        .unwrap()
        .remove("channels");
    let decoded = parse_input_toml(&toml::to_string(&document).unwrap()).unwrap();
    let Task::DftScf { basis, .. } = &decoded.task["scf"] else {
        panic!("scf task changed kind");
    };
    assert!(basis.channels.is_empty());
}

#[test]
fn header_unknown_fields_and_unknown_kinds_are_rejected() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let wrong_format = encoded.replacen("libmuffintin-input", "other-input", 1);
    assert!(matches!(
        parse_input_toml(&wrong_format),
        Err(InputError::InvalidFormat { .. })
    ));
    let wrong_version = encoded.replacen("version = 3", "version = 4", 1);
    assert!(matches!(
        parse_input_toml(&wrong_version),
        Err(InputError::UnsupportedVersion { found: 4, .. })
    ));

    let unknown_field = encoded.replacen(
        "kind = \"dft-scf\"",
        "kind = \"dft-scf\"\nunexpected = true",
        1,
    );
    assert!(matches!(
        parse_input_toml(&unknown_field),
        Err(InputError::Decode(_))
    ));
    let unknown_kind = encoded.replacen("kind = \"dft-scf\"", "kind = \"thc-fit\"", 1);
    assert!(matches!(
        parse_input_toml(&unknown_kind),
        Err(InputError::Decode(_))
    ));
}

#[test]
fn complete_v1_body_gets_the_dedicated_migration_error() {
    let v1 = r#"
format = "libmuffintin-input"
version = 1
checkpoint = "data/checkpoint.toml"

[workflow]
tasks = ["scf"]

[task.scf]
kind = "dft-scf"
electron-count = 1.0
state-overrides = []

[task.scf.k-mesh]
mesh = [1, 1, 1]
shift = [0.0, 0.0, 0.0]

[task.scf.basis]
plane-wave-cutoff = 0.5
l-max = 1
local-orbitals = []

[task.scf.occupations]
kind = "fermi-dirac"
temperature = 0.02

[task.scf.xc]
kind = "lda-pw92"

[task.scf.mixing]
kind = "linear"
beta = 1.0

[task.scf.relativity]
kind = "scalar"

[task.scf.convergence]
energy-tolerance = 1e-8
density-tolerance = 1e-7
max-iterations = 20
"#;
    let error = parse_input_toml(v1).unwrap_err();
    assert!(matches!(error, InputError::V1MigrationRequired));
    let message = error.to_string();
    assert!(message.contains("version = 3"));
    assert!(message.contains("envelope"));
    assert!(message.contains("channels"));
    assert!(message.contains("local-orbitals/state-overrides"));

    let mut in_memory = sample_input();
    in_memory.version = 1;
    assert!(matches!(
        in_memory.validate(),
        Err(InputError::V1MigrationRequired)
    ));
}

#[test]
fn v3_rejects_each_removed_flat_orbital_field() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let flat_cutoff = encoded.replacen(
        "[task.scf.basis]\n",
        "[task.scf.basis]\nplane-wave-cutoff = 4.0\n",
        1,
    );
    let local_orbitals = encoded.replacen(
        "[task.scf.basis]\n",
        "[task.scf.basis]\nlocal-orbitals = []\n",
        1,
    );
    let state_overrides = encoded.replacen(
        "kind = \"dft-scf\"\n",
        "kind = \"dft-scf\"\nstate-overrides = []\n",
        1,
    );
    for removed in [flat_cutoff, local_orbitals, state_overrides] {
        assert!(matches!(
            parse_input_toml(&removed),
            Err(InputError::Decode(_))
        ));
    }
}

#[test]
fn envelope_kind_and_cutoff_are_strict() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let unknown_kind = encoded.replacen("kind = \"plane-wave\"", "kind = \"mto\"", 1);
    assert!(matches!(
        parse_input_toml(&unknown_kind),
        Err(InputError::Decode(_))
    ));

    let nonpositive = encoded.replacen("g-cutoff = 4.0", "g-cutoff = 0.0", 1);
    assert!(matches!(
        parse_input_toml(&nonpositive),
        Err(InputError::Validation(
            InputValidationError::NotPositive { .. }
        ))
    ));

    let missing = encoded.replace("g-cutoff = 4.0\n", "");
    assert!(matches!(
        parse_input_toml(&missing),
        Err(InputError::Validation(
            InputValidationError::MissingPlaneWaveCutoff { .. }
        ))
    ));

    let conflicting = encoded.replacen("g-cutoff = 4.0", "g-cutoff = 4.0\nenergy-cutoff = 8.0", 1);
    assert!(matches!(
        parse_input_toml(&conflicting),
        Err(InputError::Validation(
            InputValidationError::ConflictingPlaneWaveCutoffs { .. }
        ))
    ));
}

#[test]
fn v2_gets_the_dedicated_cutoff_migration_error() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let v2 = encoded.replacen("version = 3", "version = 2", 1);
    assert!(matches!(
        parse_input_toml(&v2),
        Err(InputError::V2MigrationRequired)
    ));
}

#[test]
fn energy_cutoff_round_trip_preserves_its_spelling() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let energy = encoded.replacen("g-cutoff = 4.0", "energy-cutoff = 8.0", 1);
    let decoded = parse_input_toml(&energy).unwrap();
    let reencoded = input_to_toml(&decoded).unwrap();
    assert!(reencoded.contains("energy-cutoff = 8.0"));
    assert!(!reencoded.contains("g-cutoff"));
    assert_eq!(parse_input_toml(&reencoded).unwrap(), decoded);
}

#[test]
fn recipe_path_must_be_nonempty_and_relative() {
    let mut absolute = sample_input();
    let Task::DftScf { basis, .. } = absolute.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.recipe = Some(PathBuf::from("/tmp/si.toml"));
    assert!(matches!(
        absolute.validate(),
        Err(InputError::Validation(
            InputValidationError::AbsoluteRecipePath { .. }
        ))
    ));

    let mut empty = sample_input();
    let Task::DftScf { basis, .. } = empty.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.recipe = Some(PathBuf::new());
    assert!(matches!(
        empty.validate(),
        Err(InputError::Validation(
            InputValidationError::EmptyRecipePath
        ))
    ));
}

#[test]
fn recipe_ir_rejects_negative_derivative_order() {
    let record = ChannelRecipeRecord {
        scope: ChannelScope::Species {
            name: "Si".to_owned(),
        },
        identity: ChannelIdentity::ScalarL { n: 3, l: 2 },
        treatment: ChannelTreatment::Hdlo,
        derivative_order: 3,
        generator: ChannelEnergyGenerator::Explicit,
        seed: Some(Hartree(-0.15)),
        provenance: ChannelProvenance::ExternalRecipe {
            source: Some("recipes/si.toml".to_owned()),
        },
    };
    let encoded = toml::to_string(&record).unwrap();
    let negative = encoded.replacen("derivative-order = 3", "derivative-order = -1", 1);
    assert!(toml::from_str::<ChannelRecipeRecord>(&negative).is_err());
}

#[test]
fn workflow_and_task_blocks_must_match_exactly() {
    let mut missing = sample_input();
    missing.task.remove("bands");
    assert!(matches!(
        missing.validate(),
        Err(InputError::Validation(
            InputValidationError::MissingTaskBlock { id }
        )) if id == "bands"
    ));

    let mut orphan = sample_input();
    orphan.workflow.tasks.pop();
    assert!(matches!(
        orphan.validate(),
        Err(InputError::Validation(
            InputValidationError::OrphanTaskBlock { id }
        )) if id == "dos"
    ));

    let mut duplicate = sample_input();
    duplicate.workflow.tasks.insert(1, "scf".to_owned());
    assert!(matches!(
        duplicate.validate(),
        Err(InputError::Validation(
            InputValidationError::DuplicateTaskId { id }
        )) if id == "scf"
    ));
}

#[test]
fn task_ids_and_sources_are_strictly_validated() {
    let mut illegal = sample_input();
    let task = illegal.task.remove("scf").unwrap();
    illegal.task.insert("bad.id".to_owned(), task);
    illegal.workflow.tasks[0] = "bad.id".to_owned();
    assert!(matches!(
        illegal.validate(),
        Err(InputError::Validation(
            InputValidationError::InvalidTaskId { id }
        )) if id == "bad.id"
    ));

    let mut forward = sample_input();
    forward.workflow.tasks.swap(0, 1);
    assert!(matches!(
        forward.validate(),
        Err(InputError::Validation(
            InputValidationError::ForwardSource {
                task_id,
                source_task
            }
        )) if task_id == "bands" && source_task == "scf"
    ));

    let mut incompatible = sample_input();
    let Task::DftDos { source, .. } = incompatible.task.get_mut("dos").unwrap() else {
        panic!("dos task changed kind");
    };
    *source = "bands.state".to_owned();
    assert!(matches!(
        incompatible.validate(),
        Err(InputError::Validation(
            InputValidationError::IncompatibleSource { task_id, .. }
        )) if task_id == "dos"
    ));
}

#[test]
fn physical_numbers_and_nonzero_controls_are_validated() {
    let mut input = sample_input();
    let Task::DftScf { electron_count, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    *electron_count = f64::INFINITY;
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(
            InputValidationError::NonFinite { .. }
        ))
    ));

    let mut input = sample_input();
    let Task::DftScf { k_mesh, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    k_mesh.mesh[1] = 0;
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(InputValidationError::Zero { .. }))
    ));

    let mut input = sample_input();
    let Task::DftScf { mixing, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    *mixing = Mixing::Broyden2 {
        beta: 0.4,
        history: 0,
    };
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(InputValidationError::TooShort {
            minimum: 2,
            actual: 0,
            ..
        }))
    ));

    let mut input = sample_input();
    let Task::DftScf { basis, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.envelope.g_cutoff = Some(f64::NAN);
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(
            InputValidationError::NonFinite { .. }
        ))
    ));

    let mut input = sample_input();
    let Task::DftScf { basis, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.l_max = 0;
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(InputValidationError::Zero { .. }))
    ));

    let mut input = sample_input();
    let Task::DftScf { relativity, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    *relativity = Relativity::SocSecondVariation {
        band_window: [4, 4],
    };
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(
            InputValidationError::InvalidRange { .. }
        ))
    ));
}

#[test]
fn prepare_is_filesystem_free_and_resolves_sources() {
    let input = sample_input();
    let recipes = BTreeMap::from([(
        PathBuf::from("recipes/si.toml"),
        ChannelRecipeArtifact {
            channels: vec![ChannelRecipeRecord {
                scope: ChannelScope::Species {
                    name: "Si".to_owned(),
                },
                identity: ChannelIdentity::ScalarL { n: 3, l: 1 },
                treatment: ChannelTreatment::Hdlo,
                derivative_order: 2,
                generator: ChannelEnergyGenerator::Explicit,
                seed: Some(Hartree(-0.2)),
                provenance: ChannelProvenance::BuiltIn,
            }],
        },
    )]);
    let prepared =
        prepare_input_with_recipes(&input, CheckpointFile::V1(sample_checkpoint()), &recipes)
            .unwrap();
    assert_eq!(prepared.tasks.len(), 3);
    let recipe = prepared.tasks[0].channel_recipe.as_ref().unwrap();
    assert_eq!(recipe.sites.len(), 1);
    let channels = &recipe.site("Si-1").unwrap().channels;
    assert!(channels.iter().any(|channel| {
        channel.identity == ChannelIdentity::ScalarL { n: 1, l: 0 }
            && channel.treatment == ChannelTreatment::Core
            && channel.provenance == ChannelProvenance::Species
    }));
    assert!(channels.iter().any(|channel| {
        channel.identity == ChannelIdentity::ScalarL { n: 3, l: 2 }
            && channel.treatment == ChannelTreatment::Lo
            && channel.seed == Some(Hartree(-0.15))
            && channel.provenance == ChannelProvenance::Site
    }));
    assert!(channels.iter().any(|channel| {
        channel.identity == ChannelIdentity::ScalarL { n: 3, l: 1 }
            && channel.treatment == ChannelTreatment::Hdlo
            && channel.derivative_order == 2
            && channel.provenance
                == ChannelProvenance::ExternalRecipe {
                    source: Some("recipes/si.toml".to_owned()),
                }
    }));
    assert!(prepared.tasks[1].channel_recipe.is_none());
    assert!(prepared.tasks[2].channel_recipe.is_none());
    let source = prepared.tasks[1].source.as_ref().unwrap();
    assert_eq!(source.task_index, 0);
    assert_eq!(source.task_id, "scf");
    assert_eq!(source.output, "state");
}

#[test]
fn filesystem_free_prepare_requires_each_named_recipe_artifact() {
    assert!(matches!(
        prepare_input(&sample_input(), CheckpointFile::V1(sample_checkpoint())),
        Err(InputError::MissingRecipeArtifact { task_id, path })
            if task_id == "scf" && path == Path::new("recipes/si.toml")
    ));
}

#[test]
fn path_loader_resolves_checkpoint_relative_to_input_parent() {
    let fixture = FixtureDirectory::new();
    let input_path = fixture.write_workflow();
    assert_eq!(input_path.parent(), Some(fixture.root()));
    let prepared = load_input_path(&input_path).unwrap();
    assert_eq!(
        prepared.checkpoint,
        sample_checkpoint().normalize_v2().unwrap()
    );
    assert_eq!(prepared.tasks[0].id, "scf");
    assert!(prepared.tasks[0].channel_recipe.is_some());

    let mut absolute = sample_input();
    absolute.checkpoint = PathBuf::from("/tmp/checkpoint.toml");
    assert!(matches!(
        absolute.validate(),
        Err(InputError::Validation(
            InputValidationError::AbsoluteCheckpointPath { .. }
        ))
    ));
}

#[test]
fn path_loader_reports_typed_missing_and_invalid_recipe_errors() {
    let missing_fixture = FixtureDirectory::new();
    let missing_input = missing_fixture.write_workflow();
    fs::remove_file(missing_fixture.root().join("recipes/si.toml")).unwrap();
    assert!(matches!(
        load_input_path(&missing_input),
        Err(InputError::ReadRecipe {
            task_id,
            path,
            ..
        }) if task_id == "scf" && path == missing_fixture.root().join("recipes/si.toml")
    ));

    let invalid_fixture = FixtureDirectory::new();
    let invalid_input = invalid_fixture.write_workflow();
    fs::write(
        invalid_fixture.root().join("recipes/si.toml"),
        "not valid TOML = [",
    )
    .unwrap();
    let error = load_input_path(&invalid_input).unwrap_err();
    let InputError::ChannelRecipe {
        task_id,
        path: Some(path),
        source,
    } = error
    else {
        panic!("expected a contextual channel recipe error");
    };
    assert_eq!(task_id, "scf");
    assert_eq!(path, PathBuf::from("recipes/si.toml"));
    assert!(matches!(*source, ChannelRecipeError::Decode(_)));
}
