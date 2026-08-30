mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use muffintin::{
    ChannelEnergyGenerator, ChannelIdentity, ChannelProvenance, ChannelRecipeArtifact,
    ChannelRecipeRecord, ChannelScope, ChannelTreatment, InputError, SingleDftScfConfigError, Task,
    prepare_input, prepare_input_with_recipes, single_dft_scf_config,
};
use muffintin_core::Hartree;
use muffintin_io::CheckpointFile;

use common::{supported_checkpoint, supported_input};

#[test]
fn single_dft_scf_config_rejects_zero_scf_tasks() {
    let mut prepared = prepare_input(
        &supported_input(),
        CheckpointFile::V1(supported_checkpoint()),
    )
    .unwrap();
    prepared.tasks.clear();

    assert!(matches!(
        single_dft_scf_config(&prepared),
        Err(SingleDftScfConfigError::TaskCount { count: 0 })
    ));
}

#[test]
fn single_dft_scf_config_maps_one_scf_task() {
    let prepared = prepare_input(
        &supported_input(),
        CheckpointFile::V1(supported_checkpoint()),
    )
    .unwrap();

    let config = single_dft_scf_config(&prepared).unwrap();
    assert_eq!(config.electron_count, 1.0);
    assert_eq!(config.k_mesh.divisions, [1, 1, 1]);
}

#[test]
fn single_dft_scf_config_rejects_multiple_scf_tasks() {
    let mut prepared = prepare_input(
        &supported_input(),
        CheckpointFile::V1(supported_checkpoint()),
    )
    .unwrap();
    let mut second = prepared.tasks[0].clone();
    second.id = "second-scf".to_owned();
    prepared.tasks.push(second);

    assert!(matches!(
        single_dft_scf_config(&prepared),
        Err(SingleDftScfConfigError::TaskCount { count: 2 })
    ));
}

#[test]
fn single_dft_scf_config_propagates_mapper_error() {
    let mut input = supported_input();
    let Task::DftScf { basis, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    let recipe_path = PathBuf::from("recipes/h.toml");
    basis.recipe = Some(recipe_path.clone());
    let identity = ChannelIdentity::ScalarL { n: 2, l: 1 };
    let recipes = BTreeMap::from([(
        recipe_path,
        ChannelRecipeArtifact {
            channels: vec![ChannelRecipeRecord {
                scope: ChannelScope::Site {
                    name: "H-1".to_owned(),
                },
                identity,
                treatment: ChannelTreatment::Hdlo,
                derivative_order: 3,
                generator: ChannelEnergyGenerator::Explicit,
                seed: Some(Hartree(-0.1)),
                provenance: ChannelProvenance::BuiltIn,
            }],
        },
    )]);
    let prepared =
        prepare_input_with_recipes(&input, CheckpointFile::V1(supported_checkpoint()), &recipes)
            .unwrap();

    assert!(matches!(
        single_dft_scf_config(&prepared),
        Err(SingleDftScfConfigError::Input(
            InputError::DerivativeOrderNotImplemented {
                task_id,
                site,
                identity: found_identity,
                derivative_order: 3,
            }
        )) if task_id == "scf" && site == "H-1" && found_identity == identity
    ));
}
