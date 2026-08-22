mod common;

use std::path::PathBuf;

use muffintin::{
    InputError, InputValidationError, MixingV1, RelativityV1, TaskV1, input_to_toml,
    load_input_path, parse_input_toml, prepare_input,
};

use common::{FixtureDirectory, sample_input, sample_snapshot};

#[test]
fn input_round_trips_deterministically_with_header_first() {
    let input = sample_input();
    let encoded = input_to_toml(&input).unwrap();
    assert!(encoded.starts_with(
        "format = \"libmuffintin-input\"\nversion = 1\nsnapshot = \"data/snapshot.toml\"\n"
    ));
    let decoded = parse_input_toml(&encoded).unwrap();
    assert_eq!(decoded, input);
    assert_eq!(input_to_toml(&decoded).unwrap(), encoded);
}

#[test]
fn nested_arrays_and_subblocks_are_preserved() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    assert!(encoded.contains("[task.scf.k-mesh]"));
    assert!(encoded.contains("[task.scf.basis]"));
    assert!(encoded.contains("[[task.scf.basis.local-orbitals]]"));
    assert!(encoded.contains("[task.scf.xc]"));
    assert!(encoded.contains("[[task.scf.core-states]]"));
    assert!(encoded.contains("[[task.bands.path]]"));

    let document: toml::Value = toml::from_str(&encoded).unwrap();
    let scf = &document["task"]["scf"];
    assert_eq!(scf["relativity"]["band-window"][0].as_integer(), Some(0));
    assert_eq!(scf["relativity"]["band-window"][1].as_integer(), Some(12));
    assert_eq!(
        scf["basis"]["local-orbitals"][0]["kappa"].as_integer(),
        Some(1)
    );
    assert_eq!(
        scf["basis"]["local-orbitals"][0]["kind"].as_str(),
        Some("lo")
    );

    let decoded = parse_input_toml(&encoded).unwrap();
    let TaskV1::DftScf { core_states, .. } = &decoded.task["scf"] else {
        panic!("scf task changed kind");
    };
    assert_eq!(core_states.len(), 1);

    let TaskV1::DftScf {
        basis, relativity, ..
    } = &decoded.task["scf"]
    else {
        unreachable!()
    };
    assert_eq!(basis.local_orbitals.len(), 1);
    assert!(matches!(
        relativity,
        RelativityV1::SpexSecondVariation {
            band_window: [0, 12]
        }
    ));
}

#[test]
fn header_unknown_fields_and_unknown_kinds_are_rejected() {
    let encoded = input_to_toml(&sample_input()).unwrap();
    let wrong_format = encoded.replacen("libmuffintin-input", "other-input", 1);
    assert!(matches!(
        parse_input_toml(&wrong_format),
        Err(InputError::InvalidFormat { .. })
    ));
    let wrong_version = encoded.replacen("version = 1", "version = 2", 1);
    assert!(matches!(
        parse_input_toml(&wrong_version),
        Err(InputError::UnsupportedVersion { found: 2, .. })
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
    let TaskV1::DftDos { source, .. } = incompatible.task.get_mut("dos").unwrap() else {
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
    let TaskV1::DftScf { electron_count, .. } = input.task.get_mut("scf").unwrap() else {
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
    let TaskV1::DftScf { k_mesh, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    k_mesh.mesh[1] = 0;
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(InputValidationError::Zero { .. }))
    ));

    let mut input = sample_input();
    let TaskV1::DftScf { mixing, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    *mixing = MixingV1::Broyden2 {
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
    let TaskV1::DftScf { core_states, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    core_states[0].kappa = 0;
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(InputValidationError::Zero { .. }))
    ));

    let mut input = sample_input();
    let TaskV1::DftScf { basis, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.local_orbitals[0].energy = f64::NAN;
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(
            InputValidationError::NonFinite { .. }
        ))
    ));

    let mut input = sample_input();
    let TaskV1::DftScf { basis, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    basis.local_orbitals[0].kappa = 0;
    assert!(matches!(
        input.validate(),
        Err(InputError::Validation(InputValidationError::Zero { .. }))
    ));

    let mut input = sample_input();
    let TaskV1::DftScf { relativity, .. } = input.task.get_mut("scf").unwrap() else {
        panic!("scf task changed kind");
    };
    *relativity = RelativityV1::SpexSecondVariation {
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
    let prepared = prepare_input(&input, sample_snapshot()).unwrap();
    assert_eq!(prepared.tasks.len(), 3);
    let source = prepared.tasks[1].source.as_ref().unwrap();
    assert_eq!(source.task_index, 0);
    assert_eq!(source.task_id, "scf");
    assert_eq!(source.output, "state");
}

#[test]
fn path_loader_resolves_snapshot_relative_to_input_parent() {
    let fixture = FixtureDirectory::new();
    let input_path = fixture.write_workflow();
    assert_eq!(input_path.parent(), Some(fixture.root()));
    let prepared = load_input_path(&input_path).unwrap();
    assert_eq!(prepared.snapshot, sample_snapshot());
    assert_eq!(prepared.tasks[0].id, "scf");

    let mut absolute = sample_input();
    absolute.snapshot = PathBuf::from("/tmp/snapshot.toml");
    assert!(matches!(
        absolute.validate(),
        Err(InputError::Validation(
            InputValidationError::AbsoluteSnapshotPath { .. }
        ))
    ));
}
