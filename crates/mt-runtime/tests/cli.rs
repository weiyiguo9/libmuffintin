mod common;

use std::fs;
use std::process::Command;

use common::FixtureDirectory;

#[test]
fn manifest_declares_exactly_one_binary_target() {
    let manifest =
        fs::read_to_string(format!("{}/Cargo.toml", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let manifest: toml::Value = toml::from_str(&manifest).unwrap();
    assert_eq!(
        manifest["package"]["autobins"].as_bool(),
        Some(false),
        "automatic binary discovery must stay disabled"
    );
    let binaries = manifest["bin"].as_array().unwrap();
    assert_eq!(binaries.len(), 1);
    assert_eq!(binaries[0]["name"].as_str(), Some("muffintin"));
    assert_eq!(binaries[0]["path"].as_str(), Some("src/main.rs"));
}

#[test]
fn cli_requires_exactly_one_input_path() {
    let binary = env!("CARGO_BIN_EXE_muffintin");
    let no_arguments = Command::new(binary).output().unwrap();
    assert_eq!(no_arguments.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&no_arguments.stderr).contains("usage:"));

    let too_many = Command::new(binary)
        .args(["one.toml", "two.toml"])
        .output()
        .unwrap();
    assert_eq!(too_many.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&too_many.stderr).contains("usage:"));
}

#[test]
fn cli_reports_parse_failures_and_executes_a_supported_checkpoint() {
    let binary = env!("CARGO_BIN_EXE_muffintin");
    let fixture = FixtureDirectory::new();
    let bad_path = fixture.root().join("bad.toml");
    fs::write(&bad_path, "not valid TOML = [").unwrap();
    let bad = Command::new(binary).arg(&bad_path).output().unwrap();
    assert_eq!(bad.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("could not decode input TOML"));

    let input_path = fixture.write_supported_workflow();
    let valid = Command::new(binary).arg(input_path).output().unwrap();
    assert_eq!(valid.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&valid.stderr);
    assert!(!stderr.contains("needs a material physics kernel"));
    assert!(String::from_utf8_lossy(&valid.stdout).contains("task scf scf iterations="));
}
