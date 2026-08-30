//! Roundtrip and validation tests for the `libmuffintin.spexsym` v1 format.

use std::path::PathBuf;

use muffintin_io::{read_spex_symmetry_v1, write_spex_symmetry_v1};
use muffintin_symmetry::SymmetryOperation;
use muffintin_symmetry::spex::{KpointIrreps, SpexSymmetryImport, SubspaceIrreps};
use num_complex::Complex64;

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn sample_import() -> SpexSymmetryImport {
    let identity = SymmetryOperation {
        rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        translation: [0.0, 0.0, 0.0],
        time_reversal: false,
    };
    let screw = SymmetryOperation {
        rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        translation: [0.0, 0.0, 0.5],
        time_reversal: false,
    };
    let trs = SymmetryOperation {
        rotation: [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        translation: [0.0, 0.0, 0.0],
        time_reversal: true,
    };
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let phase = Complex64::new(0.0, 1.0);
    SpexSymmetryImport {
        operations: vec![identity, screw, trs],
        inverse: vec![0, 1, 2],
        atom_map: vec![vec![0, 1], vec![1, 0], vec![0, 1]],
        kpoints: vec![[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]],
        irreducible_count: 2,
        parent: vec![0, 1, 1],
        parent_operation: vec![0, 0, 2],
        irreps: vec![KpointIrreps {
            kpoint_index: 0,
            spin: 0,
            little_group: vec![0, 1],
            subspaces: vec![
                SubspaceIrreps {
                    first_band: 0,
                    dimension: 1,
                    matrices: vec![vec![one], vec![phase]],
                },
                SubspaceIrreps {
                    first_band: 1,
                    dimension: 2,
                    matrices: vec![vec![one, zero, zero, one], vec![zero, -one, one, zero]],
                },
            ],
        }],
    }
}

#[test]
fn roundtrip_preserves_the_import() {
    let path = fixture_path("libmuffintin_spexsym_v1_roundtrip.h5");
    let import = sample_import();
    write_spex_symmetry_v1(&path, "spex-6.00pre36-test", &import).unwrap();
    let file = read_spex_symmetry_v1(&path).unwrap();
    assert_eq!(file.producer_version, "spex-6.00pre36-test");
    assert_eq!(file.import, import);
    let dataset = file.import.dataset();
    assert_eq!(dataset.equivalent_atoms, vec![0, 0]);
    std::fs::remove_file(&path).ok();
}

#[test]
fn out_of_range_little_group_index_is_rejected() {
    let path = fixture_path("libmuffintin_spexsym_v1_bad_index.h5");
    let mut import = sample_import();
    import.irreps[0].little_group = vec![0, 7];
    write_spex_symmetry_v1(&path, "spex-test", &import).unwrap();
    let error = read_spex_symmetry_v1(&path).unwrap_err();
    assert!(error.to_string().contains("little_group"), "{error}");
    std::fs::remove_file(&path).ok();
}
