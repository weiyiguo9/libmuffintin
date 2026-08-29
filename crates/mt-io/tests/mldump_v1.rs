//! Focused MLDUMP v1 HDF5 roundtrip and inspectable-structure gate.

use std::f64::consts::PI;
use std::path::PathBuf;

use hdf5_metno::File;
use hdf5_metno::types::VarLenUnicode;
use muffintin_io::{
    IoError, MLDUMP_SCHEMA_NAME, MLDUMP_SCHEMA_VERSION, MLDUMP_STATUS_ABSENT_NOT_COMPUTED,
    MLDUMP_STATUS_PRESENT, MLDUMP_UNIT_ENERGY, MLDUMP_UNIT_G_UMKLAPP, MLDUMP_UNIT_INVERSE_LENGTH,
    MLDUMP_UNIT_K_Q, MLDUMP_UNIT_LENGTH, MLDUMP_UNIT_VOLUME, MldumpGeometryV1, MldumpKMinusQV1,
    MldumpKPointV1, MldumpMeshV1, MldumpMetaV1, MldumpQEntryV1, MldumpRadialMeshV1, MldumpSiteV1,
    MldumpV1, ValidationError, read_mldump_v1, write_mldump_v1,
};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn dump() -> MldumpV1 {
    MldumpV1::new(
        MldumpMetaV1 {
            producer_name: "libmuffintin-io-test".to_owned(),
            producer_version: "0.1.0".to_owned(),
            source_revision: "a9036816fd4410224832f96ea8942831ec79495f".to_owned(),
            feature_representation: "spinor_first_variation".to_owned(),
        },
        MldumpGeometryV1 {
            direct_basis_bohr: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            reciprocal_basis_inv_bohr: [
                [2.0 * PI / 8.0, 0.0, 0.0],
                [0.0, 2.0 * PI / 8.0, 0.0],
                [0.0, 0.0, 2.0 * PI / 8.0],
            ],
            cell_volume_bohr3: 512.0,
            sites: vec![
                MldumpSiteV1 {
                    species: Some("H".to_owned()),
                    label: Some("H-1".to_owned()),
                    position_bohr: [0.0, 0.0, 0.0],
                    radius_bohr: 1.0,
                    radial_mesh: MldumpRadialMeshV1 {
                        first_bohr: 1.0e-4,
                        log_increment: 0.05,
                        point_count: 61,
                    },
                },
                MldumpSiteV1 {
                    species: Some("He".to_owned()),
                    label: None,
                    position_bohr: [4.0, 0.0, 0.0],
                    radius_bohr: 1.2,
                    radial_mesh: MldumpRadialMeshV1 {
                        first_bohr: 2.0e-4,
                        log_increment: 0.04,
                        point_count: 49,
                    },
                },
            ],
        },
        MldumpMeshV1 {
            k_points: vec![
                MldumpKPointV1 {
                    fractional: [0.0, 0.0, 0.0],
                    weight: 0.5,
                },
                MldumpKPointV1 {
                    fractional: [0.5, 0.0, 0.0],
                    weight: 0.5,
                },
            ],
            q_entries: vec![
                MldumpQEntryV1 {
                    input_fractional: [0.0, 0.0, 0.0],
                    canonical_fractional: [0.0, 0.0, 0.0],
                    global_umklapp: [0, 0, 0],
                    k_minus_q: vec![
                        MldumpKMinusQV1 {
                            k_index: 0,
                            mapped_index: 0,
                            g_wrap: [0, 0, 0],
                        },
                        MldumpKMinusQV1 {
                            k_index: 1,
                            mapped_index: 1,
                            g_wrap: [0, 0, 0],
                        },
                    ],
                },
                MldumpQEntryV1 {
                    input_fractional: [1.5, 0.0, 0.0],
                    canonical_fractional: [0.5, 0.0, 0.0],
                    global_umklapp: [1, 0, 0],
                    k_minus_q: vec![
                        MldumpKMinusQV1 {
                            k_index: 0,
                            mapped_index: 1,
                            g_wrap: [-1, 0, 0],
                        },
                        MldumpKMinusQV1 {
                            k_index: 1,
                            mapped_index: 0,
                            g_wrap: [0, 0, 0],
                        },
                    ],
                },
            ],
        },
    )
}

fn attr_string(file: &File, path: &str) -> String {
    let value: VarLenUnicode = file.attr(path).unwrap().read_scalar().unwrap();
    value.as_str().to_owned()
}

fn group_status(file: &File, group: &str) -> String {
    let value: VarLenUnicode = file
        .group(group)
        .unwrap()
        .attr("status")
        .unwrap()
        .read_scalar()
        .unwrap();
    value.as_str().to_owned()
}

fn axes(file: &File, dataset: &str) -> Vec<String> {
    file.dataset(dataset)
        .unwrap()
        .attr("axes")
        .unwrap()
        .read_raw::<VarLenUnicode>()
        .unwrap()
        .iter()
        .map(|value| value.as_str().to_owned())
        .collect()
}

#[test]
fn mldump_v1_roundtrip_has_inspectable_hdf5_structure() {
    let path = fixture_path("libmuffintin-mldump-v1-fixture.h5");
    let original = dump();
    write_mldump_v1(&path, &original).unwrap();
    let read = read_mldump_v1(&path).unwrap();
    assert_eq!(read, original);

    let file = File::open(&path).unwrap();
    assert_eq!(attr_string(&file, "schema_name"), MLDUMP_SCHEMA_NAME);
    let version: u32 = file.attr("schema_version").unwrap().read_scalar().unwrap();
    assert_eq!(version, MLDUMP_SCHEMA_VERSION);
    let mut members = file.member_names().unwrap();
    members.sort();
    assert_eq!(
        members,
        [
            "coulomb", "exchange", "geometry", "mesh", "meta", "mpb", "orbitals", "products",
            "thc", "units"
        ]
    );

    assert_eq!(group_status(&file, "meta"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "units"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "geometry"), MLDUMP_STATUS_PRESENT);
    assert_eq!(group_status(&file, "mesh"), MLDUMP_STATUS_PRESENT);
    for absent in ["orbitals", "products", "mpb", "thc", "coulomb"] {
        assert_eq!(
            group_status(&file, absent),
            MLDUMP_STATUS_ABSENT_NOT_COMPUTED
        );
        assert!(file.group(absent).unwrap().datasets().unwrap().is_empty());
    }
    assert_eq!(group_status(&file, "exchange"), MLDUMP_STATUS_PRESENT);
    assert!(
        file.group("exchange")
            .unwrap()
            .datasets()
            .unwrap()
            .is_empty()
    );
    assert!(
        !file
            .group("exchange")
            .unwrap()
            .link_exists("total_relation")
    );
    for seam in ["valence", "core", "total"] {
        assert_eq!(
            group_status(&file, &format!("exchange/{seam}")),
            MLDUMP_STATUS_ABSENT_NOT_COMPUTED
        );
        assert!(
            file.group("exchange")
                .unwrap()
                .group(seam)
                .unwrap()
                .datasets()
                .unwrap()
                .is_empty()
        );
    }

    let length: VarLenUnicode = file
        .group("units")
        .unwrap()
        .attr("length")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(length.as_str(), MLDUMP_UNIT_LENGTH);
    let inverse: VarLenUnicode = file
        .group("units")
        .unwrap()
        .attr("inverse_length")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(inverse.as_str(), MLDUMP_UNIT_INVERSE_LENGTH);
    let volume: VarLenUnicode = file
        .group("units")
        .unwrap()
        .attr("volume")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(volume.as_str(), MLDUMP_UNIT_VOLUME);
    let energy: VarLenUnicode = file
        .group("units")
        .unwrap()
        .attr("energy")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(energy.as_str(), MLDUMP_UNIT_ENERGY);
    let kq: VarLenUnicode = file
        .group("units")
        .unwrap()
        .attr("k_q_coordinates")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(kq.as_str(), MLDUMP_UNIT_K_Q);
    let gum: VarLenUnicode = file
        .group("units")
        .unwrap()
        .attr("g_umklapp")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(gum.as_str(), MLDUMP_UNIT_G_UMKLAPP);

    assert_eq!(
        axes(&file, "geometry/direct_basis"),
        ["primitive_vector", "cartesian"]
    );
    assert_eq!(
        axes(&file, "geometry/reciprocal_basis"),
        ["primitive_vector", "cartesian"]
    );
    assert_eq!(
        axes(&file, "geometry/site_positions"),
        ["site", "cartesian"]
    );
    assert_eq!(axes(&file, "geometry/site_radii"), ["site"]);
    assert_eq!(axes(&file, "mesh/k_fractional"), ["k", "reciprocal_axis"]);
    assert_eq!(
        axes(&file, "mesh/q_global_umklapp"),
        ["q", "reciprocal_axis"]
    );
    assert_eq!(
        axes(&file, "mesh/k_minus_q_g_wrap"),
        ["q", "k", "reciprocal_axis"]
    );
    assert_eq!(
        file.dataset("geometry/direct_basis").unwrap().shape(),
        [3, 3]
    );
    assert_eq!(
        file.dataset("geometry/site_positions").unwrap().shape(),
        [2, 3]
    );
    assert_eq!(
        file.dataset("mesh/k_minus_q_g_wrap").unwrap().shape(),
        [2, 2, 3]
    );
    let umklapp = file
        .dataset("mesh/q_global_umklapp")
        .unwrap()
        .read_raw::<i32>()
        .unwrap();
    assert_eq!(umklapp, vec![0, 0, 0, 1, 0, 0]);
    let wraps = file
        .dataset("mesh/k_minus_q_g_wrap")
        .unwrap()
        .read_raw::<i32>()
        .unwrap();
    assert_eq!(&wraps[6..9], [-1, 0, 0]);
    let origin: i64 = file
        .group("meta")
        .unwrap()
        .attr("index_origin")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(origin, 0);
    let encoding: VarLenUnicode = file
        .group("meta")
        .unwrap()
        .attr("complex_encoding")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(encoding.as_str(), "final_re_im_axis");
    let _ = attr_string(&file, "schema_name");
}

#[test]
fn mldump_v1_rejects_unsupported_schema_version() {
    let path = fixture_path("libmuffintin-mldump-v1-bad-version.h5");
    write_mldump_v1(&path, &dump()).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.delete_attr("schema_version").unwrap();
        file.new_attr::<u32>()
            .create("schema_version")
            .unwrap()
            .write_scalar(&99_u32)
            .unwrap();
    }
    assert!(matches!(
        read_mldump_v1(&path),
        Err(IoError::UnsupportedVersion {
            format: MLDUMP_SCHEMA_NAME,
            supported: MLDUMP_SCHEMA_VERSION,
            found: 99,
        })
    ));
}

#[test]
fn mldump_v1_rejects_inconsistent_q_and_k_maps() {
    struct Case {
        label: &'static str,
        mutate: fn(&mut MldumpV1),
        path: &'static str,
    }
    let cases = [
        Case {
            label: "q_input_not_canonical_plus_global_umklapp",
            mutate: |dump| dump.mesh.q_entries[1].input_fractional[0] = 0.0,
            path: "mesh.q_entries[1].input_fractional[0]",
        },
        Case {
            label: "k_minus_q_record_not_in_k_order",
            mutate: |dump| dump.mesh.q_entries[1].k_minus_q.swap(0, 1),
            path: "mesh.q_entries[1].k_minus_q[0].k_index",
        },
        Case {
            label: "k_minus_q_wrap_identity",
            mutate: |dump| dump.mesh.q_entries[1].k_minus_q[0].g_wrap[0] = 0,
            path: "mesh.q_entries[1].k_minus_q[0].g_wrap[0]",
        },
    ];
    for case in cases {
        let mut dump = dump();
        (case.mutate)(&mut dump);
        match dump.validate() {
            Err(IoError::Validation(ValidationError::InvalidValue { path, .. })) => {
                assert_eq!(path, case.path, "{}", case.label);
            }
            other => panic!("{}: expected InvalidValue, got {other:?}", case.label),
        }
        assert!(
            write_mldump_v1(fixture_path("libmuffintin-mldump-v1-invalid-map.h5"), &dump).is_err(),
            "{}",
            case.label
        );
    }
}

#[test]
fn mldump_v1_rejects_convertible_but_wrong_numeric_dtype() {
    struct Case {
        label: &'static str,
        path_suffix: &'static str,
        expected: &'static str,
        rewrite: fn(&File),
    }
    let cases = [
        Case {
            label: "dataset_k_weights_f32",
            path_suffix: "k_weights/dtype",
            expected: "f64",
            rewrite: |file| {
                let mesh = file.group("mesh").unwrap();
                mesh.unlink("k_weights").unwrap();
                let dataset = mesh
                    .new_dataset::<f32>()
                    .shape([2])
                    .create("k_weights")
                    .unwrap();
                dataset.write_raw(&[0.5_f32, 0.5_f32]).unwrap();
                let axes = ["k"]
                    .iter()
                    .map(|axis| axis.parse::<VarLenUnicode>().unwrap())
                    .collect::<Vec<_>>();
                dataset
                    .new_attr::<VarLenUnicode>()
                    .shape([1])
                    .create("axes")
                    .unwrap()
                    .write_raw(axes.as_slice())
                    .unwrap();
            },
        },
        Case {
            label: "attr_index_origin_i32",
            path_suffix: "/meta/@index_origin/dtype",
            expected: "i64",
            rewrite: |file| {
                let meta = file.group("meta").unwrap();
                meta.delete_attr("index_origin").unwrap();
                meta.new_attr::<i32>()
                    .create("index_origin")
                    .unwrap()
                    .write_scalar(&0_i32)
                    .unwrap();
            },
        },
    ];
    for case in cases {
        let path = fixture_path(&format!(
            "libmuffintin-mldump-v1-wrong-dtype-{}.h5",
            case.label
        ));
        write_mldump_v1(&path, &dump()).unwrap();
        {
            let file = File::open_rw(&path).unwrap();
            (case.rewrite)(&file);
        }
        match read_mldump_v1(&path) {
            Err(IoError::Validation(ValidationError::InvalidValue { path, expected, .. })) => {
                assert!(path.ends_with(case.path_suffix), "{}: {path}", case.label);
                assert_eq!(expected, case.expected, "{}", case.label);
            }
            other => panic!("{}: expected dtype InvalidValue, got {other:?}", case.label),
        }
    }
}

#[test]
fn mldump_v1_rejects_nested_group_under_absent_exchange_child() {
    let path = fixture_path("libmuffintin-mldump-v1-absent-child-payload.h5");
    write_mldump_v1(&path, &dump()).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        file.group("exchange")
            .unwrap()
            .group("valence")
            .unwrap()
            .create_group("nested")
            .unwrap();
    }
    match read_mldump_v1(&path) {
        Err(IoError::Validation(ValidationError::InvalidValue { path, actual, .. })) => {
            assert_eq!(path, "/exchange/valence");
            assert!(actual.contains("nested"), "{actual}");
        }
        other => panic!("expected absent-child InvalidValue, got {other:?}"),
    }
}
