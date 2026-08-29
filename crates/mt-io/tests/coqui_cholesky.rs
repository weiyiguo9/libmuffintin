//! Neutral CoQui-native Cholesky HDF5 roundtrip. This is not MLDUMP.

use std::f64::consts::FRAC_PI_8;
use std::path::{Path, PathBuf};

use hdf5_metno::File;
use hdf5_metno::types::{TypeDescriptor, VarLenUnicode};
use muffintin_io::{
    COQUI_CHOLESKY_COMPLEX_ATTR, COQUI_CHOLESKY_COMPLEX_VALUE, COQUI_CHOLESKY_GROUP,
    CoquiCholeskyHeader, CoquiCholeskyVqRef, CoquiCholeskyWriter, IoError, ValidationError,
    read_coqui_cholesky,
};

fn fixture_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

fn header() -> CoquiCholeskyHeader {
    CoquiCholeskyHeader {
        np: 2,
        nspin: 1,
        nspin_in_basis: 1,
        nkpts: 2,
        nbnd: 2,
        nbnd_aux: 0,
        tol: 1.0e-8,
        kpts: vec![0.0, 0.0, 0.0, FRAC_PI_8, 0.0, 0.0],
        qpts: vec![0.0, 0.0, 0.0, FRAC_PI_8, 0.0, 0.0],
        qk_to_kmq: vec![0, 1, 1, 0],
    }
}

fn vq_values(header: &CoquiCholeskyHeader, q: usize) -> Vec<f64> {
    let mut values = vec![0.0; header.vq_len().unwrap()];
    let n_k = header.n_k().unwrap();
    let n_band = header.n_band().unwrap();
    let np = header.n_aux().unwrap();
    for q_aux in 0..np {
        for k in 0..n_k {
            for i in 0..n_band {
                for j in 0..n_band {
                    let base = (((q_aux * n_k + k) * n_band + i) * n_band + j) * 2;
                    values[base] = (q * 100 + q_aux * 10 + k * 4 + i * 2 + j) as f64;
                    values[base + 1] = 0.25 * values[base];
                }
            }
        }
    }
    values
}

fn write_native_file(path: &Path) -> (CoquiCholeskyHeader, Vec<f64>, Vec<f64>) {
    let header = header();
    let v0 = vq_values(&header, 0);
    let v1 = vq_values(&header, 1);
    let mut writer = CoquiCholeskyWriter::create(path, &header).unwrap();
    writer
        .write_vq(CoquiCholeskyVqRef {
            q_index: 0,
            values: &v0,
        })
        .unwrap();
    writer
        .write_vq(CoquiCholeskyVqRef {
            q_index: 1,
            values: &v1,
        })
        .unwrap();
    writer.finish().unwrap();
    (header, v0, v1)
}

#[test]
fn coqui_cholesky_native_roundtrip_and_hdf_inspection() {
    let path = fixture_path("libmuffintin-io-coqui-cholesky.h5");
    let (header, v0, v1) = write_native_file(&path);

    let read = read_coqui_cholesky(&path).unwrap();
    assert_eq!(read.header, header);
    assert_eq!(read.records.len(), 2);
    assert_eq!(read.records[0].values, v0);
    assert_eq!(read.records[1].values, v1);

    let file = File::open(&path).unwrap();
    let group = file.group(COQUI_CHOLESKY_GROUP).unwrap();
    for name in [
        "Np",
        "nspin",
        "nspin_in_basis",
        "nkpts",
        "nbnd",
        "nbnd_aux",
        "tol",
        "kpts",
        "qpts",
        "qk_to_kmq",
        "Vq0",
        "Vq1",
    ] {
        group
            .dataset(name)
            .unwrap_or_else(|_| panic!("missing {name}"));
    }
    assert!(group.dataset("Np").unwrap().dtype().unwrap().is::<i32>());
    assert!(group.dataset("nspin").unwrap().dtype().unwrap().is::<i32>());
    assert!(group.dataset("tol").unwrap().dtype().unwrap().is::<f64>());
    assert!(group.dataset("kpts").unwrap().dtype().unwrap().is::<f64>());
    assert!(group.dataset("qpts").unwrap().dtype().unwrap().is::<f64>());
    assert_eq!(
        group.dataset("Np").unwrap().read_scalar::<i32>().unwrap(),
        2
    );
    assert_eq!(
        group.dataset("qk_to_kmq").unwrap().shape().as_slice(),
        &[2, 2]
    );
    assert_eq!(group.dataset("kpts").unwrap().shape().as_slice(), &[2, 3]);
    let vq0 = group.dataset("Vq0").unwrap();
    assert_eq!(vq0.shape().as_slice(), &[2, 1, 2, 2, 2, 2]);
    assert!(vq0.dtype().unwrap().is::<f64>());
    let attr = vq0.attr(COQUI_CHOLESKY_COMPLEX_ATTR).unwrap();
    let attr_desc = attr.dtype().unwrap().to_descriptor().unwrap();
    assert_eq!(
        attr_desc,
        TypeDescriptor::VarLenUnicode,
        "CoQui __complex__ must be variable-length UTF-8, got {attr_desc}"
    );
    assert_ne!(
        attr_desc,
        TypeDescriptor::VarLenAscii,
        "variable-length ASCII is not the CoQui __complex__ wire type"
    );
    assert_ne!(
        attr_desc,
        TypeDescriptor::FixedAscii(1),
        "fixed ASCII is not the CoQui __complex__ wire type"
    );
    let tag: VarLenUnicode = attr.read_scalar().unwrap();
    assert_eq!(tag.as_str(), COQUI_CHOLESKY_COMPLEX_VALUE);
}

#[test]
fn coqui_cholesky_rejects_nspin_context_before_create() {
    let path = fixture_path("libmuffintin-io-coqui-cholesky-nspin.h5");
    let _ = std::fs::remove_file(&path);
    let mut header = header();
    header.nspin = 2;
    header.nspin_in_basis = 2;
    let error = CoquiCholeskyWriter::create(&path, &header).unwrap_err();
    match error {
        IoError::Validation(ValidationError::InvalidValue {
            path: ref field, ..
        }) => {
            assert!(field.contains("nspin"), "expected nspin path, got {field}");
        }
        other => panic!("expected nspin validation, got {other}"),
    }
    assert!(
        !path.exists(),
        "invalid CoQui nspin must not create {}",
        path.display()
    );
}

#[test]
fn coqui_cholesky_rejects_convertible_f32_vq_dtype() {
    let path = fixture_path("libmuffintin-io-coqui-cholesky-f32-vq.h5");
    write_native_file(&path);

    {
        let file = File::open_rw(&path).unwrap();
        let group = file.group(COQUI_CHOLESKY_GROUP).unwrap();
        let dataset = group.dataset("Vq0").unwrap();
        let shape = dataset.shape();
        let values: Vec<f64> = dataset.read_raw().unwrap();
        drop(dataset);
        group.unlink("Vq0").unwrap();
        let narrowed: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        group
            .new_dataset::<f32>()
            .shape(shape)
            .create("Vq0")
            .unwrap()
            .write_raw(&narrowed)
            .unwrap();
    }

    let error = read_coqui_cholesky(&path).unwrap_err();
    match error {
        IoError::Validation(ValidationError::InvalidValue {
            ref path,
            ref expected,
            ref actual,
        }) => {
            assert_eq!(path, "/Interaction/Vq0/dtype");
            assert_eq!(expected, "float64");
            assert_eq!(actual, "float32");
        }
        other => panic!("expected exact f64 dtype rejection, got {other}"),
    }
}
