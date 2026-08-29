//! CoQui-native single-file Cholesky ERI HDF5.
//!
//! This module is **not** MLDUMP. The on-disk tree matches live CoQui
//! `chol_reader_t` (branch `wg-dev` @
//! `a19774d03fb979bd852fae4f7f95c045a4cbca78`) `add_meta_data` and single-file
//! `write_Vq`: group `/Interaction`, scalar `i32` members
//! `Np,nspin,nspin_in_basis,nkpts,nbnd,nbnd_aux=0`, scalar `f64` `tol`,
//! C-order `f64` `kpts[nk,3]` / `qpts[nq,3]`, `i32` `qk_to_kmq[nq,nk]`, and
//! datasets `Vq{iq}` stored as native `f64` `[Np,nspin_in_basis,nk,nbnd,nbnd,2]`
//! with scalar variable-length UTF-8 `__complex__="1"` (`H5T_STRING` /
//! `H5T_VARIABLE` / `H5T_CSET_UTF8`, matching the checked-in CoQui `Vq`
//! fixture; not MLDUMP complex DTO and not fixed ASCII). The contract is
//! unversioned and private; a schema or dtype mismatch is a hard failure.
//! Runtime, Coulomb, and THC types stay out of this crate.

use std::path::Path;
use std::str::FromStr;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Container, File, Group, H5Type};

use crate::error::{IoError, ValidationError, finite};

/// `/Interaction` group name used by `chol_reader_t`.
pub const COQUI_CHOLESKY_GROUP: &str = "Interaction";
/// Dataset attribute written by CoQui `nda::h5_write` / pw2coqui `add_complex`.
pub const COQUI_CHOLESKY_COMPLEX_ATTR: &str = "__complex__";
/// Required value of [`COQUI_CHOLESKY_COMPLEX_ATTR`].
pub const COQUI_CHOLESKY_COMPLEX_VALUE: &str = "1";

/// Owned `/Interaction` metadata for one CoQui-native Cholesky file.
#[derive(Clone, Debug, PartialEq)]
pub struct CoquiCholeskyHeader {
    pub np: i32,
    pub nspin: i32,
    pub nspin_in_basis: i32,
    pub nkpts: i32,
    pub nbnd: i32,
    pub nbnd_aux: i32,
    pub tol: f64,
    /// C-order `[nkpts, 3]` Cartesian reciprocal coordinates.
    pub kpts: Vec<f64>,
    /// C-order `[nq, 3]` canonical Cartesian reciprocal coordinates.
    pub qpts: Vec<f64>,
    /// C-order `[nq, nkpts]` map `k -> k-q` mesh indices.
    pub qk_to_kmq: Vec<i32>,
}

impl CoquiCholeskyHeader {
    /// Number of $q$ records implied by `qpts`.
    pub fn n_q(&self) -> Result<usize, ValidationError> {
        if self.qpts.len() % 3 != 0 {
            return Err(ValidationError::InvalidValue {
                path: "/Interaction/qpts".to_owned(),
                expected: "length multiple of 3".to_owned(),
                actual: self.qpts.len().to_string(),
            });
        }
        Ok(self.qpts.len() / 3)
    }

    /// Number of $k$ points as `usize`.
    pub fn n_k(&self) -> Result<usize, ValidationError> {
        i32_as_usize("/Interaction/nkpts", self.nkpts)
    }

    /// Band count as `usize`.
    pub fn n_band(&self) -> Result<usize, ValidationError> {
        i32_as_usize("/Interaction/nbnd", self.nbnd)
    }

    /// Auxiliary Cholesky count as `usize`.
    pub fn n_aux(&self) -> Result<usize, ValidationError> {
        i32_as_usize("/Interaction/Np", self.np)
    }

    /// Reject a header that cannot be a CoQui `chol_reader_t` single-file tree.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.nspin != 1 || self.nspin_in_basis != 1 {
            return Err(ValidationError::InvalidValue {
                path: "/Interaction/nspin".to_owned(),
                expected: "nspin = nspin_in_basis = 1".to_owned(),
                actual: format!(
                    "nspin={} nspin_in_basis={}",
                    self.nspin, self.nspin_in_basis
                ),
            });
        }
        if self.nbnd_aux != 0 {
            return Err(ValidationError::InvalidValue {
                path: "/Interaction/nbnd_aux".to_owned(),
                expected: "0".to_owned(),
                actual: self.nbnd_aux.to_string(),
            });
        }
        if self.np <= 0 {
            return Err(ValidationError::NotPositive {
                path: "/Interaction/Np".to_owned(),
                value: f64::from(self.np),
            });
        }
        if self.nkpts <= 0 {
            return Err(ValidationError::NotPositive {
                path: "/Interaction/nkpts".to_owned(),
                value: f64::from(self.nkpts),
            });
        }
        if self.nbnd <= 0 {
            return Err(ValidationError::NotPositive {
                path: "/Interaction/nbnd".to_owned(),
                value: f64::from(self.nbnd),
            });
        }
        finite("/Interaction/tol", self.tol)?;
        if self.tol < 0.0 {
            return Err(ValidationError::InvalidValue {
                path: "/Interaction/tol".to_owned(),
                expected: "nonnegative finite f64".to_owned(),
                actual: self.tol.to_string(),
            });
        }
        let n_k = self.n_k()?;
        let n_q = self.n_q()?;
        if n_q == 0 {
            return Err(ValidationError::Empty {
                path: "/Interaction/qpts".to_owned(),
            });
        }
        if n_q != n_k {
            return Err(ValidationError::InvalidValue {
                path: "/Interaction/qpts".to_owned(),
                expected: format!("full-BZ n_q = n_k = {n_k}"),
                actual: n_q.to_string(),
            });
        }
        if self.kpts.len() != n_k * 3 {
            return Err(ValidationError::LengthMismatch {
                path: "/Interaction/kpts".to_owned(),
                expected: n_k * 3,
                actual: self.kpts.len(),
            });
        }
        if self.qk_to_kmq.len() != n_q * n_k {
            return Err(ValidationError::LengthMismatch {
                path: "/Interaction/qk_to_kmq".to_owned(),
                expected: n_q * n_k,
                actual: self.qk_to_kmq.len(),
            });
        }
        for (index, value) in self.kpts.iter().chain(self.qpts.iter()).enumerate() {
            finite(format!("/Interaction/kpts|qpts[{index}]"), *value)?;
        }
        let n_k_i32 = self.nkpts;
        for (index, mapped) in self.qk_to_kmq.iter().enumerate() {
            if *mapped < 0 || *mapped >= n_k_i32 {
                return Err(ValidationError::InvalidValue {
                    path: format!("/Interaction/qk_to_kmq[{index}]"),
                    expected: format!("0..{n_k_i32}"),
                    actual: mapped.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Flat length of one `Vq` dataset: `Np * 1 * nk * nbnd * nbnd * 2`.
    pub fn vq_len(&self) -> Result<usize, ValidationError> {
        let np = self.n_aux()?;
        let n_k = self.n_k()?;
        let n_band = self.n_band()?;
        np.checked_mul(n_k)
            .and_then(|value| value.checked_mul(n_band))
            .and_then(|value| value.checked_mul(n_band))
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| ValidationError::InvalidValue {
                path: "/Interaction/Vq".to_owned(),
                expected: "representable Np*nk*nbnd*nbnd*2".to_owned(),
                actual: format!("Np={} nk={} nbnd={}", self.np, self.nkpts, self.nbnd),
            })
    }

    /// HDF5 shape of one `Vq` dataset.
    pub fn vq_shape(&self) -> Result<[usize; 6], ValidationError> {
        Ok([
            self.n_aux()?,
            1,
            self.n_k()?,
            self.n_band()?,
            self.n_band()?,
            2,
        ])
    }
}

/// Borrowed one-$q$ Cholesky tensor in CoQui native packing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoquiCholeskyVqRef<'a> {
    pub q_index: usize,
    /// Native `f64` `[Np,1,nk,nbnd,nbnd,2]` C-order, final re/im axis.
    pub values: &'a [f64],
}

/// Owned one-$q$ Cholesky tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct CoquiCholeskyVq {
    pub q_index: usize,
    pub values: Vec<f64>,
}

/// Owned CoQui-native Cholesky file.
#[derive(Clone, Debug, PartialEq)]
pub struct CoquiCholeskyFile {
    pub header: CoquiCholeskyHeader,
    pub records: Vec<CoquiCholeskyVq>,
}

/// Streaming writer for one single-file CoQui Cholesky container.
#[derive(Debug)]
pub struct CoquiCholeskyWriter {
    file: File,
    header: CoquiCholeskyHeader,
    n_q: usize,
    written: usize,
}

impl CoquiCholeskyWriter {
    /// Validate `header` then create the destination file with `/Interaction`
    /// metadata. $q$ datasets follow [`Self::write_vq`].
    pub fn create(path: impl AsRef<Path>, header: &CoquiCholeskyHeader) -> Result<Self, IoError> {
        header.validate()?;
        let n_q = header.n_q()?;
        let file = File::create(path)?;
        let group = file.create_group(COQUI_CHOLESKY_GROUP)?;
        write_i32_scalar(&group, "Np", header.np)?;
        write_f64_scalar(&group, "tol", header.tol)?;
        write_i32_scalar(&group, "nspin", header.nspin)?;
        write_i32_scalar(&group, "nspin_in_basis", header.nspin_in_basis)?;
        write_i32_scalar(&group, "nkpts", header.nkpts)?;
        write_i32_scalar(&group, "nbnd", header.nbnd)?;
        write_i32_scalar(&group, "nbnd_aux", header.nbnd_aux)?;
        write_f64_matrix(&group, "kpts", header.n_k()?, 3, &header.kpts)?;
        write_f64_matrix(&group, "qpts", n_q, 3, &header.qpts)?;
        write_i32_matrix(&group, "qk_to_kmq", n_q, header.n_k()?, &header.qk_to_kmq)?;
        Ok(Self {
            file,
            header: header.clone(),
            n_q,
            written: 0,
        })
    }

    /// Write `/Interaction/Vq{q}` in required section order.
    pub fn write_vq(&mut self, record: CoquiCholeskyVqRef<'_>) -> Result<(), IoError> {
        if self.written >= self.n_q {
            return Err(ValidationError::InvalidValue {
                path: "/Interaction".to_owned(),
                expected: format!("{} Vq datasets", self.n_q),
                actual: (self.written + 1).to_string(),
            }
            .into());
        }
        if record.q_index != self.written {
            return Err(ValidationError::InvalidValue {
                path: format!("/Interaction/Vq{}", record.q_index),
                expected: format!("q_index {}", self.written),
                actual: record.q_index.to_string(),
            }
            .into());
        }
        let expected = self.header.vq_len()?;
        if record.values.len() != expected {
            return Err(ValidationError::LengthMismatch {
                path: format!("/Interaction/Vq{}", record.q_index),
                expected,
                actual: record.values.len(),
            }
            .into());
        }
        for (index, value) in record.values.iter().enumerate() {
            finite(format!("/Interaction/Vq{}/{index}", record.q_index), *value)?;
        }
        let group = self.file.group(COQUI_CHOLESKY_GROUP)?;
        let name = format!("Vq{}", record.q_index);
        let shape = self.header.vq_shape()?;
        let dataset = group
            .new_dataset::<f64>()
            .shape(shape)
            .create(name.as_str())?;
        dataset.write_raw(record.values)?;
        write_complex_attr(&dataset)?;
        self.written += 1;
        Ok(())
    }

    /// Require every `Vq` record and close the file.
    pub fn finish(self) -> Result<(), IoError> {
        if self.written != self.n_q {
            return Err(ValidationError::InvalidValue {
                path: "/Interaction".to_owned(),
                expected: format!("{} Vq datasets", self.n_q),
                actual: self.written.to_string(),
            }
            .into());
        }
        Ok(())
    }
}

/// Read a CoQui-native single-file Cholesky container.
pub fn read_coqui_cholesky(path: impl AsRef<Path>) -> Result<CoquiCholeskyFile, IoError> {
    let file = File::open(path)?;
    let group = file.group(COQUI_CHOLESKY_GROUP)?;
    let header = CoquiCholeskyHeader {
        np: read_i32_scalar(&group, "Np")?,
        nspin: read_i32_scalar(&group, "nspin")?,
        nspin_in_basis: read_i32_scalar(&group, "nspin_in_basis")?,
        nkpts: read_i32_scalar(&group, "nkpts")?,
        nbnd: read_i32_scalar(&group, "nbnd")?,
        nbnd_aux: read_i32_scalar(&group, "nbnd_aux")?,
        tol: read_exact_f64_scalar(&group, "tol")?,
        kpts: read_exact_f64_matrix(&group, "kpts")?,
        qpts: read_exact_f64_matrix(&group, "qpts")?,
        qk_to_kmq: read_i32_matrix(&group, "qk_to_kmq")?,
    };
    header.validate()?;
    let n_q = header.n_q()?;
    let expected_len = header.vq_len()?;
    let expected_shape = header.vq_shape()?;
    let mut records = Vec::with_capacity(n_q);
    for q in 0..n_q {
        let name = format!("Vq{q}");
        let dataset = group.dataset(&name)?;
        let shape = dataset.shape();
        if shape.as_slice() != expected_shape.as_slice() {
            return Err(ValidationError::InvalidValue {
                path: format!("/Interaction/{name}"),
                expected: format!("{expected_shape:?}"),
                actual: format!("{shape:?}"),
            }
            .into());
        }
        let vq_path = format!("/Interaction/{name}");
        require_exact_dtype::<f64>(&dataset, &vq_path)?;
        require_complex_attr(&dataset, &vq_path)?;
        let values: Vec<f64> = dataset.read_raw()?;
        if values.len() != expected_len {
            return Err(ValidationError::LengthMismatch {
                path: format!("/Interaction/{name}"),
                expected: expected_len,
                actual: values.len(),
            }
            .into());
        }
        records.push(CoquiCholeskyVq { q_index: q, values });
    }
    Ok(CoquiCholeskyFile { header, records })
}

fn write_i32_scalar(group: &Group, name: &str, value: i32) -> Result<(), IoError> {
    group
        .new_dataset::<i32>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

fn write_f64_scalar(group: &Group, name: &str, value: f64) -> Result<(), IoError> {
    group
        .new_dataset::<f64>()
        .create(name)?
        .write_scalar(&value)?;
    Ok(())
}

fn write_f64_matrix(
    group: &Group,
    name: &str,
    rows: usize,
    cols: usize,
    values: &[f64],
) -> Result<(), IoError> {
    if values.len() != rows * cols {
        return Err(ValidationError::LengthMismatch {
            path: format!("/Interaction/{name}"),
            expected: rows * cols,
            actual: values.len(),
        }
        .into());
    }
    let dataset = group
        .new_dataset::<f64>()
        .shape([rows, cols])
        .create(name)?;
    dataset.write_raw(values)?;
    Ok(())
}

fn write_i32_matrix(
    group: &Group,
    name: &str,
    rows: usize,
    cols: usize,
    values: &[i32],
) -> Result<(), IoError> {
    if values.len() != rows * cols {
        return Err(ValidationError::LengthMismatch {
            path: format!("/Interaction/{name}"),
            expected: rows * cols,
            actual: values.len(),
        }
        .into());
    }
    let dataset = group
        .new_dataset::<i32>()
        .shape([rows, cols])
        .create(name)?;
    dataset.write_raw(values)?;
    Ok(())
}

fn write_complex_attr(dataset: &hdf5_metno::Dataset) -> Result<(), IoError> {
    let value = VarLenUnicode::from_str(COQUI_CHOLESKY_COMPLEX_VALUE).map_err(|_| {
        ValidationError::InvalidValue {
            path: format!("@{}", COQUI_CHOLESKY_COMPLEX_ATTR),
            expected: "variable-length UTF-8 \"1\"".to_owned(),
            actual: "unencodable".to_owned(),
        }
    })?;
    dataset
        .new_attr::<VarLenUnicode>()
        .create(COQUI_CHOLESKY_COMPLEX_ATTR)?
        .write_scalar(&value)?;
    Ok(())
}

fn require_complex_attr(dataset: &hdf5_metno::Dataset, path: &str) -> Result<(), IoError> {
    let attr_path = format!("{path}/@{}", COQUI_CHOLESKY_COMPLEX_ATTR);
    let attr = dataset
        .attr(COQUI_CHOLESKY_COMPLEX_ATTR)
        .map_err(|_| ValidationError::Missing {
            path: path.to_owned(),
            key: COQUI_CHOLESKY_COMPLEX_ATTR.to_owned(),
        })?;
    require_exact_dtype::<VarLenUnicode>(&attr, &attr_path)?;
    let value: VarLenUnicode = attr.read_scalar()?;
    if value.as_str() == COQUI_CHOLESKY_COMPLEX_VALUE {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: attr_path,
            expected: COQUI_CHOLESKY_COMPLEX_VALUE.to_owned(),
            actual: value.as_str().to_owned(),
        }
        .into())
    }
}

fn read_i32_scalar(group: &Group, name: &str) -> Result<i32, IoError> {
    let path = format!("/Interaction/{name}");
    let dataset = group.dataset(name)?;
    require_exact_dtype::<i32>(&dataset, &path)?;
    Ok(dataset.read_scalar()?)
}

fn read_exact_f64_scalar(group: &Group, name: &str) -> Result<f64, IoError> {
    let path = format!("/Interaction/{name}");
    let dataset = group.dataset(name)?;
    require_exact_dtype::<f64>(&dataset, &path)?;
    Ok(dataset.read_scalar()?)
}

fn read_exact_f64_matrix(group: &Group, name: &str) -> Result<Vec<f64>, IoError> {
    let path = format!("/Interaction/{name}");
    let dataset = group.dataset(name)?;
    require_rank2(&dataset, name)?;
    require_exact_dtype::<f64>(&dataset, &path)?;
    Ok(dataset.read_raw()?)
}

fn read_i32_matrix(group: &Group, name: &str) -> Result<Vec<i32>, IoError> {
    let path = format!("/Interaction/{name}");
    let dataset = group.dataset(name)?;
    require_rank2(&dataset, name)?;
    require_exact_dtype::<i32>(&dataset, &path)?;
    Ok(dataset.read_raw()?)
}

fn require_rank2(dataset: &hdf5_metno::Dataset, name: &str) -> Result<(), IoError> {
    let shape = dataset.shape();
    if shape.len() == 2 {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("/Interaction/{name}"),
            expected: "rank-2 C-order dataset".to_owned(),
            actual: format!("rank {}", shape.len()),
        }
        .into())
    }
}

/// Exact native HDF5 dtype guard used before every numeric (and CoQui
/// `__complex__`) read. Compares [`hdf5_metno::types::TypeDescriptor`] rather
/// than `H5Tequal`: the latter ignores string charset, so variable-length
/// ASCII would otherwise pass a `VarLenUnicode` check, and `hdf5-metno`
/// otherwise allows a soft conversion such as `f32 → f64`.
fn require_exact_dtype<T: H5Type>(object: &Container, path: &str) -> Result<(), IoError> {
    let expected = T::type_descriptor();
    let dtype = object.dtype()?;
    let actual = match dtype.to_descriptor() {
        Ok(descriptor) => descriptor,
        Err(_) => {
            return Err(ValidationError::InvalidValue {
                path: format!("{path}/dtype"),
                expected: expected.to_string(),
                actual: "unreadable HDF5 datatype".to_owned(),
            }
            .into());
        }
    };
    if actual == expected {
        Ok(())
    } else {
        Err(ValidationError::InvalidValue {
            path: format!("{path}/dtype"),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
        .into())
    }
}

fn i32_as_usize(path: &str, value: i32) -> Result<usize, ValidationError> {
    usize::try_from(value).map_err(|_| ValidationError::InvalidValue {
        path: path.to_owned(),
        expected: "nonnegative i32".to_owned(),
        actual: value.to_string(),
    })
}
