//! Scalar full-BZ CoQui-native Cholesky adapter from sampled-$\zeta$ Coulomb.
//!
//! This is **not** MLDUMP and does not claim q-dependent THC compatibility with
//! CoQui. Live reader contract: CoQui `chol_reader_t.hpp` and GF2
//! `build_int` (`(pr|qs)=\sum_Q L_{Qpr}\mathrm{conj}(L_{Qsq})`) at
//! `<coqui-inspect-checkout>` `wg-dev` @
//! `a19774d03fb979bd852fae4f7f95c045a4cbca78`.

use std::path::Path;

use muffintin_auxiliary_ir::OrbitalPair;
use muffintin_io::{
    CoquiCholeskyHeader, CoquiCholeskyVqRef, CoquiCholeskyWriter, IoError, ValidationError,
};
use muffintin_thc::{L2Engine, SelectorStrategy};
use num_complex::Complex64;
use thiserror::Error;

use crate::scalar_coulomb::{
    ScalarCoulombError, ScalarCoulombResult, ScalarCoulombSpec,
    require_scalar_coulomb_export_context,
};
use crate::scalar_product::{ScalarProductInput, ScalarQSliceError, require_scalar_q_slice};
use crate::scalar_thc::ScalarThcResult;

/// Scale-aware floor for the stored Hermitian body, independent of factor `tol`.
const HERMITIAN_EQ_TOLERANCE: f64 = 1.0e-12;

/// Explicit factor tolerance written to `/Interaction/tol`.
///
/// There is no default, auto rank, or compatibility wrapper. A Hermitian
/// diagonal pivot `d` is a new factor row if `d > tolerance`, terminating
/// roundoff if `-tolerance <= d <= tolerance`, and a material negative
/// eigenvalue if `d < -tolerance`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarCoquiCholeskySpec {
    pub tolerance: f64,
}

/// Failure while preflighting or writing a CoQui-native Cholesky file.
#[derive(Debug, Error)]
pub enum ScalarCoquiCholeskyError {
    #[error(transparent)]
    Io(#[from] IoError),
    #[error(transparent)]
    Coulomb(#[from] ScalarCoulombError),
    #[error("CoQui Cholesky tolerance must be finite and nonnegative, got {0}")]
    Tolerance(f64),
    #[error("CoQui Cholesky Coulomb body at q-index {q_index} is not finite")]
    NonFiniteBody { q_index: usize },
    #[error("CoQui Cholesky Coulomb body at q-index {q_index} is not Hermitian")]
    NonHermitian { q_index: usize },
    #[error(
        "CoQui Cholesky pivot {pivot} at q-index {q_index} is materially negative for tolerance {tolerance}"
    )]
    NegativePivot {
        q_index: usize,
        pivot: f64,
        tolerance: f64,
    },
    #[error("CoQui Cholesky produced no positive factor rows")]
    EmptyFactor,
    #[error("CoQui Cholesky q-index {q_index} k-index {k_index} map is out of order or bounds")]
    KMap { q_index: usize, k_index: usize },
    #[error(
        "CoQui Cholesky vertex column {column} at q-index {q_index} is not the k-major Bloch pair"
    )]
    VertexColumn { q_index: usize, column: usize },
}

impl From<ScalarQSliceError> for ScalarCoquiCholeskyError {
    fn from(error: ScalarQSliceError) -> Self {
        Self::Coulomb(error.into())
    }
}

impl From<ValidationError> for ScalarCoquiCholeskyError {
    fn from(error: ValidationError) -> Self {
        Self::Io(error.into())
    }
}

/// Write a CoQui-native single-file Cholesky ERI from frozen scalar objects.
///
/// All recoverable q-slice, Coulomb-export, mapping, Hermitian, and pivot
/// failures occur before the HDF5 destination is created. Each $q$ is factored so
/// $V_q=B_q^\dagger B_q$ with $B_q$ row-major `(rank, n_aux)` in Rust memory:
/// $(B^\dagger B)_{ij}=\sum_Q \mathrm{conj}(B_{Qi})B_{Qj}$. Semantic vertices
/// are applied as $L_{Q,0,k,i,j}=(B_q c_{k,i,j})_Q$ in k-major
/// `PairColumnLayout` order (left at $k-q$, right at $k$), matching CoQui GF2
/// $\sum_Q L_{Qpr}\mathrm{conj}(L_{Qsq})$. Gamma contributes only the finite
/// body; the singular head is not factored. Scratch is one $q$ factor and one
/// $q$ $L$ tensor; lower-rank $q$ are zero-padded to the file-global `Np`.
pub fn write_scalar_coqui_cholesky(
    path: &Path,
    inputs: &[ScalarProductInput],
    thc: &ScalarThcResult,
    coulomb: &ScalarCoulombResult,
    coulomb_spec: &ScalarCoulombSpec,
    factor: ScalarCoquiCholeskySpec,
) -> Result<(), ScalarCoquiCholeskyError> {
    let prepared = preflight_scalar_coqui_cholesky(inputs, thc, coulomb, coulomb_spec, factor)?;
    let mut writer = CoquiCholeskyWriter::create(path, &prepared.header)?;
    let mut l_scratch = vec![0.0; prepared.header.vq_len()?];
    for (q, record) in coulomb.records.iter().enumerate() {
        let factored = factor_hermitian_psd(
            record.operator.matrix(),
            record.operator.dimension(),
            factor.tolerance,
            q,
        )?;
        pack_vq(
            &factored,
            record,
            prepared.n_k,
            prepared.n_band,
            prepared.np,
            &mut l_scratch,
        )?;
        writer.write_vq(CoquiCholeskyVqRef {
            q_index: q,
            values: &l_scratch,
        })?;
    }
    writer.finish()?;
    Ok(())
}

struct PreparedCoqui {
    header: CoquiCholeskyHeader,
    n_k: usize,
    n_band: usize,
    np: usize,
}

fn preflight_scalar_coqui_cholesky(
    inputs: &[ScalarProductInput],
    thc: &ScalarThcResult,
    coulomb: &ScalarCoulombResult,
    coulomb_spec: &ScalarCoulombSpec,
    factor: ScalarCoquiCholeskySpec,
) -> Result<PreparedCoqui, ScalarCoquiCholeskyError> {
    if !factor.tolerance.is_finite() || factor.tolerance < 0.0 {
        return Err(ScalarCoquiCholeskyError::Tolerance(factor.tolerance));
    }
    let first = require_scalar_q_slice(inputs)?;
    require_scalar_coulomb_export_context(inputs, thc, coulomb, coulomb_spec)?;
    if thc.selection.provenance.strategy != SelectorStrategy::AllQL2 {
        return Err(ScalarCoulombError::UnsupportedStrategy.into());
    }
    match thc.selection.provenance.engine {
        L2Engine::FullColumnPivotedQr | L2Engine::FullPivotedCholesky => {}
        other => return Err(ScalarCoulombError::UnsupportedEngine(other).into()),
    }
    let n_k = first.orbitals.k_fractional.len();
    let n_band = first.orbitals.band_window.count;
    if first.pair_columns.n_k != n_k || first.pair_columns.n_orb != n_band {
        return Err(ScalarCoulombError::IncompatibleInputs.into());
    }
    let n_columns = first
        .pair_columns
        .n_columns()
        .map_err(ScalarCoulombError::from)?;
    let mut kpts = Vec::with_capacity(n_k * 3);
    for fractional in &first.orbitals.k_fractional {
        kpts.extend(cartesian_from_fractional(
            first.reciprocal.basis(),
            *fractional,
        ));
    }
    if kpts.iter().any(|value| !value.is_finite()) {
        return Err(ScalarCoulombError::IncompatibleInputs.into());
    }
    let mut qpts = Vec::with_capacity(inputs.len() * 3);
    let mut qk_to_kmq = Vec::with_capacity(inputs.len() * n_k);
    for (iq, input) in inputs.iter().enumerate() {
        if input.k_minus_q.len() != n_k {
            return Err(ScalarCoquiCholeskyError::KMap {
                q_index: iq,
                k_index: 0,
            });
        }
        for component in input.source.q.cartesian {
            if !component.get().is_finite() {
                return Err(ScalarCoquiCholeskyError::NonFiniteBody { q_index: iq });
            }
            qpts.push(component.get());
        }
        for (k, mapped) in input.k_minus_q.iter().enumerate() {
            if mapped.k_index != k || mapped.kq_index >= n_k {
                return Err(ScalarCoquiCholeskyError::KMap {
                    q_index: iq,
                    k_index: k,
                });
            }
            qk_to_kmq.push(i32::try_from(mapped.kq_index).map_err(|_| {
                ScalarCoquiCholeskyError::KMap {
                    q_index: iq,
                    k_index: k,
                }
            })?);
        }
        let record = &coulomb.records[iq];
        if record.vertices.len() != n_columns {
            return Err(ScalarCoquiCholeskyError::VertexColumn {
                q_index: iq,
                column: record.vertices.len(),
            });
        }
        for (column, vertex) in record.vertices.iter().enumerate() {
            match vertex.pair() {
                OrbitalPair::Bloch {
                    k_index,
                    left,
                    right,
                } if record.layout.decode(column) == (k_index, left, right)
                    && vertex.coefficients().len() == record.operator.dimension() => {}
                _ => {
                    return Err(ScalarCoquiCholeskyError::VertexColumn {
                        q_index: iq,
                        column,
                    });
                }
            }
        }
    }
    let mut np = 0usize;
    for (q, record) in coulomb.records.iter().enumerate() {
        let rank = factor_hermitian_psd(
            record.operator.matrix(),
            record.operator.dimension(),
            factor.tolerance,
            q,
        )?
        .rank;
        np = np.max(rank);
    }
    if np == 0 {
        return Err(ScalarCoquiCholeskyError::EmptyFactor);
    }
    let header = CoquiCholeskyHeader {
        np: usize_as_i32("/Interaction/Np", np)?,
        nspin: 1,
        nspin_in_basis: 1,
        nkpts: usize_as_i32("/Interaction/nkpts", n_k)?,
        nbnd: usize_as_i32("/Interaction/nbnd", n_band)?,
        nbnd_aux: 0,
        tol: factor.tolerance,
        kpts,
        qpts,
        qk_to_kmq,
    };
    header.validate()?;
    Ok(PreparedCoqui {
        header,
        n_k,
        n_band,
        np,
    })
}

#[derive(Debug)]
struct CholeskyFactor {
    rank: usize,
    n: usize,
    /// Row-major `(rank, n)` with $V=B^\dagger B$.
    b: Vec<Complex64>,
}

fn factor_hermitian_psd(
    matrix: &[Complex64],
    n: usize,
    tolerance: f64,
    q_index: usize,
) -> Result<CholeskyFactor, ScalarCoquiCholeskyError> {
    if n == 0 || matrix.len() != n * n {
        return Err(ScalarCoquiCholeskyError::NonHermitian { q_index });
    }
    require_hermitian_finite(matrix, n, q_index)?;
    let mut a = matrix.to_vec();
    let mut remaining = (0..n).collect::<Vec<_>>();
    let mut rows = Vec::new();
    while !remaining.is_empty() {
        let mut best_slot = 0;
        let mut best_index = remaining[0];
        let mut best_diag = a[best_index * n + best_index].re;
        for (slot, &index) in remaining.iter().enumerate().skip(1) {
            let diag = a[index * n + index].re;
            if diag > best_diag {
                best_slot = slot;
                best_index = index;
                best_diag = diag;
            }
        }
        if best_diag < -tolerance {
            return Err(ScalarCoquiCholeskyError::NegativePivot {
                q_index,
                pivot: best_diag,
                tolerance,
            });
        }
        if best_diag <= tolerance {
            break;
        }
        remaining.swap_remove(best_slot);
        let scale = best_diag.sqrt();
        let mut row = vec![Complex64::default(); n];
        row[best_index] = Complex64::new(scale, 0.0);
        for &index in &remaining {
            row[index] = a[best_index * n + index] / scale;
        }
        for (slot_i, &i) in remaining.iter().enumerate() {
            for &j in remaining.iter().skip(slot_i) {
                let update = row[i].conj() * row[j];
                a[i * n + j] -= update;
                if i != j {
                    a[j * n + i] -= update.conj();
                }
            }
        }
        rows.push(row);
    }
    let rank = rows.len();
    let mut b = vec![Complex64::default(); rank * n];
    for (q, row) in rows.iter().enumerate() {
        b[q * n..(q + 1) * n].copy_from_slice(row);
    }
    Ok(CholeskyFactor { rank, n, b })
}

fn require_hermitian_finite(
    matrix: &[Complex64],
    n: usize,
    q_index: usize,
) -> Result<(), ScalarCoquiCholeskyError> {
    for row in 0..n {
        for column in row..n {
            let value = matrix[row * n + column];
            let conjugate = matrix[column * n + row].conj();
            if !value.re.is_finite()
                || !value.im.is_finite()
                || !conjugate.re.is_finite()
                || !conjugate.im.is_finite()
            {
                return Err(ScalarCoquiCholeskyError::NonFiniteBody { q_index });
            }
            let scale = value.norm().max(conjugate.norm()).max(1.0);
            if (value - conjugate).norm() > HERMITIAN_EQ_TOLERANCE * scale {
                return Err(ScalarCoquiCholeskyError::NonHermitian { q_index });
            }
        }
    }
    Ok(())
}

fn pack_vq(
    factor: &CholeskyFactor,
    record: &crate::scalar_coulomb::ScalarCoulombQRecord,
    n_k: usize,
    n_band: usize,
    np: usize,
    out: &mut [f64],
) -> Result<(), ScalarCoquiCholeskyError> {
    let expected = np
        .checked_mul(n_k)
        .and_then(|value| value.checked_mul(n_band))
        .and_then(|value| value.checked_mul(n_band))
        .and_then(|value| value.checked_mul(2))
        .ok_or(ScalarCoquiCholeskyError::EmptyFactor)?;
    if out.len() != expected {
        return Err(ScalarCoquiCholeskyError::EmptyFactor);
    }
    out.fill(0.0);
    let n_aux = factor.n;
    for (column, vertex) in record.vertices.iter().enumerate() {
        let (k, i, j) = record.layout.decode(column);
        if k >= n_k || i >= n_band || j >= n_band {
            return Err(ScalarCoquiCholeskyError::VertexColumn {
                q_index: record.q_index,
                column,
            });
        }
        let coefficients = vertex.coefficients();
        for q in 0..factor.rank {
            let mut acc = Complex64::default();
            let row = &factor.b[q * n_aux..(q + 1) * n_aux];
            for (mu, coefficient) in coefficients.iter().enumerate() {
                acc += row[mu] * coefficient;
            }
            let base = ((q * n_k + k) * n_band + i) * n_band + j;
            let slot = base * 2;
            out[slot] = acc.re;
            out[slot + 1] = acc.im;
        }
    }
    Ok(())
}

fn cartesian_from_fractional(
    reciprocal: &[[muffintin_core::InverseBohr; 3]; 3],
    fractional: [f64; 3],
) -> [f64; 3] {
    std::array::from_fn(|axis| {
        fractional
            .iter()
            .zip(reciprocal.iter())
            .map(|(&coefficient, vector)| coefficient * vector[axis].get())
            .sum()
    })
}

fn usize_as_i32(path: &str, value: usize) -> Result<i32, ScalarCoquiCholeskyError> {
    i32::try_from(value).map_err(|_| {
        ValidationError::InvalidValue {
            path: path.to_owned(),
            expected: "i32".to_owned(),
            actual: value.to_string(),
        }
        .into()
    })
}

#[cfg(test)]
mod factor_oracles {
    use super::{HERMITIAN_EQ_TOLERANCE, factor_hermitian_psd};
    use num_complex::Complex64;

    #[test]
    fn pivot_equality_boundaries_follow_tolerance_policy() {
        let tol = 1.0e-8;
        let zero = factor_hermitian_psd(&[Complex64::new(-tol, 0.0)], 1, tol, 0).unwrap();
        assert_eq!(zero.rank, 0);
        let plus = factor_hermitian_psd(&[Complex64::new(tol, 0.0)], 1, tol, 0).unwrap();
        assert_eq!(plus.rank, 0);
        let accepted =
            factor_hermitian_psd(&[Complex64::new(tol + 1.0e-12, 0.0)], 1, tol, 0).unwrap();
        assert_eq!(accepted.rank, 1);
        let err =
            factor_hermitian_psd(&[Complex64::new(-tol - 1.0e-12, 0.0)], 1, tol, 3).unwrap_err();
        match err {
            super::ScalarCoquiCholeskyError::NegativePivot {
                q_index: 3,
                pivot,
                tolerance,
            } => {
                assert_eq!(tolerance, tol);
                assert!(pivot < -tol);
            }
            other => panic!("expected negative pivot, got {other}"),
        }
    }

    #[test]
    fn factor_satisfies_dagger_b_on_a_rank_two_hermitian() {
        let n = 2;
        let matrix = [
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, 0.0),
        ];
        let factor = factor_hermitian_psd(&matrix, n, 1.0e-12, 0).unwrap();
        assert_eq!(factor.rank, 2);
        for row in 0..n {
            for column in 0..n {
                let mut acc = Complex64::default();
                for q in 0..factor.rank {
                    acc += factor.b[q * n + row].conj() * factor.b[q * n + column];
                }
                let delta = (acc - matrix[row * n + column]).norm();
                assert!(
                    delta <= HERMITIAN_EQ_TOLERANCE
                        || delta / matrix[row * n + column].norm().max(1.0)
                            <= HERMITIAN_EQ_TOLERANCE,
                    "V=B^H B failed at {row},{column}: {acc} vs {}",
                    matrix[row * n + column]
                );
            }
        }
    }
}
