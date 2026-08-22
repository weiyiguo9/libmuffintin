//! Deterministic QRCP, pivoted Cholesky, least squares, and Hermitian square roots.

use crate::ThcError;
use faer::{Mat, Side};
use num_complex::Complex64;

/// Relative singular-value cutoff matching NumPy `lstsq(..., rcond=1e-12)`.
const LSTSQ_RCOND: f64 = 1.0e-12;

/// Column-pivoted QR of a row-major `nrows × ncols` matrix.
///
/// Returned pivots `p` satisfy $A P = QR$ with `p[j]` the original column
/// moved to position $j$, matching SciPy `qr(..., pivoting=True)`.
pub fn column_pivots(
    matrix: &[Complex64],
    nrows: usize,
    ncols: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    if nrows == 0 || ncols == 0 {
        return Err(ThcError::LinearAlgebra("empty QRCP matrix"));
    }
    if matrix.len() != nrows * ncols {
        return Err(ThcError::PairBlockLength {
            expected: nrows * ncols,
            actual: matrix.len(),
        });
    }
    if matrix
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ThcError::LinearAlgebra("non-finite QRCP entry"));
    }
    let mat = Mat::<Complex64>::from_fn(nrows, ncols, |row, column| matrix[row * ncols + column]);
    let qr = mat.col_piv_qr();
    let (forward, _) = qr.P().arrays();
    let pivots = forward.to_vec();
    let rank = nrows.min(ncols);
    let r = qr.R();
    let r_diag = (0..rank).map(|index| r[(index, index)].norm()).collect();
    Ok((pivots, r_diag))
}

/// Matrix-free pivoted Cholesky of the column Gram $A^\dagger A$.
///
/// The input is a row-major `nrows × ncols` matrix. The routine never forms
/// the dense `ncols × ncols` Gram: each selected Gram column is contracted
/// directly from `matrix`. Returned diagonals are square roots of the Gram
/// residual pivots, so they have the same scale as the QRCP $|R_{kk}|$
/// diagnostics consumed by [`crate::select::RankPolicy`].
pub fn pivoted_cholesky_pivots(
    matrix: &[Complex64],
    nrows: usize,
    ncols: usize,
    n_pivots: usize,
) -> Result<(Vec<usize>, Vec<f64>), ThcError> {
    if nrows == 0 || ncols == 0 || n_pivots == 0 {
        return Err(ThcError::LinearAlgebra("empty pivoted-Cholesky matrix"));
    }
    if matrix.len() != nrows * ncols {
        return Err(ThcError::PairBlockLength {
            expected: nrows * ncols,
            actual: matrix.len(),
        });
    }
    if matrix
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ThcError::LinearAlgebra("non-finite pivoted-Cholesky entry"));
    }

    let rank = n_pivots.min(ncols);
    let mut residual = vec![0.0_f64; ncols];
    for column in 0..ncols {
        residual[column] = (0..nrows)
            .map(|row| matrix[row * ncols + column].norm_sqr())
            .sum();
        if !residual[column].is_finite() {
            return Err(ThcError::LinearAlgebra(
                "non-finite pivoted-Cholesky diagonal",
            ));
        }
    }
    let leading_scale = residual.iter().copied().fold(0.0_f64, f64::max);

    let mut factors = vec![Complex64::default(); ncols * rank];
    let mut selected = vec![false; ncols];
    let mut pivots = Vec::with_capacity(rank);
    let mut diagonal = Vec::with_capacity(rank);
    for step in 0..rank {
        let mut pivot = None;
        for column in 0..ncols {
            if selected[column] {
                continue;
            }
            if pivot.is_none_or(|current| residual[column] > residual[current]) {
                pivot = Some(column);
            }
        }
        let pivot = pivot.expect("rank is capped by the number of columns");
        let pivot_value = residual[pivot].max(0.0);
        let pivot_sqrt = pivot_value.sqrt();
        selected[pivot] = true;
        pivots.push(pivot);
        diagonal.push(pivot_sqrt);

        if pivot_sqrt == 0.0 {
            continue;
        }
        factors[pivot * rank + step] = Complex64::new(pivot_sqrt, 0.0);
        for column in 0..ncols {
            if selected[column] {
                continue;
            }
            let mut gram = Complex64::default();
            for row in 0..nrows {
                gram += matrix[row * ncols + column].conj() * matrix[row * ncols + pivot];
            }
            for previous in 0..step {
                gram -= factors[column * rank + previous] * factors[pivot * rank + previous].conj();
            }
            let factor = gram / pivot_sqrt;
            if !factor.re.is_finite() || !factor.im.is_finite() {
                return Err(ThcError::LinearAlgebra(
                    "non-finite pivoted-Cholesky factor",
                ));
            }
            factors[column * rank + step] = factor;
            let updated = residual[column] - factor.norm_sqr();
            if !updated.is_finite() {
                return Err(ThcError::LinearAlgebra(
                    "non-finite pivoted-Cholesky residual",
                ));
            }
            let tolerance = 64.0 * f64::EPSILON * leading_scale * (nrows + step + 1) as f64;
            if updated < -tolerance {
                return Err(ThcError::LinearAlgebra(
                    "negative pivoted-Cholesky residual",
                ));
            }
            residual[column] = updated.max(0.0);
        }
        residual[pivot] = 0.0;
    }
    Ok((pivots, diagonal))
}

/// Least squares `min ||A X - B||` for tall or square $A$ (`m × n`), $B$ (`m × nrhs`).
///
/// $A$ and $B$ are row-major. The result is row-major $n × nrhs$. Singular
/// values smaller than $10^{-12}$ times the largest are treated as zero,
/// matching NumPy `lstsq(..., rcond=1e-12)` used by
/// `scratch/thc_lapw_end_to_end_test.py`. Unpivoted QR is not used: the
/// source-equivalent collocation is rank-deficient enough that an
/// untruncated solve is non-finite.
pub fn lstsq(
    a: &[Complex64],
    a_rows: usize,
    a_cols: usize,
    b: &[Complex64],
    b_cols: usize,
) -> Result<Vec<Complex64>, ThcError> {
    if a_rows == 0 || a_cols == 0 || b_cols == 0 {
        return Err(ThcError::LinearAlgebra("empty least-squares system"));
    }
    if a.len() != a_rows * a_cols || b.len() != a_rows * b_cols {
        return Err(ThcError::LinearAlgebra("least-squares shape"));
    }
    if a_rows < a_cols {
        return Err(ThcError::LinearAlgebra(
            "least-squares requires at least as many rows as columns",
        ));
    }
    if a.iter()
        .chain(b.iter())
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ThcError::LinearAlgebra("non-finite least-squares entry"));
    }
    let a_mat = Mat::<Complex64>::from_fn(a_rows, a_cols, |row, column| a[row * a_cols + column]);
    let b_mat = Mat::<Complex64>::from_fn(a_rows, b_cols, |row, column| b[row * b_cols + column]);
    let svd = a_mat
        .thin_svd()
        .map_err(|_| ThcError::LinearAlgebra("SVD least-squares"))?;
    let rank = a_rows.min(a_cols);
    let u = svd.U();
    let v = svd.V();
    let singular = svd.S();
    let s_max = (0..rank)
        .map(|index| singular[index].re.max(0.0))
        .fold(0.0_f64, f64::max);
    let cutoff = LSTSQ_RCOND * s_max;
    let mut tmp = vec![Complex64::default(); rank * b_cols];
    for k in 0..rank {
        let sigma = singular[k].re.max(0.0);
        let scale = if sigma > cutoff { 1.0 / sigma } else { 0.0 };
        for column in 0..b_cols {
            let mut acc = Complex64::default();
            for row in 0..a_rows {
                acc += u[(row, k)].conj() * b_mat[(row, column)];
            }
            tmp[k * b_cols + column] = acc * scale;
        }
    }
    let mut out = vec![Complex64::default(); a_cols * b_cols];
    for row in 0..a_cols {
        for column in 0..b_cols {
            let mut acc = Complex64::default();
            for k in 0..rank {
                acc += v[(row, k)] * tmp[k * b_cols + column];
            }
            out[row * b_cols + column] = acc;
        }
    }
    if out
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(ThcError::LinearAlgebra("non-finite least-squares solution"));
    }
    Ok(out)
}

/// Hermitian square root of a row-major $n × n$ matrix, clipping tiny negative
/// eigenvalues to zero.
pub fn hermitian_sqrt(matrix: &[Complex64], n: usize) -> Result<Vec<Complex64>, ThcError> {
    let (values, vectors) = hermitian_eigensystem(matrix, n)?;
    let mut sqrt_diag = vec![0.0; n];
    for (index, &value) in values.iter().enumerate() {
        sqrt_diag[index] = value.max(0.0).sqrt();
    }
    let mut out = vec![Complex64::default(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::default();
            for k in 0..n {
                sum += vectors[i * n + k] * sqrt_diag[k] * vectors[j * n + k].conj();
            }
            out[i * n + j] = sum;
        }
    }
    Ok(out)
}

/// Eigenvalues (nondecreasing) and column-major-as-row-major $U$ of a Hermitian matrix.
pub fn hermitian_eigensystem(
    matrix: &[Complex64],
    n: usize,
) -> Result<(Vec<f64>, Vec<Complex64>), ThcError> {
    if n == 0 {
        return Err(ThcError::LinearAlgebra("empty Hermitian eigenproblem"));
    }
    if matrix.len() != n * n {
        return Err(ThcError::LinearAlgebra("Hermitian shape"));
    }
    let packed = Mat::<Complex64>::from_fn(n, n, |row, column| {
        0.5 * (matrix[row * n + column] + matrix[column * n + row].conj())
    });
    let eigen = packed
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| ThcError::LinearAlgebra("Hermitian eigensolver"))?;
    let mut values = Vec::with_capacity(n);
    let mut vectors = vec![Complex64::default(); n * n];
    for column in 0..n {
        values.push(eigen.S()[column].re);
        for row in 0..n {
            vectors[row * n + column] = eigen.U()[(row, column)];
        }
    }
    Ok((values, vectors))
}

/// Frobenius norm of a complex vector.
pub fn frobenius(values: &[Complex64]) -> f64 {
    values
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}
