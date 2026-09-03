//! Overlap-filtered generalized Hermitian eigensolver.

use crate::OperatorError;
use muffintin_core::Hartree;
use muffintin_tensor::{
    Axis, ComplexTensor, DenseEigenvectors, DenseHermitianMatrix, TensorError, einsum,
};
use num_complex::Complex64;

/// Residual diagnostic for one generalized eigenpair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EigenpairResidual {
    pub band_index: usize,
    /// Euclidean norm of `H c - S c epsilon` in the variational coordinates.
    /// An embedded solve uses the supplied active coordinates here.
    pub absolute: f64,
    /// Absolute residual divided by `max(||Hc||, |epsilon| ||Sc||)`.
    pub relative: f64,
}

/// Result of the filtered dense Hermitian generalized eigensolve.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedEigensolution {
    /// Eigenvalues in nondecreasing Hartree order.
    pub eigenvalues: Vec<Hartree>,
    /// Column-major eigenvector columns on axes `[GlobalBasis, Band]`.
    pub eigenvectors: DenseEigenvectors,
    pub retained_dimension: usize,
    /// Overlap directions discarded from the variational input space; an
    /// explicit embedding's eliminated coordinates are not counted here.
    pub filtered_dimension: usize,
    pub residuals: Vec<EigenpairResidual>,
}

/// Independent spin-up and spin-down values for a collinear, no-SOC problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Collinear<T> {
    pub up: T,
    pub down: T,
}

impl<T> Collinear<T> {
    pub const fn new(up: T, down: T) -> Self {
        Self { up, down }
    }
}

/// Solve `H C = S C epsilon` after removing near-linearly-dependent overlap
/// directions.  An overlap eigenvalue is retained when it is positive and
/// greater than `relative_overlap_threshold * max(eigenvalue(S))`.
pub fn solve_generalized_hermitian(
    hamiltonian: &DenseHermitianMatrix,
    overlap: &DenseHermitianMatrix,
    relative_overlap_threshold: f64,
) -> Result<GeneralizedEigensolution, OperatorError> {
    use faer::{Mat, Side};

    if hamiltonian.axis() != Axis::GlobalBasis {
        return Err(OperatorError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::GlobalBasis,
            actual: hamiltonian.axis(),
        }));
    }
    if overlap.axis() != Axis::GlobalBasis {
        return Err(OperatorError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::GlobalBasis,
            actual: overlap.axis(),
        }));
    }
    if hamiltonian.dimension() != overlap.dimension() {
        return Err(OperatorError::MatrixDimensionMismatch {
            hamiltonian: hamiltonian.dimension(),
            overlap: overlap.dimension(),
        });
    }
    if !relative_overlap_threshold.is_finite() || relative_overlap_threshold < 0.0 {
        return Err(OperatorError::InvalidOverlapThreshold(
            relative_overlap_threshold,
        ));
    }
    let n = overlap.dimension();
    if n == 0 {
        return Err(OperatorError::EmptyOverlapSubspace);
    }
    let s_matrix = Mat::from_fn(n, n, |row, column| overlap.at(row, column));
    let s_eigen = s_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| OperatorError::Eigensolver)?;
    let spectral_scale = (0..n)
        .map(|index| s_eigen.S()[index].re)
        .map(f64::abs)
        .fold(0.0, f64::max);
    let cutoff = relative_overlap_threshold * spectral_scale;
    let negative_noise_tolerance = 1024.0 * f64::EPSILON * spectral_scale;
    if let Some(eigenvalue) = (0..n)
        .map(|index| s_eigen.S()[index].re)
        .find(|&eigenvalue| eigenvalue < -negative_noise_tolerance)
    {
        return Err(OperatorError::IndefiniteOverlap { eigenvalue });
    }
    let retained = (0..n)
        .filter(|&index| {
            let eigenvalue = s_eigen.S()[index].re;
            eigenvalue > 0.0 && eigenvalue > cutoff
        })
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Err(OperatorError::EmptyOverlapSubspace);
    }
    let r = retained.len();

    // X = U_keep diag(s_keep^{-1/2}), so X^H S X = I. Filtering stays here;
    // the products are einsum.
    let mut u_keep = vec![Complex64::default(); n * r];
    let mut scales = vec![Complex64::default(); r];
    for (column, &source_column) in retained.iter().enumerate() {
        scales[column] = Complex64::new(1.0 / s_eigen.S()[source_column].re.sqrt(), 0.0);
        for row in 0..n {
            u_keep[row * r + column] = s_eigen.U()[(row, source_column)];
        }
    }
    let u_keep =
        ComplexTensor::from_host_row_major(&[n, r], &[Axis::GlobalBasis, Axis::Reduced], u_keep)?;
    let scales = ComplexTensor::from_host_row_major(&[r], &[Axis::Reduced], scales)?;
    let x = einsum("ik,k->ik", &[&u_keep, &scales])?;

    let x_conj = x.conjugate();
    let reduced = DenseHermitianMatrix::from_tensor(einsum(
        "ir,ij,js->rs",
        &[&x_conj, hamiltonian.as_tensor(), &x],
    )?)?;
    let reduced_matrix = Mat::from_fn(r, r, |row, column| {
        reduced
            .get(row, column)
            .expect("reduced Hermitian block is square")
    });
    let reduced_eigen = reduced_matrix
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| OperatorError::Eigensolver)?;

    let eigenvalues = (0..r)
        .map(|band| Hartree(reduced_eigen.S()[band].re))
        .collect::<Vec<_>>();
    let mut z = vec![Complex64::default(); r * r];
    for row in 0..r {
        for band in 0..r {
            z[row * r + band] = reduced_eigen.U()[(row, band)];
        }
    }
    let z = ComplexTensor::from_host_row_major(&[r, r], &[Axis::Reduced, Axis::Band], z)?;
    let vectors = einsum("ir,rb->ib", &[&x, &z])?;

    let hc = einsum("ij,jb->ib", &[hamiltonian.as_tensor(), &vectors])?;
    let sc = einsum("ij,jb->ib", &[overlap.as_tensor(), &vectors])?;
    let epsilon = ComplexTensor::from_host_row_major(
        &[r],
        &[Axis::Band],
        eigenvalues
            .iter()
            .map(|value| Complex64::new(value.get(), 0.0))
            .collect(),
    )?;
    let sc_eps = einsum("ib,b->ib", &[&sc, &epsilon])?;
    let residual = hc.sub(&sc_eps)?;
    let residual_conj = residual.conjugate();
    let residual_sq = einsum("ib,ib->b", &[&residual_conj, &residual])?;
    let hc_conj = hc.conjugate();
    let hc_sq = einsum("ib,ib->b", &[&hc_conj, &hc])?;
    let sc_conj = sc.conjugate();
    let sc_sq = einsum("ib,ib->b", &[&sc_conj, &sc])?;
    let residuals = (0..r)
        .map(|band| {
            let absolute = residual_sq
                .get(&[band])
                .expect("band residual")
                .re
                .max(0.0)
                .sqrt();
            let hc_norm = hc_sq.get(&[band]).expect("Hc norm").re.max(0.0).sqrt();
            let sc_norm = sc_sq.get(&[band]).expect("Sc norm").re.max(0.0).sqrt();
            let denominator = hc_norm.max(eigenvalues[band].get().abs() * sc_norm);
            EigenpairResidual {
                band_index: band,
                absolute,
                relative: if denominator == 0.0 {
                    absolute
                } else {
                    absolute / denominator
                },
            }
        })
        .collect();

    Ok(GeneralizedEigensolution {
        eigenvalues,
        eigenvectors: DenseEigenvectors::from_tensor(vectors)?,
        retained_dimension: r,
        filtered_dimension: n - r,
        residuals,
    })
}

/// Solve in the explicitly allowed space `C = T Z`, retaining the physical
/// global eigenvectors for density and operator contractions.
///
/// `embedding` has axes `[GlobalBasis, Reduced]`. The returned residuals and
/// filtering counts refer to `T^H H T Z = T^H S T Z epsilon`; forbidden-space
/// components of the unconstrained residual are Lagrange forces, not errors.
pub fn solve_generalized_hermitian_embedded(
    hamiltonian: &DenseHermitianMatrix,
    overlap: &DenseHermitianMatrix,
    embedding: &ComplexTensor,
    relative_overlap_threshold: f64,
) -> Result<GeneralizedEigensolution, OperatorError> {
    if embedding.shape().len() != 2 {
        return Err(TensorError::Rank {
            expected: 2,
            actual: embedding.shape().len(),
        }
        .into());
    }
    for (index, expected) in [Axis::GlobalBasis, Axis::Reduced].into_iter().enumerate() {
        if embedding.axes()[index] != expected {
            return Err(TensorError::Axis {
                index,
                expected,
                actual: embedding.axes()[index],
            }
            .into());
        }
    }
    let dimension = embedding.shape()[1];
    let conjugate = embedding.conjugate();
    let reduce = |matrix: &DenseHermitianMatrix| -> Result<DenseHermitianMatrix, OperatorError> {
        let projected = einsum("ia,ij,jb->ab", &[&conjugate, matrix.as_tensor(), embedding])?;
        Ok(DenseHermitianMatrix::from_host_row_major(
            dimension,
            Axis::GlobalBasis,
            projected.to_host_row_major(),
        )?)
    };
    let mut solved = solve_generalized_hermitian(
        &reduce(hamiltonian)?,
        &reduce(overlap)?,
        relative_overlap_threshold,
    )?;
    let coefficients = ComplexTensor::from_host_row_major(
        &[dimension, solved.retained_dimension],
        &[Axis::Reduced, Axis::Band],
        solved.eigenvectors.as_tensor().to_host_row_major(),
    )?;
    solved.eigenvectors =
        DenseEigenvectors::from_tensor(einsum("ia,ab->ib", &[embedding, &coefficients])?)?;
    Ok(solved)
}

/// Lift a Hermitian band-space feedback operator into the original global
/// nonorthogonal basis as `S C K C^H S`.
///
/// `C` must contain `S`-orthonormal generalized-eigenvector columns. The
/// returned matrix can be added to the original global Hamiltonian before a
/// fresh generalized solve; it must not be added to already shifted band
/// eigenvalues.
pub fn lift_band_hermitian_feedback(
    overlap: &DenseHermitianMatrix,
    eigenvectors: &DenseEigenvectors,
    feedback: &DenseHermitianMatrix,
) -> Result<DenseHermitianMatrix, OperatorError> {
    if overlap.axis() != Axis::GlobalBasis {
        return Err(OperatorError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::GlobalBasis,
            actual: overlap.axis(),
        }));
    }
    if feedback.axis() != Axis::Band {
        return Err(OperatorError::Tensor(TensorError::Axis {
            index: 0,
            expected: Axis::Band,
            actual: feedback.axis(),
        }));
    }
    if overlap.dimension() != eigenvectors.rows() {
        return Err(OperatorError::EigenvectorBasisCount {
            expected: overlap.dimension(),
            actual: eigenvectors.rows(),
        });
    }
    if feedback.dimension() != eigenvectors.columns() {
        return Err(OperatorError::BandFeedbackDimensionMismatch {
            feedback: feedback.dimension(),
            bands: eigenvectors.columns(),
        });
    }
    let sc = einsum(
        "ij,jb->ib",
        &[overlap.as_tensor(), eigenvectors.as_tensor()],
    )?;
    let sc_conjugate = sc.conjugate();
    Ok(DenseHermitianMatrix::from_tensor(einsum(
        "ib,bc,jc->ij",
        &[&sc, feedback.as_tensor(), &sc_conjugate],
    )?)?)
}

/// Real-symmetric eigendecomposition of a host matrix.
///
/// `element(row, column)` is queried for the upper triangle, including the
/// diagonal. Eigenvalues are nondecreasing. Eigenvectors are real and
/// column-major.
#[derive(Clone, Debug, PartialEq)]
pub struct RealSymmetricEigensolution {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<f64>,
    pub dimension: usize,
}

/// Diagonalize a real symmetric matrix without an overlap cutoff.
pub fn solve_real_symmetric(
    dimension: usize,
    mut element: impl FnMut(usize, usize) -> f64,
) -> Result<RealSymmetricEigensolution, OperatorError> {
    use faer::{Mat, Side};

    if dimension == 0 {
        return Err(OperatorError::EmptyOverlapSubspace);
    }
    let mut values = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in row..dimension {
            let value = element(row, column);
            if !value.is_finite() {
                return Err(OperatorError::Eigensolver);
            }
            values[row * dimension + column] = value;
            values[column * dimension + row] = value;
        }
    }
    let packed = Mat::<f64>::from_fn(dimension, dimension, |row, column| {
        values[row * dimension + column]
    });
    let eigen = packed
        .self_adjoint_eigen(Side::Lower)
        .map_err(|_| OperatorError::Eigensolver)?;
    let mut eigenvalues = Vec::with_capacity(dimension);
    let mut eigenvectors = vec![0.0; dimension * dimension];
    for column in 0..dimension {
        eigenvalues.push(eigen.S()[column]);
        for row in 0..dimension {
            eigenvectors[row + column * dimension] = eigen.U()[(row, column)];
        }
    }
    Ok(RealSymmetricEigensolution {
        eigenvalues,
        eigenvectors,
        dimension,
    })
}
